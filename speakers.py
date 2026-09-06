from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MIN_SEG_SEC = 0.3
MAX_SEG_SEC = 2.0
STREAM_CHUNK_SEC = 0.4

VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
EMB_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx"
)
VAD_NAME = "silero_vad.onnx"
EMB_NAME = "wespeaker_en_voxceleb_CAM++.onnx"


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def tracking_enabled() -> bool:
    return _env_bool("SPEAKER_TRACKING", True)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 100_000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def ensure_models(models_dir: Path, *, need_embedding: bool = True) -> tuple[Path, Path | None]:
    vad = _download(VAD_URL, models_dir / VAD_NAME)
    emb = _download(EMB_URL, models_dir / EMB_NAME) if need_embedding else None
    return vad, emb


def _normalize(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(out))
    if norm < 1e-8:
        return out
    return out / norm


class SpeechVad:
    """sherpa-onnx Silero VAD; feed chunks, yield completed speech segments."""

    def __init__(self, model_path: Path, sample_rate: int = SAMPLE_RATE):
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model_path)
        config.silero_vad.min_silence_duration = 0.15
        config.silero_vad.min_speech_duration = 0.25
        config.sample_rate = sample_rate
        self.window_size = int(config.silero_vad.window_size)
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self.buf = np.zeros((0,), dtype=np.float32)
        self._parts: list[np.ndarray] = []
        self._n = 0
        self.sample_rate = sample_rate
        self.min_samples = int(MIN_SEG_SEC * sample_rate)
        self.max_samples = int(MAX_SEG_SEC * sample_rate)
        chunk_sec = float(os.environ.get("STT_CHUNK_SEC", str(STREAM_CHUNK_SEC)))
        self.chunk_samples = int(chunk_sec * sample_rate) if chunk_sec > 0 else 0
        self._emitted = 0

    def _take(self, n: int | None = None) -> np.ndarray:
        blob = np.concatenate(self._parts) if self._parts else np.zeros((0,), dtype=np.float32)
        if n is None or n >= blob.size:
            self._parts = []
            self._n = 0
            return blob
        out, rest = blob[:n], blob[n:]
        self._parts = [rest] if rest.size else []
        self._n = rest.size
        return out

    def accept(self, samples: np.ndarray) -> list[np.ndarray]:
        chunk = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []
        if self.buf.size:
            chunk = np.concatenate([self.buf, chunk])
        segs: list[np.ndarray] = []
        i = 0
        n = chunk.size
        w = self.window_size
        while i + w < n:
            win = chunk[i : i + w]
            i += w
            self.vad.accept_waveform(win)
            while not self.vad.empty():
                self.vad.pop()
            if self.vad.is_speech_detected():
                self._parts.append(win)
                self._n += win.size
                if self.chunk_samples > 0:
                    while self._n > self.max_samples:
                        old = self._parts.pop(0)
                        self._n -= old.size
                        self._emitted = max(0, self._emitted - old.size)
                    if self._n >= self.min_samples and self._n - self._emitted >= self.chunk_samples:
                        segs.append(np.concatenate(self._parts))
                        self._emitted = self._n
                else:
                    while self._n >= self.max_samples:
                        segs.append(self._take(self.max_samples))
            else:
                if self._n >= self.min_samples:
                    segs.append(self._take())
                else:
                    self._parts = []
                    self._n = 0
                self._emitted = 0
        self.buf = chunk[i:]
        return segs

    def flush(self) -> list[np.ndarray]:
        if hasattr(self.vad, "flush"):
            self.vad.flush()
        while not self.vad.empty():
            self.vad.pop()
        if self._n >= self.min_samples:
            segs = [self._take()]
            self._emitted = 0
            return segs
        self._parts = []
        self._n = 0
        self._emitted = 0
        return []


class OnlineSpeakerTracker:
    """Online IDs per device: MIC-0/MIC-1 and EXT-0/EXT-1 (separate pools)."""

    def __init__(
        self,
        model_path: Path,
        *,
        threshold: float | None = None,
        max_speakers: int | None = None,
        num_threads: int = 2,
    ):
        import time
        import sherpa_onnx

        self.threshold = (
            float(os.environ.get("SPEAKER_THRESHOLD", "0.50")) if threshold is None else threshold
        )
        self.max_speakers = (
            int(os.environ.get("SPEAKER_MAX", "4")) if max_speakers is None else max_speakers
        )
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        )
        if not config.validate():
            raise SystemExit(f"Invalid speaker embedding config: {config}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self._pool: dict[str, list[str]] = {}
        self._last_name: dict[str, str] = {}
        self._last_t: dict[str, float] = {}
        self._now = time.time
        self._new_max = self.threshold - 0.12  # must be clearly different to spawn MIC-1
        self._new_min_sec = 1.0
        self._stick_sec = 3.0

    def embed(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray | None:
        if samples.size < int(MIN_SEG_SEC * sample_rate):
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return _normalize(np.array(self.extractor.compute(stream), dtype=np.float32))

    def assign(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, *, prefix: str = "EXT") -> str:
        fallback = f"{prefix}-0"
        names = self._pool.setdefault(prefix, [])
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        dur = waveform.size / float(sample_rate)
        emb = self.embed(waveform, sample_rate)
        now = self._now()
        if emb is None:
            return self._last_name.get(prefix) or fallback
        if not names:
            self.centroids[fallback] = emb
            self.counts[fallback] = 1
            names.append(fallback)
            self._last_name[prefix] = fallback
            self._last_t[prefix] = now
            return fallback

        best = ""
        score = -1.0
        for name in names:
            sim = float(np.dot(emb, self.centroids[name]))
            if sim > score:
                best, score = name, sim

        last = self._last_name.get(prefix, "")
        gap = now - self._last_t.get(prefix, 0.0)
        can_create = (
            dur >= self._new_min_sec
            and len(names) < self.max_speakers
            and score < self._new_max
        )

        # MIC-0 / MIC-1: same index = same voice. Short "Hello" must not mint MIC-2.
        if last and gap < self._stick_sec and not (can_create and best != last):
            name = last
        elif score >= self.threshold:
            name = best
        elif can_create:
            name = f"{prefix}-{len(names)}"
            self.centroids[name] = emb
            self.counts[name] = 1
            names.append(name)
            self._last_name[prefix] = name
            self._last_t[prefix] = now
            return name
        else:
            name = best or last or fallback

        if score >= self.threshold:
            n = self.counts.get(name, 1)
            self.centroids[name] = _normalize(self.centroids[name] * n + emb)
            self.counts[name] = n + 1
        self._last_name[prefix] = name
        self._last_t[prefix] = now
        return name
