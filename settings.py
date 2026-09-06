from __future__ import annotations

import os
import threading
from pathlib import Path

from events import BUS
from stt import MODEL_ALIASES, Transcriber, resolve_backend, resolve_model_name

WHISPER_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "distil-small",
    "distil-medium",
    "distil-large",
]

_WHISPER_SHORT = {
    "tiny": "tiny",
    "tiny.en": "tiny",
    "base": "base",
    "base.en": "base",
    "small": "small",
    "small.en": "small",
    "medium": "medium",
    "medium.en": "medium",
    "large": "large",
    "large-v3": "large",
    "distil": "distil-small",
    "distil-small": "distil-small",
    "distil-small.en": "distil-small",
    "distil-medium": "distil-medium",
    "distil-medium.en": "distil-medium",
    "distil-large": "distil-large",
    "distil-large-v3": "distil-large",
}


def resolve_env_path(store: Path, app_dir: Path) -> Path:
    for path in (app_dir / ".env", store / ".env"):
        if path.is_file():
            return path
    return app_dir / ".env"


def upsert_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        raw = line.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            key = raw.partition("=")[0].strip()
            if key in updates:
                out.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        out.append(line)
    if seen != set(updates) and out and out[-1].strip():
        out.append("")
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def whisper_key(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get("WHISPER_MODEL", "medium")).strip().lower()
    return _WHISPER_SHORT.get(value, "medium")


class Runtime:
    """Live overlay settings: persist to .env and apply without a full restart."""

    def __init__(self, env_path: Path) -> None:
        self.env_path = env_path
        self.extractor = None
        self.worker = None
        self._lock = threading.Lock()
        self._whisper_gen = 0
        self.whisper_loading = False

    def snapshot(self) -> dict:
        return {
            "llm_model": (os.environ.get("LLM_MODEL") or "qwen3-next-80b-a3b").strip(),
            "whisper_model": whisper_key(),
            "whisper_options": list(WHISPER_OPTIONS),
            "whisper_loading": self.whisper_loading,
        }

    def apply(self, *, llm_model: str | None = None, whisper_model: str | None = None) -> dict:
        updates: dict[str, str] = {}
        if llm_model is not None:
            model = llm_model.strip()
            if not model or "\n" in model:
                raise ValueError("LLM_MODEL is empty")
            updates["LLM_MODEL"] = model
            os.environ["LLM_MODEL"] = model
            if self.extractor is not None:
                self.extractor.set_model(model)
        if whisper_model is not None:
            raw = whisper_model.strip().lower()
            if raw not in MODEL_ALIASES:
                raise ValueError(f"Unknown WHISPER_MODEL={whisper_model!r}")
            key = whisper_key(raw)
            updates["WHISPER_MODEL"] = key
            os.environ["WHISPER_MODEL"] = key
            self._reload_whisper(key)
        if updates:
            upsert_env(self.env_path, updates)
        snap = self.snapshot()
        BUS.publish("config", **snap)
        return snap

    def forget_question(self, question: str) -> None:
        q = (question or "").strip()
        if not q:
            self.clear_questions()
            return
        if self.extractor is not None:
            self.extractor.forget(q)
        BUS.drop_qa(q)
        BUS.publish("qa_gone", question=q)

    def clear_questions(self) -> None:
        if self.extractor is not None:
            self.extractor.forget_all()
        BUS.drop_qa(None)
        BUS.publish("qa_gone", question="")

    def _reload_whisper(self, key: str) -> None:
        if self.worker is None:
            return
        name = resolve_model_name(key)
        current = getattr(self.worker.stt, "model_name", "")
        if current == name and not self.whisper_loading:
            return
        with self._lock:
            self._whisper_gen += 1
            gen = self._whisper_gen
            self.whisper_loading = True
        BUS.publish("status", text=f"loading whisper {key}…")

        def load() -> None:
            try:
                backend = resolve_backend(name)
                stt = Transcriber(name, backend)
                with self._lock:
                    if gen != self._whisper_gen:
                        return
                    self.worker.set_transcriber(stt)
                    self.whisper_loading = False
                BUS.publish("status", text=f"whisper {key}/{backend}")
                BUS.publish("config", **self.snapshot())
            except Exception as exc:
                with self._lock:
                    if gen == self._whisper_gen:
                        self.whisper_loading = False
                BUS.publish("status", text=f"whisper failed: {exc}")
                BUS.publish("config", **self.snapshot())

        threading.Thread(target=load, daemon=True, name="whisper-reload").start()
