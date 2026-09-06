from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from events import BUS
from stt import MODEL_ALIASES, Transcriber, resolve_backend, resolve_model_name

THEME_OPTIONS = ("black", "white")
THEME_ALIASES = {
    "black": "black",
    "mac-black": "black",
    "mac black": "black",
    "dark": "black",
    "white": "white",
    "mac-white": "white",
    "mac white": "white",
    "light": "white",
}

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


def theme_key(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get("OVERLAY_THEME", "black")).strip().lower()
    return THEME_ALIASES.get(value, "black")


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
        self._load_prompt()

    def prompt_path(self) -> Path:
        return self.env_path.parent / "qa_prompt.txt"

    def _load_prompt(self) -> None:
        path = self.prompt_path()
        if path.is_file():
            os.environ["QA_PROMPT"] = path.read_text(encoding="utf-8").strip()

    def snapshot(self) -> dict:
        from llm.prompt import effective_prompt

        return {
            "llm_model": (os.environ.get("LLM_MODEL") or "qwen3-next-80b-a3b").strip(),
            "whisper_model": whisper_key(),
            "whisper_options": list(WHISPER_OPTIONS),
            "whisper_loading": self.whisper_loading,
            "theme": theme_key(),
            "theme_options": list(THEME_OPTIONS),
            "prompt": effective_prompt(),
        }

    def apply(
        self,
        *,
        llm_model: str | None = None,
        whisper_model: str | None = None,
        theme: str | None = None,
        prompt: str | None = None,
    ) -> dict:
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
        if theme is not None:
            key = theme_key(theme)
            updates["OVERLAY_THEME"] = key
            os.environ["OVERLAY_THEME"] = key
        if prompt is not None:
            text = str(prompt).replace("\0", "").strip()
            if len(text) > 8000:
                raise ValueError("Prompt is too long")
            os.environ["QA_PROMPT"] = text
            path = self.prompt_path()
            if text:
                path.write_text(text + "\n", encoding="utf-8")
            elif path.is_file():
                path.unlink()
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

    def start_summary(self) -> dict:
        if self.extractor is None or not getattr(self.extractor, "llm", None):
            raise ValueError("runtime unavailable")
        if not self.extractor.llm.enabled:
            raise ValueError("LLM is not configured")
        threading.Thread(target=self._run_summary, daemon=True, name="summary").start()
        BUS.publish("summary", text="Starting summary…", done=False)
        return {"ok": True}

    def _read_text(self, path: Path) -> str:
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
        return ""

    def _session_context(self) -> tuple[str, str]:
        qa = ""
        transcript = ""
        if self.extractor is not None:
            qpath = Path(self.extractor.out_path)
            qa = self._read_text(qpath)
            tname = qpath.name
            if tname.endswith("_questions.txt"):
                tpath = qpath.with_name(tname[: -len("_questions.txt")] + "_transcribe.txt")
            else:
                tpath = qpath.with_name(qpath.stem + "_transcribe.txt")
            transcript = self._read_text(tpath)
            shown_q = getattr(self.extractor, "_shown_q", "") or ""
            shown_a = getattr(self.extractor, "_shown_a", "") or ""
            live = f"Q: {shown_q}\nA: {shown_a}".strip()
            if shown_q and shown_a and shown_q not in qa:
                qa = f"{qa}\n\n[{time.strftime('%H:%M:%S')}] {live}".strip()
        if not transcript:
            lines: list[str] = []
            for ev in BUS.snapshot():
                if ev.get("type") == "transcript" and ev.get("text"):
                    lines.append(f"[{ev.get('ts') or ''}] [{ev.get('label') or ''}] {ev.get('text')}")
            transcript = "\n".join(lines)
        if not qa and self.extractor is not None:
            qa = self.extractor.memory.block()
        return qa.strip(), transcript.strip()

    def _run_summary(self) -> None:
        from llm.client import strip_think
        from llm.prompt import summary_prompt

        qa, transcript = self._session_context()
        from llm.prompt import compact_qa, compact_transcript

        qa = compact_qa(qa)
        transcript = compact_transcript(transcript)
        if not qa and not transcript:
            BUS.publish("summary", text="Nothing to summarize yet.", done=True)
            return
        try:
            buf = ""
            last = 0.0
            for delta in self.extractor.llm.client.chat_stream(
                summary_prompt(qa=qa, transcript=transcript),
                max_tokens=900,
            ):
                if not delta:
                    continue
                buf += delta
                now = time.time()
                if now - last >= 0.12:
                    last = now
                    BUS.publish("summary", text=strip_think(buf), done=False)
            text = strip_think(buf).strip() or "Summary was empty."
            BUS.publish("summary", text=text, done=True)
        except Exception as exc:
            BUS.publish("summary", text=f"Summary failed: {exc}", done=True)

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
