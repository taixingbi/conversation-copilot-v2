from __future__ import annotations

import json
import sys
from pathlib import Path


def _prompt_path() -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return root / "prompt" / "qa_instructions.json"


_PATH = _prompt_path()


def _load() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def _build(template: str, focus: str) -> str:
    return template.replace("{focus}", focus)


def _focus_for(ext_lines: list[str], data: dict) -> str:
    blob = " ".join(ext_lines).lower()
    keys = [k for k in data.get("keywords", {}) if k in blob]
    return data["keywords"][max(keys, key=len)] if keys else data["default"]


def _instructions_for(ext_lines: list[str], *, kind: str = "base") -> str:
    data = _load()
    template = data.get(kind) or data["base"]
    return _build(template, _focus_for(ext_lines, data))


def qa_prompt(
    ext_lines: list[str],
    *,
    kind: str = "base",
    history: str = "",
    background: str = "",
) -> str:
    lines = "\n".join(f"- {line}" for line in ext_lines if line.strip())
    extra = ""
    if background.strip():
        extra += f"\n\nCandidate background (use if relevant; do not invent facts):\n{background.strip()}\n"
    if history.strip():
        extra += f"\nRecent Q&A (keep follow-ups consistent):\n{history.strip()}\n"
    return f"{_instructions_for(ext_lines, kind=kind)}{extra}\n\nInterviewer lines:\n{lines or '(none)'}\n"


def gate_prompt(ext_lines: list[str]) -> str:
    lines = "\n".join(f"- {line}" for line in ext_lines if line.strip())
    return (
        "You gate a live interview copilot. The lines are noisy ASR in any language.\n"
        "Reply with one word only:\n"
        "YES — the interviewer asked something the candidate should answer\n"
        "NO — greeting, filler, statement, or an incomplete fragment\n\n"
        f"Lines:\n{lines or '(none)'}\n"
    )
