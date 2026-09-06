from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_LINE = re.compile(r"^(?:\[[^\]]+\]\s*){0,2}(.*)$")
_JUNK = re.compile(
    r"^(hi|hello|hey|ok|okay|yeah|yes|yep|no|nah|um+|uh+|ah+|hmm+|thanks|thank you|"
    r"bye|good|nice|cool|right|sure|please)[?.!]*$",
    re.I,
)
_NORM = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


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


def default_base() -> str:
    return str(_load().get("base") or "")


def user_prompt() -> str:
    return (os.environ.get("QA_PROMPT") or "").strip()


def effective_prompt() -> str:
    return user_prompt() or default_base()


def _instructions_for(ext_lines: list[str], *, kind: str = "base") -> str:
    data = _load()
    override = user_prompt()
    if override and kind in {"base", "brief", "detailed"}:
        template = override
    else:
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


def _norm_line(text: str) -> str:
    return _NORM.sub(" ", (text or "").lower()).strip()


def _body(line: str) -> str:
    m = _LINE.match((line or "").strip())
    return (m.group(1) if m else line).strip()


def compact_transcript(raw: str) -> str:
    """Drop greetings/repeats so the summary model sees topics, not ASR noise."""
    kept: list[str] = []
    last = ""
    for raw_line in (raw or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        body = _body(line)
        norm = _norm_line(body)
        if not norm or _JUNK.match(norm) or len(norm) < 8:
            continue
        if last and (norm == last or norm in last or last in norm):
            if len(norm) > len(last):
                kept[-1] = line
                last = norm
            continue
        kept.append(line)
        last = norm
    return "\n".join(kept)


def compact_qa(raw: str) -> str:
    blocks: list[str] = []
    seen: set[str] = set()
    buf: list[str] = []
    for line in (raw or "").splitlines():
        if not line.strip():
            if buf:
                block = "\n".join(buf).strip()
                key = _norm_line(block)
                if key and key not in seen:
                    seen.add(key)
                    blocks.append(block)
                buf = []
            continue
        buf.append(line.rstrip())
    if buf:
        block = "\n".join(buf).strip()
        key = _norm_line(block)
        if key and key not in seen:
            blocks.append(block)
    return "\n\n".join(blocks)


def summary_prompt(*, qa: str = "", transcript: str = "") -> str:
    qa = compact_qa(qa) or "(none)"
    transcript = compact_transcript(transcript) or "(none)"
    return (
        "You write a useful recap for a candidate in a live conversation.\n"
        "The transcript is noisy ASR. People repeat. Greetings and fragments are junk.\n\n"
        "Rules:\n"
        "- Reconstruct the real topics. Do not list every line.\n"
        "- Never write 'user said X multiple times' or count repeats.\n"
        "- Ignore hellos, filler, one-word noise, and unclear scraps.\n"
        "- If the same question was repeated, mention it once.\n"
        "- Prefer interviewer lines (EXT / EXT-*) over the candidate repeating (MIC).\n"
        "- Write at most 8 short bullets:\n"
        "  • Question asked (the real question, in clear English)\n"
        "  • Answer given, if any\n"
        "  • What is still open\n"
        "- If there was no real question, say that in ONE short line. Do not narrate the junk.\n"
        "- Simple words. No preamble. No heading.\n\n"
        f"Q&A (may be empty):\n{qa}\n\n"
        f"Transcript (cleaned):\n{transcript}\n"
    )


def gate_prompt(ext_lines: list[str]) -> str:
    lines = "\n".join(f"- {line}" for line in ext_lines if line.strip())
    return (
        "You gate a live interview copilot. The lines are noisy ASR in any language.\n"
        "Reply with one word only:\n"
        "YES — the interviewer asked something the candidate should answer\n"
        "NO — greeting, filler, statement, or an incomplete fragment\n\n"
        f"Lines:\n{lines or '(none)'}\n"
    )
