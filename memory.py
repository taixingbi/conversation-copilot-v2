from __future__ import annotations

import math
import os
import re
from collections import deque
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not tokens:
        return out
    n = float(len(tokens))
    for t in tokens:
        out[t] = out.get(t, 0.0) + 1.0
    for t in out:
        out[t] /= n
    return out


def _cos(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


def split_chunks(text: str, *, max_chars: int = 700) -> list[str]:
    blocks = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buf = ""
    for block in blocks:
        piece = block.strip()
        if not piece:
            continue
        if len(buf) + len(piece) + 2 <= max_chars:
            buf = f"{buf}\n\n{piece}".strip()
            continue
        if buf:
            chunks.append(buf)
        buf = piece if len(piece) <= max_chars else piece[:max_chars]
    if buf:
        chunks.append(buf)
    return chunks


class SessionMemory:
    """Last N finalized Q&A turns for follow-up continuity."""

    def __init__(self, n: int = 6) -> None:
        self.turns: deque[tuple[str, str]] = deque(maxlen=n)

    def add(self, question: str, answer: str) -> None:
        q, a = (question or "").strip(), (answer or "").strip()
        if q and a:
            self.turns.append((q, a))

    def forget(self, question: str) -> None:
        needle = re.sub(r"\s+", " ", (question or "").strip().lower())
        if not needle:
            return
        kept = [
            (q, a)
            for q, a in self.turns
            if re.sub(r"\s+", " ", q.strip().lower()) != needle
        ]
        self.turns = deque(kept, maxlen=self.turns.maxlen)

    def clear(self) -> None:
        self.turns.clear()

    def block(self) -> str:
        if not self.turns:
            return ""
        lines = [f"Q: {q}\nA: {a}" for q, a in self.turns]
        return "\n\n".join(lines)


class ProfileIndex:
    """Lexical retrieval over profile/*.md STAR notes. No extra embedding deps."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.chunks: list[tuple[str, dict[str, float]]] = []
        self.reload()

    def reload(self) -> None:
        self.chunks = []
        if not self.root.is_dir():
            return
        include_example = (os.environ.get("PROFILE_INCLUDE_EXAMPLE") or "").strip() in {"1", "true", "yes"}
        for path in sorted(self.root.glob("*.md")):
            if path.name.lower() in {"readme.md"}:
                continue
            if path.name.lower() == "example.md" and not include_example:
                continue
            text = path.read_text(encoding="utf-8")
            for chunk in split_chunks(text):
                self.chunks.append((chunk, _tf(tokenize(chunk))))

    def retrieve(self, query: str, *, k: int = 3) -> list[str]:
        q = _tf(tokenize(query))
        if not q or not self.chunks:
            return []
        ranked = sorted(((_cos(q, vec), body) for body, vec in self.chunks), reverse=True)
        out = []
        for score, body in ranked:
            if score <= 0:
                break
            out.append(body)
            if len(out) >= k:
                break
        return out
