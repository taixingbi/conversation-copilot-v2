from __future__ import annotations

import re
from collections.abc import Iterator

from llm.client import ChatClient, strip_think
from llm.prompt import gate_prompt, qa_prompt

_Q = re.compile(r"^Q:\s*", re.I)
_A = re.compile(r"^A:\s*", re.I)

KIND_PROMPT = {"draft": "draft", "fast": "brief", "final": "detailed"}
KIND_TOKENS = {"draft": 80, "fast": 80, "final": 280}


def short_answer(text: str, *, max_sentences: int = 2) -> str:
    """Keep the first few sentences."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"^A:\s*", "", t, flags=re.I)
    if not t:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", t)
    return " ".join(parts[:max_sentences]).strip()


def _visible(buf: str) -> str:
    low = buf.lower()
    if "<think>" in low and "</think>" not in low:
        return ""
    return strip_think(buf)


def parse_qa(content: str, *, max_sentences: int = 2) -> tuple[str, str]:
    raw = _visible(content or "").strip()
    if not raw or raw.upper() == "SKIP":
        return "", ""
    question, answer_parts = "", []
    for line in raw.splitlines():
        t = line.strip()
        if t.upper() == "SKIP":
            return "", ""
        if _Q.match(t):
            question = _Q.sub("", t).strip()
            answer_parts = []
        elif _A.match(t):
            answer_parts.append(_A.sub("", t).strip())
        elif question:
            answer_parts.append(t)
    answer = " ".join(p for p in answer_parts if p).strip()
    if max_sentences:
        answer = short_answer(answer, max_sentences=max_sentences)
    return question, answer


def parse_qa_partial(content: str) -> tuple[str, str]:
    """Parse a possibly incomplete stream. Does not truncate the answer."""
    return parse_qa(content, max_sentences=0)


class LlmAnswerer:
    """Reconstruct Q from EXT lines and answer it. Supports streaming + two models."""

    def __init__(
        self,
        *,
        function_url: str,
        api_key: str,
        model: str,
        fast_model: str = "",
        memory=None,
        profile=None,
    ) -> None:
        self.client = ChatClient(function_url, api_key, model)
        self.model = model
        self.fast_model = (fast_model or "").strip() or model
        self.memory = memory
        self.profile = profile

    def set_model(self, model: str, *, sync_fast: bool = False) -> None:
        name = (model or "").strip()
        if not name:
            return
        self.model = name
        self.client.model = name
        if sync_fast:
            self.fast_model = name

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def _model_for(self, kind: str) -> str:
        if kind in {"draft", "fast"}:
            return self.fast_model
        return self.model

    def _context(self, ext_lines: list[str]) -> tuple[str, str]:
        query = " ".join(ext_lines)
        history = self.memory.block() if self.memory else ""
        background = ""
        if self.profile:
            hits = self.profile.retrieve(query)
            background = "\n\n".join(hits)
        return history, background

    def should_answer(self, ext_lines: list[str]) -> bool:
        raw = self.client.chat(
            gate_prompt(ext_lines),
            max_tokens=6,
            model=self.fast_model,
        )
        return (raw or "").strip().upper().startswith("YES")

    def qa_from_ext(self, ext_lines: list[str], *, kind: str = "final") -> tuple[str, str]:
        prompt_kind = KIND_PROMPT.get(kind, "base")
        max_tokens = KIND_TOKENS.get(kind, 220)
        max_sent = 1 if kind in {"draft", "fast"} else 4
        history, background = self._context(ext_lines)
        content = self.client.chat(
            qa_prompt(ext_lines, kind=prompt_kind, history=history, background=background),
            max_tokens=max_tokens,
            model=self._model_for(kind),
        )
        return parse_qa(content, max_sentences=max_sent)

    def iter_qa(self, ext_lines: list[str], *, kind: str = "final") -> Iterator[tuple[str, str]]:
        """Yield (question, answer_so_far) as tokens arrive. Last yield is cleaned."""
        prompt_kind = KIND_PROMPT.get(kind, "base")
        max_tokens = KIND_TOKENS.get(kind, 220)
        max_sent = 1 if kind in {"draft", "fast"} else 4
        history, background = self._context(ext_lines)
        buf = ""
        last: tuple[str, str] = ("", "")
        for delta in self.client.chat_stream(
            qa_prompt(ext_lines, kind=prompt_kind, history=history, background=background),
            max_tokens=max_tokens,
            model=self._model_for(kind),
        ):
            if not delta:
                continue
            buf += delta
            parsed = parse_qa_partial(buf)
            if parsed != last and (parsed[0] or parsed[1]):
                last = parsed
                yield parsed
        final = parse_qa(buf, max_sentences=max_sent)
        if final != last:
            yield final
