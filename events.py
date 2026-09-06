from __future__ import annotations

import queue
import re
import threading
import time
from collections import deque


def _norm_q(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (text or "").lower()).strip()


class EventBus:
    """Fan-out live transcript / QA events to WebSocket clients."""

    def __init__(self, *, keep: int = 80) -> None:
        self._lock = threading.Lock()
        self._subs: list[queue.Queue] = []
        self._keep = keep
        self._recent: deque = deque(maxlen=keep)

    def publish(self, typ: str, **payload) -> dict:
        ev = {"type": typ, "t": time.time(), **payload}
        with self._lock:
            if typ == "qa":
                nq = _norm_q(payload.get("question") or "")
                if nq:
                    kept = [
                        e
                        for e in self._recent
                        if not (e.get("type") == "qa" and _norm_q(e.get("question") or "") == nq)
                    ]
                    self._recent = deque(kept, maxlen=self._keep)
            elif typ == "summary":
                kept = [e for e in self._recent if e.get("type") != "summary"]
                self._recent = deque(kept, maxlen=self._keep)
            self._recent.append(ev)
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass
        return ev

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self._lock:
            for ev in self._recent:
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    break
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._recent)

    def drop_summary(self) -> None:
        with self._lock:
            kept = [e for e in self._recent if e.get("type") not in ("summary", "summary_gone")]
            self._recent = deque(kept, maxlen=self._keep)

    def drop_qa(self, question: str | None = None) -> None:
        with self._lock:
            if not question:
                kept = [e for e in self._recent if e.get("type") != "qa"]
            else:
                nq = _norm_q(question)
                kept = [
                    e
                    for e in self._recent
                    if not (e.get("type") == "qa" and _norm_q(e.get("question") or "") == nq)
                ]
            self._recent = deque(kept, maxlen=self._keep)


BUS = EventBus()
