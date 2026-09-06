from __future__ import annotations

import time
from collections import defaultdict, deque


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


class LatencyTracker:
    """Rolling latency histograms. Key metric: EXT end → first answer shown."""

    def __init__(self, *, keep: int = 80) -> None:
        self.keep = keep
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=keep))

    def observe(self, name: str, ms: float) -> None:
        self._samples[name].append(float(ms))

    def values(self, name: str) -> list[float]:
        return list(self._samples.get(name, ()))

    def p50(self, name: str) -> float:
        return percentile(self.values(name), 50)

    def p95(self, name: str) -> float:
        return percentile(self.values(name), 95)

    def line(self, name: str, *, label: str | None = None) -> str:
        xs = self.values(name)
        tag = label or name
        if not xs:
            return f"{tag}  n=0"
        last = xs[-1]
        return (
            f"{tag}  {last:.0f}ms  p50 {self.p50(name):.0f}  "
            f"p95 {self.p95(name):.0f}  n={len(xs)}"
        )

    def summary(self) -> str:
        keys = [
            ("ext_to_answer_ms", "ext→answer"),
            ("ext_to_draft_ms", "ext→draft"),
            ("stt_ms", "stt"),
            ("llm_ttft_ms", "llm-ttft"),
            ("llm_total_ms", "llm-total"),
        ]
        parts = [self.line(k, label=lab) for k, lab in keys if self.values(k)]
        return " | ".join(parts) if parts else "lat  n=0"


def now_mono() -> float:
    return time.monotonic()
