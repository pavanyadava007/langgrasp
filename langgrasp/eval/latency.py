"""Per-stage latency tracing for the in-process pipeline (mirrors the ROS 2 latency_tracer node)."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

from langgrasp.eval.stats import latency_summary


class LatencyTracer:
    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._t_start: float | None = None

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter() - t0) * 1000)

    def add(self, name: str, ms: float):
        self.samples[name].append(ms)

    def begin_command(self):
        self._t_start = time.perf_counter()

    def end_command(self, name: str = "end_to_end"):
        if self._t_start is not None:
            self.samples[name].append((time.perf_counter() - self._t_start) * 1000)
            self._t_start = None

    def summary(self, warmup: int = 0) -> dict:
        return {k: latency_summary(v, warmup=min(warmup, max(0, len(v) - 1))) for k, v in self.samples.items()}
