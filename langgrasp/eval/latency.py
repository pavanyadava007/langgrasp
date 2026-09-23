"""Per-stage latency tracing for the in-process pipeline (mirrors the ROS 2 latency_tracer node)."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

from langgrasp.eval.stats import latency_summary


class LatencyTracer:
    """Records the wall-clock of named stages.

    ``listener``, if given, is called as ``listener(name, ms)`` after each sample is recorded, so a GUI can
    watch stages it does not own (the oracle executor, an ACT rollout) without the timed code knowing. It is
    called after the measurement is taken, so the sample itself is unaffected; it must not raise.
    """

    def __init__(self, listener=None):
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._t_start: float | None = None
        self.listener = listener

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1000
            self.samples[name].append(ms)
            if self.listener is not None:
                self.listener(name, ms)

    def add(self, name: str, ms: float):
        self.samples[name].append(ms)

    def begin_command(self):
        self._t_start = time.perf_counter()

    def end_command(self, name: str = "end_to_end"):
        if self._t_start is not None:
            ms = (time.perf_counter() - self._t_start) * 1000
            self.samples[name].append(ms)
            self._t_start = None
            if self.listener is not None:
                self.listener(name, ms)

    def summary(self, warmup: int = 0) -> dict:
        return {k: latency_summary(v, warmup=min(warmup, max(0, len(v) - 1))) for k, v in self.samples.items()}
