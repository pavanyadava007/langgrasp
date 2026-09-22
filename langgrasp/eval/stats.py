"""Statistics helpers for the evaluation protocol: Wilson score intervals and latency summaries."""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (p_hat, lo, hi) of the Wilson score interval for k successes in n trials."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "n/a"
    return f"{100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] (n={n})"


def latency_summary(samples_ms: list[float] | np.ndarray, warmup: int = 0) -> dict:
    """median / p90 / p99 / mean / n after dropping warmup samples."""
    x = np.asarray(samples_ms, dtype=float)[warmup:]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "median_ms": float(np.median(x)),
        "p90_ms": float(np.percentile(x, 90)),
        "p99_ms": float(np.percentile(x, 99)),
        "mean_ms": float(x.mean()),
        "max_ms": float(x.max()),
    }
