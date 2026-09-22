import math

from langgrasp.eval.stats import fmt_rate, latency_summary, wilson


def test_wilson_known_values():
    p, lo, hi = wilson(50, 100)
    assert abs(p - 0.5) < 1e-9
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    p, lo, hi = wilson(0, 10)
    assert p == 0 and lo == 0 and 0.27 < hi < 0.29
    p, lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.71 < lo < 0.73


def test_wilson_empty():
    assert all(math.isnan(v) for v in wilson(0, 0))
    assert fmt_rate(0, 0) == "n/a"


def test_latency_summary():
    s = latency_summary([5, 1, 2, 3, 4, 100], warmup=1)
    assert s["n"] == 5 and s["median_ms"] == 3 and s["max_ms"] == 100
    assert latency_summary([], warmup=0) == {"n": 0}
