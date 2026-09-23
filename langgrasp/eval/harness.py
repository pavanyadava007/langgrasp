"""Stratified evaluation protocol: seen / unseen / language-variation trials with Wilson intervals.

Every trial is a deterministic scenario (seed), so approaches (modular, ACT, oracle) share the exact same
scenes and commands. Results are written as JSON and rendered into docs/RESULTS.md by langgrasp.eval.report.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langgrasp.eval.stats import latency_summary, wilson
from langgrasp.sim.scenarios import STRATA, make_scenario


@dataclass
class Trial:
    seed: int
    stratum: str
    lang_variant: str
    lighting: str
    command: str
    target: str
    kind: str
    color: str
    grounding_correct: bool | None
    grasped: bool
    lifted: bool
    placed: bool
    aborted: str | None
    latency_ms: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def hardware_label() -> str:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    except Exception:
        gpu = "unknown"
    return f"{gpu} (x86 EC2 host, {platform.machine()}), simulation only, not Jetson"


def trial_seeds(n_per_stratum: dict[str, int], base: int = 5000) -> list[tuple[int, str]]:
    out = []
    for s in STRATA:
        n = n_per_stratum.get(s, 0)
        off = STRATA.index(s) * 1000
        out += [(base + off + i, s) for i in range(n)]
    return out


def summarize(trials: list[Trial]) -> dict:
    def rate(sel, key):
        k = sum(1 for t in sel if getattr(t, key))
        p, lo, hi = wilson(k, len(sel))
        return {"k": k, "n": len(sel), "p": p, "lo": lo, "hi": hi}

    def grounding(sel):
        s2 = [t for t in sel if t.grounding_correct is not None]
        k = sum(1 for t in s2 if t.grounding_correct)
        p, lo, hi = wilson(k, len(s2))
        return {"k": k, "n": len(s2), "p": p, "lo": lo, "hi": hi}

    out: dict = {"n_trials": len(trials), "strata": {}, "lang_variants": {}, "lighting": {}, "kinds": {}, "aborts": {}}
    groups = {"all": trials}
    for s in STRATA:
        groups[s] = [t for t in trials if t.stratum == s]
    for name, sel in groups.items():
        if not sel:
            continue
        out["strata"][name] = {"grasp": rate(sel, "lifted"), "place": rate(sel, "placed"), "grounding": grounding(sel)}
    for v in sorted({t.lang_variant for t in trials}):
        sel = [t for t in trials if t.lang_variant == v]
        out["lang_variants"][v] = {"grasp": rate(sel, "lifted"), "place": rate(sel, "placed"), "grounding": grounding(sel)}
    for v in sorted({t.lighting for t in trials}):
        sel = [t for t in trials if t.lighting == v]
        out["lighting"][v] = {"grasp": rate(sel, "lifted"), "place": rate(sel, "placed"), "grounding": grounding(sel)}
    for v in sorted({t.kind for t in trials}):
        sel = [t for t in trials if t.kind == v]
        out["kinds"][v] = {"grasp": rate(sel, "lifted"), "place": rate(sel, "placed")}
    for t in trials:
        if t.aborted:
            out["aborts"][t.aborted] = out["aborts"].get(t.aborted, 0) + 1
    lat: dict[str, list[float]] = {}
    for t in trials:
        for k, v in t.latency_ms.items():
            lat.setdefault(k, []).append(v)
    out["latency_ms"] = {k: latency_summary(v, warmup=1 if len(v) > 3 else 0) for k, v in lat.items()}
    return out


def run_protocol(run_one, n_per_stratum: dict[str, int], approach: str, out_path: str | Path, notes: str = "", base_seed: int = 5000, log_every: int = 10, make=None) -> dict:
    """run_one(scenario) -> dict with keys grounding_correct, grasped, lifted, placed, aborted, latency_ms, extra.
    make(seed, stratum) -> Scenario overrides the default scenario generator (e.g. fixed-goal scenes)."""
    trials: list[Trial] = []
    t0 = time.time()
    seeds = trial_seeds(n_per_stratum, base_seed)
    for i, (seed, stratum) in enumerate(seeds):
        sc = (make or make_scenario)(seed, stratum)
        r = run_one(sc)
        tobj = sc.target_obj
        trials.append(
            Trial(
                seed=seed,
                stratum=stratum,
                lang_variant=sc.lang_variant,
                lighting=sc.lighting,
                command=sc.command,
                target=sc.target,
                kind=tobj.kind,
                color=tobj.color,
                grounding_correct=r.get("grounding_correct"),
                grasped=bool(r.get("grasped")),
                lifted=bool(r.get("lifted")),
                placed=bool(r.get("placed")),
                aborted=r.get("aborted"),
                latency_ms=r.get("latency_ms", {}),
                extra=r.get("extra", {}),
            )
        )
        if log_every and (i + 1) % log_every == 0:
            done = trials
            print(f"[{approach}] {i + 1}/{len(seeds)} lifted {sum(t.lifted for t in done)} placed {sum(t.placed for t in done)} ({time.time() - t0:.0f}s)", flush=True)
    summary = summarize(trials)
    result = {
        "approach": approach,
        "notes": notes,
        "hardware": hardware_label(),
        "n_per_stratum": n_per_stratum,
        "minutes": (time.time() - t0) / 60,
        "summary": summary,
        "trials": [asdict(t) for t in trials],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    return result
