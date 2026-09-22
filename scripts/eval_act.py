"""Closed-loop evaluation of the ACT policy and of the scripted oracle on 100 identical fixed-goal scenes.

Scenes: make_scenario(5000+i, "seen", target_kind="cube", target_color="red"), i in 0..99, the same seeds as
the shared protocol's "seen" stratum so the modular pipeline can be compared on the identical scenes later.
The loop mirrors langgrasp.eval.harness.run_protocol (same Trial fields, summarize() and JSON layout) because
run_protocol builds its own scenarios without the fixed-goal arguments. MuJoCo simulation only.

Usage:
  MUJOCO_GL=egl .venv/bin/python scripts/eval_act.py --n 100 --horizon 90
  MUJOCO_GL=egl .venv/bin/python scripts/eval_act.py --oracle-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgrasp.eval.harness import Trial, hardware_label, summarize  # noqa: E402
from langgrasp.eval.stats import latency_summary  # noqa: E402
from langgrasp.policies.act.data import FRONT_KEY, STATE_KEY, WRIST_KEY, observe  # noqa: E402

BASE_SEED = 5000
STRATUM = "seen"
TARGET_KIND, TARGET_COLOR = "cube", "red"


def fixed_goal_scenarios(n: int, base: int = BASE_SEED):
    from langgrasp.sim.scenarios import make_scenario

    return [make_scenario(base + i, STRATUM, target_kind=TARGET_KIND, target_color=TARGET_COLOR) for i in range(n)]


def run_oracle(env, sc) -> dict:
    from langgrasp.sim.controller import PickPlaceController
    from langgrasp.sim.scene import OBJECT_KINDS

    env.reset(sc)
    ctl = PickPlaceController(env)
    t0 = time.perf_counter()
    grasp_xyz, psi = env.grasp_point(sc.target)
    r = ctl.run(grasp_xyz, psi, sc.target, width=OBJECT_KINDS[sc.target_obj.kind]["width"])
    ms = (time.perf_counter() - t0) * 1e3
    return {"grasped": r.grasped, "lifted": r.lifted, "placed": r.placed, "aborted": None, "latency_ms": {"episode": ms}, "extra": {"ticks": r.steps, "ik_ok": r.ik_ok}}


def run_act(env, sc, runner, size: tuple[int, int], horizon: int, settle_ticks: int = 5) -> dict:
    env.reset(sc)
    runner.reset()
    name = sc.target
    lifted = grasped = False
    call_ms: list[float] = []
    ticks = 0
    released_in_tray_at = None
    for t in range(horizon):
        obs = observe(env, size)
        n_before = len(runner.latencies_ms)
        a = runner.act(obs[FRONT_KEY], obs[WRIST_KEY], obs[STATE_KEY])
        call_ms.append(runner.latencies_ms[n_before])
        env.step(np.asarray(a[:5], dtype=float), float(a[5]))
        ticks = t + 1
        if env.is_grasped(name):
            grasped = True
        if env.is_lifted(name) and env.is_grasped(name):
            lifted = True
        if env.in_tray(name) and not env.is_grasped(name):
            if released_in_tray_at is None:
                released_in_tray_at = ticks
            elif ticks - released_in_tray_at >= settle_ticks:
                break  # placed and released; stop early like the oracle's settle phase
        else:
            released_in_tray_at = None
    placed = bool(env.in_tray(name))
    return {
        "grasped": grasped,
        "lifted": lifted,
        "placed": placed,
        "aborted": None if placed else "horizon",
        "latency_ms": {"policy_call": float(np.median(call_ms))},
        "extra": {"ticks": ticks, "policy_call_p90_ms": float(np.percentile(call_ms, 90)), "final_height_m": env.object_height(name)},
    }


def evaluate(run_one, scenarios, approach: str, out_path: Path, notes: str, extra_top: dict | None = None, log_every: int = 10) -> dict:
    trials: list[Trial] = []
    t0 = time.time()
    for i, sc in enumerate(scenarios):
        r = run_one(sc)
        tobj = sc.target_obj
        trials.append(
            Trial(
                seed=sc.seed,
                stratum=sc.stratum,
                lang_variant=sc.lang_variant,
                lighting=sc.lighting,
                command=sc.command,
                target=sc.target,
                kind=tobj.kind,
                color=tobj.color,
                grounding_correct=None,
                grasped=bool(r.get("grasped")),
                lifted=bool(r.get("lifted")),
                placed=bool(r.get("placed")),
                aborted=r.get("aborted"),
                latency_ms=r.get("latency_ms", {}),
                extra=r.get("extra", {}),
            )
        )
        if log_every and (i + 1) % log_every == 0:
            print(f"[{approach}] {i + 1}/{len(scenarios)} lifted {sum(t.lifted for t in trials)} placed {sum(t.placed for t in trials)} ({time.time() - t0:.0f}s)", flush=True)
    result = {
        "approach": approach,
        "notes": notes,
        "hardware": hardware_label(),
        "n_per_stratum": {STRATUM: len(scenarios)},
        "fixed_goal": {"target_kind": TARGET_KIND, "target_color": TARGET_COLOR, "task": "pick the red cube and place it in the tray", "seeds": [scenarios[0].seed, scenarios[-1].seed]},
        "minutes": (time.time() - t0) / 60,
        "summary": summarize(trials),
        "trials": [asdict(t) for t in trials],
    }
    if extra_top:
        result.update(extra_top)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    s = result["summary"]["strata"]["all"]
    print(f"[{approach}] grasp {s['grasp']['k']}/{s['grasp']['n']} place {s['place']['k']}/{s['place']['n']} p={s['place']['p']:.3f} [{s['place']['lo']:.3f}, {s['place']['hi']:.3f}] -> {out_path}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints/act_pick"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--act-only", action="store_true")
    ap.add_argument("--out-act", default=str(ROOT / "results/act_eval.json"))
    ap.add_argument("--out-oracle", default=str(ROOT / "results/oracle_fixed_goal.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from langgrasp.sim.env import LangGraspEnv

    scenarios = fixed_goal_scenarios(args.n)
    env = LangGraspEnv(seed=0)
    if not args.act_only:
        evaluate(lambda sc: run_oracle(env, sc), scenarios, "oracle_fixed_goal", Path(args.out_oracle), "scripted PickPlaceController with ground-truth grasp point on the fixed-goal scenes (upper bound for ACT)")
    if not args.oracle_only:
        from langgrasp.policies.act.policy import ACTRunner

        runner = ACTRunner(args.ckpt, device=args.device)
        size = runner.image_hw
        cfg = runner.config
        evaluate(
            lambda sc: run_act(env, sc, runner, size, args.horizon),
            scenarios,
            "act_fixed_goal",
            Path(args.out_act),
            f"LeRobot 0.4.4 ACT, closed loop at 10 Hz, horizon {args.horizon} ticks, chunk {cfg.chunk_size}, n_action_steps {cfg.n_action_steps}, images {size[0]}x{size[1]}; no language input (fixed goal). {args.tag}".strip(),
            extra_top={
                "policy_latency_ms": latency_summary(runner.latencies_ms),
                "policy_warmup_ms": latency_summary(runner.warmup_ms),
                "checkpoint": str(Path(args.ckpt).relative_to(ROOT)) if Path(args.ckpt).is_relative_to(ROOT) else args.ckpt,
                "horizon_ticks": args.horizon,
                "device": runner.device,
            },
        )
    env.close()


if __name__ == "__main__":
    main()
