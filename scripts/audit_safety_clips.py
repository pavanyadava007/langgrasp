"""How much would the safety monitor change the executor's motion if it enforced its limits?

The evaluation scripts put only the grounding gate in the loop; the ROS 2 safety node also clips joint
targets. The GUI has to choose, so this script measures, over the 120 protocol scenes:

* how many ticks the monitor would clip at its default limits (monitor mode: reported, not applied);
* the peak per-joint speed the scripted executor actually commands, against the 1.5 rad/s default;
* whether enforcing the clip changes the outcome of the pick.

Writes results/safety_clip_audit.json. This is a property of the executor and the monitor, not a pipeline
metric: it uses the oracle grasp pose so that perception plays no part.

    MUJOCO_GL=egl .venv/bin/python scripts/audit_safety_clips.py --n 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from langgrasp.eval.harness import hardware_label, trial_seeds
from langgrasp.safety import SafetyConfig, SafetyMonitor
from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.env import CONTROL_HZ, LangGraspEnv
from langgrasp.sim.scenarios import make_scenario
from langgrasp.sim.scene import OBJECT_KINDS

DT = 1.0 / CONTROL_HZ


def run_trial(env, mon: SafetyMonitor, scenario, enforce: bool) -> dict:
    """One oracle pick with the monitor in the loop. Returns the clip bookkeeping and the outcome."""
    stats = {"ticks": 0, "limit": 0, "velocity": 0, "peak_rad_per_s": 0.0, "clipped_ticks": 0}
    orig_step = env.step

    def guarded(q_arm=None, jaw=None):
        now = time.monotonic()
        for topic in ("camera", "joint_states", "command"):
            mon.heartbeat(topic, now)
        if q_arm is not None:
            mon.check_tcp(env.obs()["tcp_pos"])
            q_now = env.q_arm
            stats["peak_rad_per_s"] = max(stats["peak_rad_per_s"], float(np.max(np.abs(np.asarray(q_arm) - q_now)) / DT))
            before = (mon.n_clipped_limit, mon.n_clipped_vel)
            q_safe, _ = mon.check_joint_command(q_arm, q_now, dt=DT)
            dl = mon.n_clipped_limit - before[0]
            dv = mon.n_clipped_vel - before[1]
            stats["limit"] += dl
            stats["velocity"] += dv
            stats["clipped_ticks"] += int(bool(dl or dv))
            if enforce:
                q_arm = q_safe
        stats["ticks"] += 1
        mon.check_staleness(time.monotonic())
        return orig_step(q_arm, jaw)

    env.step = guarded
    try:
        env.reset(scenario)
        p, psi = env.grasp_point(scenario.target)
        r = PickPlaceController(env).run(np.asarray(p), float(psi), scenario.target, width=float(OBJECT_KINDS[scenario.target_obj.kind]["width"]))
    finally:
        env.step = orig_step
    stats.update(placed=bool(r.placed), lifted=bool(r.lifted), grasped=bool(r.grasped), state=mon.state.value)
    return stats


def summarize(rows: list[dict]) -> dict:
    vel = [r["velocity"] for r in rows]
    peak = [r["peak_rad_per_s"] for r in rows]
    return {
        "n_trials": len(rows),
        "placed": sum(r["placed"] for r in rows),
        "trials_with_any_clip": sum(1 for r in rows if r["velocity"] or r["limit"]),
        "velocity_clips_total": int(sum(vel)),
        "velocity_clips_per_trial_median": float(np.median(vel)) if vel else 0.0,
        "velocity_clips_per_trial_max": int(max(vel)) if vel else 0,
        "joint_limit_clips_total": int(sum(r["limit"] for r in rows)),
        "ticks_total": int(sum(r["ticks"] for r in rows)),
        "clipped_tick_fraction": float(sum(r["clipped_ticks"] for r in rows) / max(1, sum(r["ticks"] for r in rows))),
        "commanded_peak_rad_per_s_median": float(np.median(peak)) if peak else 0.0,
        "commanded_peak_rad_per_s_max": float(max(peak)) if peak else 0.0,
        "states_at_end": sorted({r["state"] for r in rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="trials per stratum")
    ap.add_argument("--out", default="results/safety_clip_audit.json")
    ap.add_argument("--max-joint-vel", type=float, default=None, help="override the monitor's velocity limit")
    a = ap.parse_args()

    env = LangGraspEnv(seed=0)
    cfg_kw = {"joint_lo": tuple(env.kin.lo), "joint_hi": tuple(env.kin.hi)}
    if a.max_joint_vel is not None:
        cfg_kw["max_joint_vel"] = a.max_joint_vel
    limit = SafetyConfig(**cfg_kw).max_joint_vel
    seeds = trial_seeds({"seen": a.n, "unseen": a.n, "langvar": a.n})
    out: dict = {
        "hardware": hardware_label(),
        "question": "would the GUI change the executor's motion by enforcing the safety monitor's joint limits?",
        "controller": "scripted oracle pick and place (ground-truth grasp pose, no perception)",
        "max_joint_vel_rad_per_s": float(limit),
        "control_hz": CONTROL_HZ,
        "modes": {},
    }
    for mode in ("monitor", "enforce"):
        rows = []
        t0 = time.time()
        for seed, stratum in seeds:
            sc = make_scenario(seed, stratum)
            mon = SafetyMonitor(SafetyConfig(**cfg_kw))
            rows.append({"seed": seed, "stratum": stratum, "kind": sc.target_obj.kind, **run_trial(env, mon, sc, enforce=mode == "enforce")})
        out["modes"][mode] = {"minutes": (time.time() - t0) / 60, "summary": summarize(rows), "trials": rows}
        s = out["modes"][mode]["summary"]
        print(f"{mode:8s} placed {s['placed']}/{s['n_trials']}  velocity clips {s['velocity_clips_total']} on {s['trials_with_any_clip']} trials  peak {s['commanded_peak_rad_per_s_max']:.2f} rad/s", flush=True)
    m, e = out["modes"]["monitor"]["summary"], out["modes"]["enforce"]["summary"]
    out["conclusion"] = (
        f"The scripted executor commands up to {m['commanded_peak_rad_per_s_max']:.2f} rad/s per joint at {CONTROL_HZ} Hz, "
        f"above the monitor's {limit} rad/s default, so enforcing the clip is not free: it would touch "
        f"{100 * e['clipped_tick_fraction']:.1f}% of ticks. Placed {m['placed']}/{m['n_trials']} observing versus "
        f"{e['placed']}/{e['n_trials']} enforcing. The GUI therefore defaults to observing, so a live run follows the "
        f"same trajectory as the evaluation runs in results/, and offers enforcing as a labelled mode."
    )
    print(out["conclusion"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)
    env.close()


if __name__ == "__main__":
    main()
