"""Evaluate the reach baselines (SAC, OSC, MPC) with exactly the protocol of scripts/eval_ppo.py.

    MUJOCO_GL=egl .venv/bin/python scripts/eval_baselines.py --methods sac osc mpc

Conditions, seeds and episode counts are those of eval_ppo.py (nominal / shifted / shifted_latency2, 200
deterministic episodes each, 50 envs, eval seed 12345, Wilson 95 % CIs): sim-to-sim gap, no real robot.
OSC and MPC parameters are read from results/model_based_tuning.json (scripts/tune_model_based.py, nominal
sim, tuning seed 777). OSC and MPC always compute with the NOMINAL model, so in the shifted conditions they
face the same model mismatch as the learned policies. Per-step compute is measured twice: amortised over the
50-env batch during the evaluation, and for a single env (n_envs=1, nominal) in a separate timing run.

Writes results/sac_sim2sim_gap.json, results/osc_sim2sim_gap.json, results/mpc_sim2sim_gap.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgrasp.policies.rl.baseline_eval import (  # noqa: E402
    CONDITIONS,
    EVAL_EPISODES,
    EVAL_N_ENVS,
    EVAL_SEED,
    HARDWARE,
    LABEL,
    evaluate_all_conditions,
    evaluate_controller,
)
from langgrasp.policies.rl.envs import DomainRandomization, VecArmEnv, build_rl_model  # noqa: E402
from langgrasp.policies.rl.model_based import MPPIController, OSCController  # noqa: E402


def _single_env_timing(make_ctrl, model, episodes: int) -> dict:
    env = VecArmEnv(1, "reach", seed=EVAL_SEED, dr=DomainRandomization.none(), n_threads=1, model=model)
    r = evaluate_controller(make_ctrl(1), env, episodes)
    env.close()
    return {"episodes": episodes, "control_ticks": r["act_batch_ms"]["n_calls"], "per_tick_ms": {k: r["act_batch_ms"][k] for k in ("median", "p90", "p99")},
            "note": "one env, nominal, wall time of one act() call (plan + bookkeeping), control period is 100 ms (10 Hz)"}


def _write(path: str, method: str, rows: list[dict], extra: dict):
    out = {
        "label": LABEL,
        "note": "The 'shifted' rows are a deliberately perturbed simulator standing in for a real robot; there is no real robot in this project.",
        "method": method,
        "hardware": HARDWARE,
        "episodes_per_cell": EVAL_EPISODES,
        "eval_seed": EVAL_SEED,
        "n_envs": EVAL_N_ENVS,
        "conditions": {k: mk().to_dict() for k, mk in CONDITIONS.items()},
        **extra,
        "rows": rows,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["sac", "osc", "mpc"], choices=["sac", "osc", "mpc"])
    ap.add_argument("--sac-checkpoints", nargs="+", default=["checkpoints/sac_reach_nodr.zip", "checkpoints/sac_reach_dr.zip"])
    ap.add_argument("--mpc-threads", type=int, default=16)
    a = ap.parse_args()
    model = build_rl_model()
    tuning = None
    if os.path.exists("results/model_based_tuning.json"):
        with open("results/model_based_tuning.json") as f:
            tuning = json.load(f)
    if "sac" in a.methods:
        from langgrasp.policies.rl.sac import load_sac_controller

        rows = []
        for ck in a.sac_checkpoints:
            dr = "_dr" in os.path.basename(ck)
            ctrl = load_sac_controller(ck)
            rows += [{"checkpoint": ck, "train_dr": dr, **r} for r in evaluate_all_conditions(lambda n, c=ctrl: c, f"sac {'dr' if dr else 'nodr'}", model)]
        _write("results/sac_sim2sim_gap.json", "sac", rows, {"deterministic_policy": True})
    if "osc" in a.methods:
        assert tuning is not None, "run scripts/tune_model_based.py first"
        p = {k: tuning["osc"]["chosen"][k] for k in ("k_task", "k_null")}
        rows = evaluate_all_conditions(lambda n: OSCController(model, **p), "osc", model)
        timing = _single_env_timing(lambda n: OSCController(model, **p), model, 10)
        _write("results/osc_sim2sim_gap.json", "osc", rows, {"params": p, "params_from": "results/model_based_tuning.json", "single_env_timing": timing,
                                                              "planning_model": "nominal (build_rl_model), Jacobian and mass matrix at the reported joint angles"})
    if "mpc" in a.methods:
        assert tuning is not None, "run scripts/tune_model_based.py first"
        c = tuning["mpc"]["chosen"]
        p = {k: c[k] for k in ("horizon", "samples", "sigma", "temperature", "iterations")}
        t0 = time.time()
        rows = evaluate_all_conditions(lambda n: MPPIController(model, n, n_threads=a.mpc_threads, **p), "mpc", model)
        for r in rows:
            r["act_batch_ms_per_env"] = {k: r["act_batch_ms"][k] / r["act_batch_ms"]["batch"] for k in ("median", "p90", "p99")}
        timing = _single_env_timing(lambda n: MPPIController(model, n, n_threads=a.mpc_threads, **p), model, 5)
        _write("results/mpc_sim2sim_gap.json", "mpc_mppi", rows, {
            "params": p, "params_from": "results/model_based_tuning.json", "rollout_threads": a.mpc_threads, "single_env_timing": timing,
            "physics_steps_per_tick": int(p["samples"] * p["horizon"] * 50 * p["iterations"]), "total_seconds": time.time() - t0,
            "planning_model": "nominal (build_rl_model) in every condition: privileged (exact model and state) in 'nominal', model mismatch in the shifted rows",
        })
    print("done")


if __name__ == "__main__":
    main()
