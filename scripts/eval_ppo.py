"""Evaluate PPO checkpoints in the nominal env and in deliberately shifted sims (sim-to-sim gap, no real robot).

    MUJOCO_GL=egl .venv/bin/python scripts/eval_ppo.py --checkpoints checkpoints/ppo_reach_nodr.pt checkpoints/ppo_reach_dr.pt

Conditions
    nominal            no DR
    shifted            joint-position noise 5 deg + fixed 1-tick latency + mass x1.3 + friction x0.7 + kp x0.8 + 1.5 deg joint offset
    shifted_latency2   same with 2-tick latency
Writes results/ppo_sim2sim_gap.json and fills final_success_nominal in each checkpoint's results JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgrasp.policies.rl.envs import DomainRandomization, VecArmEnv, build_rl_model  # noqa: E402
from langgrasp.policies.rl.ppo import HARDWARE, evaluate, load_checkpoint  # noqa: E402

LABEL = "sim-to-sim gap, no real robot"
CONDITIONS = {
    "nominal": lambda: DomainRandomization.none(),
    "shifted": lambda: DomainRandomization.shifted(latency_ticks=1),
    "shifted_latency2": lambda: DomainRandomization.shifted(latency_ticks=2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n-envs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-threads", type=int, default=8)
    ap.add_argument("--out", default="results/ppo_sim2sim_gap.json")
    a = ap.parse_args()
    model_xml = build_rl_model()
    rows = []
    for ck_path in a.checkpoints:
        model, rms, ck = load_checkpoint(ck_path, "cpu")
        task, dr = ck["config"]["task"], ck["config"]["dr"]
        for cond, mk in CONDITIONS.items():
            env = VecArmEnv(a.n_envs, task, seed=a.seed, dr=mk(), n_threads=a.n_threads, model=model_xml)
            t0 = time.time()
            r = evaluate(model, rms, env, n_episodes=a.episodes, deterministic=True, device="cpu")
            env.close()
            row = {"checkpoint": ck_path, "task": task, "train_dr": dr, "condition": cond, "shift": mk().to_dict(), "seconds": time.time() - t0, **r}
            rows.append(row)
            ci = row["ci95"]
            extra = f" hold {row['hold']:.3f}" if "hold" in row else ""
            print(f"{os.path.basename(ck_path):24s} {cond:18s} success {row['success']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (k={row['k']}/{row['n']}){extra}", flush=True)
            if cond == "nominal":
                res_path = os.path.join("results", os.path.basename(ck_path).replace(".pt", ".json"))
                if os.path.exists(res_path):
                    with open(res_path) as f:
                        d = json.load(f)
                    d["final_success_nominal"] = row["success"]
                    d["final_success_nominal_ci95"] = ci
                    d["final_success_nominal_n"] = row["n"]
                    with open(res_path, "w") as f:
                        json.dump(d, f, indent=1)
    out = {
        "label": LABEL,
        "note": "The 'shifted' rows are a deliberately perturbed simulator standing in for a real robot; there is no real robot in this project.",
        "hardware": HARDWARE,
        "episodes_per_cell": a.episodes,
        "deterministic_policy": True,
        "eval_seed": a.seed,
        "conditions": {k: mk().to_dict() for k, mk in CONDITIONS.items()},
        "rows": rows,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
