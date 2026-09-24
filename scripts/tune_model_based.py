"""Pick the OSC and MPC (MPPI) parameters on a TUNING seed in the NOMINAL sim only.

    MUJOCO_GL=egl .venv/bin/python scripts/tune_model_based.py

Tuning seed 777 (the final evaluation uses seed 12345), nominal condition only: the shifted conditions are never
used to choose parameters, as the PPO / SAC policies never saw the calibration offset or the 2-tick latency.
Selection rule: highest success, then highest final-tick hold rate, then highest mean return.
Writes results/model_based_tuning.json.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgrasp.policies.rl.baseline_eval import HARDWARE, evaluate_controller  # noqa: E402
from langgrasp.policies.rl.envs import DomainRandomization, VecArmEnv, build_rl_model  # noqa: E402
from langgrasp.policies.rl.model_based import MPPIController, OSCController  # noqa: E402

TUNE_SEED = 777


def _run(make, model, episodes):
    env = VecArmEnv(50, "reach", seed=TUNE_SEED, dr=DomainRandomization.none(), n_threads=4, model=model)
    t0 = time.time()
    r = evaluate_controller(make(), env, episodes)
    env.close()
    return {k: r[k] for k in ("n", "k", "success", "hold", "mean_return")} | {"act_batch_ms_median": r["act_batch_ms"]["median"], "seconds": time.time() - t0}


def _best(rows):
    return max(rows, key=lambda r: (r["success"], r["hold"], r["mean_return"]))


def main():
    model = build_rl_model()
    osc_rows = []
    for k_task, k_null in itertools.product([0.5, 0.8, 1.0, 1.5], [0.0, 0.1]):
        r = _run(lambda k_task=k_task, k_null=k_null: OSCController(model, k_task=k_task, k_null=k_null), model, 100)
        osc_rows.append({"k_task": k_task, "k_null": k_null, **r})
        print("osc", k_task, k_null, r, flush=True)
    mpc_rows = []
    for horizon, sigma in itertools.product([4, 6, 10], [0.3, 0.5]):
        r = _run(lambda horizon=horizon, sigma=sigma: MPPIController(model, 50, horizon=horizon, samples=64, sigma=sigma, temperature=0.02, n_threads=16), model, 50)
        mpc_rows.append({"horizon": horizon, "samples": 64, "sigma": sigma, "temperature": 0.02, "iterations": 1, **r})
        print("mpc", horizon, sigma, r, flush=True)
    out = {
        "label": "nominal sim only, tuning seed (not the evaluation seed)",
        "tune_seed": TUNE_SEED,
        "hardware": HARDWARE,
        "selection": "max (success, hold, mean_return)",
        "osc": {"rows": osc_rows, "chosen": _best(osc_rows)},
        "mpc": {"rows": mpc_rows, "chosen": _best(mpc_rows)},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/model_based_tuning.json", "w") as f:
        json.dump(out, f, indent=1)
    print("chosen osc", out["osc"]["chosen"])
    print("chosen mpc", out["mpc"]["chosen"])


if __name__ == "__main__":
    main()
