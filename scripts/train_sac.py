"""Train SAC (Stable Baselines 3) on the reach task, CPU, with or without the PPO domain randomisation.

    MUJOCO_GL=egl .venv/bin/python scripts/train_sac.py --no-dr --max-minutes 40 --seed 0
    MUJOCO_GL=egl .venv/bin/python scripts/train_sac.py --dr    --max-minutes 40 --seed 0

Writes checkpoints/sac_reach_<dr|nodr>.zip (+ _obs_rms.pkl) and results/sac_reach_<dr|nodr>.json.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgrasp.policies.rl.sac import SACConfig, train_sac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", dest="dr", action="store_true")
    ap.add_argument("--no-dr", dest="dr", action="store_false")
    ap.set_defaults(dr=False)
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--gradient-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--n-threads", type=int, default=2)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = SACConfig(dr=a.dr, n_envs=a.n_envs, total_steps=a.steps, max_minutes=a.max_minutes, seed=a.seed, gradient_steps=a.gradient_steps,
                    eval_every=a.eval_every, torch_threads=a.torch_threads, n_threads=a.n_threads)
    name = f"sac_reach_{'dr' if a.dr else 'nodr'}{a.tag}"
    train_sac(cfg, results_path=os.path.join("results", f"{name}.json"), ckpt_path=os.path.join("checkpoints", f"{name}.zip"))


if __name__ == "__main__":
    main()
