"""Train PPO on the vectorised state-based arm env.

    MUJOCO_GL=egl .venv/bin/python scripts/train_ppo.py --task reach --dr --steps 1500000 --n-envs 64 --seed 0

Writes checkpoints/ppo_<task>_<dr|nodr>.pt and results/ppo_<task>_<dr|nodr>.json.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgrasp.policies.rl.ppo import PPOConfig, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["reach", "lift"], default="reach")
    ap.add_argument("--dr", dest="dr", action="store_true")
    ap.add_argument("--no-dr", dest="dr", action="store_false")
    ap.set_defaults(dr=False)
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--n-steps", type=int, default=64, help="rollout length per env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-threads", type=int, default=8, help="physics threads")
    ap.add_argument("--tag", default="", help="optional suffix for output names")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--init-ckpt", default="", help="warm start from this checkpoint (curriculum)")
    ap.add_argument("--dr-ramp", type=float, default=0.0, help="ramp the randomisation from 0 to 1 over this fraction of --steps")
    a = ap.parse_args()
    cfg = PPOConfig(
        task=a.task, dr=a.dr, n_envs=a.n_envs, n_steps=a.n_steps, total_steps=a.steps, max_minutes=a.max_minutes,
        seed=a.seed, device=a.device, n_threads=a.n_threads, init_ckpt=a.init_ckpt, dr_ramp_frac=a.dr_ramp,
    )
    name = f"ppo_{a.task}_{'dr' if a.dr else 'nodr'}{a.tag}"
    res = train(cfg, results_path=os.path.join(a.out_dir, "results", f"{name}.json"), ckpt_path=os.path.join(a.out_dir, "checkpoints", f"{name}.pt"))
    print(f"done: {res['total_steps']} steps in {res['minutes']:.1f} min, {res['throughput_steps_per_s']:.0f} steps/s, final train-env success {res['final_success_train_env']}")


if __name__ == "__main__":
    main()
