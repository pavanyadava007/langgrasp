"""Diagnostic: which part of the 'shifted' condition hurts each reach method? (sim-to-sim gap, no real robot)

    MUJOCO_GL=egl .venv/bin/python scripts/ablate_shift_factors.py

Each factor of DomainRandomization.shifted() is switched on alone; same 200 episodes, 50 envs, eval seed 12345 and
deterministic policies as scripts/eval_ppo.py. Not used to tune anything. Methods: PPO (no DR, DR), SAC (no DR,
DR) if their checkpoints exist, OSC and MPC with the parameters from results/model_based_tuning.json.
Writes results/reach_shift_factors.json.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgrasp.policies.rl.baseline_eval import HARDWARE, LABEL, evaluate_all_conditions  # noqa: E402
from langgrasp.policies.rl.envs import DomainRandomization as DR  # noqa: E402
from langgrasp.policies.rl.envs import build_rl_model  # noqa: E402
from langgrasp.policies.rl.model_based import MPPIController, OSCController  # noqa: E402
from langgrasp.policies.rl.ppo import load_checkpoint  # noqa: E402

FACTORS = {
    "q_noise_5deg": lambda: DR(enabled=True, q_noise_deg=5.0),
    "latency_1tick": lambda: DR(enabled=True, latency_prob=1.0, latency_ticks=1),
    "latency_2tick": lambda: DR(enabled=True, latency_prob=1.0, latency_ticks=2),
    "dynamics_mass1.3_fric0.7_kp0.8": lambda: DR(enabled=True, mass_range=(1.3, 1.3), friction_range=(0.7, 0.7), kp_range=(0.8, 0.8)),
    "q_offset_1.5deg": lambda: DR(enabled=True, q_offset_deg=1.5),
}


class _PPOCtrl:
    def __init__(self, path):
        self.model, self.rms, _ = load_checkpoint(path, "cpu")

    def reset(self, obs):
        pass

    def act(self, obs):
        a, _, _ = self.model.act(torch.as_tensor(self.rms.normalize(obs)), deterministic=True)
        return a.numpy()


def main():
    import langgrasp.policies.rl.baseline_eval as be

    be.CONDITIONS.clear()
    be.CONDITIONS.update(FACTORS)
    model = build_rl_model()
    with open("results/model_based_tuning.json") as f:
        tun = json.load(f)
    po = {k: tun["osc"]["chosen"][k] for k in ("k_task", "k_null")}
    pm = {k: tun["mpc"]["chosen"][k] for k in ("horizon", "samples", "sigma", "temperature", "iterations")}
    methods = {}
    for name, path in (("ppo_nodr", "checkpoints/ppo_reach_nodr.pt"), ("ppo_dr", "checkpoints/ppo_reach_dr.pt")):
        if os.path.exists(path):
            methods[name] = lambda n, p=path: _PPOCtrl(p)
    for name, path in (("sac_nodr", "checkpoints/sac_reach_nodr.zip"), ("sac_dr", "checkpoints/sac_reach_dr.zip")):
        if os.path.exists(path):
            from langgrasp.policies.rl.sac import load_sac_controller

            methods[name] = lambda n, p=path: load_sac_controller(p)
    methods["osc"] = lambda n: OSCController(model, **po)
    methods["mpc"] = lambda n: MPPIController(model, n, n_threads=16, **pm)
    rows = []
    for name, mk in methods.items():
        rows += [{"method": name, **{k: v for k, v in r.items() if k != "method"}} for r in evaluate_all_conditions(mk, name, model)]
    out = {"label": LABEL, "hardware": HARDWARE, "note": "single-factor diagnostic of the shifted condition; not used for tuning",
           "episodes_per_cell": 200, "eval_seed": 12345, "factors": {k: v().to_dict() for k, v in FACTORS.items()}, "rows": rows}
    with open("results/reach_shift_factors.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results/reach_shift_factors.json")


if __name__ == "__main__":
    main()
