"""SAC (Stable Baselines 3) on the reach task, with and without the PPO domain randomisation.

Training logs two curves so the result can be compared with PPO:
  * train_curve   success of the episodes that finished in each block of 4096 env steps, stochastic policy, in
                  the training env (DR on or off). One block = one PPO iteration (64 envs x 64 steps), so the
                  "first block with success >= x" milestones are directly comparable with results/ppo_reach_*.json.
  * eval_curve    every eval_every env steps: 200 deterministic episodes in the NOMINAL env (seed 999, not the
                  final eval seed 12345), with wall-clock minutes. This gives steps and time to the PPO nominal
                  level (200/200).
Final evaluation is done separately by scripts/eval_baselines.py with the exact PPO protocol.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from langgrasp.policies.rl.baseline_eval import HARDWARE, evaluate_controller
from langgrasp.policies.rl.envs import DomainRandomization, VecArmEnv, build_rl_model
from langgrasp.policies.rl.sb3_env import VecArmSB3

BLOCK = 4096  # env steps per train-curve block, = one PPO iteration
CURVE_EVAL_SEED = 999


@dataclass
class SACConfig:
    dr: bool = False
    n_envs: int = 16
    total_steps: int = 1_500_000
    max_minutes: float = 40.0
    seed: int = 0
    lr: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_starts: int = 5000
    gradient_steps: int = 8  # per vectorised step of n_envs transitions: update-to-data ratio gradient_steps / n_envs
    net_arch: tuple[int, int] = (256, 256)
    norm_obs: bool = True
    eval_every: int = 20_000
    eval_episodes: int = 200
    n_threads: int = 2  # physics threads
    torch_threads: int = 4
    device: str = "cpu"


class SACPolicyController:
    """Deterministic SAC policy behind the baseline_eval Controller interface (raw obs in, action out)."""

    def __init__(self, model: SAC, obs_rms: dict | None, clip_obs: float = 10.0):
        self.model = model
        self.rms = obs_rms
        self.clip = clip_obs

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        if self.rms is None:
            return obs.astype(np.float32)
        return np.clip((obs - self.rms["mean"]) / np.sqrt(self.rms["var"] + 1e-8), -self.clip, self.clip).astype(np.float32)

    def reset(self, obs: np.ndarray) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        a, _ = self.model.predict(self._norm(obs), deterministic=True)
        return a


def _obs_rms(venv: VecNormalize | None) -> dict | None:
    if venv is None:
        return None
    return {"mean": venv.obs_rms.mean.copy(), "var": venv.obs_rms.var.copy(), "count": float(venv.obs_rms.count)}


class _CurveCallback(BaseCallback):
    def __init__(self, cfg: SACConfig, raw_env: VecArmSB3, norm_env: VecNormalize | None, t0: float, model_xml):
        super().__init__()
        self.cfg, self.raw, self.norm, self.t0, self.model_xml = cfg, raw_env, norm_env, t0, model_xml
        self.train_curve: list[dict] = []
        self.eval_curve: list[dict] = []
        self.eval_seconds = 0.0
        self._next_block = BLOCK
        self._next_eval = cfg.eval_every
        self._block_start = 0

    def _minutes(self) -> float:
        return (time.time() - self.t0) / 60.0

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n >= self._next_block:
            s = self.raw.recent_success[self._block_start:]
            r = self.raw.recent_return[self._block_start:]
            if s:
                self.train_curve.append({"steps": n, "success_rate": float(np.mean(s)), "mean_return": float(np.mean(r)), "n_episodes": len(s), "minutes": self._minutes()})
            self._block_start = len(self.raw.recent_success)
            self._next_block += BLOCK
        if n >= self._next_eval:
            te = time.time()
            env = VecArmEnv(50, "reach", seed=CURVE_EVAL_SEED, dr=DomainRandomization.none(), n_threads=self.cfg.n_threads, model=self.model_xml)
            r = evaluate_controller(SACPolicyController(self.model, _obs_rms(self.norm)), env, self.cfg.eval_episodes)
            env.close()
            self.eval_seconds += time.time() - te
            pt = {"steps": n, "minutes": self._minutes(), "minutes_excl_eval": self._minutes() - self.eval_seconds / 60.0, "success": r["success"], "k": r["k"], "n": r["n"], "ci95": r["ci95"], "hold": r["hold"]}
            self.eval_curve.append(pt)
            tr = self.train_curve[-1]["success_rate"] if self.train_curve else float("nan")
            print(f"[sac {'dr' if self.cfg.dr else 'nodr'}] steps {n:8d} {pt['minutes']:5.1f} min  train-env succ {tr:.3f}  nominal eval {r['k']}/{r['n']} hold {r['hold']:.2f}", flush=True)
            self._next_eval += self.cfg.eval_every
        if self._minutes() >= self.cfg.max_minutes:
            return False
        return True


def first_reach(curve: list[dict], key: str, level: float) -> dict | None:
    for p in curve:
        if p[key] >= level:
            return p
    return None


def train_sac(cfg: SACConfig, results_path: str | None = None, ckpt_path: str | None = None, verbose: bool = True) -> dict:
    torch.set_num_threads(cfg.torch_threads)
    model_xml = build_rl_model()
    dr = DomainRandomization.train() if cfg.dr else DomainRandomization.none()
    raw = VecArmSB3(cfg.n_envs, "reach", seed=cfg.seed, dr=dr, n_threads=cfg.n_threads, model=model_xml)
    norm = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma) if cfg.norm_obs else None
    env = norm if norm is not None else raw
    model = SAC(
        "MlpPolicy", env, learning_rate=cfg.lr, buffer_size=cfg.buffer_size, batch_size=cfg.batch_size, gamma=cfg.gamma, tau=cfg.tau,
        learning_starts=cfg.learning_starts, train_freq=1, gradient_steps=cfg.gradient_steps, policy_kwargs={"net_arch": list(cfg.net_arch)},
        device=cfg.device, seed=cfg.seed, verbose=0,
    )
    t0 = time.time()
    cb = _CurveCallback(cfg, raw, norm, t0, model_xml)
    model.learn(total_timesteps=cfg.total_steps, callback=cb)
    minutes = (time.time() - t0) / 60.0
    steps = int(model.num_timesteps)
    raw.close()
    tail = cb.train_curve[-5:]
    milestones = {}
    for lv in (0.5, 0.9, 0.99):
        p = first_reach(cb.train_curve, "success_rate", lv)
        milestones[f"train_env_success_ge_{lv}"] = {"steps": p["steps"], "minutes": p["minutes"]} if p else None
    for lv in (0.99, 1.0):
        p = first_reach(cb.eval_curve, "success", lv)
        milestones[f"nominal_eval_success_ge_{lv}"] = {"steps": p["steps"], "minutes": p["minutes"], "minutes_excl_eval": p["minutes_excl_eval"]} if p else None
    result = {
        "method": "sac",
        "library": "stable-baselines3",
        "task": "reach",
        "dr": cfg.dr,
        "dr_spec": dr.to_dict(),
        "total_steps": steps,
        "gradient_steps": int(model._n_updates),
        "minutes": minutes,
        "eval_minutes_inside_training": cb.eval_seconds / 60.0,
        "throughput_env_steps_per_s": steps / max(minutes * 60.0, 1e-9),
        "stopped_by": "steps" if steps >= cfg.total_steps else "time_box",
        "train_curve": cb.train_curve,
        "eval_curve": cb.eval_curve,
        "milestones": milestones,
        "final_success_train_env": float(np.mean([p["success_rate"] for p in tail])) if tail else None,
        "horizon_handling": "time-limit truncation with bootstrap (PPO: treated as terminal)",
        "hardware": HARDWARE,
        "device": cfg.device,
        "config": asdict(cfg),
    }
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        model.save(ckpt_path)
        with open(ckpt_path.replace(".zip", "_obs_rms.pkl"), "wb") as f:
            pickle.dump(_obs_rms(norm), f)
        result["checkpoint"] = ckpt_path
    if results_path:
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(result, f, indent=1)
    if verbose:
        print(f"done: {steps} steps, {result['gradient_steps']} gradient steps, {minutes:.1f} min, final train-env success {result['final_success_train_env']}", flush=True)
    result["_model"] = model
    return result


def load_sac_controller(ckpt_path: str) -> SACPolicyController:
    model = SAC.load(ckpt_path, device="cpu")
    rms_path = ckpt_path.replace(".zip", "_obs_rms.pkl")
    rms = None
    if os.path.exists(rms_path):
        with open(rms_path, "rb") as f:
            rms = pickle.load(f)
    return SACPolicyController(model, rms)
