"""Shared evaluation for the reach baselines (SAC, OSC, MPC) with exactly the PPO protocol of scripts/eval_ppo.py.

Same conditions (nominal / shifted / shifted_latency2), same VecArmEnv construction (50 envs, eval seed 12345,
8 physics threads), same episode count (200), same success test and Wilson 95 % interval. A controller never
touches the env RNG, so every method sees the same start poses, targets and observation-noise draws.

A controller is any object with
    reset(obs: (n, 14) array) -> None     called at the start of every batch of episodes
    act(obs: (n, 14) array) -> (n, 5)     actions in [-1, 1]
"""

from __future__ import annotations

import time
from typing import Protocol

import mujoco
import numpy as np

from langgrasp.policies.rl.envs import DomainRandomization, VecArmEnv
from langgrasp.policies.rl.ppo import wilson

LABEL = "sim-to-sim gap, no real robot"
HARDWARE = "NVIDIA L4 host / CPU, simulation"
EVAL_SEED = 12345
EVAL_EPISODES = 200
EVAL_N_ENVS = 50
CONDITIONS = {
    "nominal": lambda: DomainRandomization.none(),
    "shifted": lambda: DomainRandomization.shifted(latency_ticks=1),
    "shifted_latency2": lambda: DomainRandomization.shifted(latency_ticks=2),
}


class Controller(Protocol):
    def reset(self, obs: np.ndarray) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...


def evaluate_controller(ctrl: Controller, env: VecArmEnv, n_episodes: int = EVAL_EPISODES) -> dict:
    """Mirror of ppo.evaluate for an arbitrary controller; also times every act() call (whole batch)."""
    successes: list[bool] = []
    holds: list[bool] = []
    returns: list[float] = []
    act_s: list[float] = []
    raw = env.reset()
    while len(successes) < n_episodes:
        ctrl.reset(raw)
        for _ in range(env.horizon):
            t0 = time.perf_counter()
            a = ctrl.act(raw)
            act_s.append(time.perf_counter() - t0)
            raw, r, d, info = env.step(a)
        assert d.all()
        need = n_episodes - len(successes)
        successes += info["success"][:need].tolist()
        returns += info["episode_return"][:need].tolist()
        holds += info["hold"][:need].tolist()
    k, n = int(sum(successes)), len(successes)
    lo, hi = wilson(k, n)
    kh = int(sum(holds))
    ms = 1000.0 * np.array(act_s)
    return {
        "n": n,
        "k": k,
        "success": k / n,
        "ci95": [lo, hi],
        "mean_return": float(np.mean(returns)),
        "hold": kh / n,
        "hold_ci95": list(wilson(kh, n)),
        "act_batch_ms": {"median": float(np.median(ms)), "p90": float(np.percentile(ms, 90)), "p99": float(np.percentile(ms, 99)), "n_calls": len(act_s), "batch": env.n},
    }


def evaluate_all_conditions(make_ctrl, name: str, model: mujoco.MjModel, n_episodes: int = EVAL_EPISODES, n_envs: int = EVAL_N_ENVS,
                            seed: int = EVAL_SEED, n_threads: int = 8, verbose: bool = True) -> list[dict]:
    """make_ctrl(n_envs) -> Controller (a fresh one per condition). Returns one row per condition."""
    rows = []
    for cond, mk in CONDITIONS.items():
        env = VecArmEnv(n_envs, "reach", seed=seed, dr=mk(), n_threads=n_threads, model=model)
        ctrl = make_ctrl(n_envs)
        t0 = time.time()
        r = evaluate_controller(ctrl, env, n_episodes)
        env.close()
        row = {"method": name, "task": "reach", "condition": cond, "shift": mk().to_dict(), "seconds": time.time() - t0, **r}
        rows.append(row)
        if verbose:
            ci = row["ci95"]
            print(f"{name:22s} {cond:18s} success {row['success']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (k={row['k']}/{row['n']}) hold {row['hold']:.3f}", flush=True)
    return rows
