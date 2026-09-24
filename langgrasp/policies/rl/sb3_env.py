"""Gymnasium and Stable Baselines 3 adapters around VecArmEnv (reach task by default).

Both adapters leave the task untouched: same observation (14-D for reach), same [-1, 1]^5 action integrated
into joint targets, same reward (-dist + 2 [dist < 1.5 cm]), same 40-tick horizon and the same success test
(true dist < 1.5 cm at any tick). The one difference to the PPO learner is how the horizon is signalled:
PPO treats the horizon cut as terminal, here it is a time-limit *truncation* (gymnasium `truncated=True`,
SB3 `TimeLimit.truncated`) so that SAC bootstraps through it, which is the standard choice for off-policy
learners. The transitions themselves are identical (tests/test_sb3_baselines.py checks reward parity).

    ArmReachGymEnv  gymnasium.Env, one environment (VecArmEnv with n_envs=1). Use for checks and single-env loops.
    VecArmSB3       stable_baselines3 VecEnv over an n-env VecArmEnv, for fast training (no per-env Python
                    wrappers, physics stepped in the VecArmEnv thread pool).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from langgrasp.policies.rl.envs import ACT_DIM, HORIZON, OBS_DIM, DomainRandomization, VecArmEnv


def _spaces(task: str) -> tuple[gym.spaces.Box, gym.spaces.Box]:
    obs = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM[task],), dtype=np.float32)
    act = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM[task],), dtype=np.float32)
    return obs, act


class ArmReachGymEnv(gym.Env):
    """Single-environment gymnasium view of VecArmEnv. reset(seed=s) reseeds the env RNG exactly like
    VecArmEnv(seed=s), so ArmReachGymEnv().reset(seed=s) and VecArmEnv(1, seed=s).reset() agree."""

    metadata = {"render_modes": []}

    def __init__(self, task: str = "reach", dr: DomainRandomization | None = None, model: mujoco.MjModel | None = None, seed: int = 0):
        super().__init__()
        self.task = task
        self.venv = VecArmEnv(1, task, seed=seed, dr=dr, n_threads=1, model=model)
        self.observation_space, self.action_space = _spaces(task)
        self.horizon = HORIZON[task]
        self._pending_obs: np.ndarray | None = None  # first obs of the episode VecArmEnv auto-reset into

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None and self._pending_obs is not None:
            # VecArmEnv already auto-reset at the time limit: start that episode instead of drawing a new one,
            # so the episode stream is the same as VecArmEnv's
            obs, self._pending_obs = self._pending_obs, None
            return obs, {}
        if seed is not None:
            self.venv.rng = np.random.default_rng(int(seed))
        self._pending_obs = None
        obs = self.venv.reset()
        return obs[0], {}

    def step(self, action):
        obs, r, d, info = self.venv.step(np.asarray(action, dtype=np.float64)[None])
        truncated = bool(d[0])
        out = {"success": bool(info["success"][0]), "is_success": bool(info["success"][0]), "dist": float(info["dist"][0])}
        if truncated:
            out["episode_return"] = float(info["episode_return"][0])
            # the auto-reset already happened inside VecArmEnv; hand back the true last observation
            self._pending_obs = obs[0].copy()
            obs = info["terminal_obs"]
        else:
            self._pending_obs = None
        return obs[0], float(r[0]), False, truncated, out

    def close(self):
        self.venv.close()


class VecArmSB3(VecEnv):
    """SB3 VecEnv over VecArmEnv. Episodes end only by the time limit, reported as truncation."""

    def __init__(self, n_envs: int = 16, task: str = "reach", seed: int = 0, dr: DomainRandomization | None = None, n_threads: int = 4, model: mujoco.MjModel | None = None):
        self.venv = VecArmEnv(n_envs, task, seed=seed, dr=dr, n_threads=n_threads, model=model)
        obs_space, act_space = _spaces(task)
        super().__init__(n_envs, obs_space, act_space)
        self._actions: np.ndarray | None = None
        self.episodes_done = 0
        self.recent_success: list[bool] = []
        self.recent_return: list[float] = []

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions)

    def step_wait(self):
        obs, r, d, info = self.venv.step(self._actions)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(d):
            infos[i] = {
                "terminal_observation": info["terminal_obs"][i],
                "TimeLimit.truncated": True,
                "is_success": bool(info["success"][i]),
                "episode": {"r": float(info["episode_return"][i]), "l": int(self.venv.horizon)},
            }
            self.recent_success.append(bool(info["success"][i]))
            self.recent_return.append(float(info["episode_return"][i]))
            self.episodes_done += 1
        return obs, r, d.copy(), infos

    def close(self) -> None:
        self.venv.close()

    # VecEnv plumbing that SB3 may call; the attribute interface is not used by SAC on this env.
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def _get_indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
