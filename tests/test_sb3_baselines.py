"""Tests for the reach baselines: gymnasium / SB3 wrappers (shapes, determinism, reward parity with VecArmEnv),
one quick episode each for the OSC and MPC controllers, and a tiny SAC training loop. CPU only."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

pytest.importorskip("stable_baselines3")

from langgrasp.policies.rl.baseline_eval import evaluate_controller  # noqa: E402
from langgrasp.policies.rl.envs import HORIZON, OBS_DIM, DomainRandomization, VecArmEnv, build_rl_model  # noqa: E402
from langgrasp.policies.rl.model_based import MPPIController, OSCController  # noqa: E402
from langgrasp.policies.rl.sb3_env import ArmReachGymEnv, VecArmSB3  # noqa: E402


@pytest.fixture(scope="module")
def model_xml():
    return build_rl_model()


def test_gym_env_spaces_and_shapes(model_xml):
    env = ArmReachGymEnv(model=model_xml)
    obs, info = env.reset(seed=3)
    assert env.observation_space.shape == (OBS_DIM["reach"],) and env.action_space.shape == (5,)
    assert obs.shape == (14,) and obs.dtype == np.float32 and env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (14,) and isinstance(r, float) and term is False and trunc is False
    assert {"success", "is_success", "dist"} <= set(info)
    env.close()


def test_gym_env_passes_sb3_checker(model_xml):
    from stable_baselines3.common.env_checker import check_env

    env = ArmReachGymEnv(model=model_xml)
    check_env(env, warn=False)
    env.close()


def test_gym_env_determinism(model_xml):
    a, b = ArmReachGymEnv(model=model_xml), ArmReachGymEnv(model=model_xml)
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.array_equal(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(HORIZON["reach"] + 5):
        act = rng.uniform(-1, 1, 5).astype(np.float32)
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1] and ra[3] == rb[3]
    a.close()
    b.close()


@pytest.mark.parametrize("dr", [DomainRandomization.none(), DomainRandomization.train()])
def test_gym_env_reward_parity_with_vecarmenv(model_xml, dr):
    """Same seed, same actions: identical observations, rewards, horizon and success to VecArmEnv(n_envs=1)."""
    g = ArmReachGymEnv(model=model_xml, dr=dr, seed=5)
    v = VecArmEnv(1, "reach", seed=5, dr=dr, n_threads=1, model=model_xml)
    og, _ = g.reset()
    ov = v.reset()
    assert np.array_equal(og, ov[0])
    rng = np.random.default_rng(1)
    for t in range(2 * HORIZON["reach"]):
        act = rng.uniform(-1, 1, 5)
        og, rg, term, trunc, info = g.step(act)
        ov, rv, dv, iv = v.step(act[None])
        assert rg == pytest.approx(float(rv[0]), abs=0.0)
        assert trunc == bool(dv[0]) and trunc == ((t + 1) % HORIZON["reach"] == 0) and term is False
        assert info["success"] == bool(iv["success"][0])
        if trunc:
            assert np.array_equal(og, iv["terminal_obs"][0])  # the true last obs, not the reset obs
            og, _ = g.reset()  # gymnasium protocol: reset after truncation starts VecArmEnv's auto-reset episode
            assert np.array_equal(og, ov[0])
        else:
            assert np.array_equal(og, ov[0])
    g.close()
    v.close()


def test_sb3_vecenv_parity_and_truncation_infos(model_xml):
    s = VecArmSB3(3, "reach", seed=2, n_threads=1, model=model_xml)
    v = VecArmEnv(3, "reach", seed=2, n_threads=1, model=model_xml)
    assert np.array_equal(s.reset(), v.reset())
    rng = np.random.default_rng(4)
    for _t in range(HORIZON["reach"]):
        act = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
        s.step_async(act)
        os_, rs, ds, infos = s.step_wait()
        ov, rv, dv, iv = v.step(act)
        assert np.array_equal(os_, ov) and np.array_equal(rs, rv) and np.array_equal(ds, dv)
    assert ds.all()
    for i, inf in enumerate(infos):
        assert inf["TimeLimit.truncated"] is True and inf["terminal_observation"].shape == (14,)
        assert inf["is_success"] == bool(iv["success"][i])
    assert s.episodes_done == 3
    s.close()
    v.close()


def test_terminal_obs_does_not_change_env_stream(model_xml):
    """Exposing the terminal observation draws no random numbers: two envs stay in lockstep across resets."""
    a = VecArmEnv(2, "reach", seed=9, dr=DomainRandomization.train(), n_threads=1, model=model_xml)
    b = VecArmEnv(2, "reach", seed=9, dr=DomainRandomization.train(), n_threads=1, model=model_xml)
    a.reset()
    b.reset()
    for _ in range(HORIZON["reach"] * 2):
        act = np.zeros((2, 5))
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
        assert np.array_equal(oa, ob)
    a.close()
    b.close()


def test_osc_one_quick_episode(model_xml):
    env = VecArmEnv(4, "reach", seed=21, n_threads=1, model=model_xml)
    ctrl = OSCController(model_xml)
    obs = env.reset()
    a = ctrl.act(obs)
    assert a.shape == (4, 5) and np.isfinite(a).all() and np.abs(a).max() <= 1.0
    r = evaluate_controller(ctrl, env, 4)
    env.close()
    assert r["n"] == 4 and r["k"] == 4  # nominal reach is easy for an exact-kinematics controller


def test_mpc_one_quick_episode(model_xml):
    env = VecArmEnv(2, "reach", seed=21, n_threads=1, model=model_xml)
    ctrl = MPPIController(model_xml, 2, horizon=3, samples=16, n_threads=2)
    r = evaluate_controller(ctrl, env, 2)
    env.close()
    assert r["n"] == 2 and np.isfinite(r["mean_return"])
    assert len(ctrl.tick_seconds) == HORIZON["reach"] and ctrl.plan.shape == (2, 3, 5)
    assert r["k"] >= 1


def test_sac_tiny_training_loop():
    from langgrasp.policies.rl.sac import SACConfig, train_sac

    cfg = SACConfig(n_envs=4, total_steps=800, learning_starts=200, gradient_steps=1, eval_every=10**9, n_threads=1, torch_threads=1, batch_size=64, buffer_size=5000)
    res = train_sac(cfg, verbose=False)
    assert res["total_steps"] >= 800 and res["gradient_steps"] > 0 and res["train_curve"] == []
    assert res["horizon_handling"].startswith("time-limit truncation")
