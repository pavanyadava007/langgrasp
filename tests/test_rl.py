"""Fast tests for the RL track (env, PPO update, checkpoint roundtrip). CPU only, a few hundred env steps."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from langgrasp.policies.rl.envs import (  # noqa: E402
    ACT_DIM,
    HORIZON,
    OBS_DIM,
    REACH_TOL,
    DomainRandomization,
    VecArmEnv,
    build_rl_model,
)
from langgrasp.policies.rl.ppo import (  # noqa: E402
    ActorCritic,
    PPOConfig,
    RunningMeanStd,
    compute_gae,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
    train,
    wilson,
)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model_xml():
    return build_rl_model()


@pytest.mark.parametrize("task", ["reach", "lift"])
def test_reset_step_shapes(model_xml, task):
    env = VecArmEnv(4, task, seed=0, model=model_xml, n_threads=2)
    obs = env.reset()
    assert obs.shape == (4, OBS_DIM[task]) and obs.dtype == np.float32
    a = np.zeros((4, ACT_DIM[task]))
    obs, r, d, info = env.step(a)
    assert obs.shape == (4, OBS_DIM[task]) and r.shape == (4,) and d.shape == (4,)
    assert np.isfinite(obs).all() and np.isfinite(r).all()
    assert not d.any() and "success" in info
    env.close()


def test_determinism_and_autoreset(model_xml):
    a = VecArmEnv(3, "reach", seed=7, model=model_xml, n_threads=2)
    b = VecArmEnv(3, "reach", seed=7, model=model_xml, n_threads=1)
    oa, ob = a.reset(), b.reset()
    assert np.allclose(oa, ob)
    rng = np.random.default_rng(0)
    for t in range(HORIZON["reach"]):
        act = rng.uniform(-1, 1, (3, 5))
        oa, ra, da, _ = a.step(act)
        ob, rb, db, _ = b.step(act)
        assert np.allclose(oa, ob) and np.allclose(ra, rb)
        assert da.all() == (t == HORIZON["reach"] - 1)
    assert (a.t == 0).all()  # auto-reset happened
    a.close()
    b.close()


def test_dr_env_changes_physics(model_xml):
    env = VecArmEnv(4, "lift", seed=1, dr=DomainRandomization.train(), model=model_xml, n_threads=1)
    env.reset()
    masses = np.array([m.body_mass[env.cube_bid] for m in env.models])
    kps = np.array([m.actuator_gainprm[env.arm_act[0], 0] for m in env.models])
    assert masses.std() > 0 and kps.std() > 0
    assert (masses >= 0.7 * env.mass0 - 1e-9).all() and (masses <= 1.3 * env.mass0 + 1e-9).all()
    shifted = VecArmEnv(2, "reach", seed=1, dr=DomainRandomization.shifted(), model=model_xml, n_threads=1)
    shifted.reset()
    assert (shifted.latency == 1).all() and np.allclose(shifted.q_offset, np.deg2rad(1.5))
    env.close()
    shifted.close()


def test_reward_increases_when_moving_toward_target(model_xml):
    """Greedy joint-space descent on the true distance must raise the per-step reward and eventually hit the target."""
    env = VecArmEnv(2, "reach", seed=3, model=model_xml, n_threads=1)
    env.reset()
    first = None
    for _ in range(HORIZON["reach"] - 1):
        act = np.zeros((2, 5))
        for i in range(2):
            q = env.q_target[i]
            best = None
            for j in range(5):
                for s in (-1.0, 1.0):
                    qq = q.copy()
                    qq[j] = np.clip(qq[j] + s * 0.12, env.lo[j], env.hi[j])
                    dist = np.linalg.norm(env.fk(qq) - env.target[i])
                    if best is None or dist < best[0]:
                        best = (dist, j, s)
            act[i, best[1]] = best[2]
        _, r, _, info = env.step(act)
        first = r.copy() if first is None else first
    assert (r > first).all()
    assert (info["dist"] < 3 * REACH_TOL).all()
    env.close()


def test_wilson_and_gae():
    lo, hi = wilson(0, 200)
    assert lo == 0.0 and 0.0 < hi < 0.03
    lo, hi = wilson(180, 200)
    assert 0.85 < lo < 0.9 < hi < 0.94
    rewards = np.ones((3, 2), dtype=np.float32)
    values = np.zeros((3, 2), dtype=np.float32)
    dones = np.zeros((3, 2), dtype=np.float32)
    dones[-1] = 1
    adv, ret = compute_gae(rewards, values, dones, np.zeros(2, dtype=np.float32), gamma=1.0, lam=1.0)
    assert np.allclose(ret[:, 0], [3, 2, 1])


def test_ppo_update_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = ActorCritic(14, 5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    cfg = PPOConfig(minibatch=64, epochs=2)
    n = 128
    obs = torch.randn(n, 14)
    with torch.no_grad():
        d = model.dist(obs)
        act = d.sample()
        logp = d.log_prob(act).sum(-1)
    batch = {"obs": obs, "act": act, "logp": logp, "adv": torch.randn(n), "ret": torch.randn(n)}
    before = model.actor[0].weight.detach().clone()
    st = ppo_update(model, opt, batch, cfg)
    assert np.isfinite(st["policy_loss"]) and np.isfinite(st["value_loss"])
    assert not torch.allclose(before, model.actor[0].weight)
    rms = RunningMeanStd((14,))
    rms.update(np.random.default_rng(0).normal(size=(50, 14)))
    path = str(tmp_path / "ck.pt")
    save_checkpoint(path, model, rms, cfg, DomainRandomization.train(), meta={"x": 1})
    m2, rms2, ck = load_checkpoint(path)
    with torch.no_grad():
        assert torch.allclose(model.dist(obs).mean, m2.dist(obs).mean)
    assert np.allclose(rms.mean, rms2.mean) and np.allclose(rms.var, rms2.var)
    assert ck["dr"]["enabled"] is True and ck["meta"]["x"] == 1


def test_tiny_train_loop(model_xml, tmp_path):
    env = VecArmEnv(4, "reach", seed=0, model=model_xml, n_threads=2)
    cfg = PPOConfig(task="reach", n_envs=4, n_steps=40, total_steps=320, minibatch=64, epochs=1, device="cpu", max_minutes=1.0)
    res = train(cfg, results_path=str(tmp_path / "r.json"), ckpt_path=str(tmp_path / "c.pt"), env=env, verbose=False)
    assert res["total_steps"] == 320 and len(res["curve"]) >= 1
    assert os.path.exists(tmp_path / "c.pt") and os.path.exists(tmp_path / "r.json")
    env.close()
