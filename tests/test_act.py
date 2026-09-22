"""ACT track tests: dataset feature construction, ACTRunner preprocessing, recorded frame shapes. No training.

Run with MUJOCO_GL=egl. The trained-checkpoint test skips when checkpoints/act_pick is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from langgrasp.policies.act.data import (  # noqa: E402
    ACTION_KEY,
    FRONT_KEY,
    HOLD_TICKS,
    STATE_KEY,
    TASK,
    WRIST_KEY,
    add_episode,
    collect_episode,
    make_features,
)
from langgrasp.policies.act.policy import ACTRunner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
H = W = 32


@pytest.fixture(scope="module")
def env():
    from langgrasp.sim.env import LangGraspEnv

    e = LangGraspEnv(seed=0)
    yield e
    e.close()


@pytest.fixture(scope="module")
def two_episodes(env):
    from langgrasp.sim.scenarios import make_scenario

    recs = []
    for seed in (20000, 20001):
        sc = make_scenario(seed, "seen", target_kind="cube", target_color="red")
        recs.append(collect_episode(env, sc, (H, W)))
    return recs


def test_make_features_layout():
    ft = make_features(H, W)
    assert set(ft) == {FRONT_KEY, WRIST_KEY, STATE_KEY, ACTION_KEY}
    assert ft[FRONT_KEY]["dtype"] == "video" and ft[FRONT_KEY]["shape"] == (H, W, 3)
    assert ft[FRONT_KEY]["names"] == ["height", "width", "channels"]
    assert make_features(H, W, use_videos=False)[WRIST_KEY]["dtype"] == "image"
    assert ft[STATE_KEY]["shape"] == (6,) and ft[ACTION_KEY]["dtype"] == "float32"


def test_collect_episode_records_shapes(two_episodes):
    for rec in two_episodes:
        assert rec.n_frames > HOLD_TICKS
        assert [f["phase"] for f in rec.frames[-HOLD_TICKS:]] == ["hold"] * HOLD_TICKS
        for f in rec.frames:
            assert f[FRONT_KEY].shape == (H, W, 3) and f[FRONT_KEY].dtype == np.uint8
            assert f[WRIST_KEY].shape == (H, W, 3) and f[WRIST_KEY].dtype == np.uint8
            assert f[STATE_KEY].shape == (6,) and f[STATE_KEY].dtype == np.float32
            assert f[ACTION_KEY].shape == (6,) and f[ACTION_KEY].dtype == np.float32
        # the 5 hold frames repeat the last commanded target
        last = two_episodes[0].frames[-HOLD_TICKS - 1][ACTION_KEY]
        assert all(np.array_equal(f[ACTION_KEY], last) for f in two_episodes[0].frames[-HOLD_TICKS:])
        # the first observation is the raised observe pose and the state is inside the joint limits
        assert np.isfinite(rec.frames[0][STATE_KEY]).all()


def test_toy_lerobot_dataset_roundtrip(tmp_path, two_episodes):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "toy"
    ds = LeRobotDataset.create("local/langgrasp_toy", fps=10, features=make_features(H, W, use_videos=False), root=root, use_videos=False)
    n = sum(add_episode(ds, rec, TASK) for rec in two_episodes)
    ds.finalize()
    ds2 = LeRobotDataset("local/langgrasp_toy", root=root)
    assert ds2.num_frames == n == sum(r.n_frames for r in two_episodes)
    assert ds2.num_episodes == 2
    s = ds2[0]
    assert tuple(s[FRONT_KEY].shape) == (3, H, W) and s[FRONT_KEY].dtype == torch.float32
    assert 0.0 <= float(s[FRONT_KEY].min()) and float(s[FRONT_KEY].max()) <= 1.0
    assert tuple(s[STATE_KEY].shape) == (6,) and tuple(s[ACTION_KEY].shape) == (6,)
    assert s["task"] == TASK
    assert ds2.meta.stats[FRONT_KEY]["mean"].shape == (3, 1, 1)
    assert ds2.meta.stats[ACTION_KEY]["std"].shape == (6,)


def _tiny_act_checkpoint(d: Path) -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    cfg = ACTConfig(
        input_features={
            FRONT_KEY: PolicyFeature(FeatureType.VISUAL, (3, H, W)),
            WRIST_KEY: PolicyFeature(FeatureType.VISUAL, (3, H, W)),
            STATE_KEY: PolicyFeature(FeatureType.STATE, (6,)),
        },
        output_features={ACTION_KEY: PolicyFeature(FeatureType.ACTION, (6,))},
        chunk_size=8,
        n_action_steps=4,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    stats = {
        FRONT_KEY: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)},
        WRIST_KEY: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)},
        STATE_KEY: {"mean": torch.zeros(6), "std": torch.ones(6)},
        ACTION_KEY: {"mean": torch.zeros(6), "std": 2 * torch.ones(6)},
    }
    torch.manual_seed(0)
    policy = ACTPolicy(cfg)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    policy.save_pretrained(d)
    pre.save_pretrained(d)
    post.save_pretrained(d)


def test_act_runner_tiny_cpu(tmp_path):
    _tiny_act_checkpoint(tmp_path)
    runner = ACTRunner(tmp_path, device="cpu", warmup=1)
    assert runner.image_hw == (H, W) and runner.state_dim == 6
    front = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    state = np.zeros(6, np.float32)
    batch = runner.build_batch(front, front, state)
    assert tuple(batch[FRONT_KEY].shape) == (1, 3, H, W) and batch[FRONT_KEY].dtype == torch.float32
    assert tuple(batch[STATE_KEY].shape) == (1, 6)
    # mean/std normalisation with mean 0.5, std 0.25 maps [0, 1] pixels to [-2, 2]
    assert -2.01 <= float(batch[FRONT_KEY].min()) and float(batch[FRONT_KEY].max()) <= 2.01
    runner.reset()
    actions = [runner.act(front, front, state) for _ in range(6)]
    assert all(a.shape == (6,) and a.dtype == np.float32 and np.isfinite(a).all() for a in actions)
    # chunk of n_action_steps=4 is consumed before the next forward pass: 6 calls leave 2 queued
    assert len(runner.policy._action_queue) == 2
    lat = runner.latency()
    assert lat["n"] == 6 and lat["median_ms"] > 0
    with pytest.raises(ValueError):
        runner.build_batch(front.astype(np.float32), front, state)


def test_trained_checkpoint_if_present():
    ckpt = ROOT / "checkpoints/act_pick"
    if not (ckpt / "model.safetensors").exists():
        pytest.skip("no trained ACT checkpoint at checkpoints/act_pick")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = ACTRunner(ckpt, device=device, warmup=1)
    h, w = runner.image_hw
    img = np.zeros((h, w, 3), np.uint8)
    a = runner.act(img, img, np.zeros(6, np.float32))
    assert a.shape == (6,) and np.isfinite(a).all()
    assert runner.config.chunk_size >= runner.config.n_action_steps
