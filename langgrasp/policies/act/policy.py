"""Closed-loop runner for a trained LeRobot 0.4.4 ACT policy on the LangGrasp simulator.

Preprocessing mirrors lerobot/scripts/lerobot_eval.py: raw uint8 HWC images become float CHW in [0, 1]
(lerobot.envs.utils.preprocess_observation), then the saved policy_preprocessor pipeline (rename, add batch dim,
device, mean/std normalisation with the training-dataset stats) runs before select_action, and the saved
policy_postprocessor un-normalises the action and moves it to the CPU.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from langgrasp.policies.act.data import ACTION_KEY, FRONT_KEY, STATE_KEY, WRIST_KEY


def _to_chw_float(img: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) -> float32 (3, H, W) in [0, 1] (same as lerobot preprocess_observation)."""
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"expected uint8 pixels, got {img.dtype}")
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


class ACTRunner:
    """Wraps ACTPolicy + its processor pipelines. act() returns the next 6-D joint target (5 arm + jaw)."""

    def __init__(self, ckpt_dir: str | Path, device: str = "cuda", warmup: int = 3):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.ckpt_dir = Path(ckpt_dir)
        if not (self.ckpt_dir / "config.json").exists():
            raise FileNotFoundError(f"no config.json in {self.ckpt_dir}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        cfg = PreTrainedConfig.from_pretrained(str(self.ckpt_dir))
        cfg.device = device
        cfg.pretrained_path = self.ckpt_dir
        self.config = cfg
        self.policy: ACTPolicy = ACTPolicy.from_pretrained(str(self.ckpt_dir), config=cfg)
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(self.ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.image_hw = self._image_hw()
        self.state_dim = int(cfg.input_features[STATE_KEY].shape[0]) if STATE_KEY in cfg.input_features else 0
        self.latencies_ms: list[float] = []
        self.warmup_ms: list[float] = []
        self.reset()
        for _ in range(warmup):
            h, w = self.image_hw
            self._act(np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 3), np.uint8), np.zeros(self.state_dim, np.float32), record=False)
        self.reset()

    def _image_hw(self) -> tuple[int, int]:
        ft = self.config.input_features[FRONT_KEY]
        c, h, w = ft.shape
        return int(h), int(w)

    def reset(self) -> None:
        """Clear the action-chunk queue; call at every episode start."""
        self.policy.reset()

    def build_batch(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray) -> dict:
        """Raw observation -> normalised, batched, on-device dict as the policy expects it."""
        obs = {
            FRONT_KEY: _to_chw_float(front_rgb),
            WRIST_KEY: _to_chw_float(wrist_rgb),
            STATE_KEY: torch.as_tensor(np.asarray(state, dtype=np.float32)),
        }
        return self.pre(obs)

    def _act(self, front_rgb, wrist_rgb, state, record: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        batch = self.build_batch(front_rgb, wrist_rgb, state)
        with torch.inference_mode():
            a = self.policy.select_action(batch)
        a = self.post(a)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        (self.latencies_ms if record else self.warmup_ms).append(ms)
        return a[0].detach().cpu().numpy().astype(np.float32)

    def act(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray) -> np.ndarray:
        """One control tick: returns the 6-D joint target (5 arm joints + jaw). Chunks are re-planned every
        n_action_steps ticks by ACTPolicy.select_action; intermediate ticks pop from the queue."""
        return self._act(front_rgb, wrist_rgb, state, record=True)

    def latency(self) -> dict:
        """median / p90 per-call latency in ms over all recorded act() calls (warmup excluded)."""
        from langgrasp.eval.stats import latency_summary

        return latency_summary(self.latencies_ms)

    @property
    def action_key(self) -> str:
        return ACTION_KEY
