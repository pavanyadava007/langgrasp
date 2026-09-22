"""Dataset construction for the ACT track: scripted-expert demonstrations written as a LeRobotDataset (v3.0).

The "teleoperator" is the scripted Cartesian PickPlaceController (langgrasp/sim/controller.py), not a human.
Observations are recorded BEFORE each 10 Hz control tick (the controller's record_fn contract), so frame t
pairs the images/state seen at tick t with the joint target commanded at tick t.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from langgrasp.sim.scene import OBJECT_KINDS

TASK = "pick the red cube and place it in the tray"
FRONT_KEY = "observation.images.front"
WRIST_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "jaw"]
HOLD_TICKS = 5  # extra ticks holding the last target so every episode ends at rest


def make_features(h: int, w: int, use_videos: bool = True) -> dict:
    """LeRobotDataset feature dict: two RGB cameras (video or png), 6-D state, 6-D action."""
    img = {"dtype": "video" if use_videos else "image", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return {
        FRONT_KEY: dict(img),
        WRIST_KEY: dict(img),
        STATE_KEY: {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
        ACTION_KEY: {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
    }


def observe(env, size: tuple[int, int]) -> dict:
    """Policy observation: front + wrist RGB (H, W, 3 uint8) and the 6-D joint state (5 arm + jaw)."""
    return {
        FRONT_KEY: env.render("front", size=size),
        WRIST_KEY: env.render("wrist", size=size),
        STATE_KEY: np.concatenate([env.q_arm, [env.obs()["jaw"]]]).astype(np.float32),
    }


@dataclass
class EpisodeRecord:
    seed: int
    frames: list = field(default_factory=list)  # dicts with the 4 feature keys + "phase"
    grasped: bool = False
    lifted: bool = False
    placed: bool = False
    seconds: float = 0.0

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def collect_episode(env, scenario, size: tuple[int, int], hold_ticks: int = HOLD_TICKS) -> EpisodeRecord:
    """Run the scripted expert on one scenario and return the recorded (obs, action) frames.

    The episode is appended with `hold_ticks` extra frames that repeat the last commanded target so the arm
    comes to rest; the success flags are re-checked after the hold.
    """
    from langgrasp.sim.controller import PickPlaceController

    t0 = time.time()
    env.reset(scenario)
    kind = scenario.target_obj.kind
    ctl = PickPlaceController(env, record=True, record_fn=lambda e, phase: observe(e, size))
    grasp_xyz, psi = env.grasp_point(scenario.target)
    r = ctl.run(grasp_xyz, psi, scenario.target, width=OBJECT_KINDS[kind]["width"])
    frames = list(r.trajectory)
    last = frames[-1][ACTION_KEY]
    for _ in range(hold_ticks):
        f = observe(env, size)
        f[ACTION_KEY] = last.copy()
        f["phase"] = "hold"
        frames.append(f)
        env.step(last[:5], float(last[5]))
    rec = EpisodeRecord(seed=scenario.seed, frames=frames, grasped=bool(r.grasped), lifted=bool(r.lifted))
    rec.placed = bool(r.placed and env.in_tray(scenario.target))
    rec.seconds = time.time() - t0
    return rec


def add_episode(ds, rec: EpisodeRecord, task: str = TASK) -> int:
    """Write one recorded episode into a LeRobotDataset created with make_features(). Returns frame count."""
    for f in rec.frames:
        ds.add_frame(
            {
                FRONT_KEY: np.ascontiguousarray(f[FRONT_KEY]),
                WRIST_KEY: np.ascontiguousarray(f[WRIST_KEY]),
                STATE_KEY: np.asarray(f[STATE_KEY], dtype=np.float32),
                ACTION_KEY: np.asarray(f[ACTION_KEY], dtype=np.float32),
                "task": task,
            }
        )
    ds.save_episode()
    return rec.n_frames
