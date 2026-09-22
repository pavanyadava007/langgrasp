"""Scripted Cartesian pick-and-place controller (the "modular" policy executor and the demo generator)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from langgrasp.sim.kinematics import JAW_CLOSED, JAW_OPEN
from langgrasp.sim.scene import TRAY_CENTER

# Heights are TCP (fingertip) heights. Measured reachability (scripts/smoke_pick.py, kinematics grid):
# strict top-down pose reachable up to z=0.065 within r<=0.27 m; relaxed (tilted) up to z=0.085.
PRE_HOVER_Z = 0.085
HOVER_Z = 0.065
LIFT_Z = 0.08
PLACE_TIP_Z = 0.02
RELAXED_ROT_W = 0.02  # transit phases tolerate a tilted gripper (5-DOF arm cannot stay top-down high up)
MAX_STEP_M = 0.02  # per control tick (10 Hz) -> 0.2 m/s
MIN_TIP_Z = 0.012  # the jaw collision mesh touches the table when the TCP site is at 9.4 mm
FIXED_PAD_X = 0.009  # inner face of the fixed jaw pads, along +x of the TCP frame
PAD_CLEARANCE = 0.010
TRACK_GAIN = 0.7  # integral correction of the servo tracking error (gravity sag, lag), clipped
TRACK_CLIP = 0.02
TRACK_MASK = np.array([1.0, 1.0, 0.0])  # correct x,y only: a z correction would push the fingertip into the table
DESCEND_STEP_M = 0.01  # slower vertical approach: the position servos lag about 1 cm at full speed


@dataclass
class RunResult:
    grasped: bool = False
    lifted: bool = False
    placed: bool = False
    ik_ok: bool = True
    steps: int = 0
    max_pos_err: float = 0.0
    rot_err: float = 0.0
    trajectory: list = field(default_factory=list)  # per-tick dicts if recording


class PickPlaceController:
    def __init__(self, env, record: bool = False, record_fn=None):
        self.env = env
        self.kin = env.kin
        self.record = record
        self.record_fn = record_fn
        self.result = RunResult()
        self.psi = 0.0

    def _tick(self, q_arm: np.ndarray, jaw: float, phase: str):
        if self.record:
            frame = self.record_fn(self.env, phase) if self.record_fn else {}
            frame.update({"action": np.concatenate([q_arm, [jaw]]).astype(np.float32), "phase": phase})
            self.result.trajectory.append(frame)
        self.env.step(q_arm, jaw)
        self.result.steps += 1

    def move_to(self, target: np.ndarray, jaw: float, phase: str, hold_ticks: int = 0, strict: bool = True, step_m: float = MAX_STEP_M) -> bool:
        """Straight-line Cartesian move of the TCP with IK every tick. strict=False relaxes the orientation term."""
        q = self.env.q_arm
        rot_w = 0.3 if strict else RELAXED_ROT_W
        cur = self.kin.fk(q)[0]
        d = np.linalg.norm(target - cur)
        n = max(1, int(np.ceil(d / step_m)))
        corr = np.zeros(3)
        pe = 0.0
        for i in range(1, n + 1):
            wp = cur + (target - cur) * (i / n)
            q, pe, _ = self.kin.solve(wp + corr, self.psi, q, rot_w=rot_w)
            self.result.max_pos_err = max(self.result.max_pos_err, pe)
            self._tick(q, jaw, phase)
            corr = np.clip(corr + TRACK_GAIN * TRACK_MASK * (wp - self.env.obs()["tcp_pos"]), -TRACK_CLIP, TRACK_CLIP)
        for _ in range(hold_ticks):
            q, pe, _ = self.kin.solve(target + corr, self.psi, q, rot_w=rot_w)
            self._tick(q, jaw, phase)
            corr = np.clip(corr + TRACK_GAIN * TRACK_MASK * (target - self.env.obs()["tcp_pos"]), -TRACK_CLIP, TRACK_CLIP)
        # ik_ok refers to the final waypoint only (intermediate waypoints near the start pose may be unreachable)
        return pe < 0.01

    def run(self, grasp_xyz: np.ndarray, psi: float, target_name: str | None = None, place: bool = True, width: float = 0.025) -> RunResult:
        """grasp_xyz: object grasp centre (world). psi: jaw-axis yaw. width: object extent along the jaw axis.

        The fixed jaw does not move, so the TCP is offset along +jaw-axis such that the fixed pad clears the
        object; closing then pushes the object against the fixed pad.
        """
        env = self.env
        self.result = RunResult()
        gx0, gy0, _ = np.asarray(grasp_xyz, dtype=float)
        self.psi, rot_err = self.kin.best_psi(psi, env.q_arm, np.array([gx0, gy0, HOVER_Z]))
        self.result.rot_err = rot_err
        jaw_dir = np.array([np.cos(self.psi), np.sin(self.psi), 0.0])
        offset = max(0.0, width / 2 + PAD_CLEARANCE - FIXED_PAD_X)
        gx, gy, gz = np.asarray(grasp_xyz, dtype=float) + offset * jaw_dir
        tip_z = max(MIN_TIP_Z, gz - 0.006)
        ok = self.move_to(np.array([gx, gy, PRE_HOVER_Z]), JAW_OPEN, "approach", strict=False)
        ok &= self.move_to(np.array([gx, gy, HOVER_Z]), JAW_OPEN, "hover", hold_ticks=3)
        ok &= self.move_to(np.array([gx, gy, tip_z]), JAW_OPEN, "descend", hold_ticks=3, step_m=DESCEND_STEP_M)
        for _ in range(6):
            self._tick(env.q_arm, JAW_CLOSED, "close")
        if target_name:
            self.result.grasped = env.is_grasped(target_name)
        ok &= self.move_to(np.array([gx, gy, LIFT_Z]), JAW_CLOSED, "lift", hold_ticks=3, strict=False)
        if target_name:
            self.result.lifted = env.is_lifted(target_name) and env.is_grasped(target_name)
        if place:
            tx, ty, _ = TRAY_CENTER
            ok &= self.move_to(np.array([tx, ty, LIFT_Z]), JAW_CLOSED, "transport", strict=False)
            ok &= self.move_to(np.array([tx, ty, max(tip_z, PLACE_TIP_Z)]), JAW_CLOSED, "lower")
            for _ in range(4):
                self._tick(env.q_arm, JAW_OPEN, "release")
            ok &= self.move_to(np.array([tx, ty, LIFT_Z]), JAW_OPEN, "retreat", strict=False)
            for _ in range(3):
                self._tick(env.q_arm, JAW_OPEN, "settle")
            if target_name:
                self.result.placed = env.in_tray(target_name)
        self.result.ik_ok = ok
        return self.result

    def go_home(self, ticks: int = 8):
        for _ in range(ticks):
            self._tick(self.env.home_q[:5], JAW_OPEN, "home")

    def go_observe(self, ticks: int = 8):
        for _ in range(ticks):
            self._tick(self.env.observe_q, JAW_OPEN, "observe")
