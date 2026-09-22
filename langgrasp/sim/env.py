"""MuJoCo environment for language-guided pick-and-place on the SO-ARM100."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

from langgrasp.sim.kinematics import ARM_JOINTS, JAW_JOINT, ArmKinematics
from langgrasp.sim.scenarios import Scenario
from langgrasp.sim.scene import (
    ALL_COLORS,
    ASSET_DIR,
    OBJECT_KINDS,
    PARK_POS,
    POOL,
    TRAY_CENTER,
    TRAY_HALF,
    SceneConfig,
    arm_xml_with_tcp,
    build_scene_xml,
)

CONTROL_HZ = 10
# Retracted, raised pose used while the cameras look at the workspace (the home pose occludes the front view).
OBSERVE_Q = [0.0, -3.2, 2.0, 1.6, -1.57]  # raised: every link above z=0.14 and behind y=-0.155 (scan in scripts/smoke_pick.py history)
PAD_GEOMS = [f"fixed_jaw_pad_{i}" for i in range(1, 5)] + [f"moving_jaw_pad_{i}" for i in range(1, 5)]


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class LangGraspEnv:
    def __init__(self, cfg: SceneConfig | None = None, seed: int = 0, render: bool = True):
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg or SceneConfig()
        self.rng = np.random.default_rng(seed)
        self._tmp = tempfile.mkdtemp(prefix="langgrasp_")
        arm_xml = arm_xml_with_tcp(self.cfg).replace('meshdir="assets/"', f'meshdir="{os.path.join(ASSET_DIR, "assets")}/"')
        with open(os.path.join(self._tmp, "so_arm100.xml"), "w") as f:
            f.write(arm_xml)
        with open(os.path.join(self._tmp, "scene.xml"), "w") as f:
            f.write(build_scene_xml(self.cfg))
        self.model = mujoco.MjModel.from_xml_path(os.path.join(self._tmp, "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.kin = ArmKinematics(self.model)
        self.n_substeps = int(round(1.0 / (CONTROL_HZ * self.model.opt.timestep)))
        self.arm_qadr = self.kin.qpos_adr
        self.arm_act = np.array([self.model.actuator(n).id for n in ARM_JOINTS])
        self.jaw_act = self.model.actuator(JAW_JOINT).id
        self.jaw_qadr = self.kin.jaw_qpos_adr
        self.home_q = self.model.key("home").qpos[:6].copy()
        self.observe_q = np.array(OBSERVE_Q)
        self.pad_ids = [self.model.geom(g).id for g in PAD_GEOMS]
        self.fixed_pad_ids = self.pad_ids[:4]
        self.moving_pad_ids = self.pad_ids[4:]
        self.obj_geoms = {f"{k}_{i}": [self.model.geom(f"{k}_{i}_g0").id] + ([self.model.geom(f"{k}_{i}_g1").id] if k == "screwdriver" else []) for k, i in POOL}
        self.geom_to_obj = {g: n for n, gs in self.obj_geoms.items() for g in gs}
        self.table_gid = self.model.geom("table").id
        self._renderers: dict[tuple[int, int], Any] = {}
        self.light_diffuse0 = self.model.light_diffuse.copy()
        self.light_pos0 = self.model.light_pos.copy()
        self.scenario: Scenario | None = None
        self.t = 0
        self.render_enabled = render

    # ------------------------------------------------------------------ setup
    def _renderer(self, h: int, w: int):
        key = (h, w)
        if key not in self._renderers:
            self._renderers[key] = self.mujoco.Renderer(self.model, height=h, width=w)
        return self._renderers[key]

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    def _park_all(self):
        for k, i in POOL:
            name = f"{k}_{i}"
            j = self.model.joint(f"{name}_free")
            adr = j.qposadr[0]
            z = OBJECT_KINDS[k]["grasp_z"]
            self.data.qpos[adr : adr + 3] = [PARK_POS[0] + 0.1 * i, PARK_POS[1] + 0.1 * (list(OBJECT_KINDS).index(k)), z]
            self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
            self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0

    def _place(self, name: str, kind: str, color: str, xy: tuple, yaw: float):
        j = self.model.joint(f"{name}_free")
        adr = j.qposadr[0]
        z = OBJECT_KINDS[kind]["grasp_z"] + 0.002
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3 : adr + 7] = yaw_to_quat(yaw)
        self.data.qvel[j.dofadr[0] : j.dofadr[0] + 6] = 0
        gid = self.obj_geoms[name][0]
        self.model.geom_rgba[gid] = ALL_COLORS[color]

    def _set_lighting(self, mode: str, rng: np.random.Generator):
        self.model.light_diffuse[:] = self.light_diffuse0
        self.model.light_pos[:] = self.light_pos0
        if mode == "degraded":
            scale = rng.uniform(0.25, 0.45)
            self.model.light_diffuse[:] = self.light_diffuse0 * scale
            self.model.light_pos[:, :2] += rng.uniform(-0.3, 0.3, size=(self.model.nlight, 2))
        else:
            self.model.light_diffuse[:] = self.light_diffuse0 * rng.uniform(0.85, 1.15)
            self.model.light_pos[:, :2] += rng.uniform(-0.08, 0.08, size=(self.model.nlight, 2))

    def reset(self, scenario: Scenario | None = None, settle_steps: int = 5, observe_pose: bool = True) -> dict:
        self.mujoco.mj_resetData(self.model, self.data)
        q0 = np.concatenate([self.observe_q, [self.home_q[5]]]) if observe_pose else self.home_q
        self.data.qpos[:6] = q0
        self.data.ctrl[:6] = q0
        self._park_all()
        self.scenario = scenario
        rng = np.random.default_rng(scenario.seed if scenario else int(self.rng.integers(1 << 31)))
        if scenario is not None:
            for o in scenario.objects:
                self._place(o.name, o.kind, o.color, o.pos, o.yaw)
            self._set_lighting(scenario.lighting, rng)
        self.mujoco.mj_forward(self.model, self.data)
        for _ in range(settle_steps):
            self.step()
        self.t = 0
        return self.obs()

    # ------------------------------------------------------------------ stepping
    def set_targets(self, q_arm: np.ndarray | None = None, jaw: float | None = None):
        if q_arm is not None:
            self.data.ctrl[self.arm_act] = np.clip(q_arm, self.kin.lo, self.kin.hi)
        if jaw is not None:
            self.data.ctrl[self.jaw_act] = float(np.clip(jaw, *self.kin.jaw_range))

    def step(self, q_arm: np.ndarray | None = None, jaw: float | None = None) -> dict:
        self.set_targets(q_arm, jaw)
        self.mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.t += 1
        return self.obs()

    def obs(self) -> dict:
        return {
            "q_arm": self.data.qpos[self.arm_qadr].copy(),
            "jaw": float(self.data.qpos[self.jaw_qadr]),
            "ctrl": self.data.ctrl[:6].copy(),
            "tcp_pos": self.data.site_xpos[self.kin.site_id].copy(),
            "t": self.t,
        }

    @property
    def q_arm(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    # ------------------------------------------------------------------ objects and success
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.model.body(name).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def grasp_point(self, name: str) -> tuple[np.ndarray, float]:
        """Ground-truth grasp point (world) and jaw yaw psi for an object (oracle for demos / upper bound)."""
        kind = name.rsplit("_", 1)[0]
        p, R = self.object_pose(name)
        long_axis = R[:, 0]
        yaw = float(np.arctan2(long_axis[1], long_axis[0]))
        if kind == "screwdriver":
            p = p + R @ np.array([-0.03, 0.0, 0.0])  # handle centre
            return p, yaw + np.pi / 2
        if kind == "bar":
            return p, yaw + np.pi / 2
        if kind == "cube":
            return p, yaw + np.pi / 2
        return p, 0.0

    def contacts_with(self, name: str) -> tuple[bool, bool]:
        """(fixed_pad_touching, moving_pad_touching) for an object."""
        gs = set(self.obj_geoms[name])
        fixed = moving = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            a, b = c.geom1, c.geom2
            if a in gs or b in gs:
                other = b if a in gs else a
                if other in self.fixed_pad_ids:
                    fixed = True
                if other in self.moving_pad_ids:
                    moving = True
        return fixed, moving

    def object_height(self, name: str) -> float:
        kind = name.rsplit("_", 1)[0]
        return float(self.object_pose(name)[0][2] - OBJECT_KINDS[kind]["grasp_z"])

    def is_lifted(self, name: str, min_height: float = 0.05) -> bool:
        return self.object_height(name) > min_height

    def is_grasped(self, name: str) -> bool:
        f, m = self.contacts_with(name)
        return f and m

    def in_tray(self, name: str) -> bool:
        p, _ = self.object_pose(name)
        return bool(abs(p[0] - TRAY_CENTER[0]) < TRAY_HALF and abs(p[1] - TRAY_CENTER[1]) < TRAY_HALF and p[2] < 0.08)

    # ------------------------------------------------------------------ rendering
    def render(self, camera: str = "front", size: tuple[int, int] | None = None, depth: bool = False, seg: bool = False) -> np.ndarray:
        h, w = size or (self.cfg.cam_height, self.cfg.cam_width)
        r = self._renderer(h, w)
        if depth:
            r.enable_depth_rendering()
        elif seg:
            r.enable_segmentation_rendering()
        try:
            r.update_scene(self.data, camera=camera)
            out = r.render().copy()
        finally:
            if depth:
                r.disable_depth_rendering()
            if seg:
                r.disable_segmentation_rendering()
        return out

    def render_rgbd_seg(self, camera: str = "front", size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB (H,W,3 uint8), depth in metres (H,W float32), per-pixel object label (H,W int: -1 background, else pool index)."""
        rgb = self.render(camera, size)
        depth = self.render(camera, size, depth=True)
        seg = self.render(camera, size, seg=True)
        geom_ids = seg[..., 0]
        label = np.full(geom_ids.shape, -1, dtype=np.int32)
        names = list(self.obj_geoms)
        for gid, obj in self.geom_to_obj.items():
            label[geom_ids == gid] = names.index(obj)
        return rgb, depth.astype(np.float32), label

    def camera_intrinsics(self, camera: str, size: tuple[int, int] | None = None) -> np.ndarray:
        h, w = size or (self.cfg.cam_height, self.cfg.cam_width)
        fovy = float(self.model.cam_fovy[self.model.camera(camera).id])
        fy = 0.5 * h / np.tan(np.deg2rad(fovy) / 2)
        return np.array([[fy, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]])

    def camera_extrinsics(self, camera: str) -> np.ndarray:
        """4x4 T_world_cam (MuJoCo camera convention: looks along -z, y up)."""
        cid = self.model.camera(camera).id
        T = np.eye(4)
        T[:3, :3] = self.data.cam_xmat[cid].reshape(3, 3)
        T[:3, 3] = self.data.cam_xpos[cid]
        return T

    def object_names(self) -> list[str]:
        return list(self.obj_geoms)
