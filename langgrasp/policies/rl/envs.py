"""Fast vectorised, state-based MuJoCo environments for PPO on the simulated SO-101 arm.

One MjModel per env (cheap copies of one compiled model, about 1 MB each) so that physical domain
randomisation (object mass / friction, actuator kp) can be applied per env; one MjData per env;
physics is stepped in a thread pool (mj_step releases the GIL). No renderer: observations are joint
positions plus forward kinematics, the target / cube position is assumed to come from perception.

The RL model is the same arm and table as langgrasp.sim.scene, with only cube_0 kept from the object
pool (the other nine pool objects are parked out of view in the full scene anyway; stripping them
cuts nq from 76 to 13 and gives about a 4x speed-up per physics step).

Episodes start from the model's "home" keyframe (TCP at about (0, -0.24, 0.08) m, inside the workspace)
plus uniform +-0.15 rad joint noise.

Tasks
    reach : move the TCP to a random point in the workspace box, z in [0.03, 0.09]. Horizon 40 ticks.
    lift  : grasp cube_0 (random pose in the workspace, yaw in +-45 deg) and lift it 5 cm. Horizon 60 ticks.
            obs (19) = q(5), tcp(3), cube_pos(3), cube - tcp(3), jaw(1), jaw axis xy(2), cube x-axis xy(2).
            reward = -|tcp - grasp_point| - 0.3 * jaw_closed_fraction * [dist > 2 cm] + 0.1 * point_down
                     + 0.5 * exp(-dist / 1 cm) + 0.3 * jaw_closed_fraction * exp(-dist / 1 cm)
                     + 0.5 * grasped + 20 * cube_height * grasped + 5 * lifted
            grasp_point = cube centre - 2.5 mm in z + 8.5 mm along the jaw axis (fixed-pad clearance, as
            in the scripted controller); grasped = both jaw pad groups touch the cube.

Actions are in [-1, 1]^k and are integrated into joint-position targets (0.12 rad per 10 Hz tick for
the arm, 0.5 rad per tick for the jaw), clipped to the joint limits.

Domain randomisation (Squint-style: physical + proprioceptive, no vision)
    q_noise_deg      per-tick Gaussian noise on the reported joint positions (5 deg)
    action_noise     Gaussian noise on the normalised action (0.1)
    latency          with probability latency_prob per episode the applied action is delayed by
                     latency_ticks control ticks
    mass_range       cube mass (and inertia) scale, per episode
    friction_range   cube sliding friction scale, per episode
    kp_range         arm position-actuator kp scale (gainprm[0] and biasprm[1]; kv unchanged), per episode
    q_offset_deg     constant calibration offset: the true joint angle is (commanded - offset) and the
                     reported joint angle is (true + offset). Not used in training, only in the shifted
                     evaluation env.
The reported TCP position in the observation is forward kinematics of the REPORTED joint angles
(noise and offset included), as it would be on a real robot with encoders only. Gravity is unchanged.

Success definitions (also in docs/RL.md)
    reach : true |tcp - target| < 0.015 m at ANY tick of the episode ("reached"). The final-tick version
            ("hold") is reported separately in info.
    lift  : cube lifted > 0.05 m above its rest height while both jaw pads touch it, at any tick.
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from langgrasp.sim.env import CONTROL_HZ, PAD_GEOMS
from langgrasp.sim.kinematics import ARM_JOINTS, JAW_CLOSED, JAW_JOINT, JAW_OPEN, ArmKinematics
from langgrasp.sim.scene import (
    ASSET_DIR,
    OBJECT_KINDS,
    PARK_POS,
    POOL,
    WORKSPACE,
    SceneConfig,
    arm_xml_with_tcp,
    build_scene_xml,
)

TASKS = ("reach", "lift")
CUBE = "cube_0"
ARM_STEP_RAD = 0.12
JAW_STEP_RAD = 0.5
REACH_TOL = 0.015
LIFT_HEIGHT = 0.05
HORIZON = {"reach": 40, "lift": 60}
OBS_DIM = {"reach": 14, "lift": 19}
ACT_DIM = {"reach": 5, "lift": 6}
TARGET_Z = (0.03, 0.09)
CUBE_HALF = OBJECT_KINDS["cube"]["grasp_z"]  # 0.0125 m
GRASP_TCP_DZ = -0.0025  # TCP (fingertip) sits 2.5 mm below the cube centre when grasping (controller uses tip_z = gz - 0.006, floor 0.010)
GRASP_JAW_OFFSET = 0.0085  # TCP offset along the jaw axis so the fixed pad clears the cube (controller: w/2 + clearance - FIXED_PAD_X)
CLOSE_RADIUS = 0.02  # closing the jaw further than this from the grasp point is penalised


@dataclass
class DomainRandomization:
    """Ranges are sampled per episode per env. Collapse a range to a point (a, a) for a fixed shift."""

    enabled: bool = False
    q_noise_deg: float = 0.0
    action_noise: float = 0.0
    latency_prob: float = 0.0
    latency_ticks: int = 1
    mass_range: tuple[float, float] = (1.0, 1.0)
    friction_range: tuple[float, float] = (1.0, 1.0)
    kp_range: tuple[float, float] = (1.0, 1.0)
    q_offset_deg: float = 0.0

    @classmethod
    def none(cls) -> DomainRandomization:
        return cls()

    @classmethod
    def train(cls) -> DomainRandomization:
        return cls(
            enabled=True,
            q_noise_deg=5.0,
            action_noise=0.1,
            latency_prob=0.5,
            latency_ticks=1,
            mass_range=(0.7, 1.3),
            friction_range=(0.7, 1.3),
            kp_range=(0.8, 1.2),
            q_offset_deg=0.0,
        )

    @classmethod
    def shifted(cls, latency_ticks: int = 1) -> DomainRandomization:
        """Deliberately shifted sim standing in for a plausible real-robot mismatch (sim-to-sim gap, no real robot)."""
        return cls(
            enabled=True,
            q_noise_deg=5.0,
            action_noise=0.0,
            latency_prob=1.0,
            latency_ticks=latency_ticks,
            mass_range=(1.3, 1.3),
            friction_range=(0.7, 0.7),
            kp_range=(0.8, 0.8),
            q_offset_deg=1.5,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DomainRandomization:
        d = dict(d)
        for k in ("mass_range", "friction_range", "kp_range"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


def build_rl_model(cfg: SceneConfig | None = None) -> mujoco.MjModel:
    """Compile the arm + table + cube_0 model (other pool objects stripped from the scene XML)."""
    cfg = cfg or SceneConfig()
    tmp = tempfile.mkdtemp(prefix="langgrasp_rl_")
    arm_xml = arm_xml_with_tcp(cfg).replace('meshdir="assets/"', f'meshdir="{os.path.join(ASSET_DIR, "assets")}/"')
    with open(os.path.join(tmp, "so_arm100.xml"), "w") as f:
        f.write(arm_xml)
    xml = build_scene_xml(cfg)
    for k, i in POOL:
        name = f"{k}_{i}"
        if name == CUBE:
            continue
        xml, n = re.subn(rf'<body name="{name}" .*?</body>', "", xml, flags=re.S)
        assert n == 1, name
    with open(os.path.join(tmp, "scene.xml"), "w") as f:
        f.write(xml)
    return mujoco.MjModel.from_xml_path(os.path.join(tmp, "scene.xml"))


class VecArmEnv:
    def __init__(
        self,
        n_envs: int = 64,
        task: str = "reach",
        seed: int = 0,
        dr: DomainRandomization | None = None,
        n_threads: int = 8,
        model: mujoco.MjModel | None = None,
    ):
        assert task in TASKS, task
        self.n = int(n_envs)
        self.task = task
        self.seed = int(seed)
        self.dr = dr or DomainRandomization.none()
        self.rng = np.random.default_rng(self.seed)
        base = model if model is not None else build_rl_model()
        self.base_model = base
        self.models = [copy.copy(base) for _ in range(self.n)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.kin = ArmKinematics(base)
        self.n_substeps = int(round(1.0 / (CONTROL_HZ * base.opt.timestep)))
        self.horizon = HORIZON[task]
        self.obs_dim = OBS_DIM[task]
        self.act_dim = ACT_DIM[task]
        self.arm_qadr = self.kin.qpos_adr
        self.arm_dofadr = self.kin.dof_adr
        self.jaw_qadr = self.kin.jaw_qpos_adr
        self.arm_act = np.array([base.actuator(n).id for n in ARM_JOINTS])
        self.jaw_act = base.actuator(JAW_JOINT).id
        self.lo, self.hi = self.kin.lo.copy(), self.kin.hi.copy()
        # episodes start from the model's 'home' keyframe (TCP about (0, -0.24, 0.08), inside the workspace) + noise
        self.start_q = np.array(base.key("home").qpos[:5], dtype=float)
        self.site_id = self.kin.site_id
        self.cube_bid = base.body(CUBE).id
        self.cube_gid = base.geom(f"{CUBE}_g0").id
        j = base.joint(f"{CUBE}_free")
        self.cube_qadr, self.cube_dofadr = int(j.qposadr[0]), int(j.dofadr[0])
        pad_ids = [base.geom(g).id for g in PAD_GEOMS]
        self.fixed_pads = np.array(pad_ids[:4])
        self.moving_pads = np.array(pad_ids[4:])
        self.mass0 = float(base.body_mass[self.cube_bid])
        self.inertia0 = base.body_inertia[self.cube_bid].copy()
        self.friction0 = float(base.geom_friction[self.cube_gid, 0])
        self.kp0 = base.actuator_gainprm[self.arm_act, 0].copy()
        self.n_threads = max(1, int(n_threads))
        self.pool = ThreadPoolExecutor(self.n_threads) if self.n_threads > 1 else None
        self._chunks = np.array_split(np.arange(self.n), min(self.n_threads, self.n))
        self._fk_data = mujoco.MjData(base)
        # per-env state
        self.t = np.zeros(self.n, dtype=np.int64)
        self.q_target = np.zeros((self.n, 5))
        self.jaw_target = np.full(self.n, JAW_OPEN)
        self.target = np.zeros((self.n, 3))
        self.ep_return = np.zeros(self.n)
        self.reached_any = np.zeros(self.n, dtype=bool)
        self.lifted_any = np.zeros(self.n, dtype=bool)
        self.latency = np.zeros(self.n, dtype=np.int64)
        self.act_queue = np.zeros((self.n, max(1, self.dr.latency_ticks), self.act_dim))
        self.q_offset = np.zeros((self.n, 5))
        self.true_q = np.zeros((self.n, 5))
        self.true_tcp = np.zeros((self.n, 3))
        self.true_jaw = np.zeros(self.n)
        self.cube_pos = np.zeros((self.n, 3))
        self.grasped = np.zeros(self.n, dtype=bool)
        self.jaw_dir = np.zeros((self.n, 3))  # TCP x axis (jaw axis), world frame
        self.point_down = np.zeros(self.n)  # world-z component of the TCP y axis, 1 = strictly top-down
        self.cube_axis = np.zeros((self.n, 2))  # cube body x axis, horizontal components
        self.episode_count = 0
        self.dr_samples: dict[str, np.ndarray] = {k: np.ones(self.n) for k in ("mass", "friction", "kp")}

    # ------------------------------------------------------------------ helpers
    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.pool = None

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        d = self._fk_data
        d.qpos[self.arm_qadr] = q_arm
        mujoco.mj_kinematics(self.base_model, d)
        return d.site_xpos[self.site_id].copy()

    def _sample_target(self, rng: np.random.Generator) -> np.ndarray:
        (x0, x1), (y0, y1) = WORKSPACE["x"], WORKSPACE["y"]
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(*TARGET_Z)])

    def _apply_physical_dr(self, i: int):
        m, d = self.models[i], self.datas[i]
        dr = self.dr
        if dr.enabled:
            ms = float(self.rng.uniform(*dr.mass_range))
            fs = float(self.rng.uniform(*dr.friction_range))
            ks = float(self.rng.uniform(*dr.kp_range))
        else:
            ms = fs = ks = 1.0
        m.body_mass[self.cube_bid] = self.mass0 * ms
        m.body_inertia[self.cube_bid] = self.inertia0 * ms
        m.geom_friction[self.cube_gid, 0] = self.friction0 * fs
        m.actuator_gainprm[self.arm_act, 0] = self.kp0 * ks
        m.actuator_biasprm[self.arm_act, 1] = -self.kp0 * ks
        mujoco.mj_setConst(m, d)
        self.dr_samples["mass"][i], self.dr_samples["friction"][i], self.dr_samples["kp"][i] = ms, fs, ks

    def _reset_env(self, i: int):
        m, d = self.models[i], self.datas[i]
        rng = self.rng
        mujoco.mj_resetData(m, d)
        self._apply_physical_dr(i)
        q = np.clip(self.start_q + rng.uniform(-0.15, 0.15, size=5), self.lo, self.hi)
        d.qpos[self.arm_qadr] = q
        d.qpos[self.jaw_qadr] = JAW_OPEN
        self.q_target[i] = q
        self.jaw_target[i] = JAW_OPEN
        self.q_offset[i] = np.deg2rad(self.dr.q_offset_deg) if self.dr.enabled else 0.0
        if self.task == "reach":
            self.target[i] = self._sample_target(rng)
            d.qpos[self.cube_qadr : self.cube_qadr + 3] = [PARK_POS[0], PARK_POS[1], CUBE_HALF]
        else:
            (x0, x1), (y0, y1) = WORKSPACE["x"], WORKSPACE["y"]
            yaw = rng.uniform(-np.pi / 4, np.pi / 4)
            d.qpos[self.cube_qadr : self.cube_qadr + 3] = [rng.uniform(x0, x1), rng.uniform(y0, y1), CUBE_HALF + 0.0005]
            d.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[:] = 0.0
        d.ctrl[self.arm_act] = q - self.q_offset[i]
        d.ctrl[self.jaw_act] = JAW_OPEN
        mujoco.mj_forward(m, d)
        self.latency[i] = self.dr.latency_ticks if (self.dr.enabled and rng.random() < self.dr.latency_prob) else 0
        self.act_queue[i] = 0.0
        self.t[i] = 0
        self.ep_return[i] = 0.0
        self.reached_any[i] = False
        self.lifted_any[i] = False
        self.episode_count += 1

    def _read_state(self, idx: np.ndarray | None = None):
        idx = np.arange(self.n) if idx is None else idx
        for i in idx:
            d = self.datas[i]
            self.true_q[i] = d.qpos[self.arm_qadr]
            self.true_jaw[i] = d.qpos[self.jaw_qadr]
            self.true_tcp[i] = d.site_xpos[self.site_id]
            self.cube_pos[i] = d.xpos[self.cube_bid]
            if self.task == "lift":
                self.grasped[i] = self._is_grasped(d)
                R = d.site_xmat[self.site_id].reshape(3, 3)
                self.jaw_dir[i] = R[:, 0]
                self.point_down[i] = R[2, 1]
                self.cube_axis[i] = d.xmat[self.cube_bid].reshape(3, 3)[:2, 0]

    def _is_grasped(self, d: mujoco.MjData) -> bool:
        n = d.ncon
        if n == 0:
            return False
        g = d.contact.geom[:n]
        cube = (g[:, 0] == self.cube_gid) | (g[:, 1] == self.cube_gid)
        if not cube.any():
            return False
        other = np.where(g[cube, 0] == self.cube_gid, g[cube, 1], g[cube, 0])
        return bool(np.isin(other, self.fixed_pads).any() and np.isin(other, self.moving_pads).any())

    def _observe(self, idx: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(self.n) if idx is None else idx
        q_rep = self.true_q[idx] + self.q_offset[idx]
        if self.dr.enabled and self.dr.q_noise_deg > 0:
            q_rep = q_rep + self.rng.normal(0.0, np.deg2rad(self.dr.q_noise_deg), size=q_rep.shape)
        tcp_rep = np.stack([self.fk(q) for q in q_rep])
        if self.task == "reach":
            obs = np.concatenate([q_rep, tcp_rep, self.target[idx], self.target[idx] - tcp_rep], axis=1)
        else:
            cube = self.cube_pos[idx]
            obs = np.concatenate(
                [q_rep, tcp_rep, cube, cube - tcp_rep, self.true_jaw[idx, None], self.jaw_dir[idx, :2], self.cube_axis[idx]], axis=1
            )
        return obs.astype(np.float32)

    def _step_chunk(self, idx: np.ndarray):
        for i in idx:
            mujoco.mj_step(self.models[i], self.datas[i], nstep=self.n_substeps)

    def _physics(self):
        if self.pool is None:
            self._step_chunk(np.arange(self.n))
        else:
            list(self.pool.map(self._step_chunk, self._chunks))

    # ------------------------------------------------------------------ API
    def reset(self) -> np.ndarray:
        for i in range(self.n):
            self._reset_env(i)
        self._read_state()
        return self._observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(self.n, self.act_dim), -1.0, 1.0)
        if self.dr.enabled and self.dr.action_noise > 0:
            a = np.clip(a + self.rng.normal(0.0, self.dr.action_noise, size=a.shape), -1.0, 1.0)
        applied = a.copy()
        delayed = self.latency > 0
        if delayed.any():
            # queue[:, 0] is the oldest pending action; latency k means the action applied now was issued k ticks ago
            for i in np.flatnonzero(delayed):
                k = self.latency[i]
                applied[i] = self.act_queue[i, -k]
                self.act_queue[i, :-1] = self.act_queue[i, 1:]
                self.act_queue[i, -1] = a[i]
        self.q_target = np.clip(self.q_target + applied[:, :5] * ARM_STEP_RAD, self.lo, self.hi)
        if self.task == "lift":
            self.jaw_target = np.clip(self.jaw_target + applied[:, 5] * JAW_STEP_RAD, JAW_CLOSED, JAW_OPEN)
        for i in range(self.n):
            d = self.datas[i]
            d.ctrl[self.arm_act] = self.q_target[i] - self.q_offset[i]
            d.ctrl[self.jaw_act] = self.jaw_target[i]
        self._physics()
        self._read_state()
        self.t += 1
        if self.task == "reach":
            dist = np.linalg.norm(self.true_tcp - self.target, axis=1)
            hit = dist < REACH_TOL
            reward = -dist + 2.0 * hit
            self.reached_any |= hit
            success_now = self.reached_any
            info = {"dist": dist, "hit": hit, "hold": hit.copy()}
        else:
            jaw_h = self.jaw_dir.copy()
            jaw_h[:, 2] = 0.0
            jaw_h /= np.linalg.norm(jaw_h, axis=1, keepdims=True) + 1e-9
            gp = self.cube_pos + np.array([0.0, 0.0, GRASP_TCP_DZ]) + GRASP_JAW_OFFSET * jaw_h
            dist = np.linalg.norm(self.true_tcp - gp, axis=1)
            closed_frac = np.clip((JAW_OPEN - self.true_jaw) / (JAW_OPEN - JAW_CLOSED), 0.0, 1.0)
            height = np.clip(self.cube_pos[:, 2] - CUBE_HALF, 0.0, 0.1)
            lifted = (height > LIFT_HEIGHT) & self.grasped
            at_point = np.exp(-dist / 0.01)
            reward = (
                -dist
                - 0.3 * closed_frac * (dist > CLOSE_RADIUS)
                + 0.3 * closed_frac * at_point
                + 0.1 * self.point_down
                + 0.5 * at_point
                + 0.5 * self.grasped
                + 20.0 * height * self.grasped
                + 5.0 * lifted
            )
            self.lifted_any |= lifted
            success_now = self.lifted_any
            info = {"dist": dist, "grasped": self.grasped.copy(), "height": height, "lifted": lifted}
        self.ep_return += reward
        done = self.t >= self.horizon
        info["success"] = success_now.copy()
        info["episode_return"] = self.ep_return.copy()
        info["done"] = done
        obs = self._observe()
        if done.any():
            idx = np.flatnonzero(done)
            for i in idx:
                self._reset_env(i)
            self._read_state(idx)
            obs[idx] = self._observe(idx)
        return obs, reward.astype(np.float32), done, info
