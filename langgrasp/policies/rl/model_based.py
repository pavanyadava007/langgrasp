"""Model-based reach baselines, no learning: an operational-space controller and a sampling-based MPC (MPPI).

Both act through exactly the same interface as PPO and SAC: they read the 14-D reach observation (reported joint
angles, FK of the reported angles, target, target - tcp) and return an action in [-1, 1]^5 that VecArmEnv
integrates into the position-servo targets (0.12 rad per 10 Hz tick). Noise, latency, calibration offset and the
kp / mass / friction shift are therefore applied by the env to them exactly as to the learned policies. Both use
the NOMINAL MjModel (build_rl_model()) for every internal computation, whatever the plant is.

OSCController
    Operational-space control in its velocity-level form for position servos. At the reported configuration q:
    translational TCP Jacobian J (mj_jacSite, 3x5 arm block) and joint-space inertia M (mj_fullM, 5x5 arm block),
    task-space inertia Lambda = (J M^-1 J^T + damping I)^-1, dynamically consistent inverse Jbar = M^-1 J^T Lambda.
        dq = Jbar (k_task * e) + (I - Jbar J) k_null (q_home - q),   e = target - tcp_reported
        action = clip(dq / 0.12, -1, 1)
    The torque-level loop of classical OSC runs inside the MuJoCo position servos (kp 50); this controller sets
    their targets. Because the env integrates the action into the target, the loop has integral action, which
    removes the gravity sag without an explicit gravity term. No torque-level variant is included.

MPPIController
    Model predictive path integral control on a copy of the nominal MuJoCo model. Each tick, for every env it
    samples K action sequences of H ticks around the shifted previous plan, rolls them out with
    mujoco.rollout (all physics substeps, multi-threaded), scores cost = sum_h (dist_h - 2 [dist_h < 1.5 cm]),
    i.e. the negative env reward, plus a small action-rate term, and takes the exponentially weighted mean.
    Planning state: the reported joint angles from the observation, the joint velocities predicted by a shadow
    copy of the nominal model stepped with the controller's own commands, and the servo target the controller
    believes it has commanded (it does not know about latency or the calibration offset). In the nominal
    condition model and state are exact, so the nominal MPC row is a privileged upper reference; in the shifted
    conditions it plans with the nominal model while the plant is shifted (the model-mismatch test).
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
from mujoco import rollout

from langgrasp.policies.rl.envs import ARM_STEP_RAD, CUBE, JAW_OPEN, REACH_TOL
from langgrasp.sim.env import CONTROL_HZ
from langgrasp.sim.kinematics import ARM_JOINTS, JAW_JOINT
from langgrasp.sim.scene import PARK_POS

ARM = 5


class _ModelInfo:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.site = model.site("tcp").id
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in ARM_JOINTS])
        self.dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in ARM_JOINTS])
        self.jaw_qadr = int(model.jnt_qposadr[model.joint(JAW_JOINT).id])
        self.arm_act = np.array([model.actuator(n).id for n in ARM_JOINTS])
        self.jaw_act = model.actuator(JAW_JOINT).id
        self.lo = np.array([model.jnt_range[model.joint(n).id][0] for n in ARM_JOINTS])
        self.hi = np.array([model.jnt_range[model.joint(n).id][1] for n in ARM_JOINTS])
        self.q_home = np.array(model.key("home").qpos[:ARM], dtype=float)
        j = model.joint(f"{CUBE}_free")
        self.cube_qadr = int(j.qposadr[0])
        self.nsub = int(round(1.0 / (CONTROL_HZ * model.opt.timestep)))

    def set_config(self, d: mujoco.MjData, q: np.ndarray):
        d.qpos[:] = self.m.qpos0
        d.qpos[self.qadr] = q
        d.qpos[self.jaw_qadr] = JAW_OPEN
        d.qpos[self.cube_qadr : self.cube_qadr + 3] = [PARK_POS[0], PARK_POS[1], 0.0125]


class OSCController:
    def __init__(self, model: mujoco.MjModel, k_task: float = 1.0, k_null: float = 0.1, damping: float = 1e-4):
        self.info = _ModelInfo(model)
        self.d = mujoco.MjData(model)
        self.k_task, self.k_null, self.damping = k_task, k_null, damping
        self._jacp = np.zeros((3, model.nv))
        self._M = np.zeros((model.nv, model.nv))

    def reset(self, obs: np.ndarray) -> None:
        pass

    def joint_step(self, q: np.ndarray, e: np.ndarray) -> np.ndarray:
        inf, m, d = self.info, self.info.m, self.d
        inf.set_config(d, q)
        mujoco.mj_fwdPosition(m, d)  # kinematics, com, composite-rigid-body inertia (qM)
        mujoco.mj_jacSite(m, d, self._jacp, None, inf.site)
        mujoco.mj_fullM(m, self._M, d.qM)
        J = self._jacp[:, inf.dadr]
        M = self._M[np.ix_(inf.dadr, inf.dadr)]
        Minv = np.linalg.inv(M)
        Lam = np.linalg.inv(J @ Minv @ J.T + self.damping * np.eye(3))
        Jbar = Minv @ J.T @ Lam
        N = np.eye(ARM) - Jbar @ J
        return Jbar @ (self.k_task * e) + N @ (self.k_null * (inf.q_home - q))

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        out = np.zeros((obs.shape[0], ARM))
        for i in range(obs.shape[0]):
            dq = self.joint_step(obs[i, :ARM], obs[i, 11:14])
            out[i] = np.clip(dq / ARM_STEP_RAD, -1.0, 1.0)
        return out


class MPPIController:
    def __init__(self, model: mujoco.MjModel, n_envs: int, horizon: int = 6, samples: int = 64, sigma: float = 0.5, temperature: float = 0.02,
                 iterations: int = 1, rate_cost: float = 0.001, n_threads: int = 16, seed: int = 0):
        self.info = _ModelInfo(model)
        self.m = model
        self.n, self.H, self.K = n_envs, horizon, samples
        self.sigma, self.lam, self.iters, self.rate_cost = sigma, temperature, iterations, rate_cost
        self.rng = np.random.default_rng(seed)  # the controller's own RNG, never the env's
        self.shadow = [mujoco.MjData(model) for _ in range(n_envs)]
        self.thread_datas = [mujoco.MjData(model) for _ in range(n_threads)]
        self._kin = mujoco.MjData(model)
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(model, self.spec)
        self.plan = np.zeros((n_envs, horizon, ARM))
        self.q_target = np.zeros((n_envs, ARM))
        self.targets = np.zeros((n_envs, 3))
        self.tick_seconds: list[float] = []

    def reset(self, obs: np.ndarray) -> None:
        obs = np.asarray(obs, dtype=np.float64)
        inf = self.info
        for i, d in enumerate(self.shadow):
            mujoco.mj_resetData(self.m, d)
            inf.set_config(d, obs[i, :ARM])
            d.qvel[:] = 0.0
            d.ctrl[inf.arm_act] = obs[i, :ARM]
            d.ctrl[inf.jaw_act] = JAW_OPEN
            mujoco.mj_forward(self.m, d)
        self.q_target = obs[:, :ARM].copy()
        self.plan[:] = 0.0

    def _targets_seq(self, q_target: np.ndarray, U: np.ndarray) -> np.ndarray:
        """U (B, H, 5) actions -> servo targets (B, H, 5), integrated and clipped per tick exactly like the env."""
        qt = np.repeat(q_target[None], U.shape[0], 0) if q_target.ndim == 1 else q_target.copy()
        out = np.empty_like(U)
        for h in range(U.shape[1]):
            qt = np.clip(qt + U[:, h] * ARM_STEP_RAD, self.info.lo, self.info.hi)
            out[:, h] = qt
        return out

    def _tcp(self, qpos: np.ndarray) -> np.ndarray:
        d, inf = self._kin, self.info
        out = np.empty((qpos.shape[0], 3))
        for j in range(qpos.shape[0]):
            d.qpos[:] = qpos[j]
            mujoco.mj_kinematics(self.m, d)
            out[j] = d.site_xpos[inf.site]
        return out

    def act(self, obs: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        obs = np.asarray(obs, dtype=np.float64)
        inf, n, H, K, S = self.info, obs.shape[0], self.H, self.K, self.info.nsub
        # planning start states: measured q, predicted qvel, believed servo target
        x0 = np.empty((n, self.nstate))
        for i, d in enumerate(self.shadow):
            d.qpos[inf.qadr] = obs[i, :ARM]
            d.ctrl[inf.arm_act] = self.q_target[i]
            mujoco.mj_getState(self.m, d, x0[i], self.spec)
        target = obs[:, 8:11]
        for _ in range(self.iters):
            noise = self.rng.normal(0.0, self.sigma, size=(n, K, H, ARM))
            noise[:, 0] = 0.0  # always evaluate the unperturbed previous plan
            U = np.clip(self.plan[:, None] + noise, -1.0, 1.0)  # (n, K, H, 5)
            qt = np.stack([self._targets_seq(self.q_target[i], U[i]) for i in range(n)])  # (n, K, H, 5)
            ctrl = np.empty((n * K, H * S, self.m.nu))
            ctrl[:, :, inf.arm_act] = np.repeat(qt.reshape(n * K, H, ARM), S, axis=1)
            ctrl[:, :, inf.jaw_act] = JAW_OPEN
            init = np.repeat(x0, K, axis=0)
            states, _ = rollout.rollout(self.m, self.thread_datas, init, ctrl, persistent_pool=True)
            tick_end = states[:, S - 1 :: S, 1 : 1 + self.m.nq]  # (n*K, H, nq); state[0] is time
            tcp = self._tcp(tick_end.reshape(-1, self.m.nq)).reshape(n, K, H, 3)
            dist = np.linalg.norm(tcp - target[:, None, None, :], axis=-1)
            rate = np.sum(np.diff(np.concatenate([self.plan[:, None, :1].repeat(K, 1), U], axis=2), axis=2) ** 2, axis=(2, 3))
            cost = np.sum(dist - 2.0 * (dist < REACH_TOL), axis=2) + self.rate_cost * rate  # (n, K)
            w = np.exp(-(cost - cost.min(axis=1, keepdims=True)) / self.lam)
            w /= w.sum(axis=1, keepdims=True)
            self.plan = np.einsum("nk,nkhj->nhj", w, U)
        a = np.clip(self.plan[:, 0].copy(), -1.0, 1.0)
        # commit: believed servo target, shadow prediction for the next tick's velocity estimate, shift the plan
        self.q_target = np.clip(self.q_target + a * ARM_STEP_RAD, inf.lo, inf.hi)
        for i, d in enumerate(self.shadow):
            d.ctrl[inf.arm_act] = self.q_target[i]
            mujoco.mj_step(self.m, d, nstep=S)
        self.plan = np.concatenate([self.plan[:, 1:], self.plan[:, -1:]], axis=1)
        self.tick_seconds.append(time.perf_counter() - t0)
        return a
