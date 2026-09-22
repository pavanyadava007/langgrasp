"""Numerical inverse kinematics for the 5-DOF SO-ARM100 (top-down grasps).

Damped least squares on a 6-D residual (position + orientation) over the five arm joints. A top-down grasp with
jaw yaw psi is reachable for this arm because the wrist-roll axis coincides with the pointing axis, so the
6-D target is consistent with 5 joints (base yaw is fixed by x,y; pitch/elbow by radius/height; wrist pitch by
"point down"; wrist roll by psi).
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
JAW_JOINT = "Jaw"
JAW_OPEN = 0.55  # pre-grasp opening about 5.5 cm: a wider opening lifts the moving finger tip 3 cm and rolls cylinders away (measured 60/60 vs 56/60 at 1.2)
JAW_CLOSED = -0.17


def top_down_rotation(psi: float) -> np.ndarray:
    """Rotation matrix of the TCP frame for a gripper pointing down with jaw axis at yaw psi."""
    c, s = np.cos(psi), np.sin(psi)
    x = np.array([c, s, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def _rot_error(R_target: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Small-angle rotation vector taking R_cur to R_target (world frame)."""
    R_err = R_target @ R_cur.T
    w = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    cos_t = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-6:
        return 0.5 * w
    return w * (theta / (2.0 * np.sin(theta)))


class ArmKinematics:
    def __init__(self, model, site_name: str = "tcp"):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site_name).id
        self.joint_ids = [model.joint(n).id for n in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids])
        self.jaw_qpos_adr = model.jnt_qposadr[model.joint(JAW_JOINT).id]
        self.lo = np.array([model.jnt_range[j][0] for j in self.joint_ids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.joint_ids])
        self.jaw_range = tuple(model.jnt_range[model.joint(JAW_JOINT).id])

    def fk(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[:] = 0.0
        d.qpos[self.qpos_adr] = q_arm
        self.mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site_id].copy(), d.site_xmat[self.site_id].reshape(3, 3).copy()

    def solve(
        self,
        target_pos: np.ndarray,
        psi: float,
        q_init: np.ndarray,
        iters: int = 60,
        damping: float = 0.02,
        pos_w: float = 1.0,
        rot_w: float = 0.3,
        step: float = 0.7,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, float, float]:
        """Return (q_arm, position_error_m, rotation_error_rad)."""
        mujoco = self.mujoco
        d = self.data
        R_t = top_down_rotation(psi)
        q = np.clip(np.array(q_init, dtype=float), self.lo, self.hi)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        pos_err = rot_err = np.inf
        for _ in range(iters):
            d.qpos[:] = 0.0
            d.qpos[self.qpos_adr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p = d.site_xpos[self.site_id]
            R = d.site_xmat[self.site_id].reshape(3, 3)
            e_p = target_pos - p
            e_r = _rot_error(R_t, R)
            pos_err, rot_err = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pos_err < tol and rot_err < 5e-3:
                break
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site_id)
            J = np.vstack([pos_w * jacp[:, self.dof_adr], rot_w * jacr[:, self.dof_adr]])
            e = np.concatenate([pos_w * e_p, rot_w * e_r])
            JJt = J @ J.T + (damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            q = np.clip(q + step * dq, self.lo, self.hi)
        return q, pos_err, rot_err

    def best_psi(self, psi: float, q_ref: np.ndarray, target_pos: np.ndarray) -> tuple[float, float]:
        """The jaw axis is symmetric under 180 deg. Pick the representative (psi + k*pi) that the wrist roll
        can actually reach at target_pos (strict top-down IK), preferring the smallest rotation error, then the
        smallest joint motion from q_ref. Returns (psi, rotation_error_rad)."""
        best, best_key = psi, (np.inf, np.inf)
        for k in (-2, -1, 0, 1, 2):
            c = psi + k * np.pi
            q, pe, re = self.solve(np.asarray(target_pos, dtype=float), c, q_ref, iters=80)
            key = (round(re, 3) + (1.0 if pe > 5e-3 else 0.0), float(np.abs(q - q_ref).sum()))
            if key < best_key:
                best, best_key = c, key
        return best, best_key[0]
