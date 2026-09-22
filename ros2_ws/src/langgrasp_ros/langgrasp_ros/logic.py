"""Message-free logic shared by the ROS 2 nodes. Importable without rclpy (tested on the host in
tests/test_ros2_msgs.py). Only numpy and the langgrasp base package are required; MuJoCo is imported lazily
in load_arm_kinematics()."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from langgrasp.sim import controller as _ctl
from langgrasp.sim.controller import FIXED_PAD_X, HOVER_Z, LIFT_Z, MAX_STEP_M, MIN_TIP_Z, PAD_CLEARANCE, PLACE_TIP_Z
from langgrasp.sim.kinematics import ARM_JOINTS, JAW_CLOSED, JAW_JOINT, JAW_OPEN
from langgrasp.sim.scenarios import KIND_SYNONYMS
from langgrasp.sim.scene import OBJECT_KINDS, TRAY_CENTER

JOINT_NAMES = list(ARM_JOINTS) + [JAW_JOINT]
# Controller tuning constants (with fallbacks so this module survives renames in langgrasp.sim.controller).
PRE_HOVER_Z = getattr(_ctl, "PRE_HOVER_Z", 0.085)
DESCEND_STEP_M = getattr(_ctl, "DESCEND_STEP_M", 0.01)
TRACK_GAIN = getattr(_ctl, "TRACK_GAIN", 0.7)
TRACK_CLIP = getattr(_ctl, "TRACK_CLIP", 0.02)
RELAXED_ROT_W = getattr(_ctl, "RELAXED_ROT_W", 0.02)
STAGES = ["camera", "detections", "target_point", "grasp_pose", "joint_command", "joint_command_safe"]


# ---------------------------------------------------------------------- sensor message fields
def image_fields(rgb: np.ndarray) -> dict:
    """sensor_msgs/Image fields for an (H, W, 3) uint8 array, encoding rgb8."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = a.shape[:2]
    return {"height": h, "width": w, "encoding": "rgb8", "is_bigendian": 0, "step": w * 3, "data": a.tobytes()}


def depth_fields(depth_m: np.ndarray) -> dict:
    """sensor_msgs/Image fields for an (H, W) float32 depth array in metres, encoding 32FC1."""
    a = np.ascontiguousarray(depth_m, dtype=np.float32)
    h, w = a.shape[:2]
    return {"height": h, "width": w, "encoding": "32FC1", "is_bigendian": 0, "step": w * 4, "data": a.tobytes()}


def camera_info_fields(K: np.ndarray, height: int, width: int) -> dict:
    """sensor_msgs/CameraInfo fields (plumb_bob, zero distortion) from a 3x3 intrinsic matrix."""
    K = np.asarray(K, dtype=float)
    P = np.zeros((3, 4))
    P[:, :3] = K
    return {
        "height": int(height),
        "width": int(width),
        "distortion_model": "plumb_bob",
        "d": [0.0] * 5,
        "k": K.reshape(-1).tolist(),
        "r": np.eye(3).reshape(-1).tolist(),
        "p": P.reshape(-1).tolist(),
    }


def project_point(K: np.ndarray, T_world_cam: np.ndarray, p_world) -> tuple[float, float, float]:
    """Project a world point into pixel (u, v) for a MuJoCo camera (looks along -z, y up). Returns (u, v, depth)."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T_world_cam, dtype=float)
    p_cam = T[:3, :3].T @ (np.asarray(p_world, dtype=float) - T[:3, 3])
    z = -p_cam[2]
    u = K[0, 2] + K[0, 0] * p_cam[0] / z
    v = K[1, 2] - K[1, 1] * p_cam[1] / z
    return float(u), float(v), float(z)


def unproject_pixel(K: np.ndarray, T_world_cam: np.ndarray, u: float, v: float, depth: float) -> np.ndarray:
    """Inverse of project_point: pixel + depth (metres along the optical axis) to a world point."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T_world_cam, dtype=float)
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = -(v - K[1, 2]) * depth / K[1, 1]
    p_cam = np.array([x, y, -depth])
    return T[:3, :3] @ p_cam + T[:3, 3]


# ---------------------------------------------------------------------- scene JSON (driver -> oracle grounding)
def scene_to_json(scenario, grasp_points: dict, K: np.ndarray, T_world_cam: np.ndarray, image_hw: tuple) -> dict:
    """Ground truth handed from the sim driver to the oracle grounding node through a file.

    grasp_points: name -> (xyz, psi) from env.grasp_point.
    """
    objs = []
    for o in scenario.objects:
        xyz, psi = grasp_points[o.name]
        u, v, d = project_point(K, T_world_cam, xyz)
        objs.append(
            {
                "name": o.name,
                "kind": o.kind,
                "color": o.color,
                "pos": [float(o.pos[0]), float(o.pos[1])],
                "yaw": float(o.yaw),
                "grasp_xyz": [float(x) for x in xyz],
                "psi": float(psi),
                "width": float(OBJECT_KINDS[o.kind]["width"]),
                "pixel": [u, v],
                "depth": d,
            }
        )
    return {
        "seed": int(scenario.seed),
        "stratum": scenario.stratum,
        "target": scenario.target,
        "command": scenario.command,
        "lighting": scenario.lighting,
        "objects": objs,
        "camera": {"K": np.asarray(K).tolist(), "T_world_cam": np.asarray(T_world_cam).tolist(), "height": int(image_hw[0]), "width": int(image_hw[1])},
    }


def write_scene_json(scene: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(scene, f, indent=1)
    os.replace(tmp, path)


def read_scene_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------- oracle grounding
def lexical_scores(command: str, objects: list[dict]) -> list[float]:
    """Cheap lexical match of the command against (color, kind) of each object: 0.5 per matched attribute.

    This is the "oracle" grounding used in the CPU-only ROS 2 smoke run; it deliberately produces ties for
    spatial commands ("the red cube on the left") so the safety gate's ambiguity check has something to catch.
    """
    words = set(command.lower().replace(",", " ").split())
    scores = []
    for o in objects:
        s = 0.0
        if o["color"] in words:
            s += 0.5
        if any(syn in words for syn in KIND_SYNONYMS.get(o["kind"], [o["kind"]])):
            s += 0.5
        scores.append(s)
    return scores


def oracle_ground(scene: dict, command: str, box_px: float = 40.0) -> dict:
    """Return detections (bbox centre/size in pixels, score, name) and the chosen target index.

    The chosen target is the argmax of the lexical scores; if the command does not match anything the scenario
    ground-truth target is used with score 0.0 (so the gate blocks it, which is the honest outcome).
    """
    objs = scene["objects"]
    scores = lexical_scores(command, objs)
    dets = []
    for o, s in zip(objs, scores, strict=True):
        dets.append({"name": o["name"], "kind": o["kind"], "color": o["color"], "score": float(s), "cx": o["pixel"][0], "cy": o["pixel"][1], "w": box_px, "h": box_px})
    if max(scores) > 0:
        idx = int(np.argmax(scores))
    else:
        idx = next(i for i, o in enumerate(objs) if o["name"] == scene["target"])
    return {"detections": dets, "target_index": idx, "scores": scores, "target": objs[idx]}


# ---------------------------------------------------------------------- grasp pose
def top_down_quaternion(psi: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) of the TCP frame for a gripper pointing down with jaw axis yaw psi.

    Same frame as langgrasp.sim.kinematics.top_down_rotation: x = jaw axis, y = world +z, z = x cross y.
    """
    c, s = math.cos(psi), math.sin(psi)
    x = np.array([c, s, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    return rotation_to_quaternion(R)


def rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    return float(x), float(y), float(z), float(w)


def quaternion_to_psi(qx: float, qy: float, qz: float, qw: float) -> float:
    """Recover the jaw yaw psi from a top_down_quaternion (first column of R is (cos psi, sin psi, 0))."""
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r10 = 2 * (qx * qy + qz * qw)
    return math.atan2(r10, r00)


def grasp_from_target(target: dict) -> dict:
    """Grasp pose dict from an oracle target entry (or a detection with grasp_xyz / psi / width)."""
    xyz = [float(v) for v in target["grasp_xyz"]]
    psi = float(target.get("psi", 0.0))
    return {"xyz": xyz, "psi": psi, "quat_xyzw": top_down_quaternion(psi), "width": float(target.get("width", 0.025)), "name": target.get("name", "")}


# ---------------------------------------------------------------------- Cartesian pick plan (mirrors PickPlaceController.run)
@dataclass
class Segment:
    target: np.ndarray
    jaw: float
    phase: str
    strict: bool = True
    hold_ticks: int = 0
    stay_ticks: int = 0  # ticks at the current joint pose (jaw-only phases)
    step_m: float = MAX_STEP_M


def plan_pick_segments(grasp_xyz, psi: float, width: float, place: bool = True) -> list[Segment]:
    """The same approach / hover / descend / close / lift / transport / lower / release / retreat / settle
    sequence as langgrasp.sim.controller.PickPlaceController.run, expressed as data."""
    jaw_dir = np.array([math.cos(psi), math.sin(psi), 0.0])
    offset = max(0.0, width / 2 + PAD_CLEARANCE - FIXED_PAD_X)
    gx, gy, gz = np.asarray(grasp_xyz, dtype=float) + offset * jaw_dir
    tip_z = max(MIN_TIP_Z, gz - 0.006)
    segs = [
        Segment(np.array([gx, gy, PRE_HOVER_Z]), JAW_OPEN, "approach", strict=False),
        Segment(np.array([gx, gy, HOVER_Z]), JAW_OPEN, "hover", hold_ticks=3),
        Segment(np.array([gx, gy, tip_z]), JAW_OPEN, "descend", hold_ticks=3, step_m=DESCEND_STEP_M),
        Segment(np.zeros(3), JAW_CLOSED, "close", stay_ticks=6),
        Segment(np.array([gx, gy, LIFT_Z]), JAW_CLOSED, "lift", strict=False, hold_ticks=3),
    ]
    if place:
        tx, ty, _ = TRAY_CENTER
        segs += [
            Segment(np.array([tx, ty, LIFT_Z]), JAW_CLOSED, "transport", strict=False),
            Segment(np.array([tx, ty, max(tip_z, PLACE_TIP_Z)]), JAW_CLOSED, "lower"),
            Segment(np.zeros(3), JAW_OPEN, "release", stay_ticks=4),
            Segment(np.array([tx, ty, LIFT_Z]), JAW_OPEN, "retreat", strict=False),
            Segment(np.zeros(3), JAW_OPEN, "settle", stay_ticks=3),
        ]
    return segs


def _best_psi(kin, psi: float, q_ref: np.ndarray, hover_xyz: np.ndarray) -> float:
    """ArmKinematics.best_psi across its two signatures: (psi, q_ref) -> psi or (psi, q_ref, target_pos) -> (psi, rot_err)."""
    try:
        out = kin.best_psi(psi, q_ref, hover_xyz)
    except TypeError:
        out = kin.best_psi(psi, q_ref)
    return float(out[0]) if isinstance(out, tuple) else float(out)


class PickExecutor:
    """Tick-by-tick executor of plan_pick_segments driven by joint_states callbacks.

    kin must provide fk(q) -> (pos, R), solve(target, psi, q_init, rot_w=...) -> (q, pos_err, rot_err) and
    best_psi(psi, q). Each call to tick(q_measured) returns (q_arm_target, jaw, phase) or None when done.
    Mirrors PickPlaceController.move_to: straight-line Cartesian waypoints from the measured TCP at segment
    start, IK seeded with the previously commanded joint vector, an integral tracking correction from the
    measured TCP (fk of the measured joints), hold ticks after a move, jaw-only "stay" segments.
    """

    def __init__(self, kin, grasp_xyz, psi: float, width: float, place: bool = True):
        self.kin = kin
        self.psi = float(psi)
        self.width = float(width)
        self.grasp_xyz = np.asarray(grasp_xyz, dtype=float)
        self.place = place
        self.segments: list[Segment] = []
        self.i_seg = -1
        self._queue: list = []  # ("wp", xyz) | ("hold", target) | ("stay", None)
        self._q: np.ndarray | None = None
        self._corr = np.zeros(3)
        self._last_wp: np.ndarray | None = None
        self.max_pos_err = 0.0
        self.ik_ok = True
        self.done = False
        self.phase = "idle"
        self.ticks = 0

    def start(self, q_measured) -> None:
        q = np.asarray(q_measured, dtype=float)
        self.psi = float(_best_psi(self.kin, self.psi, q, np.array([self.grasp_xyz[0], self.grasp_xyz[1], HOVER_Z])))
        self.segments = plan_pick_segments(self.grasp_xyz, self.psi, self.width, self.place)
        self._q = q.copy()
        self.i_seg = -1
        self._advance(q)

    def _advance(self, q_measured: np.ndarray) -> None:
        self.i_seg += 1
        self._corr = np.zeros(3)
        self._last_wp = None
        if self.i_seg >= len(self.segments):
            self.done = True
            self.phase = "done"
            self._queue = []
            return
        seg = self.segments[self.i_seg]
        self.phase = seg.phase
        if seg.stay_ticks:
            self._queue = [("stay", None)] * seg.stay_ticks
            return
        cur = self.kin.fk(q_measured)[0]
        d = float(np.linalg.norm(seg.target - cur))
        n = max(1, int(math.ceil(d / seg.step_m)))
        self._queue = [("wp", cur + (seg.target - cur) * (i / n)) for i in range(1, n + 1)] + [("hold", seg.target)] * seg.hold_ticks

    def tick(self, q_measured) -> tuple[np.ndarray, float, str] | None:
        if self.done:
            return None
        q_meas = np.asarray(q_measured, dtype=float)
        if not self.segments:
            self.start(q_meas)
        seg = self.segments[self.i_seg]
        # tracking correction from the previous waypoint of this segment (same as the controller's corr update)
        if self._last_wp is not None:
            tcp = self.kin.fk(q_meas)[0]
            self._corr = np.clip(self._corr + TRACK_GAIN * (self._last_wp - tcp), -TRACK_CLIP, TRACK_CLIP)
        kind, payload = self._queue.pop(0)
        if kind in ("wp", "hold"):
            rot_w = 0.3 if seg.strict else RELAXED_ROT_W
            q, pe, _ = self.kin.solve(payload + self._corr, self.psi, self._q, rot_w=rot_w)
            self.max_pos_err = max(self.max_pos_err, float(pe))
            if not self._queue and pe > 0.01:  # ik_ok refers to the final waypoint of each segment
                self.ik_ok = False
            self._q = q
            self._last_wp = payload
        else:
            self._q = q_meas.copy()
        out = (self._q.copy(), seg.jaw, seg.phase)
        self.ticks += 1
        if not self._queue:
            self._advance(q_meas)
        return out


def load_arm_kinematics():
    """ArmKinematics on the SO-ARM100 model from the langgrasp package assets (no env, no rendering)."""
    import mujoco

    from langgrasp.sim.kinematics import ArmKinematics
    from langgrasp.sim.scene import ASSET_DIR, SceneConfig, arm_xml_with_tcp, build_scene_xml

    cfg = SceneConfig()
    tmp = tempfile.mkdtemp(prefix="langgrasp_ros_")
    arm_xml = arm_xml_with_tcp(cfg).replace('meshdir="assets/"', f'meshdir="{os.path.join(ASSET_DIR, "assets")}/"')
    with open(os.path.join(tmp, "so_arm100.xml"), "w") as f:
        f.write(arm_xml)
    with open(os.path.join(tmp, "scene.xml"), "w") as f:
        f.write(build_scene_xml(cfg))
    model = mujoco.MjModel.from_xml_path(os.path.join(tmp, "scene.xml"))
    return ArmKinematics(model)


# ---------------------------------------------------------------------- latency tracer
@dataclass
class LatencyTracker:
    """Per-stage latency from header stamps. Every message in the chain carries the capture stamp of the camera
    frame it was derived from; record(stage, stamp_ns, arrival_ns) keeps the first arrival per (stage, stamp)."""

    stages: list = field(default_factory=lambda: list(STAGES))
    arrivals: dict = field(default_factory=dict)  # stage -> {stamp_ns: arrival_ns}
    n_messages: dict = field(default_factory=dict)

    def record(self, stage: str, stamp_ns: int, arrival_ns: int) -> None:
        self.n_messages[stage] = self.n_messages.get(stage, 0) + 1
        self.arrivals.setdefault(stage, {}).setdefault(int(stamp_ns), int(arrival_ns))

    @staticmethod
    def _stats(values_ms: list[float]) -> dict:
        if not values_ms:
            return {"n": 0}
        a = np.asarray(values_ms, dtype=float)
        return {"n": int(a.size), "median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)), "max_ms": float(a.max())}

    def summary(self) -> dict:
        since_capture = {}
        per_stage = {}
        for k, stage in enumerate(self.stages):
            arr = self.arrivals.get(stage, {})
            since_capture[stage] = self._stats([(t - s) / 1e6 for s, t in arr.items() if s > 0])
            if k > 0:
                prev = self.arrivals.get(self.stages[k - 1], {})
                per_stage[f"{self.stages[k - 1]}->{stage}"] = self._stats([(t - prev[s]) / 1e6 for s, t in arr.items() if s in prev])
        first, last = self.stages[0], self.stages[-1]
        e2e = [(t - self.arrivals[first][s]) / 1e6 for s, t in self.arrivals.get(last, {}).items() if s in self.arrivals.get(first, {})]
        return {"since_capture_ms": since_capture, "per_stage_ms": per_stage, "end_to_end_ms": self._stats(e2e), "n_messages": dict(self.n_messages)}


def stamp_to_ns(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)
