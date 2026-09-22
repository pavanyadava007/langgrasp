"""Modular language-guided pick-and-place: STT/text -> parse -> ground -> segment -> depth fusion -> grasp -> execute.

The stages are plain functions so the in-process pipeline (this file), the ROS 2 nodes and the evaluation
harness share one implementation. Every stage is timed by a LatencyTracer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from langgrasp.eval.latency import LatencyTracer
from langgrasp.language.parser import Intent, parse_command
from langgrasp.perception.grounding import Candidate, ground_with_fallback, select_target
from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.scene import OBJECT_KINDS


@dataclass
class PipelineConfig:
    use_color_check: bool = True
    use_yolo_mask: bool = True  # refine the grounder box with a YOLO11-seg mask when one overlaps (IoU >= 0.3)
    depth_noise: bool = True  # synthetic RealSense-like noise on the depth image
    grounding_threshold: float = 0.35
    max_grasp_width: float = 0.06
    camera: str = "front"
    image_size: tuple = (480, 640)


@dataclass
class CommandResult:
    command: str
    intent: Intent | None = None
    candidate: Candidate | None = None
    grounding_info: dict = field(default_factory=dict)
    select_info: dict = field(default_factory=dict)
    grasp: dict | None = None
    grounding_correct: bool | None = None
    grasped: bool = False
    lifted: bool = False
    placed: bool = False
    aborted: str | None = None
    latency_ms: dict = field(default_factory=dict)
    mask_source: str = "none"

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("candidate", "intent", "grasp")}
        d["intent"] = self.intent.to_dict() if self.intent else None
        d["box"] = self.candidate.box if self.candidate else None
        d["score"] = self.candidate.score if self.candidate else None
        d["grasp"] = None if self.grasp is None else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.grasp.items()}
        return d


def box_iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class ModularPipeline:
    def __init__(self, env, grounder, segmenter=None, cfg: PipelineConfig | None = None, safety=None, seed: int = 0):
        self.env = env
        self.grounder = grounder
        self.segmenter = segmenter
        self.cfg = cfg or PipelineConfig()
        self.safety = safety
        self.tracer = LatencyTracer()
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- stages
    def perceive(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from langgrasp.perception.depth_fusion import realsense_like_noise

        with self.tracer.stage("capture"):
            rgb = self.env.render(self.cfg.camera, self.cfg.image_size)
            depth = self.env.render(self.cfg.camera, self.cfg.image_size, depth=True).astype(np.float32)
            if self.cfg.depth_noise:
                depth = realsense_like_noise(depth, self.rng)
            K = self.env.camera_intrinsics(self.cfg.camera, self.cfg.image_size)
            T = self.env.camera_extrinsics(self.cfg.camera)
        return rgb, depth, K, T

    def ground(self, rgb: np.ndarray, intent: Intent) -> tuple[Candidate | None, dict, dict]:
        with self.tracer.stage("grounding"):
            cands, ginfo = ground_with_fallback(self.grounder, rgb, intent.phrase, intent.color, intent.generic, self.cfg.use_color_check)
            cand, sinfo = select_target(cands, rgb, intent.color, intent.spatial, self.cfg.use_color_check)
        return cand, ginfo, sinfo

    def mask_for(self, rgb: np.ndarray, cand: Candidate) -> tuple[np.ndarray, str]:
        from langgrasp.perception.depth_fusion import mask_from_box

        if cand.mask is not None:
            return cand.mask.astype(bool), "oracle"
        if self.segmenter is not None and self.cfg.use_yolo_mask:
            with self.tracer.stage("segmentation"):
                dets = self.segmenter.detect(rgb)
            best, best_iou = None, 0.3
            for d in dets:
                iou = box_iou(d["box"], cand.box)
                if iou > best_iou:
                    best, best_iou = d, iou
            if best is not None:
                return best["mask"].astype(bool), f"yolo:{best['name']}"
        return mask_from_box(cand.box, rgb.shape[:2]), "box"

    def grasp_from(self, depth: np.ndarray, K: np.ndarray, T: np.ndarray, mask: np.ndarray) -> dict | None:
        from langgrasp.perception.depth_fusion import depth_to_points, grasp_from_points, segment_object_points

        with self.tracer.stage("depth_fusion"):
            pts = depth_to_points(depth, K, T, mask)
            pts = segment_object_points(pts)
            if len(pts) < 20:
                return None
            g = grasp_from_points(pts)
        return g

    # ---------------------------------------------------------------- full command
    def run_command(self, command: str, target_name: str | None = None) -> CommandResult:
        res = CommandResult(command=command)
        self.tracer.begin_command()
        with self.tracer.stage("parse"):
            res.intent = parse_command(command)
        rgb, depth, K, T = self.perceive()
        cand, res.grounding_info, res.select_info = self.ground(rgb, res.intent)
        res.candidate = cand
        if cand is None:
            res.aborted = "no_candidate"
            self.tracer.end_command()
            return self._finish(res)
        if self.safety is not None:
            allowed, reason = self.safety.gate_grounding(cand.score, res.select_info.get("n_candidates", 1))
            if not allowed:
                res.aborted = f"safety:{reason}"
                self.tracer.end_command()
                return self._finish(res)
        elif cand.score < self.cfg.grounding_threshold:
            res.aborted = "low_confidence"
            self.tracer.end_command()
            return self._finish(res)
        if target_name is not None:
            res.grounding_correct = self._box_hits_object(cand.box, target_name)
        mask, res.mask_source = self.mask_for(rgb, cand)
        g = self.grasp_from(depth, K, T, mask)
        if g is None or not np.isfinite(g["center"]).all() or g["width"] > self.cfg.max_grasp_width:
            res.aborted = "no_grasp"
            self.tracer.end_command()
            return self._finish(res)
        res.grasp = g
        with self.tracer.stage("execute"):
            ctl = PickPlaceController(self.env)
            r = ctl.run(np.asarray(g["center"]), float(g["psi"]), target_name, width=float(g["width"]))
        res.grasped, res.lifted, res.placed = r.grasped, r.lifted, r.placed
        self.tracer.end_command()
        return self._finish(res)

    def _finish(self, res: CommandResult) -> CommandResult:
        res.latency_ms = {k: v[-1] for k, v in self.tracer.samples.items() if v}
        return res

    def _box_hits_object(self, box: list, name: str) -> bool:
        """Grounding is correct if the box centre lies on the target's label-map pixels (or within 12 px)."""
        _, _, label = self.env.render_rgbd_seg(self.cfg.camera, self.cfg.image_size)
        idx = self.env.object_names().index(name)
        cx, cy = int(0.5 * (box[0] + box[2])), int(0.5 * (box[1] + box[3]))
        H, W = label.shape
        y0, y1, x0, x1 = max(0, cy - 12), min(H, cy + 13), max(0, cx - 12), min(W, cx + 13)
        return bool((label[y0:y1, x0:x1] == idx).any())


def oracle_grasp_width(kind: str) -> float:
    return OBJECT_KINDS[kind]["width"]
