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
    """The stage functions composed into one command.

    ``hooks`` and ``controller_factory`` are optional observation points added for the web GUI
    (``langgrasp/gui/trace.py``). Both default to None and every hook call is guarded, so with them unset
    this class runs exactly the code it ran before they existed. Hook calls sit outside the ``tracer.stage``
    blocks, so the latency the tracer records stays the latency the evaluation scripts record.
    """

    def __init__(self, env, grounder, segmenter=None, cfg: PipelineConfig | None = None, safety=None, seed: int = 0, hooks=None, controller_factory=None):
        self.env = env
        self.grounder = grounder
        self.segmenter = segmenter
        self.cfg = cfg or PipelineConfig()
        self.safety = safety
        self.tracer = LatencyTracer()
        self.rng = np.random.default_rng(seed)
        self.hooks = hooks
        self.controller_factory = controller_factory

    def _stage_ms(self, tracer_name: str) -> float | None:
        """Latency of the last sample of a tracer stage, for the hooks. None if the stage never ran."""
        s = self.tracer.samples.get(tracer_name)
        return s[-1] if s else None

    # ---------------------------------------------------------------- stages
    def perceive(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from langgrasp.perception.depth_fusion import realsense_like_noise

        if self.hooks is not None:
            self.hooks.stage_started("capture", camera=self.cfg.camera, depth_noise=self.cfg.depth_noise)
        with self.tracer.stage("capture"):
            rgb = self.env.render(self.cfg.camera, self.cfg.image_size)
            depth = self.env.render(self.cfg.camera, self.cfg.image_size, depth=True).astype(np.float32)
            if self.cfg.depth_noise:
                depth = realsense_like_noise(depth, self.rng)
            K = self.env.camera_intrinsics(self.cfg.camera, self.cfg.image_size)
            T = self.env.camera_extrinsics(self.cfg.camera)
        if self.hooks is not None:
            finite = np.isfinite(depth)
            self.hooks.stage_finished(
                "capture",
                latency_ms=self._stage_ms("capture"),
                camera=self.cfg.camera,
                size=[int(rgb.shape[0]), int(rgb.shape[1])],
                depth_noise=self.cfg.depth_noise,
                fx_px=float(K[0, 0]),
                depth_min_m=float(depth[finite].min()) if finite.any() else None,
                depth_max_m=float(depth[finite].max()) if finite.any() else None,
                depth_invalid_px=int((~finite).sum()),
                images={"rgb": rgb, "depth": depth},
            )
        return rgb, depth, K, T

    def ground(self, rgb: np.ndarray, intent: Intent) -> tuple[Candidate | None, dict, dict]:
        if self.hooks is not None:
            self.hooks.stage_started("grounding", query=intent.phrase, color=intent.color, generic=intent.generic, use_color_check=self.cfg.use_color_check)
        with self.tracer.stage("grounding"):
            cands, ginfo = ground_with_fallback(self.grounder, rgb, intent.phrase, intent.color, intent.generic, self.cfg.use_color_check)
            cand, sinfo = select_target(cands, rgb, intent.color, intent.spatial, self.cfg.use_color_check)
        if self.hooks is not None:
            self._emit_grounding(cands, cand, ginfo, sinfo, intent)
        return cand, ginfo, sinfo

    def _emit_grounding(self, cands: list, cand, ginfo: dict, sinfo: dict, intent: Intent) -> None:
        """Hook-only: the candidate table and the selection decision.

        ``select_target`` runs inside the "grounding" tracer stage (that is how the reported grounding
        latency is defined), so the select stage reports no latency of its own.
        """
        winner = None
        rows = []
        for i, c in enumerate(cands):
            if cand is not None and c is cand:
                winner = i
            rows.append({"box": [float(v) for v in c.box], "score": float(c.score), "label": c.label, "color_frac": float(c.color_frac), "area_px": float(c.area)})
        thr = getattr(self.grounder, "box_threshold", None)
        self.hooks.stage_finished(
            "grounding",
            status="ok" if cands else "warn",
            latency_ms=self._stage_ms("grounding"),
            message=None if cands else f"The grounder returned no candidate above {thr} for '{ginfo.get('query')}'.",
            candidates=rows,
            query=ginfo.get("query"),
            fallback=bool(ginfo.get("fallback")),
            grounder_latency_ms=ginfo.get("latency_ms"),
            box_threshold=thr,
            text_threshold=getattr(self.grounder, "text_threshold", None),
        )
        why = []
        if sinfo.get("color_filtered"):
            why.append(f"colour check dropped {sinfo['color_filtered']} of {sinfo.get('n_candidates')}")
        if sinfo.get("spatial_used"):
            why.append(f"spatial reference '{intent.spatial}' resolved in image space")
        if not why:
            why.append("highest score")
        self.hooks.stage_finished(
            "select",
            status="ok" if cand is not None else "fail",
            latency_ms=None,
            message=None if cand is not None else "No candidate survived selection.",
            winner=winner,
            box=[float(v) for v in cand.box] if cand is not None else None,
            score=float(cand.score) if cand is not None else None,
            color_frac=float(cand.color_frac) if cand is not None else None,
            label=cand.label if cand is not None else None,
            spatial=intent.spatial,
            rule=", ".join(why),
            **{k: sinfo[k] for k in sinfo},
        )

    def mask_for(self, rgb: np.ndarray, cand: Candidate) -> tuple[np.ndarray, str]:
        from langgrasp.perception.depth_fusion import mask_from_box

        if self.hooks is not None:
            self.hooks.stage_started("segment", use_yolo_mask=self.cfg.use_yolo_mask, backend=getattr(self.segmenter, "backend", None))
        info: dict = {}
        mask = source = None
        if cand.mask is not None:
            mask, source = cand.mask.astype(bool), "oracle"
        elif self.segmenter is not None and self.cfg.use_yolo_mask:
            with self.tracer.stage("segmentation"):
                dets = self.segmenter.detect(rgb)
            best, best_iou = None, 0.3
            for d in dets:
                iou = box_iou(d["box"], cand.box)
                if iou > best_iou:
                    best, best_iou = d, iou
            if self.hooks is not None:
                info = {
                    "n_detections": len(dets),
                    "detections": [{"name": d["name"], "conf": float(d["conf"]), "box": [float(v) for v in d["box"]], "iou_with_box": float(box_iou(d["box"], cand.box))} for d in dets],
                    "iou_threshold": 0.3,
                    "best_iou": float(best_iou) if best is not None else None,
                    "timing": dict(getattr(self.segmenter, "last_timing", {}) or {}),
                }
            if best is not None:
                mask, source = best["mask"].astype(bool), f"yolo:{best['name']}"
        if mask is None:
            mask, source = mask_from_box(cand.box, rgb.shape[:2]), "box"
        if self.hooks is not None:
            msg = None
            if source == "box":
                msg = "No YOLO mask overlapped the chosen box (IoU at least 0.3). Using the box as the mask, which includes table pixels."
            elif source == "oracle":
                msg = "Ground-truth mask from the simulator label map (oracle grounder only)."
            self.hooks.stage_finished(
                "segment",
                status="ok" if source.startswith("yolo") else "warn",
                latency_ms=self._stage_ms("segmentation") if source.startswith("yolo") else None,
                message=msg,
                mask_source=source,
                mask_px=int(mask.sum()),
                images={"mask": mask},
                **info,
            )
        return mask, source

    def grasp_from(self, depth: np.ndarray, K: np.ndarray, T: np.ndarray, mask: np.ndarray) -> dict | None:
        from langgrasp.perception.depth_fusion import depth_to_points, grasp_from_points, segment_object_points

        if self.hooks is not None:
            self.hooks.stage_started("fuse")
        n_raw = None
        with self.tracer.stage("depth_fusion"):
            pts = depth_to_points(depth, K, T, mask)
            if self.hooks is not None:
                n_raw = len(pts)
            pts = segment_object_points(pts)
            n_obj = len(pts)
            g = None if n_obj < 20 else grasp_from_points(pts)
        if self.hooks is not None:
            step = max(1, n_obj // 400)
            self.hooks.stage_finished(
                "fuse",
                status="ok" if g is not None else "fail",
                latency_ms=self._stage_ms("depth_fusion"),
                message=None if g is not None else f"{n_obj} object points after table and outlier removal, the minimum is 20. No grasp pose.",
                n_points_backprojected=n_raw,
                n_points_object=n_obj,
                min_points=20,
                grasp=g,
                points_xyz=pts[::step][:400].tolist() if n_obj else [],
                points_subsample=step,
            )
        return g

    # ---------------------------------------------------------------- full command
    def run_command(self, command: str, target_name: str | None = None) -> CommandResult:
        res = CommandResult(command=command)
        self.tracer.begin_command()
        if self.hooks is not None:
            self.hooks.stage_started("parse", text=command)
        with self.tracer.stage("parse"):
            res.intent = parse_command(command)
        if self.hooks is not None:
            self.hooks.stage_finished("parse", latency_ms=self._stage_ms("parse"), **res.intent.to_dict())
        rgb, depth, K, T = self.perceive()
        cand, res.grounding_info, res.select_info = self.ground(rgb, res.intent)
        res.candidate = cand
        if cand is None:
            res.aborted = "no_candidate"
            self.tracer.end_command()
            return self._finish(res)
        if self.safety is not None:
            threshold = float(self.safety.cfg.grounding_conf_threshold)
            if self.hooks is not None:
                self.hooks.stage_started("gate", score=float(cand.score), n_candidates=res.select_info.get("n_candidates", 1), threshold=threshold, human_confirm=bool(self.safety.cfg.require_human_confirm))
            allowed, reason = self.safety.gate_grounding(cand.score, res.select_info.get("n_candidates", 1))
            if self.hooks is not None:
                self.hooks.stage_finished(
                    "gate",
                    status="ok" if allowed else "fail",
                    latency_ms=None,
                    message=None if allowed else f"Safety gate refused: {reason}. Nothing moved.",
                    decision="allow" if allowed else "refuse",
                    reason=reason,
                    score=float(cand.score),
                    threshold=threshold,
                    ambiguity_margin=float(self.safety.cfg.ambiguity_margin),
                    select_flagged_ambiguous=bool(res.select_info.get("ambiguous", False)),
                    gate_input="top-1 score only: the pipeline passes one score, so the monitor's top-2 ambiguity margin is not reached on this path",
                )
            if not allowed:
                res.aborted = f"safety:{reason}"
                self.tracer.end_command()
                return self._finish(res)
        elif cand.score < self.cfg.grounding_threshold:
            if self.hooks is not None:
                self.hooks.stage_finished("gate", status="fail", latency_ms=None, message=f"Confidence {cand.score:.3f} is below the threshold {self.cfg.grounding_threshold:.2f}. Nothing moved.", decision="refuse", reason="low_confidence", score=float(cand.score), threshold=float(self.cfg.grounding_threshold), gate_input="pipeline threshold (no safety monitor attached)")
            res.aborted = "low_confidence"
            self.tracer.end_command()
            return self._finish(res)
        elif self.hooks is not None:
            self.hooks.stage_finished("gate", status="ok", latency_ms=None, decision="allow", reason="ok", score=float(cand.score), threshold=float(self.cfg.grounding_threshold), gate_input="pipeline threshold (no safety monitor attached)")
        if target_name is not None:
            res.grounding_correct = self._box_hits_object(cand.box, target_name)
        mask, res.mask_source = self.mask_for(rgb, cand)
        g = self.grasp_from(depth, K, T, mask)
        if g is None or not np.isfinite(g["center"]).all() or g["width"] > self.cfg.max_grasp_width:
            if self.hooks is not None and g is not None:
                bad_width = g["width"] > self.cfg.max_grasp_width
                self.hooks.stage_finished(
                    "fuse",
                    status="fail",
                    latency_ms=None,
                    message=(f"Grasp width {g['width']:.3f} m exceeds the jaw maximum {self.cfg.max_grasp_width:.3f} m. No grasp." if bad_width else "The fitted grasp centre is not finite. No grasp."),
                    grasp=g,
                    max_grasp_width=float(self.cfg.max_grasp_width),
                )
            res.aborted = "no_grasp"
            self.tracer.end_command()
            return self._finish(res)
        res.grasp = g
        if self.hooks is not None:
            self.hooks.stage_started("execute", center=[float(v) for v in np.asarray(g["center"])], psi=float(g["psi"]), width=float(g["width"]), target=target_name)
        with self.tracer.stage("execute"):
            ctl = PickPlaceController(self.env) if self.controller_factory is None else self.controller_factory(self.env)
            r = ctl.run(np.asarray(g["center"]), float(g["psi"]), target_name, width=float(g["width"]))
        res.grasped, res.lifted, res.placed = r.grasped, r.lifted, r.placed
        if self.hooks is not None:
            self.hooks.stage_finished(
                "execute",
                status="ok" if res.placed else ("warn" if res.lifted else "fail"),
                latency_ms=self._stage_ms("execute"),
                message=None if res.placed else ("Lifted but not placed in the tray." if res.lifted else "The object was not lifted."),
                steps=int(r.steps),
                ik_ok=bool(r.ik_ok),
                max_pos_err_m=float(r.max_pos_err),
                rot_err_rad=float(r.rot_err),
                psi_used=float(getattr(ctl, "psi", float("nan"))),
                grasped=bool(r.grasped),
                lifted=bool(r.lifted),
                placed=bool(r.placed),
            )
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
