"""Open-vocabulary grounding: a text phrase -> candidate boxes in the front camera image.

Backends:
  - "gdino": Grounding DINO tiny (IDEA-Research/grounding-dino-tiny via transformers), the self-hostable
    open checkpoint. Run once per command, not in the control loop.
  - "oracle": uses the simulator's label map (for unit tests and for isolating downstream failures).

Colour verification: Grounding DINO tiny scores "blue screwdriver" and "yellow screwdriver" almost equally
on our renders (measured 0.78 vs 0.77), so an optional HSV colour check re-ranks candidates by the fraction
of pixels inside the box (or mask) that match the colour word. Both variants are evaluated separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

# HSV hue ranges (OpenCV convention, H in [0,180)) for the colour words used in commands.
HUE_RANGES = {
    "red": [(0, 8), (170, 180)],
    "orange": [(8, 20)],
    "yellow": [(20, 35)],
    "green": [(35, 85)],
    "cyan": [(85, 100)],
    "blue": [(100, 130)],
    "purple": [(130, 155)],
    "pink": [(155, 170)],
}


@dataclass
class Candidate:
    box: list  # x1, y1, x2, y2 in pixels
    score: float
    label: str
    color_frac: float = 0.0
    mask: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3]))

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def color_fraction(rgb: np.ndarray, box: list, color: str, mask: np.ndarray | None = None) -> float:
    """Fraction of saturated pixels in the box (or mask) whose hue matches the colour word."""
    import cv2

    if color not in HUE_RANGES:
        return 0.0
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2HSV)
    sel = (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    if mask is not None:
        sel &= mask[y1:y2, x1:x2].astype(bool)
    if sel.sum() < 10:
        return 0.0
    h = hsv[..., 0][sel]
    hit = np.zeros_like(h, dtype=bool)
    for lo, hi in HUE_RANGES[color]:
        hit |= (h >= lo) & (h < hi)
    return float(hit.mean())


class GroundingDINO:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str | None = None, box_threshold: float = 0.25, text_threshold: float = 0.2):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.last_latency_ms = float("nan")

    def warmup(self, shape=(480, 640, 3)):
        self.ground(np.zeros(shape, dtype=np.uint8), "object")

    def ground(self, rgb: np.ndarray, phrase: str) -> list[Candidate]:
        import torch
        from PIL import Image

        text = phrase.strip().rstrip(".") + "."
        t0 = time.perf_counter()
        inputs = self.proc(images=Image.fromarray(rgb), text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold, target_sizes=[rgb.shape[:2]]
        )[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        labels = res.get("text_labels", res.get("labels"))
        cands = [Candidate([float(v) for v in b], float(s), str(lab)) for b, s, lab in zip(res["boxes"], res["scores"], labels, strict=False)]
        cands.sort(key=lambda c: -c.score)
        return cands


class OracleGrounder:
    """Boxes from the simulator label map; label = kind, score = 1.0 for the true target kind, 0.9 for others."""

    def __init__(self, env):
        self.env = env
        self.last_latency_ms = 0.0

    def ground(self, rgb: np.ndarray, phrase: str) -> list[Candidate]:
        _, _, label = self.env.render_rgbd_seg("front", rgb.shape[:2])
        names = self.env.object_names()
        cands = []
        for idx in np.unique(label):
            if idx < 0:
                continue
            ys, xs = np.nonzero(label == idx)
            if len(xs) < 30:
                continue
            kind = names[idx].rsplit("_", 1)[0]
            score = 1.0 if kind in phrase else 0.5
            cands.append(Candidate([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], score, kind, mask=(label == idx), extra={"name": names[idx]}))
        cands.sort(key=lambda c: -c.score)
        return cands


def ground_with_fallback(grounder, rgb: np.ndarray, phrase: str, color: str | None, generic: bool = False, use_color_check: bool = True) -> tuple[list[Candidate], dict]:
    """Ground the phrase; if a colour word is present and no candidate matches that colour (typical for a
    synonym the grounder does not know, e.g. "driver"), re-ground the generic phrase "object" and keep the
    colour filter. Returns (candidates, info) with the query that was finally used."""
    info = {"query": phrase, "fallback": False, "latency_ms": 0.0}
    cands = [] if generic else grounder.ground(rgb, phrase)
    info["latency_ms"] += grounder.last_latency_ms
    if color and use_color_check:
        for c in cands:
            c.color_frac = color_fraction(rgb, c.box, color, c.mask)
        H, W = rgb.shape[:2]
        if not any(c.color_frac >= 0.3 and c.area < 0.35 * H * W for c in cands):
            cands = grounder.ground(rgb, "object")
            info["latency_ms"] += grounder.last_latency_ms
            info["fallback"] = True
            info["query"] = "object"
    elif generic:
        cands = grounder.ground(rgb, "object")
        info["latency_ms"] += grounder.last_latency_ms
        info["query"] = "object"
    return cands, info


def select_target(cands: list[Candidate], rgb: np.ndarray, color: str | None, spatial: str | None, use_color_check: bool = True, min_color_frac: float = 0.3) -> tuple[Candidate | None, dict]:
    """Pick one candidate: colour re-ranking (optional), then spatial resolution in image space, then top score.

    Image-left is world -x for the front camera (camera looks along +y). "front"/"back" use image-bottom/top.
    Returns (candidate, info) where info holds the decision path for the evaluation log.
    """
    info = {"n_candidates": len(cands), "color_filtered": 0, "spatial_used": False, "ambiguous": False}
    if not cands:
        return None, info
    pool = list(cands)
    # drop huge boxes (the arm or the table) : more than 35% of the image
    H, W = rgb.shape[:2]
    pool = [c for c in pool if c.area < 0.35 * H * W] or pool
    if color and use_color_check:
        for c in pool:
            c.color_frac = color_fraction(rgb, c.box, color, c.mask)
        matching = [c for c in pool if c.color_frac >= min_color_frac]
        info["color_filtered"] = len(pool) - len(matching)
        if matching:
            pool = matching
    if spatial and len(pool) > 1:
        info["spatial_used"] = True
        key = {"left": lambda c: c.center[0], "right": lambda c: -c.center[0], "front": lambda c: -c.center[1], "back": lambda c: c.center[1]}[spatial]
        pool.sort(key=key)
        return pool[0], info
    pool.sort(key=lambda c: -(c.score + (0.5 * c.color_frac if color and use_color_check else 0.0)))
    if len(pool) > 1 and abs(pool[0].score - pool[1].score) < 0.05 and (not color or not use_color_check or abs(pool[0].color_frac - pool[1].color_frac) < 0.2):
        info["ambiguous"] = True
    return pool[0], info
