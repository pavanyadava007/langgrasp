"""Render a YOLO-seg dataset from the MuJoCo sim.

Only the three SEEN kinds (cube, screwdriver, can) and the four SEEN colours are used. Unseen colours and the
"bar" kind are held out on purpose so the evaluation can measure generalisation. Every image is 640x480 and
comes from one of the three cameras (front, wrist, side). Labels are YOLO segmentation polygons taken from the
renderer's per-pixel object label map (largest external contour per instance, normalised coordinates).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from langgrasp.perception.classes import CLASS_NAMES
from langgrasp.sim.scene import POOL, SEEN_COLORS, SEEN_KINDS, WORKSPACE

CAMERAS = ("front", "wrist", "side")
IMG_H, IMG_W = 480, 640
MIN_INSTANCE_PX = 60
PLACE_MARGIN = 0.05  # metres added around WORKSPACE for the dataset (wider than the reach-verified region)
MIN_DIST = 0.06  # metres between object centres (screwdriver is 0.15 m long, so overlaps still happen on purpose)


@dataclass
class SynthConfig:
    n_train_scenes: int = 534  # x 3 cameras = 1602 images
    n_val_scenes: int = 100  # x 3 cameras = 300 images
    min_objects: int = 1
    max_objects: int = 5
    p_degraded: float = 0.35
    p_arm_move: float = 0.4  # fraction of scenes where the arm is moved off the observe pose
    arm_ticks_max: int = 6
    n_previews: int = 6
    seed: int = 0


def _sample_positions(rng: np.random.Generator, n: int) -> list[tuple[float, float]]:
    (x0, x1), (y0, y1) = WORKSPACE["x"], WORKSPACE["y"]
    x0, x1, y0, y1 = x0 - PLACE_MARGIN, x1 + PLACE_MARGIN, y0 - PLACE_MARGIN, y1 + PLACE_MARGIN
    pts: list[tuple[float, float]] = []
    for _ in range(2000):
        p = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= MIN_DIST for q in pts):
            pts.append(p)
        if len(pts) == n:
            break
    return pts


def sample_scene(rng: np.random.Generator, cfg: SynthConfig):
    """Random scene with 1..5 seen-kind objects in seen colours. Returns a Scenario the env can reset to."""
    from langgrasp.sim.scenarios import PlacedObject, Scenario

    seen_slots = [f"{k}_{i}" for k, i in POOL if k in SEEN_KINDS]
    n = int(rng.integers(cfg.min_objects, cfg.max_objects + 1))
    names = [str(s) for s in rng.choice(seen_slots, size=n, replace=False)]
    positions = _sample_positions(rng, n)
    objs = []
    for name, pos in zip(names, positions, strict=True):
        kind = name.rsplit("_", 1)[0]
        color = str(rng.choice(list(SEEN_COLORS)))
        objs.append(PlacedObject(name, kind, color, pos, float(rng.uniform(-np.pi, np.pi))))
    lighting = "degraded" if rng.random() < cfg.p_degraded else "nominal"
    return Scenario(stratum="seen", objects=objs, target=objs[0].name, command="", lighting=lighting, seed=int(rng.integers(1 << 31)))


def maybe_move_arm(env, rng: np.random.Generator, cfg: SynthConfig) -> bool:
    """With probability p_arm_move drive the arm a few ticks toward a random pose between observe and home."""
    if rng.random() >= cfg.p_arm_move:
        return False
    a = rng.uniform(0.0, 1.0)
    q = (1 - a) * env.observe_q + a * env.home_q[:5]
    q = q + rng.normal(0.0, 0.15, size=5)
    q = np.clip(q, env.kin.lo, env.kin.hi)
    for _ in range(int(rng.integers(1, cfg.arm_ticks_max + 1))):
        env.step(q_arm=q)
    return True


def instance_polygons(label: np.ndarray, object_names: list[str], min_px: int = MIN_INSTANCE_PX) -> list[tuple[int, np.ndarray, int]]:
    """(class_id, polygon Nx2 normalised xy, pixel_count) per visible instance, largest external contour only."""
    h, w = label.shape
    out = []
    for idx in np.unique(label):
        if idx < 0:
            continue
        kind = object_names[idx].rsplit("_", 1)[0]
        if kind not in CLASS_NAMES:
            continue
        m = (label == idx).astype(np.uint8)
        npx = int(m.sum())
        if npx < min_px:
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        poly = np.stack([c[:, 0] / w, c[:, 1] / h], axis=1).clip(0.0, 1.0)
        out.append((CLASS_NAMES.index(kind), poly, npx))
    return out


def write_label(path: str, polys: list[tuple[int, np.ndarray, int]]):
    lines = []
    for cls, poly, _ in polys:
        coords = " ".join(f"{v:.5f}" for v in poly.reshape(-1))
        lines.append(f"{cls} {coords}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def draw_preview(rgb: np.ndarray, polys: list[tuple[int, np.ndarray, int]]) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]
    colors = [(0, 255, 0), (0, 200, 255), (255, 80, 80)]
    for cls, poly, _ in polys:
        pts = np.round(poly * [w, h]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, colors[cls], 2)
        x, y = pts[0, 0]
        cv2.putText(img, CLASS_NAMES[cls], (int(x), max(12, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1)
    return img


def write_data_yaml(root: str) -> str:
    path = os.path.join(root, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {os.path.abspath(root)}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    return path


def build_dataset(root: str, cfg: SynthConfig | None = None, env=None, verbose: bool = True) -> dict:
    """Render the dataset into root/{images,labels}/{train,val}. Returns stats (also written to root/stats.json)."""
    from langgrasp.sim.env import LangGraspEnv

    cfg = cfg or SynthConfig()
    env = env or LangGraspEnv(seed=cfg.seed)
    names = env.object_names()
    t0 = time.time()
    stats: dict = {"config": cfg.__dict__, "splits": {}}
    previews_left = cfg.n_previews
    for split, n_scenes, seed_off in (("train", cfg.n_train_scenes, 0), ("val", cfg.n_val_scenes, 10_000_000)):
        rng = np.random.default_rng(cfg.seed + seed_off)
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_img = n_inst = n_empty = n_arm = 0
        per_class = [0] * len(CLASS_NAMES)
        per_cam = dict.fromkeys(CAMERAS, 0)
        n_degraded = 0
        for s in range(n_scenes):
            sc = sample_scene(rng, cfg)
            env.reset(sc)
            n_degraded += sc.lighting == "degraded"
            n_arm += maybe_move_arm(env, rng, cfg)
            for cam in CAMERAS:
                rgb, _, label = env.render_rgbd_seg(cam, (IMG_H, IMG_W))
                polys = instance_polygons(label, names)
                stem = f"{split}_{s:05d}_{cam}"
                cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
                write_label(os.path.join(lbl_dir, stem + ".txt"), polys)
                n_img += 1
                n_inst += len(polys)
                n_empty += len(polys) == 0
                for cls, _, _ in polys:
                    per_class[cls] += 1
                    per_cam[cam] += 1
                if previews_left > 0 and polys and split == "train":
                    cv2.imwrite(os.path.join(root, f"preview_{stem}.png"), draw_preview(rgb, polys))
                    previews_left -= 1
            if verbose and (s + 1) % 50 == 0:
                print(f"[{split}] {s + 1}/{n_scenes} scenes, {n_img} images, {n_inst} instances, {time.time() - t0:.0f}s", flush=True)
        stats["splits"][split] = {
            "scenes": n_scenes,
            "images": n_img,
            "instances": n_inst,
            "empty_images": n_empty,
            "scenes_with_arm_moved": n_arm,
            "scenes_degraded_lighting": n_degraded,
            "instances_per_class": dict(zip(CLASS_NAMES, per_class, strict=True)),
            "instances_per_camera": per_cam,
        }
    stats["data_yaml"] = write_data_yaml(root)
    stats["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats
