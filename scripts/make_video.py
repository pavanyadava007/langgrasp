"""Render a demo video: command -> pick -> place with a per-stage latency overlay (front camera + wrist inset).

--approach oracle : scripted executor with ground-truth grasp points (no perception), for testing the renderer
--approach modular: full language pipeline (Grounding DINO + YOLO11-seg + depth fusion + scripted executor)
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import imageio.v3 as iio
import numpy as np

from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario
from langgrasp.sim.scene import OBJECT_KINDS

ap = argparse.ArgumentParser()
ap.add_argument("--approach", default="oracle", choices=["oracle", "modular"])
ap.add_argument("--seeds", default="5000,5001,5002")
ap.add_argument("--strata", default="seen,langvar,unseen")
ap.add_argument("--out", default="media/demo.mp4")
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--no-color-check", action="store_true")
args = ap.parse_args()

env = LangGraspEnv(seed=0)
frames: list[np.ndarray] = []
FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay(img: np.ndarray, lines: list[str], box=None, box_color=(0, 255, 0)) -> np.ndarray:
    img = np.ascontiguousarray(img.copy())
    wrist = cv2.resize(env.render("wrist", (240, 320)), (160, 120))
    img[8 : 8 + 120, img.shape[1] - 168 : img.shape[1] - 8] = wrist
    cv2.rectangle(img, (img.shape[1] - 168, 8), (img.shape[1] - 8, 128), (255, 255, 255), 1)
    cv2.putText(img, "wrist", (img.shape[1] - 164, 24), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y = 22
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y), FONT, 0.5 if i else 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), FONT, 0.5 if i else 0.6, (255, 255, 255) if i else (80, 255, 120), 1, cv2.LINE_AA)
        y += 22
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    return img


def record_fn(env, phase):
    lines = [f"cmd: {current['command']}"] + current["lines"] + [f"phase: {phase}"]
    frames.append(overlay(env.render("front"), lines, current.get("box")))
    return {}


current: dict = {}
pipeline = None
if args.approach == "modular":
    from langgrasp.perception.grounding import GroundingDINO
    from langgrasp.perception.segmentation import Segmenter
    from langgrasp.policies.modular import ModularPipeline, PipelineConfig

    seg_path = "checkpoints/yolo11n-seg-langgrasp.pt"
    segmenter = Segmenter(seg_path, backend="pt") if os.path.exists(seg_path) else None
    grounder = GroundingDINO()
    grounder.warmup()
    pipeline = ModularPipeline(env, grounder, segmenter, PipelineConfig(use_color_check=not args.no_color_check))

seeds = [int(s) for s in args.seeds.split(",")]
strata = args.strata.split(",")
for i, seed in enumerate(seeds):
    sc = make_scenario(seed, strata[i % len(strata)])
    env.reset(sc)
    current = {"command": sc.command, "lines": [f"stratum: {sc.stratum} / {sc.lang_variant} / {sc.lighting}"], "box": None}
    for _ in range(args.fps):  # 1 s title card
        frames.append(overlay(env.render("front"), [f"cmd: {sc.command}"] + current["lines"] + ["listening..."]))
    if args.approach == "oracle":
        p, psi = env.grasp_point(sc.target)
        ctl = PickPlaceController(env, record=True, record_fn=record_fn)
        r = ctl.run(np.asarray(p), psi, sc.target, width=OBJECT_KINDS[sc.target_obj.kind]["width"])
        ok = r.placed
    else:
        # run the perception stages first so the overlay can show the latency budget during execution
        from langgrasp.language.parser import parse_command

        t0 = time.perf_counter()
        intent = parse_command(sc.command)
        rgb, depth, K, T = pipeline.perceive()
        cand, ginfo, sinfo = pipeline.ground(rgb, intent)
        if cand is None:
            current["lines"].append("grounding: no candidate -> abort")
            for _ in range(args.fps):
                frames.append(overlay(env.render("front"), [f"cmd: {sc.command}"] + current["lines"]))
            continue
        current["box"] = cand.box
        mask, src = pipeline.mask_for(rgb, cand)
        g = pipeline.grasp_from(depth, K, T, mask)
        lat = {k: v[-1] for k, v in pipeline.tracer.samples.items() if v}
        current["lines"] += [
            f"ground: '{ginfo['query']}' score {cand.score:.2f} color {cand.color_frac:.2f} {'fallback' if ginfo['fallback'] else ''} {lat.get('grounding', 0):.0f} ms",
            f"mask: {src}  seg {lat.get('segmentation', 0):.0f} ms  depth-fusion {lat.get('depth_fusion', 0):.0f} ms",
        ]
        if g is None:
            current["lines"].append("grasp: none -> abort")
            for _ in range(args.fps):
                frames.append(overlay(env.render("front"), [f"cmd: {sc.command}"] + current["lines"], current["box"]))
            continue
        current["lines"].append(f"grasp: xyz ({g['center'][0]:.3f}, {g['center'][1]:.3f}, {g['center'][2]:.3f}) psi {np.degrees(g['psi']) % 180:.0f} deg width {g['width']*100:.1f} cm")
        current["lines"].append(f"perception total {1000*(time.perf_counter()-t0):.0f} ms (L4 GPU, not Jetson)")
        if True:
            ctl = PickPlaceController(env, record=True, record_fn=record_fn)
            r = ctl.run(np.asarray(g["center"]), float(g["psi"]), sc.target, width=float(g["width"]))
            ok = r.placed
    for _ in range(args.fps):
        frames.append(overlay(env.render("front"), [f"cmd: {sc.command}"] + current["lines"] + [f"result: {'PLACED' if ok else 'FAILED'}"], current.get("box"), (0, 255, 0) if ok else (0, 0, 255)))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
iio.imwrite(args.out, np.stack(frames), fps=args.fps, codec="libx264", plugin="pyav")
print(f"wrote {args.out}: {len(frames)} frames at {args.fps} fps")
