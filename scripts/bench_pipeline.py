"""Clean per-stage latency benchmark of the modular pipeline (run with NO other GPU job).

Measures, per command on rendered scenes: parse, capture (render + noise), grounding (Grounding DINO tiny,
one query), segmentation (YOLO11n-seg, selectable backend), depth fusion, and the sum, plus the executor's
sim compute time. Reports median/p90/p99 over N commands after warmup. Hardware label is recorded.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from langgrasp.eval.harness import hardware_label
from langgrasp.eval.stats import latency_summary
from langgrasp.language.parser import parse_command
from langgrasp.perception.grounding import GroundingDINO
from langgrasp.policies.modular import ModularPipeline, PipelineConfig
from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--seg", default="checkpoints/yolo11n-seg-langgrasp.pt")
ap.add_argument("--backends", default="pt,engine", help="comma list of Segmenter backends to time (pt, onnx, engine)")
ap.add_argument("--out", default="results/pipeline_latency_l4.json")
args = ap.parse_args()

env = LangGraspEnv(seed=0)
grounder = GroundingDINO()
grounder.warmup()
from langgrasp.perception.segmentation import Segmenter  # noqa: E402

out = {"hardware": hardware_label(), "python": platform.python_version(), "n_commands": args.n, "warmup": args.warmup, "note": "per-command latency, batch 1, 640x480 front camera, MuJoCo render included in capture; executor time is simulation compute (10 Hz ticks), not robot motion time", "backends": {}}
for backend in args.backends.split(","):
    path = args.seg if backend == "pt" else args.seg.replace(".pt", ".onnx" if backend == "onnx" else "-fp16.engine")
    try:
        seg = Segmenter(path, backend=backend)
        seg.warmup()
    except Exception as e:  # noqa: BLE001
        out["backends"][backend] = {"error": f"{type(e).__name__}: {e}", "path": path}
        continue
    pipe = ModularPipeline(env, grounder, seg, PipelineConfig())
    samples: dict[str, list[float]] = {k: [] for k in ["parse", "capture", "grounding", "segmentation", "depth_fusion", "perception_total", "execute_sim"]}
    for i in range(args.warmup + args.n):
        sc = make_scenario(9000 + i, ["seen", "unseen", "langvar"][i % 3])
        env.reset(sc)
        t0 = time.perf_counter()
        intent = parse_command(sc.command)
        t1 = time.perf_counter()
        rgb, depth, K, T = pipe.perceive()
        t2 = time.perf_counter()
        cand, ginfo, sinfo = pipe.ground(rgb, intent)
        t3 = time.perf_counter()
        if cand is None:
            continue
        mask, src = pipe.mask_for(rgb, cand)
        t4 = time.perf_counter()
        g = pipe.grasp_from(depth, K, T, mask)
        t5 = time.perf_counter()
        te = float("nan")
        if g is not None and g["width"] <= 0.06:
            t6 = time.perf_counter()
            PickPlaceController(env).run(np.asarray(g["center"]), float(g["psi"]), sc.target, width=float(g["width"]))
            te = (time.perf_counter() - t6) * 1000
        if i >= args.warmup:
            samples["parse"].append((t1 - t0) * 1000)
            samples["capture"].append((t2 - t1) * 1000)
            samples["grounding"].append((t3 - t2) * 1000)
            samples["segmentation"].append((t4 - t3) * 1000)
            samples["depth_fusion"].append((t5 - t4) * 1000)
            samples["perception_total"].append((t5 - t0) * 1000)
            if np.isfinite(te):
                samples["execute_sim"].append(te)
    out["backends"][backend] = {"seg_path": path, **{k: latency_summary(v) for k, v in samples.items()}}
    print(backend, {k: round(v["median_ms"], 1) for k, v in out["backends"][backend].items() if isinstance(v, dict) and "median_ms" in v})
json.dump(out, open(args.out, "w"), indent=1)
print("wrote", args.out)
