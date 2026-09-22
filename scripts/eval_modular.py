"""Modular pipeline through the shared protocol: seen / unseen / language-variation, plus ablations.

Ablations: --no-color-check (Grounding DINO ranking only), --no-yolo (box-only depth crop instead of the
YOLO11-seg mask), --fixed-goal (the 100 red-cube scenes shared with ACT).
"""

from __future__ import annotations

import argparse
import os

from langgrasp.eval.harness import run_protocol
from langgrasp.perception.grounding import GroundingDINO
from langgrasp.policies.modular import ModularPipeline, PipelineConfig
from langgrasp.safety.watchdog import SafetyConfig, SafetyMonitor
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40, help="trials per stratum")
ap.add_argument("--no-color-check", action="store_true")
ap.add_argument("--no-yolo", action="store_true")
ap.add_argument("--no-depth-noise", action="store_true")
ap.add_argument("--seg", default="checkpoints/yolo11n-seg-langgrasp.pt")
ap.add_argument("--seg-backend", default="pt")
ap.add_argument("--fixed-goal", action="store_true", help="100 red-cube scenes (seeds 5000-5099) shared with ACT")
ap.add_argument("--out", default=None)
args = ap.parse_args()

env = LangGraspEnv(seed=0)
grounder = GroundingDINO()
grounder.warmup()
segmenter = None
if not args.no_yolo and os.path.exists(args.seg):
    from langgrasp.perception.segmentation import Segmenter

    segmenter = Segmenter(args.seg, backend=args.seg_backend)
    segmenter.warmup()
elif not args.no_yolo:
    print(f"WARNING: {args.seg} not found, running box-only masks")
cfg = PipelineConfig(use_color_check=not args.no_color_check, use_yolo_mask=not args.no_yolo, depth_noise=not args.no_depth_noise)
safety = SafetyMonitor(SafetyConfig(grounding_conf_threshold=0.30))
pipe = ModularPipeline(env, grounder, segmenter, cfg, safety=safety)

tag = "modular"
if args.no_color_check:
    tag += "_nocolor"
if args.no_yolo or segmenter is None:
    tag += "_noyolo"
if args.no_depth_noise:
    tag += "_nonoise"
if args.fixed_goal:
    tag += "_fixed_goal" if tag == "modular" else "_fixedgoal"
out = args.out or f"results/{tag}_protocol.json".replace("modular_fixed_goal_protocol", "modular_fixed_goal")


def run_one(sc):
    if args.fixed_goal:
        sc = make_scenario(sc.seed, "seen", target_kind="cube", target_color="red")
    env.reset(sc)
    r = pipe.run_command(sc.command, sc.target)
    d = r.to_dict()
    return {
        "grounding_correct": r.grounding_correct if r.grounding_correct is not None else (False if r.aborted in ("no_candidate",) else None),
        "grasped": r.grasped,
        "lifted": r.lifted,
        "placed": r.placed,
        "aborted": r.aborted,
        "latency_ms": r.latency_ms,
        "extra": {k: d[k] for k in ("box", "score", "grounding_info", "select_info", "mask_source", "grasp") if k in d},
    }


strata = {"seen": args.n} if args.fixed_goal else {"seen": args.n, "unseen": args.n, "langvar": args.n}
notes = f"color_check={cfg.use_color_check} yolo_mask={segmenter is not None} depth_noise={cfg.depth_noise} seg={args.seg if segmenter else None} backend={args.seg_backend}"
res = run_protocol(run_one, strata, tag, out, notes=notes)
for k, v in res["summary"]["strata"].items():
    print(k, "place", f"{v['place']['k']}/{v['place']['n']}", "grounding", f"{v['grounding']['k']}/{v['grounding']['n']}")
print("lang variants", {k: f"{v['place']['k']}/{v['place']['n']}" for k, v in res["summary"]["lang_variants"].items()})
print("aborts", res["summary"]["aborts"])
print("latency", {k: round(v["median_ms"], 1) for k, v in res["summary"]["latency_ms"].items()})
print("wrote", out)
