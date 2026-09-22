"""Run one language command on one scene and print the decision path (for live demos)."""

from __future__ import annotations

import argparse
import json
import os

from langgrasp.perception.grounding import GroundingDINO
from langgrasp.policies.modular import ModularPipeline, PipelineConfig
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=5000)
ap.add_argument("--stratum", default="seen")
ap.add_argument("--command", default=None, help="override the scenario's command")
ap.add_argument("--seg", default="checkpoints/yolo11n-seg-langgrasp.pt")
ap.add_argument("--no-color-check", action="store_true")
ap.add_argument("--save-frame", default=None)
args = ap.parse_args()

env = LangGraspEnv(seed=0)
sc = make_scenario(args.seed, args.stratum)
env.reset(sc)
print("scene:", [(o.name, o.color, [round(v, 3) for v in o.pos]) for o in sc.objects], "target:", sc.target, "lighting:", sc.lighting)
grounder = GroundingDINO()
grounder.warmup()
seg = None
if os.path.exists(args.seg):
    from langgrasp.perception.segmentation import Segmenter

    seg = Segmenter(args.seg, backend="pt")
pipe = ModularPipeline(env, grounder, seg, PipelineConfig(use_color_check=not args.no_color_check))
cmd = args.command or sc.command
res = pipe.run_command(cmd, sc.target)
d = res.to_dict()
print(json.dumps({k: d[k] for k in ("command", "intent", "box", "score", "grounding_info", "select_info", "mask_source", "grasp", "grounding_correct", "grasped", "lifted", "placed", "aborted", "latency_ms")}, indent=1, default=str))
if args.save_frame:
    import cv2
    import imageio.v3 as iio

    env.reset(sc)
    img = env.render("front").copy()
    if d["box"]:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    iio.imwrite(args.save_frame, img)
