"""Render the synthetic YOLO-seg dataset (seen kinds and seen colours only) from the MuJoCo sim.

Usage: MUJOCO_GL=egl python scripts/make_synth_dataset.py [--root data/yolo_synth] [--train-scenes 534] [--val-scenes 100]
"""

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from langgrasp.perception.synth_dataset import SynthConfig, build_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/yolo_synth")
    ap.add_argument("--train-scenes", type=int, default=534)
    ap.add_argument("--val-scenes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = SynthConfig(n_train_scenes=args.train_scenes, n_val_scenes=args.val_scenes, seed=args.seed)
    stats = build_dataset(args.root, cfg)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
