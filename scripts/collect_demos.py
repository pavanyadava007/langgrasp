"""Collect scripted-expert demonstrations of the fixed-goal task into a LeRobotDataset for ACT.

Task: "pick the red cube and place it in the tray" on random fixed-goal scenes (seeds 20000+i, disjoint from the
evaluation seeds 5000+i). Failed expert episodes are discarded but counted. Simulation only (MuJoCo).

Usage:
  MUJOCO_GL=egl .venv/bin/python scripts/collect_demos.py --n-success 120 --size 128
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgrasp.policies.act.data import TASK, add_episode, collect_episode, make_features  # noqa: E402


def hardware_label() -> str:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    except Exception:
        gpu = "unknown"
    return f"{gpu} (x86 EC2 host), MuJoCo simulation only"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/langgrasp_pick")
    ap.add_argument("--root", default=str(ROOT / "data/lerobot/langgrasp_pick"))
    ap.add_argument("--n-success", type=int, default=120)
    ap.add_argument("--max-attempts", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--size", type=int, default=128, help="square image size for both cameras")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--png", action="store_true", help="store images as png instead of AV1 video")
    ap.add_argument("--out", default=str(ROOT / "results/demos_act.json"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.0, help="start-pose jitter in rad (uniform per joint)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from langgrasp.sim.env import CONTROL_HZ, LangGraspEnv
    from langgrasp.sim.scenarios import make_scenario

    assert args.fps == CONTROL_HZ, "the dataset fps must equal the 10 Hz control rate"
    root = Path(args.root).resolve()
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    size = (args.size, args.size)
    ds = LeRobotDataset.create(args.repo_id, fps=args.fps, features=make_features(*size, use_videos=not args.png), root=root, robot_type="so101_sim", use_videos=not args.png)

    env = LangGraspEnv(seed=0)
    t0 = time.time()
    n_ok = n_att = n_frames = 0
    lengths: list[int] = []
    failed_seeds: list[int] = []
    seeds_used: list[int] = []
    while n_ok < args.n_success and n_att < args.max_attempts:
        seed = args.seed0 + n_att
        n_att += 1
        sc = make_scenario(seed, "seen", target_kind="cube", target_color="red")
        rec = collect_episode(env, sc, size, jitter=args.jitter)
        if not rec.placed:
            failed_seeds.append(seed)
            print(f"seed {seed}: expert failed (lifted={rec.lifted}), discarded", flush=True)
            continue
        n_frames += add_episode(ds, rec, TASK)
        lengths.append(rec.n_frames)
        seeds_used.append(seed)
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"{n_ok}/{args.n_success} episodes, {n_frames} frames, {time.time() - t0:.0f}s", flush=True)
    ds.finalize()
    env.close()
    minutes = (time.time() - t0) / 60

    # reload from disk and verify the frame count
    # torchcodec is installed but cannot find system ffmpeg libraries on this host; PyAV decodes the AV1 streams
    ds2 = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    assert ds2.num_frames == n_frames, (ds2.num_frames, n_frames)
    assert ds2.num_episodes == n_ok, (ds2.num_episodes, n_ok)
    sample = ds2[0]
    shapes = {k: tuple(v.shape) for k, v in sample.items() if hasattr(v, "shape")}
    size_mb = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6
    out = {
        "task": TASK,
        "repo_id": args.repo_id,
        "root": str(root.relative_to(ROOT)),
        "n_success": n_ok,
        "n_attempted": n_att,
        "expert_success_rate": n_ok / n_att,
        "failed_seeds": failed_seeds,
        "seed_range": [args.seed0, args.seed0 + n_att - 1],
        "n_frames": n_frames,
        "episode_len": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "hold_ticks": 5,
        "fps": args.fps,
        "image_size": list(size),
        "cameras": ["front", "wrist"],
        "storage": "png" if args.png else "video (libsvtav1)",
        "dataset_mb": round(size_mb, 1),
        "sample_shapes": shapes,
        "minutes": round(minutes, 2),
        "expert": "scripted PickPlaceController with ground-truth grasp point (no human teleop)",
        "hardware": hardware_label(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "failed_seeds"}, indent=1))


if __name__ == "__main__":
    main()
