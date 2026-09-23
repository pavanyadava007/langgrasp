"""Train ACT (LeRobot 0.4.4) on the scripted-demo dataset and export the final checkpoint.

Wraps the `lerobot-train` CLI (draccus config) so the LeRobot training loop, optimizer preset and checkpoint
format are used unchanged. Parses the training log for the loss curve, samples the child's VRAM with
nvidia-smi, copies checkpoints/last/pretrained_model to checkpoints/act_pick and writes results/act_train.json.

Usage (about 60-75 min on a shared NVIDIA L4):
  MUJOCO_GL=egl .venv/bin/python scripts/train_act.py --steps 20000 --batch 32
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_RE = re.compile(r"step:\S+ smpl:\S+ ep:\S+ epch:(\S+) loss:(\S+) grdn:(\S+) lr:(\S+) updt_s:(\S+) data_s:(\S+)")


def hardware_label() -> str:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    except Exception:
        gpu = "unknown"
    return f"{gpu} (x86 EC2 host), shared with other jobs"


def _vram_sampler(pid: int, stop: threading.Event, samples: list[float], every_s: float = 10.0) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                p, mem = [x.strip() for x in line.split(",")]
                if int(p) == pid:
                    samples.append(float(mem))
        except Exception:
            pass
        stop.wait(every_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/langgrasp_pick")
    ap.add_argument("--root", default=str(ROOT / "data/lerobot/langgrasp_pick"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="ACT preset is 1e-5 (AdamW, no scheduler)")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--dim-model", type=int, default=512)
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--no-pretrained-backbone", action="store_true", help="random-init resnet18 (default: ImageNet weights)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=0, help="intermediate checkpoint period (0: final only)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--run-dir", default=str(ROOT / "runs/act/pick"))
    ap.add_argument("--ckpt-out", default=str(ROOT / "checkpoints/act_pick"))
    ap.add_argument("--out", default=str(ROOT / "results/act_train.json"))
    ap.add_argument("--tag", default="", help="free-text note stored in the result JSON")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        shutil.rmtree(run_dir)  # lerobot-train refuses an existing output_dir without --resume
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    save_freq = args.save_every if args.save_every > 0 else args.steps
    cmd = [
        str(ROOT / ".venv/bin/lerobot-train"),
        "--policy.type=act",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.root}",
        "--dataset.video_backend=pyav",  # torchcodec cannot find system ffmpeg libs on this host
        f"--output_dir={run_dir}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={save_freq}",
        f"--eval_freq={args.steps + 1}",  # no gym env: never evaluate inside lerobot-train
        f"--seed={args.seed}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.dim_model={args.dim_model}",
        f"--policy.kl_weight={args.kl_weight}",
        f"--policy.optimizer_lr={args.lr}",
        f"--policy.optimizer_lr_backbone={args.lr}",
        "--wandb.enable=false",
    ]
    if args.no_pretrained_backbone:
        cmd.append("--policy.pretrained_backbone_weights=null")
    env = dict(os.environ, MUJOCO_GL="egl", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    print(" ".join(cmd), flush=True)
    log_path = run_dir.parent / f"{run_dir.name}_train.log"
    curve: list[dict] = []
    vram: list[float] = []
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, cwd=str(ROOT))
        stop = threading.Event()
        th = threading.Thread(target=_vram_sampler, args=(proc.pid, stop, vram), daemon=True)
        th.start()
        try:
            for line in proc.stdout:
                log.write(line)
                m = LOG_RE.search(line)
                if m:
                    epoch, loss, grdn, lr, updt, data = (float(x) for x in m.groups())
                    step = args.log_freq * (len(curve) + 1)
                    curve.append({"step": step, "epoch": epoch, "loss": loss, "grad_norm": grdn, "lr": lr, "update_s": updt, "data_s": data, "t_min": (time.time() - t0) / 60})
                    print(f"step {step} loss {loss:.4f} grdn {grdn:.2f} updt {updt * 1e3:.0f}ms data {data * 1e3:.0f}ms {(time.time() - t0) / 60:.1f}min", flush=True)
            rc = proc.wait()
        finally:
            stop.set()
    minutes = (time.time() - t0) / 60
    if rc != 0:
        raise SystemExit(f"lerobot-train failed with code {rc}; see {log_path}")

    src = run_dir / "checkpoints" / "last" / "pretrained_model"
    if not src.exists():
        raise SystemExit(f"no checkpoint at {src}")
    dst = Path(args.ckpt_out).resolve()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src.resolve(), dst)
    cfg = json.loads((dst / "config.json").read_text())
    keep = ["type", "n_obs_steps", "chunk_size", "n_action_steps", "vision_backbone", "pretrained_backbone_weights", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers", "n_decoder_layers", "use_vae", "latent_dim", "n_vae_encoder_layers", "dropout", "kl_weight", "optimizer_lr", "optimizer_weight_decay", "optimizer_lr_backbone", "normalization_mapping", "input_features", "output_features"]
    sub = max(1, len(curve) // 100)
    last = curve[-max(1, min(5, len(curve))) :]
    out = {
        "steps": args.steps,
        "batch": args.batch,
        "minutes": round(minutes, 2),
        "steps_per_s": round(args.steps / (minutes * 60), 3),
        "samples_per_s": round(args.steps * args.batch / (minutes * 60), 1),
        "final_loss": curve[-1]["loss"] if curve else None,
        "final_loss_mean_last5_logs": sum(c["loss"] for c in last) / len(last) if curve else None,
        "first_loss": curve[0]["loss"] if curve else None,
        "epochs": curve[-1]["epoch"] if curve else None,
        "mean_update_s": sum(c["update_s"] for c in curve) / len(curve) if curve else None,
        "mean_data_s": sum(c["data_s"] for c in curve) / len(curve) if curve else None,
        "vram_mb_peak_child": max(vram) if vram else None,
        "vram_mb_samples": len(vram),
        "log_freq": args.log_freq,
        "loss_curve": curve[::sub],
        "config": {k: cfg.get(k) for k in keep},
        "dataset": {"repo_id": args.repo_id, "root": str(Path(args.root).relative_to(ROOT)) if Path(args.root).is_relative_to(ROOT) else args.root, "video_backend": "pyav", "use_imagenet_stats": True},
        "seed": args.seed,
        "num_workers": args.num_workers,
        "checkpoint": str(dst.relative_to(ROOT)),
        "train_log": str(log_path.relative_to(ROOT)),
        "command": " ".join(cmd),
        "tag": args.tag,
        "hardware": hardware_label(),
        "notes": "MuJoCo simulation only; demos come from the scripted PickPlaceController, not human teleop",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("loss_curve", "command", "config")}, indent=1))


if __name__ == "__main__":
    main()
