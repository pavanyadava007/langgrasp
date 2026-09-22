"""Fine-tune yolo11n-seg on the synthetic LangGrasp dataset and record only measured numbers.

Usage: python scripts/train_yolo.py [--epochs 25] [--time-minutes 35] [--batch 32]
Writes checkpoints/yolo11n-seg-langgrasp.pt and results/yolo_train.json.
"""

import argparse
import json
import os
import shutil
import time

import torch
import yaml


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def count_images(root: str, split: str) -> int:
    d = os.path.join(root, "images", split)
    return len([f for f in os.listdir(d) if f.endswith((".jpg", ".png"))]) if os.path.isdir(d) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_synth/data.yaml")
    ap.add_argument("--weights", default="checkpoints/yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--time-minutes", type=float, default=35.0, help="hard time box; ultralytics stops early if exceeded")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs/yolo")
    ap.add_argument("--name", default="yolo11n-seg-langgrasp")
    ap.add_argument("--out", default="checkpoints/yolo11n-seg-langgrasp.pt")
    ap.add_argument("--results", default="results/yolo_train.json")
    ap.add_argument("--skip-train", default=None, metavar="RUN_DIR", help="reuse a finished run directory instead of training")
    args = ap.parse_args()

    from ultralytics import YOLO

    with open(args.data) as f:
        root = yaml.safe_load(f)["path"]
    model = YOLO(args.weights)
    t0 = time.time()
    budget_s = args.time_minutes * 60.0

    def time_box(trainer):
        # Hard wall-clock cap. We do not use ultralytics' own `time=` argument because it rescales the epoch
        # count (and the LR schedule) to fill the whole budget instead of capping the requested epochs.
        if time.time() - t0 > budget_s:
            print(f"time box of {args.time_minutes} min exceeded after epoch {trainer.epoch + 1}, stopping")
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", time_box)
    if args.skip_train:
        run_dir = args.skip_train
        minutes = None
    else:
        res = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            exist_ok=True,
            workers=8,
            cache="ram",
            plots=True,
            verbose=True,
        )
        minutes = (time.time() - t0) / 60.0
        # ultralytics 8.4 nests a relative project under runs/<task>/, so take the real directory from the trainer
        run_dir = str(getattr(res, "save_dir", None) or model.trainer.save_dir)
    best = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copy(best, args.out)

    # Validate the saved best weights on the val split so the reported mAP is a single measured pass.
    m = YOLO(args.out).val(data=args.data, imgsz=args.imgsz, batch=16, plots=False, verbose=False, project=args.project, name="val_best", exist_ok=True)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[names[int(c)]] = {
            "box_map50": float(m.box.ap50[i]),
            "box_map50_95": float(m.box.ap[i]),
            "mask_map50": float(m.seg.ap50[i]),
            "mask_map50_95": float(m.seg.ap[i]),
        }
    # epochs actually run (results.csv rows), which can be fewer than requested under the time box
    csv = os.path.join(run_dir, "results.csv")
    epochs_run = sum(1 for _ in open(csv)) - 1 if os.path.exists(csv) else None
    if minutes is None and os.path.exists(csv):
        import csv as csvmod

        rows = list(csvmod.DictReader(open(csv)))
        minutes = float(rows[-1]["time"]) / 60.0 if rows and "time" in rows[-1] else None
    out = {
        "weights_in": args.weights,
        "weights_out": args.out,
        "epochs_requested": args.epochs,
        "epochs": epochs_run,
        "time_box_minutes": args.time_minutes,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "train_images": count_images(root, "train"),
        "val_images": count_images(root, "val"),
        "minutes": round(minutes, 2) if minutes is not None else "not measured",
        "minutes_source": "wall clock of model.train() incl. final val" if not args.skip_train else "cumulative 'time' column of ultralytics results.csv (training only)",
        "box_map50": float(m.box.map50),
        "box_map50_95": float(m.box.map),
        "mask_map50": float(m.seg.map50),
        "mask_map50_95": float(m.seg.map),
        "per_class": per_class,
        "hardware": f"{gpu_name()} (x86 EC2 host), not Jetson Orin; shared with other training jobs during this run",
        "results_csv": csv,
    }
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
