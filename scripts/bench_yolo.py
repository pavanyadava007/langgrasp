"""Batch-1 latency and mAP of every exported YOLO11n-seg artifact on the local GPU.

Latency: 30 warmup + 300 timed detect() calls on one 640x480 val frame; inference-only (backend-reported) and
end-to-end (letterbox + inference + full-resolution boolean masks). mAP: ultralytics val on the val split at
imgsz 640 with rect=False for every backend (square letterbox, same as the static exports).
Writes results/yolo_latency_l4.json and results/yolo_export_map.json. Numbers are for the GPU actually used,
which is reported in the JSON; they are not Jetson numbers.
"""

import argparse
import glob
import json
import os
import subprocess
import time

import cv2
import numpy as np
import torch


def gpu_state() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,clocks.sm", "--format=csv,noheader"], text=True
        ).strip()
        n_proc = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        return {"nvidia_smi": out, "other_compute_processes": len([p for p in n_proc.splitlines() if p.strip()])}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def pick_frame(root: str) -> str:
    cands = sorted(glob.glob(os.path.join(root, "images", "val", "*_front.jpg")))
    for p in cands:
        lbl = p.replace("images", "labels").replace(".jpg", ".txt")
        if os.path.exists(lbl) and sum(1 for _ in open(lbl)) >= 2:
            return p
    return cands[0]


def bench_backend(seg, rgb: np.ndarray, warmup: int, iters: int, conf: float) -> dict:
    seg.warmup(1)
    for _ in range(warmup):
        seg.detect(rgb, conf=conf)
    e2e, inf = [], []
    for _ in range(iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dets = seg.detect(rgb, conf=conf)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1000)
        inf.append(seg.last_timing["inference_ms"])
    e2e, inf = np.array(e2e), np.array(inf)
    stat = lambda a: {"median_ms": round(float(np.median(a)), 3), "p90_ms": round(float(np.percentile(a, 90)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}  # noqa: E731
    return {"inference_only": stat(inf), "end_to_end": stat(e2e), "n_detections_on_frame": len(dets), "warmup": warmup, "iters": iters}


def val_map(path: str, data: str, quantize: int | None) -> dict:
    from ultralytics import YOLO

    kw = {"quantize": quantize} if quantize else {}
    m = YOLO(path, task="segment").val(data=data, imgsz=640, batch=1, rect=False, plots=False, verbose=False, project="runs/yolo", name="val_export", exist_ok=True, **kw)
    return {"box_map50": float(m.box.map50), "box_map50_95": float(m.box.map), "mask_map50": float(m.seg.map50), "mask_map50_95": float(m.seg.map)}


def decoder_agreement(seg_a, seg_b, root: str, n: int = 40, conf: float = 0.25) -> dict:
    """Compare two backends on val frames: matched detections (same class, box IoU >= 0.5), mean mask IoU."""
    paths = sorted(glob.glob(os.path.join(root, "images", "val", "*.jpg")))[:n]
    n_a = n_b = matched = 0
    ious = []
    for p in paths:
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        da, db = seg_a.detect(rgb, conf=conf), seg_b.detect(rgb, conf=conf)
        n_a += len(da)
        n_b += len(db)
        used = set()
        for a in da:
            best, bi = 0.0, -1
            for j, b in enumerate(db):
                if j in used or a["cls"] != b["cls"]:
                    continue
                x1, y1 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
                x2, y2 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area = lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1])  # noqa: E731
                iou = inter / (area(a["box"]) + area(b["box"]) - inter + 1e-9)
                if iou > best:
                    best, bi = iou, j
            if best >= 0.5:
                used.add(bi)
                matched += 1
                ma, mb = a["mask"], db[bi]["mask"]
                ious.append(float((ma & mb).sum() / max((ma | mb).sum(), 1)))
    return {"frames": len(paths), "dets_a": n_a, "dets_b": n_b, "matched": matched, "mean_mask_iou": float(np.mean(ious)) if ious else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/yolo11n-seg-langgrasp.pt")
    ap.add_argument("--data", default="data/yolo_synth/data.yaml")
    ap.add_argument("--root", default="data/yolo_synth")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()
    from langgrasp.perception.segmentation import Segmenter

    stem = os.path.splitext(args.weights)[0]
    artifacts = {
        "pt-fp32": (args.weights, {}, None),
        "pt-fp16": (args.weights, {"half": True}, 16),
        "onnx-ort-cuda": (stem + ".onnx", {}, None),
        "trt-fp16": (stem + "-fp16.engine", {}, None),
        "trt-int8": (stem + "-int8.engine", {}, None),
    }
    frame_path = pick_frame(args.root)
    rgb = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
    gpu = torch.cuda.get_device_name(0)
    hw = f"{gpu} (x86 EC2 host), not Jetson Orin"
    latency = {"hardware": hw, "frame": frame_path, "frame_hw": list(rgb.shape[:2]), "gpu_state_at_start": gpu_state(), "backends": {}}
    maps = {"hardware": hw, "val_settings": "ultralytics val, imgsz=640, rect=False, batch=1, conf=0.001, iou=0.7", "backends": {}}
    segs = {}
    for name, (path, kw, q) in artifacts.items():
        if not os.path.exists(path):
            latency["backends"][name] = {"status": f"not measured: artifact missing ({path})"}
            maps["backends"][name] = {"status": f"not measured: artifact missing ({path})"}
            continue
        print(f"== {name}: {path}", flush=True)
        seg = Segmenter(path, **kw)
        segs[name] = seg
        r = bench_backend(seg, rgb, args.warmup, args.iters, args.conf)
        r["path"] = path
        r["input_hw"] = list(getattr(seg, "input_hw", (480, 640) if name.startswith("pt") else (640, 640)))
        if name.startswith("pt"):
            r["note"] = "ultralytics pt predictor letterboxes 640x480 to 480x640 (auto pad to stride), so it runs fewer pixels than the static 640x640 exports"
        if name == "onnx-ort-cuda":
            r["providers"] = seg.providers
            r["note"] = "letterbox, NMS and mask decoding in numpy (langgrasp.perception.segmentation); NMS is not in the graph"
        latency["backends"][name] = r
        print(json.dumps(r), flush=True)
        if not args.skip_map:
            try:
                maps["backends"][name] = val_map(path, args.data, q)
            except Exception as e:  # noqa: BLE001
                maps["backends"][name] = {"status": f"not measured: ultralytics val failed: {type(e).__name__}: {e}"[:400]}
            print(json.dumps(maps["backends"][name]), flush=True)
    if "pt-fp32" in segs and "onnx-ort-cuda" in segs:
        maps["onnx_decoder_vs_pt"] = decoder_agreement(segs["pt-fp32"], segs["onnx-ort-cuda"], args.root)
        maps["onnx_decoder_vs_pt"]["note"] = "ultralytics val on the ONNX file uses its own decoder; this compares our numpy decoder against the pt backend"
    latency["gpu_state_at_end"] = gpu_state()
    os.makedirs("results", exist_ok=True)
    with open("results/yolo_latency_l4.json", "w") as f:
        json.dump(latency, f, indent=2)
    if not args.skip_map:
        with open("results/yolo_export_map.json", "w") as f:
            json.dump(maps, f, indent=2)
    print(json.dumps(latency, indent=2))
    print(json.dumps(maps, indent=2))


if __name__ == "__main__":
    main()
