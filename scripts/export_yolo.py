"""Export checkpoints/yolo11n-seg-langgrasp.pt to ONNX (opset 17, static) and TensorRT FP16 / INT8 engines.

Usage: python scripts/export_yolo.py [--weights checkpoints/yolo11n-seg-langgrasp.pt] [--no-int8]
"""

import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/yolo11n-seg-langgrasp.pt")
    ap.add_argument("--data", default="data/yolo_synth/data.yaml")
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--out", default="results/yolo_export.json")
    args = ap.parse_args()
    from langgrasp.perception.export import export_all

    info = export_all(args.weights, args.data, do_int8=not args.no_int8)
    info["hardware"] = f"{torch.cuda.get_device_name(0)} (x86 EC2 host), not Jetson Orin; engines are only valid on this GPU type"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
