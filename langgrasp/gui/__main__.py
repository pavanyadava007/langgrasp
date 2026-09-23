"""Start the whole GUI with one command: ``python -m langgrasp.gui``.

Binds to the loopback interface only. The GUI can move the (simulated) arm and latch an e-stop, so it is
reached through an SSH tunnel (``ssh -L 8000:localhost:8000 <host>``) rather than exposed on a network
interface. ``--host`` can override that, deliberately and explicitly.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

from langgrasp.gui.api import create_app
from langgrasp.gui.worker import WorkerConfig


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) == 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m langgrasp.gui", description="LangGrasp web GUI (simulation, one worker process)")
    ap.add_argument("--host", default=os.environ.get("GUI_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GUI_PORT", "8000")))
    ap.add_argument("--grounder", default="gdino", choices=["gdino", "oracle"], help="'oracle' skips the GPU grounding model and uses the simulator label map (clearly labelled in the UI)")
    ap.add_argument("--seg-weights", default="checkpoints/yolo11n-seg-langgrasp.pt")
    ap.add_argument("--seg-backend", default="pt", help="pt, pt-fp16, onnx, engine or engine-int8; only backends whose files exist are offered")
    ap.add_argument("--no-segmenter", action="store_true", help="do not load YOLO11-seg at startup (box-only masks until it is needed)")
    ap.add_argument("--gate-threshold", type=float, default=0.30, help="grounding confidence gate, 0.30 is what scripts/eval_modular.py uses")
    ap.add_argument("--safety-mode", default="monitor", choices=["monitor", "enforce"])
    ap.add_argument("--stt-model", default="base", choices=["tiny", "base", "small"])
    ap.add_argument("--no-record", action="store_true", help="do not write runs/gui/<timestamp>_<seed>/")
    ap.add_argument(
        "--stream-size",
        default=None,
        help="render the live stream at WxH instead of the pipeline's 640x480, for hosts with software rendering. It changes only the picture, never what the pipeline sees.",
    )
    ap.add_argument("--log-level", default="info")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if port_in_use(args.host, args.port):
        print(
            f"port {args.port} on {args.host} is already in use.\n"
            f"Pick another one, for example:  GUI_PORT={args.port + 10} make gui\n"
            f"(then forward that port: ssh -L {args.port + 10}:localhost:{args.port + 10} <host>)",
            file=sys.stderr,
        )
        return 2
    stream_size = None
    if args.stream_size:
        w, h = (int(v) for v in args.stream_size.lower().split("x"))
        stream_size = (h, w)
    cfg = WorkerConfig(
        stream_size=stream_size,
        grounder=args.grounder,
        seg_weights=args.seg_weights,
        seg_backend=args.seg_backend,
        load_segmenter=not args.no_segmenter,
        gate_threshold=args.gate_threshold,
        safety_mode=args.safety_mode,
        stt_model=args.stt_model,
        record=not args.no_record,
    )
    import uvicorn

    print(f"LangGrasp GUI on http://{args.host}:{args.port}  (MuJoCo simulation, not a robot, not a Jetson; the page names the machine it found)")
    print(f"forward it with:  ssh -L {args.port}:localhost:{args.port} <this host>")
    uvicorn.run(create_app(cfg=cfg), host=args.host, port=args.port, log_level=args.log_level, ws_ping_interval=20, ws_ping_timeout=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
