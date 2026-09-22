"""Command-line entry point: `langgrasp <subcommand>`."""

from __future__ import annotations

import argparse
import os
import runpy
import sys

SCRIPTS = {
    "smoke": "scripts/smoke_pick.py",
    "eval-oracle": "scripts/eval_oracle.py",
    "eval-modular": "scripts/eval_modular.py",
    "video": "scripts/make_video.py",
    "report": None,
}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="langgrasp", description="LangGrasp: language-guided pick-and-place (simulation)")
    ap.add_argument("command", choices=sorted(SCRIPTS))
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.command == "report":
        from langgrasp.eval.report import main as report_main

        return report_main()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, SCRIPTS[args.command])
    sys.argv = [path] + args.rest
    runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
