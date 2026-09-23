"""Export the GUI as a static site: the same interface, reading files instead of a worker.

A Docker Space needs a paid plan, a static one does not, so this produces a build that anyone can open: the
Results, Safety and Inspector views read the project's own JSON, and the Live Run view replays a command that
was recorded on the GPU host. Nothing in it can command the simulator, and the page says so in three places.

    python scripts/export_static_gui.py --run <runs/gui/id> --out dist-static

The frontend has to be built with the static flag first:

    cd gui && VITE_STATIC=1 npm run build -- --base ./ --outDir ../dist-static
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from langgrasp.gui.api import RESULTS_MANIFEST, _results_index, hardware_banner
from langgrasp.gui.trace import STAGE_HELP, STAGE_TITLES, STAGES, jsonable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = ROOT / "runs" / "gui"


def read_json(path: Path) -> dict:
    """results/*.json may contain bare NaN, which Python writes and no browser will parse."""
    return jsonable(json.loads(path.read_text()))


def newest_successful_run() -> Path:
    """The newest run that shows the whole pipeline working.

    It has to be a pipeline run, not an oracle one: an oracle run skips grounding, segmentation and depth
    fusion, so replaying it would show six of the nine stages greyed out, which is the opposite of the point.
    """
    fallback = None
    for d in sorted(RUNS.iterdir(), reverse=True):
        events = d / "events.jsonl"
        if not d.is_dir() or not (d / "meta.json").exists() or not events.exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("controller") != "pipeline":
            continue
        placed = False
        stages = set()
        real_grounder = False
        yolo_mask = False
        for line in events.read_text().splitlines():
            e = json.loads(line)
            if e.get("type") == "stage_finished" and e.get("status") != "skipped":
                stages.add(e["stage"])
                if e["stage"] == "grounding":
                    # The oracle grounder reads the simulator's label map and answers in tens of milliseconds;
                    # Grounding DINO takes about 275 ms. Replaying an oracle run would show the pipeline
                    # grounding in 20 ms and segmenting from ground truth, which is not what it does.
                    real_grounder = (e.get("latency_ms") or 0) > 100
                if e["stage"] == "segment":
                    yolo_mask = str((e.get("payload") or {}).get("mask_source", "")).startswith("yolo")
            if e.get("type") == "outcome":
                placed = bool(e.get("placed"))
        frames = len(list((d / "frames").glob("tick_*_front.jpg")))
        if placed and frames > 30 and real_grounder and yolo_mask and {"grounding", "segment", "fuse", "execute"} <= stages:
            return d
        fallback = fallback or d
    if fallback is None:
        raise SystemExit("no recorded pipeline runs under runs/gui: run a command in the GUI first")
    print(f"warning: no run found that used the real grounder and a YOLO mask and placed the object; replaying {fallback.name}")
    return fallback


def run_payload(run_dir: Path) -> dict:
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()]
    return {"meta": json.loads((run_dir / "meta.json").read_text()), "events": jsonable(events), "frames_url": f"runs/{run_dir.name}/frames"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist-static", help="directory the built frontend was written to")
    ap.add_argument("--run", default=None, help="the recorded run to replay (default: the newest successful one)")
    ap.add_argument("--extra-runs", type=int, default=3, help="how many further recorded runs the Inspector should list")
    a = ap.parse_args()

    out = ROOT / a.out
    if not (out / "index.html").exists():
        raise SystemExit(f"{out}/index.html is missing: build the frontend with VITE_STATIC=1 first (see this file's docstring)")
    data = out / "data"
    (data / "results").mkdir(parents=True, exist_ok=True)
    (data / "runs").mkdir(parents=True, exist_ok=True)

    # --- the measured results, exactly as the files hold them, minus the non-finite numbers
    index = _results_index()
    wanted = {f for section in RESULTS_MANIFEST.values() for f in section["files"].values()}
    wanted |= {"modular_protocol.json", "safety_clip_audit.json"}
    exported = []
    for row in index:
        if row["file"] not in wanted or not row["parses"]:
            continue
        (data / "results" / row["file"]).write_text(json.dumps(read_json(RESULTS / row["file"])))
        exported.append(row["file"])
    (data / "results_index.json").write_text(json.dumps({"files": index, "dir": "results/", "note": "Every number shown comes from one of these files, exported from the repository at build time."}))
    sections = {}
    for key, spec in RESULTS_MANIFEST.items():
        by_name = {r["file"]: r for r in index}
        sections[key] = {
            "title": spec["title"],
            "files": {label: {"file": f, "present": f in exported, "mtime": (by_name.get(f) or {}).get("mtime"), "parses": True} for label, f in spec["files"].items()},
        }
    (data / "results_manifest.json").write_text(json.dumps({"sections": sections}))

    # --- the hazard table
    fmea = yaml.safe_load((ROOT / "langgrasp" / "gui" / "fmea.yaml").read_text())
    fmea.update({"source": "langgrasp/gui/fmea.yaml", "checked_by": "tests/test_gui_fmea.py", "mtime": (ROOT / "langgrasp" / "gui" / "fmea.yaml").stat().st_mtime})
    (data / "fmea.json").write_text(json.dumps(fmea))

    # --- the recorded runs
    run_dir = (ROOT / a.run) if a.run else newest_successful_run()
    others = [d for d in sorted(RUNS.iterdir(), reverse=True) if d.is_dir() and (d / "meta.json").exists() and d != run_dir][: a.extra_runs]
    listed = []
    for d in [run_dir, *others]:
        payload = run_payload(d)
        (data / "runs" / f"{d.name}.json").write_text(json.dumps(payload))
        shutil.copytree(d / "frames", out / "runs" / d.name / "frames", dirs_exist_ok=True)
        meta = payload["meta"]
        listed.append({"id": d.name, "seed": meta.get("seed"), "command": meta.get("command"), "controller": meta.get("controller"), "started": meta.get("started"), "safety_mode": meta.get("safety_mode"), "mtime": d.stat().st_mtime, "n_events": len(payload["events"])})
    (data / "runs.json").write_text(json.dumps({"runs": listed}))

    # --- what the header and the Live Run view need
    meta = json.loads((run_dir / "meta.json").read_text())
    hardware = meta.get("hardware") or "NVIDIA L4 (x86 EC2 host), simulation only, not Jetson"
    gpu = hardware.split(" (")[0]
    scene = meta.get("scenario") or {}
    (data / "scene.json").write_text(json.dumps({"scene": scene, "examples": __import__("langgrasp.gui.api", fromlist=["EXAMPLE_COMMANDS"]).EXAMPLE_COMMANDS}))
    (data / "system.json").write_text(
        json.dumps(
            {
                "banner": hardware_banner(gpu),
                "hardware_label": hardware,
                "gpu": gpu,
                "worker": {"alive": False, "busy": False, "pid": None},
                "models": {"recorded": {"state": "warm", "load_ms": None, "detail": f"this page replays a run recorded on {gpu}; no model is loaded in your browser"}},
                "versions": {},
                "python": meta.get("python", ""),
                "engines": [],
                "stages": [{"name": s, "title": STAGE_TITLES[s], "help": STAGE_HELP[s]} for s in STAGES],
                "cameras": ["front"],
                "safety": None,
                "safety_note": "Recorded from a run where the monitor observed rather than enforced, which is the default: HOLD and ESTOP stop the arm, clipping is reported. See results/safety_clip_audit.json.",
                "watchdog_note": "The staleness timers measure how long ago the worker sampled each topic, not an independent publisher. This page is a replay, so they are frozen at the values that run recorded.",
                "hub": {},
                "results_dir": "results/",
            }
        )
    )
    (data / "replay.json").write_text(
        json.dumps({"run_id": run_dir.name, "frames_url": f"runs/{run_dir.name}/frames", "hello": {"type": "hello", "banner": hardware_banner(gpu), "stages": list(STAGES), "scene": scene, "models": {}, "run_id": run_dir.name}})
    )

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"static build in {out}: {len(exported)} results files, {len(listed)} recorded runs, replay {run_dir.name}, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
