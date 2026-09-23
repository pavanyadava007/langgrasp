"""HTTP and WebSocket API in front of the simulation worker.

This process never imports MuJoCo, torch or a model: it owns the sockets, the worker handle and the files in
``results/``. Every number it returns is either read from a file on disk (with that file's modification time
attached, so the browser can show provenance) or forwarded from a live worker event. There is no metric
constant anywhere in this module, and a missing file is reported as missing rather than filled in.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from langgrasp.gui.hub import EventHub
from langgrasp.gui.trace import STAGE_HELP, STAGE_TITLES, STAGES, jsonable
from langgrasp.gui.worker import CAMERAS, RUNS_DIR, WorkerConfig, WorkerHandle

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CHECKPOINTS = ROOT / "checkpoints"

HARDWARE_BANNER = "Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware"

# How long a request waits for the worker to answer. The worker is single threaded and a command can land
# while a 700 ms pick is running, so these are generous; they are module constants so tests can shorten them.
SCENE_TIMEOUT_S = 30.0
ESTOP_WAIT_S = 1.0
RESET_TIMEOUT_S = 10.0
STT_TIMEOUT_S = 120.0

# Which results files each dashboard section needs. The browser never hardcodes a filename, and a file that
# is absent becomes a "not run" card rather than a gap.
RESULTS_MANIFEST: dict[str, dict] = {
    "protocol": {
        "title": "Stratified protocol, place rate with Wilson 95% intervals",
        "files": {
            "Oracle executor (ground-truth grasp pose, no perception)": "oracle_protocol.json",
            "Modular: Grounding DINO + colour check + YOLO11-seg + depth fusion": "modular_protocol.json",
            "Modular, no colour check": "modular_nocolor_protocol.json",
            "Modular, box-only mask (no YOLO)": "modular_noyolo_protocol.json",
            "Modular, no depth noise": "modular_nonoise_protocol.json",
        },
    },
    "fixed_goal": {
        "title": "Fixed goal: red cube to tray, 100 identical scenes",
        "files": {
            "Oracle executor": "oracle_fixed_goal.json",
            "Modular pipeline": "modular_fixed_goal.json",
            "ACT attempt 1 (120 demos, 128 px, 20k steps)": "act_eval.json",
            "ACT attempt 2 (240 demos, 192 px, 25k steps)": "act_eval_192.json",
            "ACT attempt 3 (700 demos, cropped 192 px, 30k steps)": "act_eval_v3.json",
            "ACT attempt 3 resumed to 60k steps": "act_eval_v3_60k.json",
        },
    },
    "latency": {"title": "Latency budget per command", "files": {"Clean bench, no other GPU job": "pipeline_latency_l4.json", "During the protocol runs": "modular_protocol.json"}},
    "perception": {"title": "YOLO11n-seg: accuracy and latency by backend", "files": {"Fine-tune": "yolo_train.json", "Latency": "yolo_latency_l4.json", "mAP per export": "yolo_export_map.json", "Export": "yolo_export.json", "Depth fusion vs ground truth": "depth_fusion_accuracy.json"}},
    "rl": {"title": "PPO, sim-to-sim gap (no real robot)", "files": {"Reach": "ppo_sim2sim_gap.json", "Lift": "ppo_sim2sim_gap_lift.json", "Reach with DR": "ppo_reach_dr.json", "Reach without DR": "ppo_reach_nodr.json", "Lift with DR": "ppo_lift_dr.json", "Lift without DR": "ppo_lift_nodr.json", "Lift, DR curriculum": "ppo_lift_drcurriculum.json"}},
    "speech_ros_safety": {"title": "Speech, ROS 2 and the safety monitor", "files": {"faster-whisper latency": "stt_latency_l4.json", "ROS 2 Humble pipeline smoke": "ros2_smoke.json", "Safety clip audit": "safety_clip_audit.json"}},
}


# ---------------------------------------------------------------------------- request bodies


class SceneRequest(BaseModel):
    seed: int = 5000
    stratum: Literal["seen", "unseen", "langvar"] = "seen"
    lighting: Literal["nominal", "degraded"] | None = None
    n_distractors: int | None = Field(default=None, ge=0, le=3)
    fixed_goal: bool = False


class RunConfig(BaseModel):
    use_color_check: bool = True
    use_yolo_mask: bool = True
    depth_noise: bool = True
    seg_backend: str = "pt"
    require_human_confirm: bool = False
    safety_mode: Literal["monitor", "enforce"] = "monitor"
    speed: float = Field(default=1.0, ge=0.0, le=20.0)


class RunRequest(BaseModel):
    command: str | None = None
    controller: Literal["pipeline", "oracle", "act", "ppo_reach", "ppo_lift"] = "pipeline"
    config: RunConfig = Field(default_factory=RunConfig)
    source: Literal["typed", "stt", "chip"] = "typed"
    stt_latency_ms: float | None = None


class JobRequest(BaseModel):
    controller: Literal["pipeline", "oracle"] = "pipeline"
    strata: dict[str, int] = Field(default_factory=lambda: {"seen": 10, "unseen": 10, "langvar": 10})
    config: RunConfig = Field(default_factory=RunConfig)
    fixed_goal: bool = False
    base_seed: int = 5000
    out_path: str | None = None
    overwrite: bool = False
    tag: str | None = None


class EstopRequest(BaseModel):
    client_ts_ms: float | None = None


class ConfirmRequest(BaseModel):
    decision: Literal["confirm", "reject"]


class PauseRequest(BaseModel):
    paused: bool | None = None
    step: bool = False


# ---------------------------------------------------------------------------- helpers


def _versions() -> dict[str, str | None]:
    out = {}
    for name in ("torch", "mujoco", "ultralytics", "transformers", "tensorrt", "lerobot", "faster-whisper", "onnxruntime-gpu", "fastapi", "uvicorn", "numpy", "opencv-python-headless"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _engine_files() -> list[dict]:
    out = []
    for pattern in ("yolo11n-seg-langgrasp.pt", "yolo11n-seg-langgrasp.onnx", "yolo11n-seg-langgrasp-fp16.engine", "yolo11n-seg-langgrasp-int8.engine"):
        p = CHECKPOINTS / pattern
        out.append({"path": f"checkpoints/{pattern}", "present": p.exists(), "size_mb": round(p.stat().st_size / 1e6, 1) if p.exists() else None, "mtime": p.stat().st_mtime if p.exists() else None})
    return out


def _results_index() -> list[dict]:
    """Every results file that parses, with the facts needed to label it. No numbers are computed here."""
    out = []
    if not RESULTS_DIR.exists():
        return out
    for p in sorted(RESULTS_DIR.glob("*.json")):
        st = p.stat()
        row = {"file": p.name, "size": st.st_size, "mtime": st.st_mtime, "parses": True, "approach": None, "n_trials": None, "hardware": None}
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            row["parses"] = False
            out.append(row)
            continue
        if isinstance(d, dict):
            row["approach"] = d.get("approach") or d.get("task") or d.get("label")
            row["hardware"] = d.get("hardware")
            summary = d.get("summary")
            if isinstance(summary, dict):
                row["n_trials"] = summary.get("n_trials")
        out.append(row)
    return out


def _safe_results_path(name: str) -> Path:
    if name != Path(name).name or not name.endswith(".json"):
        raise HTTPException(status_code=400, detail={"code": "bad_filename", "message": f"'{name}' is not a results file name."})
    p = RESULTS_DIR / name
    if not p.exists():
        raise HTTPException(status_code=404, detail={"code": "not_run", "message": f"results/{name} does not exist: that measurement has not been run."})
    return p


def _run_index() -> list[dict]:
    out = []
    if not RUNS_DIR.exists():
        return out
    for d in sorted(RUNS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta_path = d / "meta.json"
        if not d.is_dir() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        events = d / "events.jsonl"
        out.append({"id": d.name, "seed": meta.get("seed"), "command": meta.get("command"), "controller": meta.get("controller"), "started": meta.get("started"), "safety_mode": meta.get("safety_mode"), "mtime": d.stat().st_mtime, "n_events": sum(1 for _ in events.open()) if events.exists() else 0})
    return out


# ---------------------------------------------------------------------------- the app


def create_app(worker=None, cfg: WorkerConfig | None = None, start_worker: bool = True) -> FastAPI:
    """Build the app. Tests pass their own worker; production gets a real spawned one."""
    handle = worker if worker is not None else WorkerHandle(cfg or WorkerConfig())
    hub = EventHub(handle)

    async def lifespan(app: FastAPI):
        hub.start()
        if start_worker:
            handle.start()
        try:
            yield
        finally:
            hub.stop()
            with contextlib.suppress(Exception):
                handle.stop()

    app = FastAPI(title="LangGrasp GUI", version="0.1.0", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.worker = handle
    app.state.hub = hub

    def require_idle() -> None:
        if handle.flags.estop.value:
            raise HTTPException(status_code=423, detail={"code": "estop_latched", "message": "The e-stop is latched. Press Reset before running a command."})
        if handle.busy:
            raise HTTPException(status_code=409, detail={"code": "busy", "message": "A run or a batch job is already using the simulator."})

    # ------------------------------------------------------------------ system and scene
    @app.get("/api/system")
    async def get_system() -> dict:
        safety = hub.last.get("safety")
        return {
            "banner": HARDWARE_BANNER,
            "hardware_label": (hub.last.get("system") or {}).get("hardware_label"),
            "gpu": (hub.last.get("system") or {}).get("gpu"),
            "worker": {"alive": handle.alive, "busy": handle.busy, "pid": getattr(getattr(handle, "proc", None), "pid", None)},
            "models": hub.models,
            "versions": _versions(),
            "python": os.sys.version.split()[0],
            "engines": _engine_files(),
            "stages": [{"name": s, "title": STAGE_TITLES[s], "help": STAGE_HELP[s]} for s in STAGES],
            "cameras": list(CAMERAS),
            "safety": safety,
            "safety_note": "The GUI observes the safety monitor by default: HOLD and ESTOP stop the arm, while joint and velocity clipping is reported but not applied, so a live run follows the same trajectory as the runs in results/. See results/safety_clip_audit.json.",
            "watchdog_note": "The staleness timers measure how long ago this worker sampled each topic, not how long ago an independent publisher spoke: there is no separate camera node here. The pipeline captures one frame per command and then executes open loop, so while a command is being carried out the camera and command timers are refreshed each tick. They still fire if the worker itself stalls, and the command timer still holds the arm when the system has been sitting idle.",
            "hub": hub.stats(),
            "results_dir": str(RESULTS_DIR.relative_to(ROOT)),
        }

    @app.get("/api/scene")
    async def get_scene() -> dict:
        return {"scene": hub.scene, "examples": EXAMPLE_COMMANDS}

    @app.post("/api/scene", status_code=202)
    async def post_scene(req: SceneRequest) -> dict:
        require_idle()
        handle.send(cmd="scene", **req.model_dump())
        event = await hub.wait_for(lambda e: e["type"] == "system" and e.get("note") == "scene reset", timeout=SCENE_TIMEOUT_S)
        if event is None:
            raise HTTPException(status_code=504, detail={"code": "worker_timeout", "message": f"The worker did not confirm the new scene within {SCENE_TIMEOUT_S:.0f} s."})
        return {"scene": event["scene"]}

    # ------------------------------------------------------------------ running
    @app.post("/api/run", status_code=202)
    async def post_run(req: RunRequest) -> dict:
        require_idle()
        if req.controller in ("act", "ppo_reach", "ppo_lift"):
            raise HTTPException(status_code=501, detail={"code": "controller_not_wired", "message": f"The {req.controller} controller is not wired into the worker yet."})
        run_id = f"{time.strftime('%Y-%m-%dT%H-%M-%S')}_{(hub.scene or {}).get('seed', 0)}"
        handle.send(cmd="run", run_id=run_id, command=req.command, controller=req.controller, config=req.config.model_dump(), source=req.source, stt_latency_ms=req.stt_latency_ms)
        return {"run_id": run_id, "watch": "/ws/live"}

    @app.post("/api/estop")
    async def post_estop(req: EstopRequest) -> dict:
        api_ts = handle.estop()
        event = await hub.wait_for(lambda e: e["type"] == "safety" and e["state"] == "ESTOP", timeout=ESTOP_WAIT_S)
        return {
            "state": event["state"] if event else "requested",
            "api_ts_monotonic": api_ts,
            "estop_latency_ms": (event or {}).get("estop_latency_ms"),
            "client_ts_ms": req.client_ts_ms,
            "note": "estop_latency_ms is measured from this request arriving to the first tick the arm was held, both on this machine's monotonic clock.",
        }

    @app.post("/api/reset")
    async def post_reset() -> dict:
        handle.reset_estop()
        event = await hub.wait_for(lambda e: e["type"] == "safety" and e["state"] != "ESTOP", timeout=RESET_TIMEOUT_S)
        return {"state": (event or {}).get("state", "unknown"), "reason": (event or {}).get("reason", "")}

    @app.post("/api/pause")
    async def post_pause(req: PauseRequest) -> dict:
        if req.step:
            handle.step_once()
            return {"paused": bool(handle.flags.paused.value), "stepped": True}
        if req.paused is None:
            raise HTTPException(status_code=400, detail={"code": "bad_request", "message": "Send either paused=true/false or step=true."})
        handle.pause(req.paused)
        return {"paused": req.paused, "stepped": False}

    @app.post("/api/confirm")
    async def post_confirm(req: ConfirmRequest) -> dict:
        handle.send(cmd="confirm", decision=req.decision)
        return {"accepted": req.decision == "confirm"}

    @app.post("/api/stt")
    async def post_stt(audio: UploadFile = File(...), model_size: str | None = None) -> dict:
        suffix = Path(audio.filename or "clip.webm").suffix or ".webm"
        blob = await audio.read()
        if not blob:
            raise HTTPException(status_code=400, detail={"code": "empty_audio", "message": "The recording was empty. Check the microphone permission and try again."})
        fd, path = tempfile.mkstemp(prefix="langgrasp_stt_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            data = await hub.request(cmd="stt", path=path, model_size=model_size, timeout=STT_TIMEOUT_S)
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail={"code": "stt_timeout", "message": str(e)}) from e
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail={"code": "stt_failed", "message": str(e)}) from e
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)
        return data

    # ------------------------------------------------------------------ batch jobs
    @app.post("/api/jobs", status_code=202)
    async def post_job(req: JobRequest) -> dict:
        require_idle()
        strata = {k: v for k, v in req.strata.items() if k in ("seen", "unseen", "langvar") and v > 0}
        if not strata:
            raise HTTPException(status_code=400, detail={"code": "no_strata", "message": "Choose at least one stratum with a trial count above zero."})
        tag = req.tag or ("oracle" if req.controller == "oracle" else "modular")
        default_name = f"gui_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        out_path = RESULTS_DIR / (Path(req.out_path).name if req.out_path else default_name)
        if not out_path.name.endswith(".json"):
            raise HTTPException(status_code=400, detail={"code": "bad_filename", "message": "The output file must end in .json and live in results/."})
        if out_path.exists() and not req.overwrite:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "file_exists",
                    "message": f"results/{out_path.name} already exists, last written {time.strftime('%Y-%m-%d %H:%M', time.localtime(out_path.stat().st_mtime))}. Confirm the overwrite, or write to results/{default_name} instead.",
                    "suggestion": default_name,
                },
            )
        job_id = f"job_{time.strftime('%Y%m%d-%H%M%S')}"
        handle.send(cmd="job", job_id=job_id, controller=req.controller, strata=strata, config=req.config.model_dump(), fixed_goal=req.fixed_goal, base_seed=req.base_seed, out_path=str(out_path), overwrite=req.overwrite, tag=tag)
        return {"job_id": job_id, "out_path": f"results/{out_path.name}", "total": sum(strata.values())}

    @app.get("/api/jobs")
    async def get_jobs() -> dict:
        return {"jobs": list(hub.jobs.values())}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = hub.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail={"code": "no_such_job", "message": f"No job '{job_id}' in this session."})
        return job

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict:
        if job_id not in hub.jobs:
            raise HTTPException(status_code=404, detail={"code": "no_such_job", "message": f"No job '{job_id}' in this session."})
        handle.send(cmd="cancel_job", job_id=job_id)
        return {"cancelling": job_id, "note": "The job stops before the next trial and writes no file."}

    # ------------------------------------------------------------------ results and runs
    @app.get("/api/results")
    async def get_results() -> dict:
        return {"files": _results_index(), "dir": "results/", "note": "Every number the dashboard shows comes from one of these files. A file that is absent is shown as 'not run'."}

    @app.get("/api/results/manifest")
    async def get_manifest() -> dict:
        index = {r["file"]: r for r in _results_index()}
        sections = {}
        for key, spec in RESULTS_MANIFEST.items():
            files = {}
            for label, fname in spec["files"].items():
                row = index.get(fname)
                files[label] = {"file": fname, "present": bool(row), "mtime": row["mtime"] if row else None, "parses": row["parses"] if row else False}
            sections[key] = {"title": spec["title"], "files": files}
        return {"sections": sections}

    @app.get("/api/results/{name}")
    async def get_results_file(name: str, raw: int = 0) -> Response:
        """Serve a results file.

        Python writes a rate of 0 successes in 0 trials as ``NaN``, which is legal for ``json.dump`` and
        illegal for the browser's JSON parser: five of these files contain it, and a browser given them
        verbatim sees a parse error and shows a measurement that exists as "not run". So the default response
        replaces every non-finite number with null and says in a header that it did. ``?raw=1`` returns the
        file's own bytes, for anyone checking provenance.
        """
        p = _safe_results_path(name)
        text = p.read_text()
        headers = {"X-Source-File": f"results/{name}", "X-Source-Mtime": str(p.stat().st_mtime)}
        if raw:
            headers["X-Json-Sanitised"] = "false"
            return Response(content=text, media_type="application/json", headers=headers)
        had_nonfinite = "NaN" in text or "Infinity" in text
        try:
            body = json.dumps(jsonable(json.loads(text)))
        except (json.JSONDecodeError, ValueError):
            headers["X-Json-Sanitised"] = "false"
            return Response(content=text, media_type="application/json", headers=headers)
        headers["X-Json-Sanitised"] = "true" if had_nonfinite else "false"
        if had_nonfinite:
            headers["X-Json-Note"] = "non-finite numbers in the file (a rate over zero trials) were replaced with null; add ?raw=1 for the file itself"
        return Response(content=body, media_type="application/json", headers=headers)

    @app.get("/api/runs")
    async def get_runs() -> dict:
        return {"runs": _run_index()}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        d = RUNS_DIR / Path(run_id).name
        if not (d / "meta.json").exists():
            raise HTTPException(status_code=404, detail={"code": "no_such_run", "message": f"No recorded run '{run_id}'."})
        events = []
        ev_path = d / "events.jsonl"
        if ev_path.exists():
            for line in ev_path.read_text().splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    events.append(json.loads(line))
        return {"meta": json.loads((d / "meta.json").read_text()), "events": events, "frames_url": f"/runs/{d.name}/frames"}

    @app.get("/runs/{run_id}/frames/{name}")
    async def get_run_frame(run_id: str, name: str) -> FileResponse:
        p = RUNS_DIR / Path(run_id).name / "frames" / Path(name).name
        if not p.exists():
            raise HTTPException(status_code=404, detail={"code": "no_such_frame", "message": f"{name} was not recorded."})
        return FileResponse(p, media_type="image/jpeg" if p.suffix == ".jpg" else "image/png")

    # ------------------------------------------------------------------ live stream
    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        cams = {c for c in (ws.query_params.get("cameras", "front").split(",")) if c in CAMERAS} or {"front"}
        depth = ws.query_params.get("depth") == "1"
        handle.send(cmd="subscribe", cameras=sorted(cams), depth=depth, bodies=ws.query_params.get("bodies") == "1")
        channel = hub.add_client(cams, depth)
        import asyncio

        async def reader() -> None:
            """Client messages: change the camera subscription, nothing that can move the arm."""
            while True:
                msg = await ws.receive_json()
                if "cameras" in msg or "depth" in msg or "bodies" in msg:
                    cams2 = {c for c in msg.get("cameras", sorted(cams)) if c in CAMERAS} or {"front"}
                    channel.cameras = cams2
                    channel.depth = bool(msg.get("depth", depth))
                    handle.send(cmd="subscribe", cameras=sorted(cams2), depth=channel.depth, bodies=bool(msg.get("bodies", False)))

        task = asyncio.ensure_future(reader())
        try:
            await ws.send_json({"type": "hello", "banner": HARDWARE_BANNER, "stages": list(STAGES), "scene": hub.scene, "models": hub.models, "run_id": hub.run_id})
            while True:
                await channel.wake.wait()
                js, frames = channel.take()
                for event in js:
                    await ws.send_json(event)
                for meta, jpeg in frames:
                    await ws.send_bytes(hub.encode_frame(meta, jpeg))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()
            hub.remove_client(channel)

    # ------------------------------------------------------------------ static frontend
    if (STATIC_DIR / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        async def placeholder() -> str:
            return (
                "<!doctype html><html><head><title>LangGrasp GUI</title>"
                '<style>body{font:16px/1.6 system-ui;margin:3rem auto;max-width:44rem;background:#0f1216;color:#e6e9ee}'
                "code{background:#1f242c;padding:.1rem .3rem;border-radius:4px}</style></head><body>"
                f"<p><strong>{HARDWARE_BANNER}</strong></p>"
                "<h1>LangGrasp GUI: API is up, the frontend is not built yet</h1>"
                "<p>The simulation worker and the API are running. The React app arrives in phase 3.</p>"
                "<p>Meanwhile: <a href='/api/docs'>/api/docs</a>, <a href='/api/system'>/api/system</a>, "
                "<a href='/api/results'>/api/results</a>, and the live stream at <code>/ws/live</code>.</p>"
                "</body></html>"
            )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        detail: Any = exc.detail
        if not isinstance(detail, dict):
            detail = {"code": "error", "message": str(detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": detail})

    return app


# Example commands, one per stratum, taken from the scenario generator's own grammar
# (langgrasp/sim/scenarios.py) rather than invented here.
EXAMPLE_COMMANDS = [
    {"stratum": "seen", "command": "pick the red cube", "why": "plain command, kind and colour both in the training sets"},
    {"stratum": "seen", "command": "pick the blue screwdriver", "why": "the colour check earns its place here: the grounder scores a yellow screwdriver almost as high"},
    {"stratum": "unseen", "command": "pick the purple can", "why": "colour never seen in the demos or the YOLO fine-tune"},
    {"stratum": "unseen", "command": "pick the green bar", "why": "held-out shape"},
    {"stratum": "langvar", "command": "grab the blue one", "why": "attribute only, no noun: falls back to grounding 'object' plus the colour filter"},
    {"stratum": "langvar", "command": "pick the yellow driver", "why": "synonym the grounder does not know"},
    {"stratum": "langvar", "command": "pick the red cube on the left", "why": "spatial reference, two identical objects"},
]
