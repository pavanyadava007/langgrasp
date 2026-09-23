"""Event vocabulary shared by the simulation worker, the API and the browser.

Design rules this module encodes:

* Nine stages, in order: command, parse, capture, grounding, select, gate, segment, fuse, execute. They are
  the stages that already exist in ``langgrasp/policies/modular.py``; this module does not invent any.
* Latency carries its own meaning. ``compute`` is the wall-clock of the same code region the evaluation
  scripts time (so it is comparable to ``results/*.json``), ``wall_paced`` includes the deliberate pacing
  sleeps and the rendering the GUI adds, and ``none`` means the stage is not timed separately.
* A stage payload says what the code decided and why. It never contains a hardcoded metric.
* Frames are JPEG bytes and travel as binary WebSocket messages; every other event is JSON.

Nothing here imports MuJoCo, torch or the pipeline: this is the contract, not the implementation.
"""

from __future__ import annotations

import json
import struct
import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------- vocabulary

STAGES: tuple[str, ...] = ("command", "parse", "capture", "grounding", "select", "gate", "segment", "fuse", "execute")

StageName = Literal["command", "parse", "capture", "grounding", "select", "gate", "segment", "fuse", "execute"]
StageStatus = Literal["running", "ok", "warn", "fail", "skipped"]
SafetyStateName = Literal["RUN", "REDUCED_SPEED", "HOLD", "ESTOP"]
SafetyMode = Literal["monitor", "enforce"]
CameraName = Literal["front", "wrist", "side"]
FrameKind = Literal["rgb", "depth"]
LatencyKind = Literal["compute", "wall_paced", "none"]
ControllerName = Literal["pipeline", "oracle", "act", "ppo_reach", "ppo_lift"]
ModelState = Literal["absent", "loading", "warm", "missing", "error"]

# Short titles for the stepper, and the one-line explanation the tooltip shows.
STAGE_TITLES: dict[str, str] = {
    "command": "Command",
    "parse": "Parse",
    "capture": "Capture",
    "grounding": "Ground",
    "select": "Select",
    "gate": "Safety gate",
    "segment": "Segment",
    "fuse": "Depth fusion",
    "execute": "Execute",
}
STAGE_HELP: dict[str, str] = {
    "command": "The text to execute, typed or transcribed. Speech is never executed without a human pressing Run.",
    "parse": "Rule-based parser: action, colour word, head noun, spatial reference, and the phrase sent to the grounder.",
    "capture": "One rendered RGB frame plus a metric depth frame from the chosen camera, with the synthetic depth noise model.",
    "grounding": "Grounding DINO tiny, once per command, with an HSV colour check and a generic 'object' fallback for unknown synonyms.",
    "select": "Picks one candidate: colour filter, then spatial reference in image space, then highest score. Timed inside grounding.",
    "gate": "SafetyMonitor.gate_grounding: confidence threshold and ambiguity margin decide whether anything is allowed to move.",
    "segment": "YOLO11n-seg mask that overlaps the chosen box (IoU at least 0.3), else the box itself becomes the mask.",
    "fuse": "Back-projects the masked depth pixels, removes the table plane, and fits a grasp centre, jaw yaw and width.",
    "execute": "Scripted Cartesian pick and place at 10 Hz with damped-least-squares IK, under the safety monitor.",
}

# The pipeline's own LatencyTracer names mapped onto the nine GUI stages.
TRACER_TO_STAGE: dict[str, str] = {
    "parse": "parse",
    "capture": "capture",
    "grounding": "grounding",
    "segmentation": "segment",
    "depth_fusion": "fuse",
    "execute": "execute",
}


# ---------------------------------------------------------------------------- json safety


def jsonable(obj: Any) -> Any:
    """Convert numpy scalars, arrays and enums into JSON-serialisable Python objects.

    Payloads come straight out of the pipeline, so they are full of ``np.float32``, ``np.bool_`` and small
    arrays. Non-finite floats become ``None`` because JSON has no NaN and the browser must show "not run"
    rather than crash.
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "value") and type(obj).__mro__[1:2] and type(obj).__name__.endswith(("State", "Enum")):
        return jsonable(obj.value)
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    return str(obj)


# ---------------------------------------------------------------------------- events


class Event(BaseModel):
    """Base event. ``t`` is ``time.monotonic()`` of the process that created it.

    Every process on this machine shares CLOCK_MONOTONIC, so worker and API timestamps are directly
    comparable; that is how the e-stop latency is measured.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    t: float = Field(default_factory=time.monotonic)
    run_id: str | None = None


class StageStarted(Event):
    type: Literal["stage_started"] = "stage_started"
    stage: StageName
    payload: dict = Field(default_factory=dict)


class StageFinished(Event):
    type: Literal["stage_finished"] = "stage_finished"
    stage: StageName
    status: StageStatus = "ok"
    latency_ms: float | None = None
    latency_kind: LatencyKind = "compute"
    message: str | None = None
    payload: dict = Field(default_factory=dict)
    images: dict[str, str] = Field(default_factory=dict)  # role -> URL path under /runs/<id>/frames


class FrameMeta(Event):
    """Metadata of a JPEG frame. The bytes travel beside it, see :func:`pack_frame`."""

    type: Literal["frame"] = "frame"
    camera: CameraName
    kind: FrameKind = "rgb"
    tick: int | None = None
    width: int
    height: int
    bytes: int


class Safety(Event):
    type: Literal["safety"] = "safety"
    state: SafetyStateName
    reason: str = ""
    mode: SafetyMode = "monitor"
    gate: dict | None = None
    watchdog: dict[str, float | None] = Field(default_factory=dict)  # topic -> age in seconds, None = never seen
    clips: dict[str, int] = Field(default_factory=dict)  # {"limit": n, "velocity": n}
    would_clip: dict[str, int] = Field(default_factory=dict)  # monitor mode: clips that were reported but not applied
    estop_latency_ms: float | None = None
    estop_latency_note: str | None = None


class Tick(Event):
    type: Literal["tick"] = "tick"
    tick: int
    phase: str
    q: list[float]
    q_target: list[float]
    jaw: float
    tcp: list[float]
    sim_t: float | None = None
    bodies: dict[str, list[float]] | None = None  # name -> [x, y, z, qw, qx, qy, qz], for the 3D view


class Outcome(Event):
    type: Literal["outcome"] = "outcome"
    grounding_correct: bool | None = None
    grasped: bool = False
    lifted: bool = False
    placed: bool = False
    aborted: str | None = None
    scored_by: Literal["ground_truth", "none"] = "ground_truth"
    message: str = ""
    latency_ms: dict[str, float] = Field(default_factory=dict)
    sim_compute_ms: float | None = None
    wall_ms: float | None = None


class GateRequest(Event):
    type: Literal["gate_request"] = "gate_request"
    box: list[float] | None = None
    score: float | None = None
    top2: list[float] = Field(default_factory=list)
    threshold: float | None = None
    ambiguous: bool = False
    needs_confirmation: bool = False
    timeout_s: float | None = None


class JobProgress(Event):
    type: Literal["job_progress"] = "job_progress"
    job_id: str
    state: Literal["queued", "running", "done", "cancelled", "error"] = "running"
    done: int = 0
    total: int = 0
    out_path: str | None = None
    last: dict | None = None
    message: str | None = None


class System(Event):
    type: Literal["system"] = "system"
    models: dict[str, dict] = Field(default_factory=dict)  # name -> {state, load_ms, detail}
    gpu: str | None = None
    hardware_label: str | None = None
    scene: dict | None = None
    note: str | None = None


class Reply(Event):
    """Answer to one command that the API is waiting on, correlated by ``reply_to``."""

    type: Literal["reply"] = "reply"
    reply_to: str
    ok: bool = True
    data: dict = Field(default_factory=dict)
    error: str | None = None


class Log(Event):
    type: Literal["log"] = "log"
    level: Literal["info", "warn", "error"] = "info"
    message: str
    detail: dict | None = None


EVENT_TYPES: dict[str, type[Event]] = {
    "stage_started": StageStarted,
    "stage_finished": StageFinished,
    "frame": FrameMeta,
    "safety": Safety,
    "tick": Tick,
    "outcome": Outcome,
    "gate_request": GateRequest,
    "job_progress": JobProgress,
    "reply": Reply,
    "system": System,
    "log": Log,
}


def parse_event(d: dict) -> Event:
    """Validate a raw event dict against its model. Raises KeyError on an unknown type."""
    return EVENT_TYPES[d["type"]].model_validate(d)


# ---------------------------------------------------------------------------- binary frame envelope

FRAME_MAGIC = b"LGF1"
_HEADER = struct.Struct("<4sI")


def pack_frame(meta: dict, jpeg: bytes) -> bytes:
    """``b"LGF1" + uint32 meta length + meta JSON + JPEG bytes``: self describing, one WebSocket message."""
    blob = json.dumps(jsonable(meta), separators=(",", ":")).encode("utf-8")
    return _HEADER.pack(FRAME_MAGIC, len(blob)) + blob + jpeg


def unpack_frame(buf: bytes) -> tuple[dict, bytes]:
    magic, n = _HEADER.unpack_from(buf, 0)
    if magic != FRAME_MAGIC:
        raise ValueError(f"not a LangGrasp frame: {magic!r}")
    start = _HEADER.size
    return json.loads(buf[start : start + n]), bytes(buf[start + n :])


# ---------------------------------------------------------------------------- producer side (hooks)


class PipelineHooks:
    """Sink for pipeline stage events. The base class does nothing, which is what "hooks off" means.

    The contract the pipeline relies on (``langgrasp/policies/modular.py``):

    * no method may raise: the pipeline does not guard these calls, and an observer must not be able to
      change the outcome of a run;
    * no method may mutate its arguments: images and payload dicts are the pipeline's own objects, passed
      without copying so that instrumentation stays cheap;
    * methods are called outside the timed regions, so the latency the tracer records is the latency the
      evaluation scripts record.
    """

    def stage_started(self, stage: str, **payload: Any) -> None:  # noqa: D401
        """A stage is about to run."""

    def stage_finished(
        self,
        stage: str,
        status: str = "ok",
        latency_ms: float | None = None,
        message: str | None = None,
        images: dict[str, Any] | None = None,
        **payload: Any,
    ) -> None:
        """A stage finished. ``images`` maps a role name to a numpy array the sink may encode."""


class RecordingHooks(PipelineHooks):
    """Collects everything in memory. Used by the tests and by anything that wants a transcript."""

    def __init__(self, keep_images: bool = False):
        self.events: list[dict] = []
        self.keep_images = keep_images

    def stage_started(self, stage: str, **payload: Any) -> None:
        self.events.append({"type": "stage_started", "stage": stage, "payload": payload})

    def stage_finished(
        self,
        stage: str,
        status: str = "ok",
        latency_ms: float | None = None,
        message: str | None = None,
        images: dict[str, Any] | None = None,
        **payload: Any,
    ) -> None:
        rec: dict[str, Any] = {
            "type": "stage_finished",
            "stage": stage,
            "status": status,
            "latency_ms": latency_ms,
            "message": message,
            "payload": payload,
            "image_roles": sorted(images) if images else [],
        }
        if self.keep_images and images:
            rec["images"] = dict(images)
        self.events.append(rec)

    @property
    def stages(self) -> list[str]:
        """Stage names in the order they finished."""
        return [e["stage"] for e in self.events if e["type"] == "stage_finished"]

    def payload(self, stage: str) -> dict:
        """Payload of the last finished event for a stage ({} if the stage never finished)."""
        for e in reversed(self.events):
            if e["type"] == "stage_finished" and e["stage"] == stage:
                return e["payload"]
        return {}
