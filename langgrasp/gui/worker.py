"""The one process that owns the simulation, the GPU models and the safety monitor.

MuJoCo renderers and the GPU models are not thread safe and loading them twice would double VRAM, so
exactly one process touches them. The API process talks to it through two queues (commands in, events out)
and a handful of shared flags (e-stop, pause, step, speed, camera subscriptions) that are read inside the
control loop, so a click does not have to wait for the queue to drain.

What this module adds to a run, and what it does not:

* it observes: hook events from the pipeline, tick events from the controller's existing ``record_fn``
  contract, and the safety monitor wrapped around ``env.step`` exactly as
  ``tests/test_safety.py::test_sim_pick_through_safety_monitor`` does;
* it paces: live runs sleep between 10 Hz ticks so a person can watch. Sleeping does not change physics, and
  ``speed = 0`` removes it;
* in ``monitor`` mode (the default) the joint targets the arm receives are the controller's own, so the
  motion is the motion the evaluation scripts measured; the monitor still reports what it would have clipped,
  and HOLD and ESTOP still stop the arm. In ``enforce`` mode the clipped target is applied, as the ROS 2
  safety node does, and the run is marked so its outcome is never mistaken for a protocol number.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from langgrasp.gui.trace import (
    FrameMeta,
    GateRequest,
    JobProgress,
    Log,
    Outcome,
    PipelineHooks,
    Reply,
    Safety,
    StageFinished,
    StageStarted,
    System,
    Tick,
    jsonable,
)

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs" / "gui"
CAMERAS = ("front", "wrist", "side")
CAM_BITS = {"front": 1, "wrist": 2, "side": 4}
DEPTH_BIT = 8
BODIES_BIT = 16
MAX_KEPT_RUNS = 50
EVENT_QUEUE_HIGH_WATER = 60  # above this the worker drops frames, never stage or safety events
IDLE_POLL_S = 0.02  # how often an idle worker looks at the command queue and the e-stop flag
PACE_SLICE_S = 0.004  # the pacing sleep is cut into slices this long so an e-stop is seen inside a tick


class MotionAborted(RuntimeError):
    """Raised inside the control loop when the operator e-stops, to stop the executor immediately."""


class JobCancelled(RuntimeError):
    """Raised inside a batch job when the operator cancels it, so no partial results file is written."""


# ---------------------------------------------------------------------------- shared flags


class ControlFlags:
    """Lock-free shared scalars. Written by the API process, read inside the control loop."""

    def __init__(self, ctx):
        self.estop = ctx.Value("i", 0)
        self.estop_t = ctx.Value("d", 0.0)  # time.monotonic() when the API received the click
        self.paused = ctx.Value("i", 0)
        self.step_once = ctx.Value("i", 0)
        self.speed = ctx.Value("d", 1.0)  # 1.0 = real time (10 Hz), 0 = as fast as the sim runs
        self.cameras = ctx.Value("i", CAM_BITS["front"])
        self.cancel_job = ctx.Value("i", 0)
        self.busy = ctx.Value("i", 0)  # 1 while a run or job is executing

    def request_estop(self, client_t: float | None = None) -> float:
        t = time.monotonic()
        self.estop_t.value = t
        self.estop.value = 1
        return t

    def clear_estop(self) -> None:
        self.estop.value = 0
        self.estop_t.value = 0.0


@dataclass
class WorkerConfig:
    """Everything the worker needs to know before it loads anything."""

    grounder: str = "gdino"  # "gdino" (the real model) or "oracle" (label map, for tests and no-GPU hosts)
    seg_weights: str = "checkpoints/yolo11n-seg-langgrasp.pt"
    seg_backend: str = "pt"
    load_segmenter: bool = True
    stt_model: str = "base"
    image_size: tuple[int, int] = (480, 640)
    jpeg_quality: int = 72
    gate_threshold: float = 0.30  # what scripts/eval_modular.py uses, not the SafetyConfig default of 0.35
    safety_mode: str = "monitor"
    command_staleness_s: float = 30.0  # FMEA H6: a command older than this is a HOLD, not an e-stop
    idle_frame_period_s: float = 0.2
    record: bool = True
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------- hooks that publish


class QueueHooks(PipelineHooks):
    """Pipeline observer that turns stage events into queue events and stage images into JPEG/PNG files."""

    def __init__(self, worker: SimWorker):
        self.w = worker

    def stage_started(self, stage: str, **payload: Any) -> None:
        self.w.emit(StageStarted(stage=stage, run_id=self.w.run_id, payload=jsonable(payload)))

    def stage_finished(
        self,
        stage: str,
        status: str = "ok",
        latency_ms: float | None = None,
        message: str | None = None,
        images: dict[str, Any] | None = None,
        **payload: Any,
    ) -> None:
        saved = self.w.save_stage_images(stage, images) if images else {}
        kind = "none" if latency_ms is None else ("wall_paced" if stage == "execute" and self.w.paced else "compute")
        self.w.emit(
            StageFinished(
                stage=stage,
                status=status,
                latency_ms=latency_ms,
                latency_kind=kind,
                message=message,
                payload=jsonable(payload),
                images=saved,
                run_id=self.w.run_id,
            )
        )
        if stage == "capture":
            self.w.heartbeat_sensors()


# ---------------------------------------------------------------------------- the worker


class SimWorker:
    def __init__(self, cmd_q, evt_q, flags: ControlFlags, cfg: WorkerConfig):
        self.cmd_q = cmd_q
        self.evt_q = evt_q
        self.flags = flags
        self.cfg = cfg
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.env = None
        self.grounder = None
        self.segmenters: dict[str, Any] = {}
        self.stt = None
        self.monitor = None
        self.scenario = None
        self.hooks = QueueHooks(self)
        self.models: dict[str, dict] = {}
        self.mode = cfg.safety_mode
        self.paced = True
        self._shutdown = False
        self._phase = "idle"
        self._next_tick_t: float | None = None
        self._sim_compute_ms = 0.0
        self._clips = {"limit": 0, "velocity": 0}
        self._would = {"limit": 0, "velocity": 0}
        self._last_safety_key: tuple | None = None
        self._last_idle_frame = 0.0
        self._tick_frame_i = 0
        self._executing = False  # True only while a controller is driving the arm
        self._command_active = False  # True from the moment a command is accepted until its motion ends
        self._events_fp = None
        self._hardware = "unknown"

    # ------------------------------------------------------------------ plumbing
    def emit(self, event, jpeg: bytes | None = None) -> None:
        """Put one event on the queue. Frames are dropped first when a client cannot keep up."""
        d = event.model_dump()
        if jpeg is not None:
            d["jpeg"] = jpeg
        try:
            if d["type"] == "frame" and self.evt_q.qsize() > EVENT_QUEUE_HIGH_WATER:
                return
        except NotImplementedError:  # pragma: no cover - qsize is available on Linux
            pass
        self.evt_q.put(d)
        self._record(d)

    def log(self, message: str, level: str = "info", **detail) -> None:
        self.emit(Log(level=level, message=message, detail=jsonable(detail) or None, run_id=self.run_id))

    def _record(self, d: dict) -> None:
        if self._events_fp is None:
            return
        row = {k: v for k, v in d.items() if k != "jpeg"}
        self._events_fp.write(json.dumps(jsonable(row), separators=(",", ":")) + "\n")

    # ------------------------------------------------------------------ model loading
    def build(self) -> None:
        from langgrasp.eval.harness import hardware_label
        from langgrasp.safety import SafetyConfig
        from langgrasp.sim.env import LangGraspEnv

        self._hardware = hardware_label()
        self._set_model("simulation", "loading")
        t0 = time.perf_counter()
        self.env = LangGraspEnv(seed=self.cfg.seed)
        self._set_model("simulation", "warm", (time.perf_counter() - t0) * 1e3, detail="MuJoCo scene, SO-ARM100 (Menagerie)")
        self.monitor = make_monitor(
            SafetyConfig(
                joint_lo=tuple(self.env.kin.lo),
                joint_hi=tuple(self.env.kin.hi),
                grounding_conf_threshold=self.cfg.gate_threshold,
                staleness_s={"camera": 0.5, "joint_states": 0.2, "command": self.cfg.command_staleness_s},
            ),
            self,
        )
        # The simulated camera and the joint bus exist from the moment the scene is compiled. Without this
        # the first env.reset() would tick the guarded step with no heartbeat on record and the camera
        # watchdog would latch an e-stop before anything had a chance to run.
        self.heartbeat_sensors()
        self._install_safety_wrapper()
        self._load_grounder()
        if self.cfg.load_segmenter:
            self._load_segmenter(self.cfg.seg_backend)
        self.reset_scene(seed=5000, stratum="seen")
        self.emit(System(models=self.models, hardware_label=self._hardware, gpu=self._gpu_name(), note="models warm"))

    def _set_model(self, name: str, state: str, load_ms: float | None = None, detail: str | None = None) -> None:
        self.models[name] = {"state": state, "load_ms": load_ms, "detail": detail}
        self.emit(System(models=self.models, hardware_label=self._hardware, gpu=self._gpu_name(), note=f"{name}: {state}"))

    def _gpu_name(self) -> str:
        return self._hardware.split(" (")[0]

    def _load_grounder(self) -> None:
        self.heartbeat_sensors()
        if self.cfg.grounder == "oracle":
            from langgrasp.perception.grounding import OracleGrounder

            self.grounder = OracleGrounder(self.env)
            self._set_model("grounder", "warm", 0.0, detail="oracle grounder: simulator label map, no GPU model")
            return
        from langgrasp.perception.grounding import GroundingDINO

        self._set_model("grounder", "loading", detail="Grounding DINO tiny")
        t0 = time.perf_counter()
        self.grounder = GroundingDINO()
        self.grounder.warmup()
        self._set_model("grounder", "warm", (time.perf_counter() - t0) * 1e3, detail=f"Grounding DINO tiny on {self.grounder.device}")
        self.heartbeat_sensors()

    def _load_segmenter(self, backend: str):
        if backend in self.segmenters:
            return self.segmenters[backend]
        path = self.seg_path(backend)
        if path is None or not Path(path).exists():
            self._set_model(f"segmenter:{backend}", "missing", detail=f"{path} not found")
            return None
        from langgrasp.perception.segmentation import Segmenter

        self._set_model(f"segmenter:{backend}", "loading", detail=path)
        try:
            t0 = time.perf_counter()
            seg = Segmenter(path, backend="pt" if backend.startswith("pt") else backend, half=backend == "pt-fp16")
            seg.warmup()
            self.segmenters[backend] = seg
            self._set_model(f"segmenter:{backend}", "warm", (time.perf_counter() - t0) * 1e3, detail=path)
            self.heartbeat_sensors()
            return seg
        except Exception as e:  # noqa: BLE001 - a missing TensorRT engine must not kill the worker
            self._set_model(f"segmenter:{backend}", "error", detail=f"{type(e).__name__}: {e}")
            return None

    def seg_path(self, backend: str) -> str | None:
        base = self.cfg.seg_weights
        return {
            "pt": base,
            "pt-fp16": base,
            "onnx": base.replace(".pt", ".onnx"),
            "engine": base.replace(".pt", "-fp16.engine"),
            "engine-int8": base.replace(".pt", "-int8.engine"),
        }.get(backend)

    def available_backends(self) -> list[dict]:
        out = []
        for b in ("pt", "pt-fp16", "onnx", "engine", "engine-int8"):
            p = self.seg_path(b)
            ok = bool(p) and Path(p).exists()
            out.append({"backend": b, "path": p, "present": ok, "state": self.models.get(f"segmenter:{b}", {}).get("state", "absent")})
        return out

    # ------------------------------------------------------------------ safety in the loop
    def _install_safety_wrapper(self) -> None:
        """Wrap env.step with the safety monitor, the pacing check and the tick events."""
        from langgrasp.safety import SafetyState
        from langgrasp.sim.env import CONTROL_HZ

        env = self.env
        orig_step = env.step
        self._orig_step = orig_step
        dt = 1.0 / CONTROL_HZ

        def guarded(q_arm=None, jaw=None):
            now = time.monotonic()
            self._check_control()
            reasons: list[str] = []
            if self._command_active:
                # A command being carried out is not a stale command. This covers every controller, including
                # batch trials, which do not go through record_fn.
                self.monitor.heartbeat("command", now)
                # The camera is refreshed too, and the reason matters. This stack captures one frame per
                # command and then executes open loop, so between the capture and the end of the motion
                # nothing reads the camera by design. The 0.5 s timeout assumes an independent publisher, as
                # there would be on hardware; here it only measures the worker's own sampling gap, and left
                # alone it latches an e-stop in the middle of every command whose motion outlasts half a
                # second, reported as a camera fault that did not happen. What still holds the arm is the
                # tick-level check itself: if the worker stalls, these heartbeats stop with it.
                self.monitor.heartbeat("camera", now)
            if q_arm is not None:
                self.monitor.heartbeat("joint_states", now)
                reasons += self.monitor.check_tcp(env.obs()["tcp_pos"])
                before = (self.monitor.n_clipped_limit, self.monitor.n_clipped_vel)
                q_safe, r2 = self.monitor.check_joint_command(q_arm, env.q_arm, dt=dt)
                reasons += r2
                counter = self._clips if self.mode == "enforce" else self._would
                counter["limit"] += self.monitor.n_clipped_limit - before[0]
                counter["velocity"] += self.monitor.n_clipped_vel - before[1]
                if self.mode == "enforce" or self.monitor.state in (SafetyState.HOLD, SafetyState.ESTOP):
                    q_arm = q_safe
            self.monitor.check_staleness(now)
            t0 = time.perf_counter()
            out = orig_step(q_arm, jaw)
            self._sim_compute_ms += (time.perf_counter() - t0) * 1e3
            self._emit_tick(out, reasons)
            return out

        env.step = guarded

    def latch_estop_if_requested(self) -> bool:
        """Latch the monitor if the operator pressed e-stop, and report the measured latency once."""
        from langgrasp.safety import SafetyState

        if not self.flags.estop.value:
            return False
        if self.monitor.state is not SafetyState.ESTOP:
            self.monitor.estop("operator e-stop (GUI)")
            requested = self.flags.estop_t.value
            lat = (time.monotonic() - requested) * 1e3 if requested else None
            self._emit_safety(
                force=True,
                estop_latency_ms=lat,
                estop_latency_note="API receipt of the click to the first held tick, one machine, CLOCK_MONOTONIC; browser to API is measured separately",
            )
        return True

    def _check_control(self) -> None:
        """Honour e-stop, pause and single-step. Raises MotionAborted when the operator e-stops."""
        f = self.flags
        if self.latch_estop_if_requested():
            raise MotionAborted("operator e-stop")
        while f.paused.value and not f.step_once.value and not self._shutdown:
            if f.estop.value:
                return self._check_control()
            time.sleep(0.01)
        if f.step_once.value:
            f.step_once.value = 0

    def heartbeat_sensors(self) -> None:
        """Record that the camera and the joint bus were sampled now.

        In this in-process simulation there is no independent camera node or servo-bus thread: the worker
        samples both synchronously, so the staleness watchdogs measure the worker's own sampling gaps rather
        than a sensor fault. They still fire when the worker stalls, which is what they are for here; on
        hardware, where the publishers run on their own threads, they measure the real thing. Without this
        the 200 ms joint-state timeout would trip during every 275 ms grounding call, and loading Grounding
        DINO (6 s) would latch an e-stop before the first command, reported as a camera fault that never
        happened. Loading a model on the GPU is not a sensor gap, so the worker re-samples afterwards. What
        the watchdogs still catch, which is what they are for here, is the sim or the render stalling while a
        run is in flight, because then the per-tick samples stop.
        """
        now = time.monotonic()
        self.monitor.heartbeat("camera", now)
        self.monitor.heartbeat("joint_states", now)

    def _pace(self) -> None:
        """Wait until this tick is due, in slices.

        Sleeping the whole remainder in one call would mean an e-stop pressed just after a tick is not seen
        until the next one, which measured 60 ms at 1x. Slicing the sleep costs a handful of wake-ups per tick
        and brings it back to single-digit milliseconds. The physics is untouched either way: this only
        decides when the next identical step is taken.
        """
        speed = float(self.flags.speed.value)
        from langgrasp.sim.env import CONTROL_HZ

        if speed <= 0:
            self.paced = False
            self._next_tick_t = None
            return
        self.paced = True
        period = (1.0 / CONTROL_HZ) / speed
        now = time.monotonic()
        nxt = self._next_tick_t if self._next_tick_t is not None else now
        while now < nxt:
            if self.flags.estop.value:
                self._check_control()  # raises MotionAborted
            if self.flags.paused.value:
                self._check_control()  # blocks until the operator resumes or steps
                self._next_tick_t = None
                return
            time.sleep(min(PACE_SLICE_S, nxt - now))
            now = time.monotonic()
        self._next_tick_t = now + period

    def record_fn(self, env, phase: str) -> dict:
        """Passed to PickPlaceController(record=True, record_fn=...): called once per tick before env.step.

        The command heartbeat is refreshed here because a command that is being carried out right now is not a
        stale command. Without it the command watchdog can fire in the middle of a motion it asked for, which
        stops the arm half way through a pick for no reason an operator would recognise.
        """
        self._phase = phase
        self._check_control()
        self._pace()
        return {}

    # ------------------------------------------------------------------ events from the loop
    def _emit_tick(self, obs: dict, reasons: list[str]) -> None:
        if not self._executing:  # settling ticks of env.reset() are not part of a run
            return
        env = self.env
        ctrl = obs["ctrl"]
        want_bodies = bool(self.flags.cameras.value & BODIES_BIT)
        self.emit(
            Tick(
                run_id=self.run_id,
                tick=int(obs["t"]),
                phase=self._phase,
                q=[float(v) for v in obs["q_arm"]],
                q_target=[float(v) for v in ctrl[:5]],
                jaw=float(obs["jaw"]),
                tcp=[float(v) for v in obs["tcp_pos"]],
                sim_t=float(env.data.time),
                bodies=self._body_poses() if want_bodies else None,
            )
        )
        self._tick_frame_i += 1
        every = 1 if self.paced else 5
        # At speed the stream is thinned, but a recorded run still gets every frame, or the Inspector's
        # scrubber would have holes at exactly the ticks someone wants to look at.
        self.publish_frames(tick=int(obs["t"]), publish=self._tick_frame_i % every == 0)
        self._emit_safety(extra_reasons=reasons)

    def _body_poses(self) -> dict[str, list[float]]:
        env = self.env
        out = {}
        for i in range(env.model.nbody):
            name = env.model.body(i).name
            if not name or name == "world":
                continue
            p, q = env.data.xpos[i], env.data.xquat[i]
            out[name] = [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        return out

    def _emit_safety(self, force: bool = False, extra_reasons: list[str] | None = None, **kw) -> None:
        mon = self.monitor
        diag = mon.diagnostics()
        state = mon.state.value
        reason = diag["message"]
        if state == "RUN" and extra_reasons:
            reason = "; ".join(extra_reasons)
        key = (state, reason, self._clips["limit"], self._clips["velocity"], self._would["limit"], self._would["velocity"])
        if not force and key == self._last_safety_key:
            return
        self._last_safety_key = key
        now = time.monotonic()
        ages = {}
        for topic in mon.cfg.staleness_s:
            last = mon._last_seen.get(topic)
            ages[topic] = None if last is None else round(now - last, 3)
        self.emit(
            Safety(
                run_id=self.run_id,
                state=state,
                reason=self._plain_reason(state, reason),
                mode=self.mode,
                watchdog=ages,
                clips=dict(self._clips),
                would_clip=dict(self._would),
                **kw,
            )
        )

    def _plain_reason(self, state: str, reason: str) -> str:
        """Turn a monitor message into something an operator can act on, without changing what it says."""
        if state == "HOLD" and "command stale" in reason:
            timeout = self.monitor.cfg.staleness_s.get("command", 30.0)
            if self.monitor._last_seen.get("command") is None:
                return f"no command yet ({timeout:.0f} s command watchdog). Sending one clears this hold."
            return f"{reason}. Sending a command clears this hold."
        return reason

    # ------------------------------------------------------------------ frames
    def publish_frames(self, tick: int | None = None, publish: bool = True) -> None:
        """Render the subscribed cameras. With publish=False the front frame is still written to the run
        directory but no event is sent: that is what keeps a recorded run scrubbable tick by tick even when
        the stream is being thinned, which it is at speeds above real time.
        """
        bits = self.flags.cameras.value
        for cam in CAMERAS:
            if bits & CAM_BITS[cam] and (publish or cam == "front"):
                rgb = self.env.render(cam, self.cfg.image_size)
                self._emit_frame(cam, "rgb", rgb, tick, publish=publish)
        if not publish:
            self.heartbeat_sensors()
            return
        if bits & DEPTH_BIT:
            depth = self.env.render("front", self.cfg.image_size, depth=True).astype(np.float32)
            self._emit_frame("front", "depth", self.colorize_depth(depth), tick)
        self.heartbeat_sensors()

    def _emit_frame(self, camera: str, kind: str, img: np.ndarray, tick: int | None, publish: bool = True) -> None:
        jpeg = self.encode_jpeg(img)
        if publish:
            self.emit(FrameMeta(run_id=self.run_id, camera=camera, kind=kind, tick=tick, width=int(img.shape[1]), height=int(img.shape[0]), bytes=len(jpeg)), jpeg=jpeg)
        if self.run_dir is not None and tick is not None and camera == "front" and kind == "rgb":
            (self.run_dir / "frames" / f"tick_{tick:05d}_{camera}.jpg").write_bytes(jpeg)

    def encode_jpeg(self, img: np.ndarray) -> bytes:
        import cv2

        bgr = np.ascontiguousarray(img[..., ::-1]) if img.ndim == 3 else img
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)])
        return buf.tobytes() if ok else b""

    @staticmethod
    def colorize_depth(depth: np.ndarray) -> np.ndarray:
        """Metres to a turbo colormap, scaled to the 2nd and 98th percentile of the finite pixels."""
        import cv2

        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        lo, hi = np.percentile(depth[finite], [2, 98])
        span = max(float(hi - lo), 1e-6)
        norm = np.clip((depth - lo) / span, 0, 1)
        norm[~finite] = 0
        bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        return bgr[..., ::-1]

    def save_stage_images(self, stage: str, images: dict[str, Any]) -> dict[str, str]:
        """Write the images a stage produced next to the run and return their URL paths."""
        out: dict[str, str] = {}
        if self.run_dir is None:
            return out
        import cv2

        for role, arr in images.items():
            a = np.asarray(arr)
            if a.dtype == bool:
                img = np.repeat((a.astype(np.uint8) * 255)[..., None], 3, axis=2)
            elif a.dtype == np.float32 or a.dtype == np.float64:
                img = self.colorize_depth(a.astype(np.float32))
            else:
                img = a
            name = f"{stage}_{role}.jpg"
            path = self.run_dir / "frames" / name
            cv2.imwrite(str(path), np.ascontiguousarray(img[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            out[role] = f"/runs/{self.run_dir.name}/frames/{name}"
        return out

    # ------------------------------------------------------------------ scene and runs
    def reset_scene(self, seed: int, stratum: str = "seen", lighting: str | None = None, n_distractors: int | None = None, fixed_goal: bool = False) -> dict:
        from langgrasp.sim.scenarios import make_scenario

        kw = {"target_kind": "cube", "target_color": "red"} if fixed_goal else {}
        sc = make_scenario(seed, stratum, n_distractors=n_distractors, lighting=lighting, **kw)
        self.scenario = sc
        self.heartbeat_sensors()  # about to sample the camera; whatever blocked before this was not a fault
        self.env.reset(sc)
        self.publish_frames()
        d = sc.to_dict()
        d["fixed_goal"] = fixed_goal
        self.emit(System(models=self.models, hardware_label=self._hardware, gpu=self._gpu_name(), scene=jsonable(d), note="scene reset"))
        return d

    def _open_run(self, run_id: str, meta: dict) -> None:
        self.run_id = run_id
        self._sim_compute_ms = 0.0
        self._clips = {"limit": 0, "velocity": 0}
        self._would = {"limit": 0, "velocity": 0}
        self._next_tick_t = None
        self._tick_frame_i = 0
        self._executing = False
        if not self.cfg.record:
            self.run_dir = None
            return
        self.run_dir = RUNS_DIR / run_id
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "meta.json").write_text(json.dumps(jsonable(meta), indent=1))
        self._events_fp = open(self.run_dir / "events.jsonl", "w")

    def _close_run(self) -> None:
        if self._events_fp is not None:
            self._events_fp.close()
            self._events_fp = None
        self.run_id = None
        self.run_dir = None
        self._prune_runs()

    def _prune_runs(self) -> None:
        try:
            dirs = sorted((d for d in RUNS_DIR.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
            for d in dirs[:-MAX_KEPT_RUNS]:
                shutil.rmtree(d, ignore_errors=True)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ commands
    def dispatch(self, msg: dict) -> None:
        cmd = msg.get("cmd")
        fn = getattr(self, f"cmd_{cmd}", None)
        if fn is None:
            self.log(f"unknown command {cmd!r}", level="warn")
            return
        try:
            fn(msg)
        except MotionAborted as e:
            self.log(f"run aborted: {e}", level="warn")
        except Exception as e:  # noqa: BLE001 - the worker must survive any single command
            self.log(f"{type(e).__name__}: {e}", level="error", traceback=traceback.format_exc()[-2000:])
            if msg.get("reply_to"):
                self.emit(Reply(reply_to=msg["reply_to"], ok=False, error=f"{type(e).__name__}: {e}"))
        finally:
            self.flags.busy.value = 0

    def cmd_shutdown(self, msg: dict) -> None:
        self._shutdown = True

    def cmd_scene(self, msg: dict) -> None:
        self.reset_scene(
            seed=int(msg.get("seed", 5000)),
            stratum=msg.get("stratum", "seen"),
            lighting=msg.get("lighting"),
            n_distractors=msg.get("n_distractors"),
            fixed_goal=bool(msg.get("fixed_goal", False)),
        )

    def cmd_reset_estop(self, msg: dict) -> None:
        self.flags.clear_estop()
        self.monitor.reset_estop()
        self.heartbeat_sensors()
        self._emit_safety(force=True)

    def cmd_confirm(self, msg: dict) -> None:
        self.log("no command is waiting for confirmation", level="warn")

    def cmd_subscribe(self, msg: dict) -> None:
        bits = 0
        for cam in msg.get("cameras", ["front"]):
            bits |= CAM_BITS.get(cam, 0)
        if msg.get("depth"):
            bits |= DEPTH_BIT
        if msg.get("bodies"):
            bits |= BODIES_BIT
        self.flags.cameras.value = bits or CAM_BITS["front"]

    # ------------------------------------------------------------------ batch jobs
    def cmd_job(self, msg: dict) -> None:
        """Run the stratified protocol, or a subset of it, through the same harness the scripts use.

        The harness is not reimplemented here: ``langgrasp.eval.harness.run_protocol`` builds the scenarios,
        summarises the trials and writes the JSON, exactly as ``scripts/eval_modular.py`` does, so a file this
        produces is the same shape as the files in results/. What this adds is a progress event per trial and
        a cancel flag.

        Batch trials are not paced and record no frames: pacing exists so a person can watch one command, and
        applying it to 120 trials would turn 90 seconds into 12 minutes. The physics is identical either way.
        """
        from langgrasp.eval.harness import run_protocol
        from langgrasp.policies.modular import ModularPipeline
        from langgrasp.sim.controller import PickPlaceController
        from langgrasp.sim.scenarios import make_scenario
        from langgrasp.sim.scene import OBJECT_KINDS

        job_id = msg["job_id"]
        strata = {k: int(v) for k, v in (msg.get("strata") or {}).items() if int(v) > 0}
        if not strata:
            self.emit(JobProgress(job_id=job_id, state="error", message="No strata selected, so there is nothing to run."))
            return
        out_path = Path(msg["out_path"])
        if out_path.exists() and not msg.get("overwrite"):
            self.emit(JobProgress(job_id=job_id, state="error", out_path=str(out_path), message=f"{out_path.name} already exists and overwrite was not confirmed."))
            return
        controller = msg.get("controller", "pipeline")
        cfgd = dict(msg.get("config") or {})
        fixed_goal = bool(msg.get("fixed_goal"))
        total = sum(strata.values())
        self.flags.busy.value = 1
        self.flags.cancel_job.value = 0
        self.mode = cfgd.get("safety_mode", self.cfg.safety_mode)
        self.run_id = job_id
        self.run_dir = None  # no per-trial frame files: a 120-trial job would write about 250 MB
        done = 0

        seg = self._load_segmenter(cfgd.get("seg_backend", self.cfg.seg_backend)) if cfgd.get("use_yolo_mask", True) else None
        pipe = ModularPipeline(self.env, self.grounder, seg, self._pipeline_config(cfgd), safety=self.monitor, seed=0, hooks=self.hooks)

        def make(seed: int, stratum: str):
            return make_scenario(seed, "seen", target_kind="cube", target_color="red") if fixed_goal else make_scenario(seed, stratum)

        def run_one(sc) -> dict:
            nonlocal done
            if self.flags.cancel_job.value:
                raise JobCancelled("cancelled by the operator")
            if self.latch_estop_if_requested():
                raise JobCancelled("e-stop latched")
            # Each trial is a command. Without this the command watchdog, which will have put the monitor in
            # HOLD while the operator was reading the results page, holds the arm for every trial of the job
            # and every trial fails while the perception stages still look perfectly healthy.
            self._command_active = True
            self.monitor.heartbeat("command", time.monotonic())
            self.monitor.check_staleness(time.monotonic())
            self.heartbeat_sensors()
            self.env.reset(sc)
            self.publish_frames()
            if controller == "oracle":
                p, psi = self.env.grasp_point(sc.target)
                r = PickPlaceController(self.env).run(np.asarray(p), float(psi), sc.target, width=float(OBJECT_KINDS[sc.target_obj.kind]["width"]))
                out = {"grounding_correct": True, "grasped": r.grasped, "lifted": r.lifted, "placed": r.placed, "aborted": None, "latency_ms": {}, "extra": {"steps": r.steps, "rot_err": r.rot_err}}
            else:
                res = pipe.run_command(sc.command, sc.target)
                d = res.to_dict()
                out = {
                    "grounding_correct": res.grounding_correct if res.grounding_correct is not None else (False if res.aborted == "no_candidate" else None),
                    "grasped": res.grasped,
                    "lifted": res.lifted,
                    "placed": res.placed,
                    "aborted": res.aborted,
                    "latency_ms": res.latency_ms,
                    "extra": {k: d[k] for k in ("box", "score", "grounding_info", "select_info", "mask_source", "grasp") if k in d},
                }
            done += 1
            self.emit(
                JobProgress(
                    job_id=job_id,
                    state="running",
                    done=done,
                    total=total,
                    out_path=rel_out,
                    last=jsonable({"seed": sc.seed, "stratum": sc.stratum, "command": sc.command, "kind": sc.target_obj.kind, "color": sc.target_obj.color, "lighting": sc.lighting, **{k: out[k] for k in ("grounding_correct", "lifted", "placed", "aborted")}}),
                )
            )
            return out

        notes = f"run from the GUI. controller={controller} color_check={cfgd.get('use_color_check', True)} yolo_mask={seg is not None} depth_noise={cfgd.get('depth_noise', True)} seg={cfgd.get('seg_backend')} safety_mode={self.mode} gate={self.monitor.cfg.grounding_conf_threshold}"
        rel_out = str(out_path.relative_to(ROOT)) if out_path.is_absolute() and str(out_path).startswith(str(ROOT)) else str(out_path)
        self.monitor.heartbeat("command", time.monotonic())
        self.monitor.check_staleness(time.monotonic())
        self._emit_safety(force=True)
        self.emit(JobProgress(job_id=job_id, state="running", done=0, total=total, out_path=rel_out, message=f"{total} trials"))
        t0 = time.monotonic()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result = run_protocol(run_one, strata, msg.get("tag", controller), out_path, notes=notes, base_seed=int(msg.get("base_seed", 5000)), log_every=0, make=make)
        except JobCancelled as e:
            self.emit(JobProgress(job_id=job_id, state="cancelled", done=done, total=total, message=f"Cancelled after {done} of {total} trials: {e}. No file was written."))
            return
        finally:
            self._command_active = False
            self.flags.busy.value = 0
            self.run_id = None
            self._emit_safety(force=True)
        summary = result["summary"]["strata"].get("all", {})
        self.emit(
            JobProgress(
                job_id=job_id,
                state="done",
                done=done,
                total=total,
                out_path=rel_out,
                message=f"{done} trials in {(time.monotonic() - t0) / 60:.1f} min, written to {rel_out}",
                last=jsonable({"placed": summary.get("place"), "grounding": summary.get("grounding")}),
            )
        )

    def cmd_cancel_job(self, msg: dict) -> None:
        self.flags.cancel_job.value = 1

    def cmd_stt(self, msg: dict) -> None:
        from langgrasp.language.stt import SpeechToText

        size = msg.get("model_size") or self.cfg.stt_model
        if self.stt is None or self.stt.model_size != size:
            self._set_model(f"stt:{size}", "loading")
            self.stt = SpeechToText(model_size=size)
            t0 = time.perf_counter()
            self.stt.load()
            self._set_model(f"stt:{size}", "warm", (time.perf_counter() - t0) * 1e3, detail=f"faster-whisper {size} on {self.stt.device}")
            self.heartbeat_sensors()
        from langgrasp.language.stt import normalize_command

        r = self.stt.transcribe(msg["path"])
        self.emit(
            Reply(
                reply_to=msg["reply_to"],
                data=jsonable(
                    {
                        "text": r["text"],
                        "normalized": normalize_command(r["text"]),
                        "latency_ms": r["latency_ms"],
                        "language": r["language"],
                        "model_size": size,
                        "device": self.stt.device,
                        "note": "speech to text only: the transcript is never executed without a person pressing Run",
                    }
                ),
            )
        )

    def cmd_run(self, msg: dict) -> None:
        from langgrasp.safety import SafetyState

        if self.monitor.state is SafetyState.ESTOP:
            self.log("e-stop is latched: press Reset before running a command", level="warn")
            return
        cfgd = dict(msg.get("config") or {})
        self.mode = cfgd.get("safety_mode", self.cfg.safety_mode)
        self.flags.speed.value = float(cfgd.get("speed", 1.0))
        controller = msg.get("controller", "pipeline")
        command = msg.get("command") or (self.scenario.command if self.scenario else "")
        seed = int(self.scenario.seed) if self.scenario else 0
        run_id = msg.get("run_id") or f"{time.strftime('%Y-%m-%dT%H-%M-%S')}_{seed}"
        self.flags.busy.value = 1
        self._open_run(
            run_id,
            {
                "run_id": run_id,
                "seed": seed,
                "command": command,
                "controller": controller,
                "config": cfgd,
                "safety_mode": self.mode,
                "scenario": self.scenario.to_dict() if self.scenario else None,
                "hardware": self._hardware,
                "note": "simulation, NVIDIA L4 class GPU shared with other processes on this host; not a hardware measurement",
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        self._command_active = True
        self.monitor.heartbeat("command", time.monotonic())
        self.monitor.check_staleness(time.monotonic())
        self._emit_safety(force=True)
        self.hooks.stage_started("command", text=command, source=msg.get("source", "typed"))
        self.hooks.stage_finished("command", text=command, source=msg.get("source", "typed"), stt_latency_ms=msg.get("stt_latency_ms"), controller=controller)
        t0 = time.monotonic()
        try:
            if controller == "pipeline":
                self._run_pipeline(command, cfgd)
            elif controller == "oracle":
                self._run_oracle(cfgd)
            else:
                self.log(f"controller '{controller}' is not enabled in this build of the worker", level="warn")
                self.emit(Outcome(run_id=self.run_id, aborted=f"controller_unavailable:{controller}", scored_by="none", message=f"The {controller} controller is not wired into the worker yet."))
        except MotionAborted as e:
            self.emit(Outcome(run_id=self.run_id, aborted="safety:estop", message=f"Run stopped: {e}.", wall_ms=(time.monotonic() - t0) * 1e3, sim_compute_ms=self._sim_compute_ms))
            self._emit_safety(force=True)
        finally:
            self._command_active = False
            self.flags.busy.value = 0
            self._close_run()

    def _pipeline_config(self, cfgd: dict):
        from langgrasp.policies.modular import PipelineConfig

        return PipelineConfig(
            use_color_check=bool(cfgd.get("use_color_check", True)),
            use_yolo_mask=bool(cfgd.get("use_yolo_mask", True)),
            depth_noise=bool(cfgd.get("depth_noise", True)),
            camera=cfgd.get("camera", "front"),
            image_size=tuple(cfgd.get("image_size", self.cfg.image_size)),
        )

    def _make_controller(self, env):
        from langgrasp.sim.controller import PickPlaceController

        self._executing = True
        self._next_tick_t = None
        self.heartbeat_sensors()
        return PickPlaceController(env, record=True, record_fn=self.record_fn)

    def _run_pipeline(self, command: str, cfgd: dict) -> None:
        from langgrasp.policies.modular import ModularPipeline

        backend = cfgd.get("seg_backend", self.cfg.seg_backend)
        seg = self._load_segmenter(backend) if cfgd.get("use_yolo_mask", True) else None
        if self.monitor.cfg.require_human_confirm != bool(cfgd.get("require_human_confirm", False)):
            self.monitor.cfg.require_human_confirm = bool(cfgd.get("require_human_confirm", False))
        self.heartbeat_sensors()
        self.env.reset(self.scenario)
        t0 = time.monotonic()
        pipe = ModularPipeline(
            self.env,
            self.grounder,
            seg,
            self._pipeline_config(cfgd),
            safety=self.monitor,
            seed=int(self.scenario.seed),
            hooks=self.hooks,
            controller_factory=self._make_controller,
        )
        try:
            res = pipe.run_command(command, self.scenario.target)
        finally:
            self._executing = False
        self._emit_outcome(res.grounding_correct, res.grasped, res.lifted, res.placed, res.aborted, res.latency_ms, time.monotonic() - t0, custom_command=self._is_custom(command))

    def _run_oracle(self, cfgd: dict) -> None:
        """The scripted upper bound: the grasp pose comes from the simulator, so stages 3 to 8 are skipped."""
        from langgrasp.eval.latency import LatencyTracer
        from langgrasp.sim.scene import OBJECT_KINDS

        self.heartbeat_sensors()
        self.env.reset(self.scenario)
        for stage in ("capture", "grounding", "select", "gate", "segment", "fuse"):
            self.hooks.stage_finished(stage, status="skipped", message="Oracle controller: the grasp pose comes from the simulator, no perception runs.")
        tracer = LatencyTracer()
        target = self.scenario.target
        kind = self.scenario.target_obj.kind
        p, psi = self.env.grasp_point(target)
        self.hooks.stage_started("execute", center=[float(v) for v in p], psi=float(psi), width=float(OBJECT_KINDS[kind]["width"]), target=target, ground_truth=True)
        t0 = time.monotonic()
        try:
            with tracer.stage("execute"):
                ctl = self._make_controller(self.env)
                r = ctl.run(np.asarray(p), float(psi), target, width=float(OBJECT_KINDS[kind]["width"]))
        finally:
            self._executing = False
        self.hooks.stage_finished(
            "execute",
            status="ok" if r.placed else ("warn" if r.lifted else "fail"),
            latency_ms=tracer.samples["execute"][-1],
            steps=int(r.steps),
            ik_ok=bool(r.ik_ok),
            max_pos_err_m=float(r.max_pos_err),
            rot_err_rad=float(r.rot_err),
            psi_used=float(ctl.psi),
            grasped=bool(r.grasped),
            lifted=bool(r.lifted),
            placed=bool(r.placed),
            ground_truth=True,
        )
        self._emit_outcome(True, r.grasped, r.lifted, r.placed, None, {"execute": tracer.samples["execute"][-1]}, time.monotonic() - t0)

    def _is_custom(self, command: str) -> bool:
        """True when the operator typed something other than the scene's own command.

        It matters for honesty, not for control: the simulator's ground truth only knows which object this
        scene designated as the target. If the command asks for a different object, then grounding cannot be
        scored (picking the object the operator asked for would count as a grounding error), so the GUI
        reports the scoring as unavailable instead of reporting a failure that is not one.
        """
        scene_command = (self.scenario.command if self.scenario else "") or ""
        return command.strip().lower() != scene_command.strip().lower()

    def _emit_outcome(self, grounding, grasped, lifted, placed, aborted, latency_ms: dict, wall_s: float, custom_command: bool = False) -> None:
        msg = "Placed in the tray." if placed else ("Lifted but not placed." if lifted else (f"Aborted: {aborted}." if aborted else "Not lifted."))
        if custom_command and self.scenario is not None:
            grounding = None
            msg += (
                f" Not scored: this scene's own command is '{self.scenario.command}' and its ground-truth target is"
                f" {self.scenario.target}, so a different command cannot be graded. The grasp and place flags refer"
                f" to {self.scenario.target}."
            )
        self.emit(
            Outcome(
                run_id=self.run_id,
                scored_by="none" if custom_command else "ground_truth",
                grounding_correct=grounding,
                grasped=bool(grasped),
                lifted=bool(lifted),
                placed=bool(placed),
                aborted=aborted,
                message=msg,
                latency_ms=jsonable({k: float(v) for k, v in (latency_ms or {}).items()}),
                sim_compute_ms=self._sim_compute_ms,
                wall_ms=wall_s * 1e3,
            )
        )
        self._emit_safety(force=True)

    def wait_for_confirmation(self, timeout: float = 60.0) -> bool:
        """Block the run until /api/confirm arrives. Only confirm and shutdown are honoured while waiting."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._check_control()
            try:
                msg = self.cmd_q.get(timeout=0.05)
            except queue.Empty:
                continue
            cmd = msg.get("cmd")
            if cmd == "confirm":
                return msg.get("decision") == "confirm"
            if cmd == "shutdown":
                self._shutdown = True
                return False
            self.log(f"ignored '{cmd}' while waiting for the operator to confirm the grounding", level="warn")
        self.log("confirmation timed out after 60 s", level="warn")
        return False

    # ------------------------------------------------------------------ main loop
    def idle(self) -> None:
        now = time.monotonic()
        self.latch_estop_if_requested()
        self.monitor.heartbeat("joint_states", now)
        if now - self._last_idle_frame >= self.cfg.idle_frame_period_s:
            self._last_idle_frame = now
            self.publish_frames()
        self.monitor.check_staleness(time.monotonic())
        self._emit_safety()

    def serve(self) -> None:
        self.build()
        # Poll fast even when nothing is running: the e-stop flag is only read on this loop while the arm is
        # idle, so the queue timeout is the idle e-stop latency. Frames are still published on their own
        # slower timer inside idle().
        while not self._shutdown:
            try:
                msg = self.cmd_q.get(timeout=IDLE_POLL_S)
            except queue.Empty:
                self.idle()
                continue
            self.dispatch(msg)
        self.log("worker stopped")


def make_monitor(cfg, worker: SimWorker):
    """SafetyMonitor plus the operator in the loop.

    Everything is the parent's behaviour except ``gate_grounding``, which announces the decision it is about
    to take (so the GUI can show the score against the threshold and the top-2 margin) and, when human
    confirmation is switched on, waits for the operator before delegating. The refusal rules themselves are
    untouched: an ambiguous top-2 is still a refusal, as FMEA H6 says.
    """
    from langgrasp.safety import SafetyMonitor

    class GuiSafetyMonitor(SafetyMonitor):
        def gate_grounding(self, score, n_candidates: int):
            scores = np.atleast_1d(np.asarray(score, dtype=float))
            top2 = [float(v) for v in np.sort(scores)[::-1][:2]]
            needs = bool(self.cfg.require_human_confirm)
            worker.emit(
                GateRequest(
                    run_id=worker.run_id,
                    score=top2[0] if top2 else None,
                    top2=top2,
                    threshold=float(self.cfg.grounding_conf_threshold),
                    ambiguous=len(top2) > 1 and (top2[0] - top2[1]) < self.cfg.ambiguity_margin,
                    needs_confirmation=needs,
                    timeout_s=60.0 if needs else None,
                )
            )
            if needs and not worker.wait_for_confirmation(60.0):
                return False, "rejected by the operator"
            if needs:
                self.confirm_human()
            ok, reason = super().gate_grounding(score, n_candidates)
            self.clear_confirmation()
            return ok, reason

    return GuiSafetyMonitor(cfg)


# ---------------------------------------------------------------------------- parent side


def _worker_main(cmd_q, evt_q, flags, cfg_dict: dict) -> None:  # pragma: no cover - runs in the child
    os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = WorkerConfig(**cfg_dict)
    w = SimWorker(cmd_q, evt_q, flags, cfg)
    try:
        w.serve()
    except Exception:  # noqa: BLE001
        try:
            evt_q.put(Log(level="error", message="worker crashed", detail={"traceback": traceback.format_exc()[-3000:]}).model_dump())
        finally:
            raise


@dataclass
class WorkerHandle:
    """Parent-side view of the worker: send commands, drain events, flip flags."""

    cfg: WorkerConfig = field(default_factory=WorkerConfig)

    def __post_init__(self):
        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.cmd_q = self.ctx.Queue()
        self.evt_q = self.ctx.Queue()
        self.flags = ControlFlags(self.ctx)
        self.proc = None

    def start(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.proc = self.ctx.Process(target=_worker_main, args=(self.cmd_q, self.evt_q, self.flags, self.cfg.to_dict()), daemon=True, name="langgrasp-sim")
        self.proc.start()

    @property
    def alive(self) -> bool:
        return bool(self.proc and self.proc.is_alive())

    @property
    def busy(self) -> bool:
        return bool(self.flags.busy.value)

    def send(self, **msg) -> None:
        self.cmd_q.put(msg)

    def next_event(self, timeout: float = 0.2) -> dict | None:
        try:
            return self.evt_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, timeout: float = 0.0) -> list[dict]:
        out = []
        deadline = time.monotonic() + timeout
        while True:
            try:
                out.append(self.evt_q.get_nowait())
            except queue.Empty:
                if time.monotonic() >= deadline:
                    return out
                time.sleep(0.005)

    def estop(self) -> float:
        return self.flags.request_estop()

    def reset_estop(self) -> None:
        self.send(cmd="reset_estop")

    def pause(self, on: bool) -> None:
        self.flags.paused.value = 1 if on else 0

    def step_once(self) -> None:
        self.flags.step_once.value = 1

    def set_speed(self, speed: float) -> None:
        self.flags.speed.value = float(speed)

    def stop(self, timeout: float = 5.0) -> None:
        if not self.alive:
            return
        self.flags.paused.value = 0
        self.send(cmd="shutdown")
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(2)
