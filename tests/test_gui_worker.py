"""The simulation worker: one process owning the sim, events out, flags in, e-stop that actually stops.

These tests spawn the real worker process and drive it the way the API will. They use the oracle grounder so
no GPU model is loaded; the model path is covered by tests/test_gui_hooks_identical.py.
"""

import pytest

from langgrasp.gui.trace import STAGES
from langgrasp.gui.worker import WorkerConfig, WorkerHandle


class Collector:
    """Drains the worker's event queue and keeps what the assertions need."""

    def __init__(self, handle: WorkerHandle):
        self.h = handle
        self.events: list[dict] = []

    def until(self, predicate, timeout: float = 90.0, label: str = "event"):
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            e = self.h.next_event(0.3)
            if e is None:
                continue
            self.events.append(e)
            if predicate(e):
                return e
        errors = [e for e in self.events if e["type"] == "log" and e["level"] == "error"]
        raise AssertionError(f"timed out after {timeout}s waiting for {label}; errors seen: {errors[-1] if errors else 'none'}")

    def of(self, type_: str) -> list[dict]:
        return [e for e in self.events if e["type"] == type_]

    def stages(self) -> list[tuple[str, str]]:
        return [(e["stage"], e["status"]) for e in self.events if e["type"] == "stage_finished"]


@pytest.fixture(scope="module")
def worker():
    h = WorkerHandle(WorkerConfig(grounder="oracle", load_segmenter=False, record=False))
    h.start()
    c = Collector(h)
    c.until(lambda e: e["type"] == "system" and e["note"] == "models warm", timeout=120, label="worker warm-up")
    yield h, c
    h.stop()


def test_worker_reports_what_it_loaded(worker):
    h, c = worker
    assert h.alive
    sysev = c.of("system")[-1]
    assert sysev["models"]["simulation"]["state"] == "warm"
    assert sysev["models"]["grounder"]["state"] == "warm"
    assert sysev["hardware_label"] and "simulation only" in sysev["hardware_label"]
    # the scene the worker starts on is announced, not assumed by the client
    scene = next(e["scene"] for e in reversed(c.of("system")) if e["scene"])
    assert scene["seed"] == 5000 and scene["target"] and scene["command"]


def test_idle_worker_streams_frames_without_a_run(worker):
    h, c = worker
    c.events.clear()
    f = c.until(lambda e: e["type"] == "frame", timeout=10, label="an idle frame")
    assert f["camera"] == "front" and f["kind"] == "rgb" and f["bytes"] > 1000
    assert f["width"] == 640 and f["height"] == 480 and f["run_id"] is None


def test_pipeline_run_emits_all_nine_stages_and_an_outcome(worker):
    h, c = worker
    c.events.clear()
    h.send(cmd="scene", seed=5000, stratum="seen")
    c.until(lambda e: e["type"] == "system" and e["note"] == "scene reset", timeout=30, label="scene reset")
    h.send(cmd="run", command="pick the blue cube", controller="pipeline", config={"use_yolo_mask": False, "speed": 0})
    out = c.until(lambda e: e["type"] == "outcome", timeout=90, label="the run outcome")
    stages = c.stages()
    assert [s for s, _ in stages] == list(STAGES), stages
    assert all(st in ("ok", "warn") for _, st in stages), stages
    assert out["placed"] and out["grounding_correct"] and out["aborted"] is None
    assert out["scored_by"] == "ground_truth"
    assert out["sim_compute_ms"] > 0 and out["wall_ms"] >= out["sim_compute_ms"]
    ticks = c.of("tick")
    assert len(ticks) > 30 and ticks[0]["tick"] < ticks[-1]["tick"]
    assert ticks[-1]["phase"] in ("settle", "retreat", "release")
    assert len(ticks[0]["q"]) == 5 and len(ticks[0]["tcp"]) == 3
    # this worker runs the oracle grounder, which hands over the simulator's own mask: the GUI has to flag
    # that as ground truth rather than pass it off as a YOLO segmentation
    seg = next(e for e in c.of("stage_finished") if e["stage"] == "segment")
    assert seg["status"] == "warn" and "ground-truth" in seg["message"].lower()
    assert seg["payload"]["mask_source"] == "oracle" and seg["payload"]["mask_px"] > 0


def test_estop_stops_the_arm_and_reports_its_own_latency(worker):
    h, c = worker
    c.events.clear()
    h.send(cmd="run", controller="oracle", config={"speed": 1.0})
    ticks = 0
    while ticks < 5:
        e = h.next_event(0.4)
        if e is None:
            continue
        c.events.append(e)
        if e["type"] == "tick":
            ticks += 1
    h.estop()
    ev = c.until(lambda e: e["type"] == "safety" and e["state"] == "ESTOP", timeout=30, label="the ESTOP state")
    assert ev["estop_latency_ms"] is not None and 0 < ev["estop_latency_ms"] < 100, ev["estop_latency_ms"]
    assert "MONOTONIC" in ev["estop_latency_note"].upper()
    out = c.until(lambda e: e["type"] == "outcome", timeout=30, label="the aborted outcome")
    assert out["aborted"] == "safety:estop" and not out["placed"]
    # the executor was interrupted: an oracle pick takes about 60 ticks, we stopped in the first few
    assert len(c.of("tick")) < 40
    # and a new command is refused while the latch is closed
    c.events.clear()
    h.send(cmd="run", controller="oracle", config={"speed": 0})
    warn = c.until(lambda e: e["type"] == "log" and e["level"] == "warn", timeout=20, label="the refusal")
    assert "e-stop is latched" in warn["message"]


def test_reset_releases_the_latch_and_the_next_run_succeeds(worker):
    h, c = worker
    c.events.clear()
    h.reset_estop()
    ev = c.until(lambda e: e["type"] == "safety" and e["state"] != "ESTOP", timeout=20, label="the state after reset")
    assert ev["state"] in ("RUN", "REDUCED_SPEED", "HOLD")
    h.send(cmd="run", controller="oracle", config={"speed": 0})
    out = c.until(lambda e: e["type"] == "outcome", timeout=90, label="the outcome after reset")
    assert out["placed"] and out["aborted"] is None
    stages = dict(c.stages())
    # the oracle controller uses the simulator's own grasp pose, so perception is skipped, not faked
    assert stages["grounding"] == "skipped" and stages["fuse"] == "skipped" and stages["execute"] == "ok"


def test_monitor_mode_reports_what_it_would_have_clipped(worker):
    """Default mode observes: the arm gets the controller's target, and the monitor says what it would clip."""
    h, c = worker
    c.events.clear()
    h.send(cmd="run", controller="oracle", config={"speed": 0, "safety_mode": "monitor"})
    c.until(lambda e: e["type"] == "outcome", timeout=90, label="the outcome")
    safety = c.of("safety")[-1]
    assert safety["mode"] == "monitor"
    assert safety["clips"] == {"limit": 0, "velocity": 0}, "monitor mode must not apply clips"
    assert safety["would_clip"]["velocity"] > 0, "the default velocity limit does bite on this trajectory"
    assert set(safety["watchdog"]) == {"camera", "joint_states", "command"}


def test_unknown_command_is_reported_not_swallowed(worker):
    h, c = worker
    c.events.clear()
    h.send(cmd="fly_to_the_moon")
    warn = c.until(lambda e: e["type"] == "log" and e["level"] == "warn", timeout=20, label="the unknown-command warning")
    assert "unknown command" in warn["message"]
    assert h.alive
