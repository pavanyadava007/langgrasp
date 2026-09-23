"""The command watchdog must hold the arm when no command arrives, and let go when one does.

This lives in its own module because it spawns its own worker: the simulator is a single resource, and these
tests run one worker at a time.
"""

import os
import time

from langgrasp.gui.worker import WorkerConfig, WorkerHandle


def drain_until(handle: WorkerHandle, predicate, timeout: float, label: str, seen: list | None = None):
    seen = seen if seen is not None else []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        e = handle.next_event(0.3)
        if e is None:
            continue
        seen.append(e)
        if predicate(e):
            return e
    errors = [x for x in seen if x["type"] == "log" and x["level"] == "error"]
    raise AssertionError(f"timed out after {timeout}s waiting for {label}; errors: {errors[-1] if errors else 'none'}")


def test_a_run_and_a_job_still_move_the_arm_after_the_command_watchdog_has_fired():
    """Issuing a command must clear the hold.

    This regression exists because a batch job did not clear it, so every trial of a job started after an idle
    spell failed while the perception stages still looked perfectly healthy.
    """
    out_file = "/tmp/langgrasp_gui_job_test.json"
    h = WorkerHandle(WorkerConfig(grounder="oracle", load_segmenter=False, record=False, command_staleness_s=0.2))
    h.start()
    events: list = []
    try:
        drain_until(h, lambda e: e["type"] == "system" and e["note"] == "models warm", 180, "the worker to warm up", events)
        hold = drain_until(h, lambda e: e["type"] == "safety" and e["state"] == "HOLD", 20, "the command watchdog to hold", events)
        assert "command" in hold["reason"], hold

        events.clear()
        h.send(cmd="run", controller="oracle", config={"speed": 0})
        out = drain_until(h, lambda e: e["type"] == "outcome", 120, "the outcome of a run after a hold", events)
        assert out["placed"], f"the arm was still held during a run: {out}"

        events.clear()
        h.send(cmd="job", job_id="jtest", controller="oracle", strata={"seen": 2}, config={}, out_path=out_file, overwrite=True)
        done = drain_until(h, lambda e: e["type"] == "job_progress" and e["state"] in ("done", "error", "cancelled"), 240, "the job to finish", events)
        assert done["state"] == "done", done
        trials = [e["last"] for e in events if e["type"] == "job_progress" and (e.get("last") or {}).get("seed") is not None]
        assert len(trials) == 2 and all(t["placed"] for t in trials), f"trials were held by the watchdog: {trials}"
    finally:
        h.stop()
        if os.path.exists(out_file):
            os.unlink(out_file)


def test_a_batch_trial_is_not_stopped_by_the_camera_watchdog_mid_motion():
    """The pipeline captures one frame per command and then executes open loop.

    A 0.5 s camera timer assumes an independent publisher; here nothing reads the camera during the motion, so
    before this was handled the timer latched an e-stop partway through the first trial of every batch job and
    every trial after it failed without moving.
    """
    out_file = "/tmp/langgrasp_gui_camera_watchdog.json"
    h = WorkerHandle(WorkerConfig(grounder="oracle", load_segmenter=False, record=False))
    h.start()
    events: list = []
    try:
        drain_until(h, lambda e: e["type"] == "system" and e["note"] == "models warm", 180, "the worker to warm up", events)
        events.clear()
        h.send(cmd="job", job_id="jcam", controller="pipeline", strata={"seen": 3}, config={"use_yolo_mask": False}, out_path=out_file, overwrite=True)
        done = drain_until(h, lambda e: e["type"] == "job_progress" and e["state"] in ("done", "error", "cancelled"), 240, "the job to finish", events)
        assert done["state"] == "done", done
        trials = [e["last"] for e in events if e["type"] == "job_progress" and (e.get("last") or {}).get("seed") is not None]
        assert len(trials) == 3
        assert all(t["placed"] for t in trials), f"the watchdog stopped trials mid motion: {trials}"
        states = {e["state"] for e in events if e["type"] == "safety"}
        assert "ESTOP" not in states, f"an e-stop latched during a batch job: {[e for e in events if e['type'] == 'safety']}"
    finally:
        h.stop()
        if os.path.exists(out_file):
            os.unlink(out_file)
