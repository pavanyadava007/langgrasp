"""The API in front of the real worker: one command from HTTP request to recorded run on disk.

This is the wiring test. It spawns the actual simulation worker (oracle grounder, so no GPU model loads),
sends a command over HTTP, watches the WebSocket, and then reads the run back out of runs/gui through the
API the way the Pipeline Inspector will.
"""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from langgrasp.gui.api import create_app
from langgrasp.gui.trace import STAGES, unpack_frame
from langgrasp.gui.worker import WorkerConfig


def wait_idle(client, timeout: float = 120.0) -> None:
    """Block until the worker has finished whatever it was doing. The simulator is a single resource."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not client.get("/api/system").json()["worker"]["busy"]:
            return
        time.sleep(0.2)
    raise AssertionError("the worker stayed busy")


@pytest.fixture(scope="module")
def live():
    app = create_app(cfg=WorkerConfig(grounder="oracle", load_segmenter=False, record=True))
    with TestClient(app) as c:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            models = c.get("/api/system").json()["models"]
            if models.get("simulation", {}).get("state") == "warm" and models.get("grounder", {}).get("state") == "warm":
                break
            time.sleep(0.25)
        else:
            pytest.fail("the worker never reported warm models")
        yield c


def test_system_knows_what_the_worker_loaded(live):
    d = live.get("/api/system").json()
    assert d["worker"]["alive"] and d["worker"]["pid"]
    assert d["models"]["simulation"]["state"] == "warm" and d["models"]["simulation"]["load_ms"] > 0
    assert "not Jetson" in d["hardware_label"]
    assert d["gpu"]


def test_scene_then_run_streams_nine_stages_and_records_the_run(live):
    scene = live.post("/api/scene", json={"seed": 5000, "stratum": "seen"}).json()["scene"]
    assert scene["seed"] == 5000 and scene["command"]

    stages, frames, outcome, safety = [], 0, None, []
    with live.websocket_connect("/ws/live?cameras=front") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["banner"].startswith("Simulation")
        run = live.post("/api/run", json={"command": "pick the blue cube", "controller": "pipeline", "config": {"use_yolo_mask": False, "speed": 0.0}})
        assert run.status_code == 202
        run_id = run.json()["run_id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and outcome is None:
            msg = ws.receive()
            if msg["type"] == "websocket.send" and msg.get("bytes"):
                meta, jpeg = unpack_frame(msg["bytes"])
                assert jpeg[:2] == b"\xff\xd8"  # a real JPEG
                frames += 1
            elif msg.get("text"):
                import json

                e = json.loads(msg["text"])
                if e["type"] == "stage_finished":
                    stages.append(e["stage"])
                elif e["type"] == "safety":
                    safety.append(e)
                elif e["type"] == "outcome":
                    outcome = e
    assert outcome is not None, f"no outcome arrived; stages seen: {stages}"
    assert [s for s in STAGES if s in stages] == list(STAGES), stages
    assert outcome["placed"] and outcome["aborted"] is None
    assert frames > 0, "no camera frames arrived over the WebSocket"
    assert safety and safety[-1]["mode"] == "monitor"

    runs = live.get("/api/runs").json()["runs"]
    row = next((r for r in runs if r["id"] == run_id), None)
    assert row is not None, [r["id"] for r in runs]
    assert row["seed"] == 5000 and row["n_events"] > 20 and row["safety_mode"] == "monitor"

    detail = live.get(f"/api/runs/{run_id}").json()
    assert detail["meta"]["command"] == "pick the blue cube"
    assert "not a hardware measurement" in detail["meta"]["note"]
    kinds = {e["type"] for e in detail["events"]}
    assert {"stage_finished", "tick", "outcome", "safety"} <= kinds
    assert not any("jpeg" in e for e in detail["events"]), "recorded events must reference frames, not embed them"

    # the stage images the drawer will show are on disk and served
    seg = next(e for e in detail["events"] if e["type"] == "stage_finished" and e["stage"] == "capture")
    assert set(seg["images"]) == {"rgb", "depth"}
    img = live.get(seg["images"]["rgb"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and len(img.content) > 1000
    assert live.get(f"/runs/{run_id}/frames/nope.jpg").status_code == 404


def test_estop_and_reset_through_http(live):
    r = live.post("/api/estop", json={"client_ts_ms": 0.0})
    assert r.status_code == 200
    d = r.json()
    assert d["state"] == "ESTOP"
    assert d["estop_latency_ms"] is not None and d["estop_latency_ms"] < 500
    assert live.post("/api/run", json={}).status_code == 423
    back = live.post("/api/reset").json()
    assert back["state"] in ("RUN", "HOLD", "REDUCED_SPEED")
    # and the simulator is usable again
    assert live.post("/api/run", json={"controller": "oracle", "config": {"speed": 0.0}}).status_code == 202
    wait_idle(live)


def test_a_batch_job_runs_through_the_real_harness_and_writes_a_results_file(live):
    """The file a job writes must be the same shape as the files in results/, because it is the same harness."""
    import json
    from pathlib import Path

    wait_idle(live)
    r = live.post("/api/jobs", json={"controller": "oracle", "strata": {"seen": 2, "langvar": 1}, "config": {"speed": 0.0}})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    out_path = Path(r.json()["out_path"])
    assert r.json()["total"] == 3

    deadline = time.monotonic() + 180
    job = {}
    while time.monotonic() < deadline:
        job = live.get(f"/api/jobs/{job_id}").json()
        if job.get("state") in ("done", "error", "cancelled"):
            break
        time.sleep(0.3)
    assert job.get("state") == "done", job
    assert job["done"] == 3 and len(job["trials"]) == 3
    assert {t["stratum"] for t in job["trials"]} == {"seen", "langvar"}
    assert all(t["seed"] >= 5000 for t in job["trials"])

    written = Path(__file__).resolve().parents[1] / out_path
    try:
        assert written.exists(), f"{written} was not written"
        d = json.loads(written.read_text())
        assert d["summary"]["n_trials"] == 3
        assert set(d["summary"]["strata"]) >= {"all", "seen", "langvar"}
        assert d["summary"]["strata"]["all"]["place"]["n"] == 3
        assert "simulation only" in d["hardware"] and "run from the GUI" in d["notes"]
        assert len(d["trials"]) == 3 and "grounding_correct" in d["trials"][0]
        # the file is listed like any other measurement, with its own timestamp
        row = next(f for f in live.get("/api/results").json()["files"] if f["file"] == written.name)
        assert row["n_trials"] == 3 and row["mtime"] > 0
    finally:
        written.unlink(missing_ok=True)


def test_a_job_will_not_overwrite_a_protocol_file_and_the_simulator_stays_usable(live):
    wait_idle(live)
    r = live.post("/api/jobs", json={"controller": "oracle", "strata": {"seen": 1}, "out_path": "modular_protocol.json"})
    assert r.status_code == 409
    assert "already exists" in r.json()["error"]["message"]
    assert live.post("/api/run", json={"controller": "oracle", "config": {"speed": 0.0}}).status_code == 202
