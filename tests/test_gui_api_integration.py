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
