"""HTTP and WebSocket API, driven against a stub worker so no simulator or GPU model is involved.

What these tests are for: that the API forwards commands faithfully, refuses what it must refuse with a
message a person can act on, serves results files with their provenance, and never invents a number.
"""

from __future__ import annotations

import json
import queue

import pytest
from starlette.testclient import TestClient

from langgrasp.gui.api import RESULTS_MANIFEST, create_app
from langgrasp.gui.hub import CLIENT_JSON_BUFFER, ClientChannel
from langgrasp.gui.trace import STAGES, FrameMeta, Safety, StageFinished, System, unpack_frame


class Flag:
    def __init__(self, value=0):
        self.value = value


class StubFlags:
    def __init__(self):
        self.estop = Flag(0)
        self.estop_t = Flag(0.0)
        self.paused = Flag(0)
        self.step_once = Flag(0)
        self.speed = Flag(1.0)
        self.cameras = Flag(1)
        self.busy = Flag(0)

    def request_estop(self):
        import time

        t = time.monotonic()
        self.estop_t.value = t
        self.estop.value = 1
        return t

    def clear_estop(self):
        self.estop.value = 0


class StubWorker:
    """Records the commands the API sends and lets a test push events back."""

    def __init__(self):
        self.flags = StubFlags()
        self.sent: list[dict] = []
        self.q: queue.Queue = queue.Queue()
        self.alive = True
        self.proc = None

    @property
    def busy(self) -> bool:
        return bool(self.flags.busy.value)

    def send(self, **msg):
        self.sent.append(msg)

    def emit(self, event, jpeg: bytes | None = None):
        d = event.model_dump()
        if jpeg is not None:
            d["jpeg"] = jpeg
        self.q.put(d)

    def next_event(self, timeout: float = 0.2):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def estop(self):
        return self.flags.request_estop()

    def reset_estop(self):
        self.send(cmd="reset_estop")

    def pause(self, on: bool):
        self.flags.paused.value = 1 if on else 0

    def step_once(self):
        self.flags.step_once.value = 1

    def start(self):
        pass

    def stop(self):
        self.alive = False

    def last(self, cmd: str) -> dict:
        return next(m for m in reversed(self.sent) if m.get("cmd") == cmd)


SCENE = {"seed": 5000, "stratum": "seen", "target": "cube_0", "command": "pick the blue cube", "lighting": "nominal", "objects": []}


@pytest.fixture
def client():
    w = StubWorker()
    app = create_app(worker=w, start_worker=False)
    with TestClient(app) as c:
        c.worker = w
        yield c


def test_system_reports_the_banner_versions_and_stages_without_inventing_any(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    d = r.json()
    assert d["banner"] == "Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware"
    assert [s["name"] for s in d["stages"]] == list(STAGES)
    assert all(s["help"] for s in d["stages"])
    assert d["versions"]["mujoco"] and d["versions"]["fastapi"]
    # nothing is known about the models until the worker says so
    assert d["models"] == {} and d["gpu"] is None and d["safety"] is None
    assert "observes the safety monitor by default" in d["safety_note"]
    assert all(set(e) == {"path", "present", "size_mb", "mtime"} for e in d["engines"])


def test_scene_post_forwards_and_waits_for_the_worker(client):
    w = client.worker

    def answer():
        w.emit(System(scene=SCENE, note="scene reset"))

    # the worker answers as soon as the command lands
    original_send = w.send

    def send_and_answer(**msg):
        original_send(**msg)
        if msg.get("cmd") == "scene":
            answer()

    w.send = send_and_answer
    r = client.post("/api/scene", json={"seed": 7001, "stratum": "langvar", "lighting": "degraded"})
    assert r.status_code == 202
    assert r.json()["scene"]["seed"] == 5000  # whatever the worker reported, not what we asked for
    sent = w.last("scene")
    assert sent["seed"] == 7001 and sent["stratum"] == "langvar" and sent["lighting"] == "degraded"


def test_scene_post_reports_a_worker_that_does_not_answer(client, monkeypatch):
    import langgrasp.gui.api as api_mod

    monkeypatch.setattr(api_mod, "SCENE_TIMEOUT_S", 0.3)
    r = client.post("/api/scene", json={"seed": 1})
    assert r.status_code == 504
    assert "did not confirm" in r.json()["error"]["message"]


def test_run_forwards_the_whole_config(client):
    w = client.worker
    body = {"command": "pick the blue screwdriver", "controller": "pipeline", "config": {"use_color_check": False, "seg_backend": "engine", "speed": 0.0, "safety_mode": "enforce"}, "source": "chip"}
    r = client.post("/api/run", json=body)
    assert r.status_code == 202 and r.json()["run_id"]
    sent = w.last("run")
    assert sent["command"] == "pick the blue screwdriver" and sent["source"] == "chip"
    assert sent["config"]["use_color_check"] is False and sent["config"]["seg_backend"] == "engine"
    assert sent["config"]["safety_mode"] == "enforce" and sent["config"]["speed"] == 0.0
    # defaults that were not sent are still explicit
    assert sent["config"]["use_yolo_mask"] is True and sent["config"]["depth_noise"] is True


def test_run_is_refused_while_busy_or_latched_with_an_actionable_message(client):
    w = client.worker
    w.flags.busy.value = 1
    r = client.post("/api/run", json={})
    assert r.status_code == 409 and r.json()["error"]["code"] == "busy"
    w.flags.busy.value = 0
    w.flags.estop.value = 1
    r = client.post("/api/run", json={})
    assert r.status_code == 423
    assert "Press Reset" in r.json()["error"]["message"]


def test_unwired_controllers_say_so_instead_of_pretending(client):
    r = client.post("/api/run", json={"controller": "act"})
    assert r.status_code == 501 and "not wired into the worker yet" in r.json()["error"]["message"]


def test_estop_returns_the_measured_latency_from_the_worker(client):
    w = client.worker
    original_send = w.send

    def send_and_answer(**msg):
        original_send(**msg)

    w.send = send_and_answer

    # the API sets the flag synchronously; the worker reports the measured latency on the event stream
    import threading

    def emit_soon():
        w.emit(Safety(state="ESTOP", reason="operator e-stop (GUI)", estop_latency_ms=7.5, estop_latency_note="CLOCK_MONOTONIC"))

    threading.Timer(0.05, emit_soon).start()
    r = client.post("/api/estop", json={"client_ts_ms": 1234.5})
    d = r.json()
    assert d["state"] == "ESTOP" and d["estop_latency_ms"] == 7.5
    assert w.flags.estop.value == 1
    assert "monotonic clock" in d["note"]


def test_estop_still_answers_if_the_worker_is_silent(client, monkeypatch):
    import langgrasp.gui.api as api_mod

    monkeypatch.setattr(api_mod, "ESTOP_WAIT_S", 0.2)
    r = client.post("/api/estop", json={})
    assert r.status_code == 200
    d = r.json()
    assert d["state"] == "requested" and d["estop_latency_ms"] is None


def test_reset_and_pause_flip_the_right_flags(client):
    w = client.worker
    import threading

    threading.Timer(0.05, lambda: w.emit(Safety(state="HOLD", reason="command stale"))).start()
    r = client.post("/api/reset")
    assert r.json()["state"] == "HOLD" and w.last("reset_estop")
    assert client.post("/api/pause", json={"paused": True}).json()["paused"] is True
    assert w.flags.paused.value == 1
    assert client.post("/api/pause", json={"step": True}).json()["stepped"] is True
    assert w.flags.step_once.value == 1
    assert client.post("/api/pause", json={}).status_code == 400


def test_confirm_is_forwarded(client):
    w = client.worker
    assert client.post("/api/confirm", json={"decision": "reject"}).json()["accepted"] is False
    assert w.last("confirm")["decision"] == "reject"


def test_results_index_is_the_real_directory_with_provenance(client):
    d = client.get("/api/results").json()
    files = {f["file"] for f in d["files"]}
    assert "modular_protocol.json" in files and "safety_clip_audit.json" in files
    row = next(f for f in d["files"] if f["file"] == "modular_protocol.json")
    assert row["mtime"] > 0 and row["parses"] and row["n_trials"] == 120
    assert "not run" in d["note"]


def test_results_file_is_served_verbatim_with_its_mtime(client):
    r = client.get("/api/results/oracle_protocol.json")
    assert r.status_code == 200
    assert r.headers["X-Source-File"] == "results/oracle_protocol.json"
    assert float(r.headers["X-Source-Mtime"]) > 0
    d = r.json()
    assert d["summary"]["n_trials"] == 120 and "simulation only" in d["hardware"]


def test_a_missing_results_file_is_not_run_and_a_bad_name_is_refused(client):
    r = client.get("/api/results/nothing_measured_this.json")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_run" and "has not been run" in r.json()["error"]["message"]
    assert client.get("/api/results/..%2F..%2Fpyproject.toml").status_code in (400, 404)
    assert client.get("/api/results/pyproject.toml").status_code == 400


def test_manifest_names_the_files_each_section_needs(client):
    d = client.get("/api/results/manifest").json()["sections"]
    assert set(d) == set(RESULTS_MANIFEST)
    protocol = d["protocol"]["files"]
    assert any("Oracle" in label for label in protocol)
    for entry in protocol.values():
        assert entry["file"].endswith(".json") and isinstance(entry["present"], bool)
    # the manifest is the only place filenames live, so the dashboard cannot hardcode one
    assert protocol["Modular, no colour check"]["file"] == "modular_nocolor_protocol.json"


def test_runs_index_exists_even_with_no_recorded_runs(client):
    assert isinstance(client.get("/api/runs").json()["runs"], list)
    assert client.get("/api/runs/not-a-run").status_code == 404


def test_websocket_sends_hello_json_events_and_binary_frames(client):
    w = client.worker
    with client.websocket_connect("/ws/live?cameras=front,wrist&depth=1") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["stages"] == list(STAGES)
        assert w.last("subscribe")["cameras"] == ["front", "wrist"] and w.last("subscribe")["depth"] is True
        w.emit(StageFinished(stage="grounding", latency_ms=281.0, payload={"query": "blue cube"}, run_id="r1"))
        event = ws.receive_json()
        assert event["type"] == "stage_finished" and event["payload"]["query"] == "blue cube"
        w.emit(FrameMeta(camera="front", kind="rgb", tick=3, width=640, height=480, bytes=4, run_id="r1"), jpeg=b"\xff\xd8ab")
        meta, jpeg = unpack_frame(ws.receive_bytes())
        assert meta["camera"] == "front" and meta["tick"] == 3 and jpeg == b"\xff\xd8ab"


def test_websocket_does_not_forward_cameras_a_client_did_not_ask_for(client):
    w = client.worker
    with client.websocket_connect("/ws/live?cameras=front") as ws:
        ws.receive_json()
        w.emit(FrameMeta(camera="side", kind="rgb", width=8, height=8, bytes=1), jpeg=b"x")
        w.emit(StageFinished(stage="parse", latency_ms=0.1))
        # the side frame is filtered out, so the next message is the JSON event
        assert ws.receive_json()["stage"] == "parse"


def test_client_channel_drops_stale_frames_not_stage_events():
    c = ClientChannel({"front"}, depth=False)
    for tick in range(5):
        c.offer(FrameMeta(camera="front", kind="rgb", tick=tick, width=1, height=1, bytes=1).model_dump(), b"j")
    for i in range(CLIENT_JSON_BUFFER + 10):
        c.offer(StageFinished(stage="parse", latency_ms=float(i)).model_dump())
    js, frames = c.take()
    assert len(frames) == 1 and frames[0][0]["tick"] == 4, "only the newest frame per camera survives"
    assert c.dropped_frames == 4
    assert len(js) == CLIENT_JSON_BUFFER and c.dropped_json == 10
    assert js[-1]["latency_ms"] == float(CLIENT_JSON_BUFFER + 9), "the newest events are the ones kept"


def test_placeholder_page_carries_the_banner_until_the_frontend_is_built(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "not real hardware" in r.text


def test_stt_failure_is_reported_as_such(client):
    r = client.post("/api/stt", files={"audio": ("clip.webm", b"", "audio/webm")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_audio"


def test_openapi_documents_every_route(client):
    spec = client.get("/api/openapi.json").json()
    paths = set(spec["paths"])
    for p in ("/api/system", "/api/scene", "/api/run", "/api/estop", "/api/reset", "/api/pause", "/api/confirm", "/api/stt", "/api/results", "/api/results/manifest", "/api/runs"):
        assert p in paths, f"{p} is not in the OpenAPI document"
    assert json.dumps(spec)
