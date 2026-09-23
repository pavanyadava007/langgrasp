"""Event schema of the GUI: validation, JSON safety, the binary frame envelope and the recording hooks."""

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from langgrasp.gui.trace import (
    STAGE_HELP,
    STAGE_TITLES,
    STAGES,
    TRACER_TO_STAGE,
    Outcome,
    PipelineHooks,
    RecordingHooks,
    Safety,
    StageFinished,
    jsonable,
    pack_frame,
    parse_event,
    unpack_frame,
)


def test_nine_stages_have_a_title_and_an_explanation():
    assert len(STAGES) == 9
    assert set(STAGE_TITLES) == set(STAGES)
    assert set(STAGE_HELP) == set(STAGES)
    # every tracer stage of the modular pipeline maps onto one of the nine
    assert set(TRACER_TO_STAGE.values()) <= set(STAGES)


def test_stage_finished_roundtrip_and_validation():
    e = StageFinished(stage="fuse", status="warn", latency_ms=3.5, message="only 21 points", payload={"n": 21}, run_id="r1")
    d = e.model_dump()
    back = parse_event(d)
    assert back.stage == "fuse" and back.status == "warn" and back.payload["n"] == 21
    assert back.latency_kind == "compute" and isinstance(back.t, float)
    with pytest.raises(ValidationError):
        StageFinished(stage="not_a_stage")
    with pytest.raises(ValidationError):
        StageFinished(stage="fuse", status="exploded")


def test_safety_and_outcome_defaults_are_honest():
    s = Safety(state="HOLD", reason="command stale")
    assert s.mode == "monitor" and s.estop_latency_ms is None and s.clips == {}
    o = Outcome()
    # nothing succeeded until something says it did
    assert (o.grasped, o.lifted, o.placed, o.aborted) == (False, False, False, None)
    assert o.grounding_correct is None and o.scored_by == "ground_truth"


def test_jsonable_handles_numpy_and_non_finite():
    out = jsonable(
        {
            "f32": np.float32(1.25),
            "i64": np.int64(7),
            "bool": np.bool_(True),
            "arr": np.arange(3, dtype=np.float32),
            "nan": float("nan"),
            "inf": np.float32("inf"),
            "nested": [{"x": np.float64(0.5)}],
            "tuple": (1, 2),
        }
    )
    assert out == {"f32": 1.25, "i64": 7, "bool": True, "arr": [0.0, 1.0, 2.0], "nan": None, "inf": None, "nested": [{"x": 0.5}], "tuple": [1, 2]}
    # the result must survive json.dumps, which is the whole point
    assert json.loads(json.dumps(out))["arr"] == [0.0, 1.0, 2.0]
    assert not any(isinstance(v, float) and math.isnan(v) for v in [out["nan"]] if v is not None)


def test_frame_envelope_roundtrip():
    jpeg = b"\xff\xd8\xff\xe0 not really a jpeg"
    buf = pack_frame({"camera": "front", "tick": 12, "score": np.float32(0.5)}, jpeg)
    meta, payload = unpack_frame(buf)
    assert meta == {"camera": "front", "tick": 12, "score": 0.5}
    assert payload == jpeg
    with pytest.raises(ValueError, match="not a LangGrasp frame"):
        unpack_frame(b"XXXX" + buf[4:])


def test_base_hooks_do_nothing_and_never_raise():
    h = PipelineHooks()
    assert h.stage_started("parse", text="x") is None
    assert h.stage_finished("parse", status="ok", latency_ms=1.0, images={"rgb": object()}) is None


def test_recording_hooks_collect_stages_and_payloads():
    h = RecordingHooks()
    h.stage_started("capture", camera="front")
    h.stage_finished("capture", latency_ms=9.0, camera="front", images={"rgb": np.zeros((2, 2, 3), np.uint8)})
    h.stage_finished("fuse", status="fail", message="no points")
    assert h.stages == ["capture", "fuse"]
    assert h.payload("capture")["camera"] == "front"
    assert h.events[1]["image_roles"] == ["rgb"] and "images" not in h.events[1]
    assert h.payload("missing") == {}
    keep = RecordingHooks(keep_images=True)
    keep.stage_finished("segment", images={"mask": np.ones((2, 2), bool)})
    assert keep.events[0]["images"]["mask"].shape == (2, 2)
