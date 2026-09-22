"""Modular pipeline with the oracle grounder (no GPU): stage plumbing, grounding-correctness check, tracer."""

import numpy as np
import pytest

from langgrasp.bus import Bus
from langgrasp.perception.grounding import Candidate, OracleGrounder, select_target
from langgrasp.policies.modular import ModularPipeline, PipelineConfig, box_iou
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.scenarios import make_scenario


@pytest.fixture(scope="module")
def env():
    e = LangGraspEnv(seed=0)
    yield e
    e.close()


def test_bus_pubsub_and_age():
    bus = Bus()
    got = []
    bus.subscribe("/a", lambda m: got.append(m.data))
    bus.publish("/a", 1)
    bus.publish("/a", 2)
    assert got == [1, 2] and bus.last("/a").seq == 2 and bus.age("/a") < 1.0 and bus.age("/missing") == float("inf")


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_select_target_spatial_and_color():
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[40:60, 20:40] = (220, 30, 30)  # red on the left
    rgb[40:60, 160:180] = (30, 30, 220)  # blue on the right
    cands = [Candidate([20, 40, 40, 60], 0.6, "cube"), Candidate([160, 40, 180, 60], 0.7, "cube")]
    c, info = select_target(cands, rgb, None, "left")
    assert c.box[0] == 20 and info["spatial_used"]
    c, info = select_target([Candidate(**x.__dict__) for x in cands], rgb, "red", None)
    assert c.box[0] == 20 and info["color_filtered"] == 1
    c, info = select_target([Candidate(**x.__dict__) for x in cands], rgb, None, None)
    assert c.box[0] == 160  # highest score


def test_pipeline_with_oracle_grounder(env):
    pipe = ModularPipeline(env, OracleGrounder(env), None, PipelineConfig(depth_noise=False))
    sc = make_scenario(5000, "seen", target_kind="cube")
    env.reset(sc)
    r = pipe.run_command(sc.command, sc.target)
    assert r.aborted is None and r.grounding_correct and r.mask_source == "oracle"
    assert r.grasp is not None and abs(r.grasp["width"] - 0.025) < 0.01
    gt, _ = env.grasp_point(sc.target)
    assert r.placed  # the oracle mask + depth fusion + executor should place a cube
    assert set(r.latency_ms) >= {"parse", "capture", "grounding", "depth_fusion", "execute", "end_to_end"}
    s = pipe.tracer.summary()
    assert s["end_to_end"]["n"] == 1
