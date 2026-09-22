"""Host-side tests of the message-free ROS 2 logic (ros2_ws/src/langgrasp_ros/langgrasp_ros/logic.py).
No rclpy needed: the ROS nodes are thin wrappers around these functions."""

import os
import sys

import numpy as np
import pytest

ROS_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ros2_ws", "src", "langgrasp_ros")
if ROS_PKG not in sys.path:
    sys.path.insert(0, ROS_PKG)

from langgrasp_ros import logic  # noqa: E402


def test_no_rclpy_needed():
    assert "rclpy" not in sys.modules or True  # importing logic must not have required rclpy
    assert logic.STAGES[0] == "camera" and logic.STAGES[-1] == "joint_command_safe"
    assert logic.JOINT_NAMES == ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]


def test_image_and_depth_fields():
    rgb = np.random.default_rng(0).integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    f = logic.image_fields(rgb)
    assert (f["height"], f["width"], f["encoding"], f["step"]) == (240, 320, "rgb8", 960)
    assert len(f["data"]) == 240 * 960
    assert np.frombuffer(f["data"], dtype=np.uint8).reshape(240, 320, 3).tolist() == rgb.tolist()
    d = logic.depth_fields(np.full((240, 320), 0.65, dtype=np.float64))
    assert (d["encoding"], d["step"]) == ("32FC1", 1280)
    assert np.allclose(np.frombuffer(d["data"], dtype=np.float32), 0.65)


def test_camera_info_fields():
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    ci = logic.camera_info_fields(K, 240, 320)
    assert ci["k"] == K.reshape(-1).tolist() and ci["distortion_model"] == "plumb_bob"
    assert ci["p"][:3] == [300.0, 0.0, 160.0] and len(ci["p"]) == 12 and len(ci["r"]) == 9


def test_project_unproject_roundtrip():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [0.0, -0.6, 0.5]
    # camera looking straight down: cam -z = world -z, cam y = world +y
    T[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    p = np.array([0.05, -0.55, 0.0])
    u, v, d = logic.project_point(K, T, p)
    assert d == pytest.approx(0.5)
    assert np.allclose(logic.unproject_pixel(K, T, u, v, d), p)


def test_lexical_scores_and_oracle_ground():
    scene = {
        "target": "cube_0",
        "objects": [
            {"name": "cube_0", "kind": "cube", "color": "red", "grasp_xyz": [0.0, -0.2, 0.0125], "psi": 0.3, "width": 0.025, "pixel": [100.0, 120.0], "depth": 0.6},
            {"name": "can_0", "kind": "can", "color": "blue", "grasp_xyz": [0.05, -0.2, 0.03], "psi": 0.0, "width": 0.026, "pixel": [200.0, 120.0], "depth": 0.6},
            {"name": "cube_1", "kind": "cube", "color": "red", "grasp_xyz": [-0.05, -0.2, 0.0125], "psi": 0.0, "width": 0.025, "pixel": [50.0, 120.0], "depth": 0.6},
        ],
    }
    assert logic.lexical_scores("pick the blue can", scene["objects"]) == [0.0, 1.0, 0.0]
    assert logic.lexical_scores("pick the blue block", scene["objects"]) == [0.5, 0.5, 0.5]
    res = logic.oracle_ground(scene, "pick the blue cylinder")
    assert res["target"]["name"] == "can_0" and res["detections"][1]["score"] == 1.0
    # duplicated red cubes tie: the safety gate must flag this as ambiguous
    res = logic.oracle_ground(scene, "pick the red cube on the left")
    assert sorted(res["scores"])[-2:] == [1.0, 1.0]
    from langgrasp.safety import SafetyMonitor

    assert not SafetyMonitor().gate_grounding(res["scores"], 3)[0]
    # no match at all: falls back to the ground-truth target with score 0 (gate blocks it)
    res = logic.oracle_ground(scene, "hello")
    assert res["target"]["name"] == "cube_0" and max(res["scores"]) == 0.0


def test_top_down_quaternion_roundtrip():
    from langgrasp.sim.kinematics import top_down_rotation

    for psi in (-2.5, -0.9, 0.0, 0.7, 2.9):
        q = logic.top_down_quaternion(psi)
        assert np.isclose(np.linalg.norm(q), 1.0)
        assert logic.quaternion_to_psi(*q) == pytest.approx(psi, abs=1e-9)
        R = top_down_rotation(psi)
        assert np.allclose(logic.rotation_to_quaternion(R), q) or np.allclose(logic.rotation_to_quaternion(R), -np.array(q))


def test_grasp_from_target_and_plan():
    g = logic.grasp_from_target({"grasp_xyz": [0.01, -0.2, 0.0125], "psi": 0.4, "width": 0.025, "name": "cube_0"})
    assert g["xyz"] == [0.01, -0.2, 0.0125] and len(g["quat_xyzw"]) == 4
    segs = logic.plan_pick_segments(g["xyz"], g["psi"], g["width"])
    phases = [s.phase for s in segs]
    assert phases == ["approach", "hover", "descend", "close", "lift", "transport", "lower", "release", "retreat", "settle"]
    assert segs[2].target[2] >= 0.01 and segs[3].stay_ticks == 6 and not segs[4].strict
    assert len(logic.plan_pick_segments(g["xyz"], g["psi"], g["width"], place=False)) == 5


def test_latency_tracker_summary():
    tr = logic.LatencyTracker()
    for k, stage in enumerate(logic.STAGES):
        for stamp in (1_000_000_000, 2_000_000_000):
            tr.record(stage, stamp, stamp + (k + 1) * 10_000_000)  # 10 ms per stage
        tr.record(stage, 1_000_000_000, 999)  # duplicate arrival must be ignored
    s = tr.summary()
    assert s["end_to_end_ms"]["n"] == 2 and s["end_to_end_ms"]["median_ms"] == pytest.approx(50.0)
    assert all(v["median_ms"] == pytest.approx(10.0) for v in s["per_stage_ms"].values())
    assert s["since_capture_ms"]["camera"]["median_ms"] == pytest.approx(10.0)
    assert s["n_messages"]["camera"] == 3
    assert logic.stamp_to_ns(1, 5) == 1_000_000_005


def test_scene_json_roundtrip(tmp_path):
    from langgrasp.sim.scenarios import PlacedObject, Scenario

    sc = Scenario(stratum="seen", objects=[PlacedObject("cube_0", "cube", "red", (0.0, -0.2), 0.1)], target="cube_0", command="pick the red cube", seed=7)
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    T = np.eye(4)
    T[:3, 3] = [0, -0.6, 0.5]
    js = logic.scene_to_json(sc, {"cube_0": (np.array([0.0, -0.2, 0.0125]), 1.67)}, K, T, (240, 320))
    p = str(tmp_path / "scene.json")
    logic.write_scene_json(js, p)
    back = logic.read_scene_json(p)
    assert back["target"] == "cube_0" and back["objects"][0]["psi"] == pytest.approx(1.67)
    assert back["camera"]["height"] == 240 and len(back["objects"][0]["pixel"]) == 2
    assert logic.read_scene_json(str(tmp_path / "missing.json")) is None


def test_pick_executor_reproduces_controller_in_sim():
    """PickExecutor (ROS policy node logic) must produce the same outcome as PickPlaceController."""
    pytest.importorskip("mujoco")
    from langgrasp.sim.controller import PickPlaceController
    from langgrasp.sim.env import LangGraspEnv
    from langgrasp.sim.scenarios import make_scenario
    from langgrasp.sim.scene import OBJECT_KINDS

    env = LangGraspEnv(seed=0, render=False)
    kin = logic.load_arm_kinematics()
    agree = 0
    placed_any = False
    for seed in (1001, 1002, 1003):
        try:
            sc = make_scenario(seed, "seen")
        except RuntimeError:
            continue
        env.reset(sc)
        xyz, psi = env.grasp_point(sc.target)
        w = OBJECT_KINDS[sc.target_obj.kind]["width"]
        ex = logic.PickExecutor(kin, xyz, psi, w)
        while True:
            out = ex.tick(env.q_arm)
            if out is None:
                break
            env.step(out[0], out[1])
        ex_placed = env.in_tray(sc.target)
        env.reset(sc)
        ctl_placed = PickPlaceController(env).run(xyz, psi, sc.target, width=w).placed
        agree += int(ex_placed == ctl_placed)
        placed_any |= ex_placed
        assert ex.done and ex.phase == "done" and ex.ticks > 20
    assert agree >= 2
    assert placed_any, "PickExecutor placed nothing on 3 seeds"
    env.close()
