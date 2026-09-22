"""Depth fusion tests (sim only, no GPU model needed)."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from langgrasp.perception.depth_fusion import (  # noqa: E402
    depth_to_points,
    grasp_from_points,
    intersect_mask_with_box,
    mask_from_box,
    realsense_like_noise,
    segment_object_points,
)


@pytest.fixture(scope="module")
def env():
    from langgrasp.sim.env import LangGraspEnv

    try:
        e = LangGraspEnv(seed=0)
        e.render("front")
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"MuJoCo rendering unavailable: {ex}")
    yield e
    e.close()


def _scene(env, seed):
    from langgrasp.sim.scenarios import make_scenario

    sc = make_scenario(seed, "seen", lighting="nominal")
    env.reset(sc)
    return sc


def _object_points(env, cam, name):
    _, depth, label = env.render_rgbd_seg(cam)
    m = label == env.object_names().index(name)
    pts = depth_to_points(depth, env.camera_intrinsics(cam), env.camera_extrinsics(cam), m)
    return segment_object_points(pts), m.sum()


def test_backprojection_centroid_matches_object_pose(env):
    checked = 0
    for seed in range(12):
        sc = _scene(env, seed)
        for o in sc.objects:
            if o.kind == "screwdriver":
                continue  # asymmetric visible surface; covered by the grasp test below
            pts, npx = _object_points(env, "front", o.name)
            if npx < 300:
                continue
            pos = env.object_pose(o.name)[0]
            mid = 0.5 * (pts[:, :2].min(0) + pts[:, :2].max(0))  # midpoint of the visible extent
            assert np.linalg.norm(mid - pos[:2]) < 0.006, (o.name, mid, pos)
            assert np.linalg.norm(pts[:, :2].mean(0) - pos[:2]) < 0.015
            # visible surface is above the body centre and below the top of the object
            assert pts[:, 2].max() > pos[2] - 0.001
            assert pts[:, 2].max() < pos[2] + 0.04
            checked += 1
    assert checked >= 6


def test_grasp_matches_oracle(env):
    n = {"cube": 0, "can": 0, "screwdriver": 0}
    for seed in range(20):
        sc = _scene(env, seed)
        for o in sc.objects:
            pts, npx = _object_points(env, "front", o.name)
            if npx < 300 or len(pts) < 50:
                continue
            g = grasp_from_points(pts)
            gt_p, gt_psi = env.grasp_point(o.name)
            assert np.linalg.norm(g["center"][:2] - gt_p[:2]) < 0.008, (o.name, g["center"], gt_p)
            assert abs(g["grasp_z"] - gt_p[2]) < 0.006
            if o.kind == "screwdriver":
                assert g["elongated"]
                d = (g["psi"] - gt_psi + np.pi / 2) % np.pi - np.pi / 2
                assert abs(np.degrees(d)) < 15
            elif o.kind == "cube":
                assert not g["elongated"]
                d = (g["psi"] - gt_psi + np.pi / 4) % (np.pi / 2) - np.pi / 4
                assert abs(np.degrees(d)) < 15
                assert 0.018 < g["width"] < 0.032
            n[o.kind] += 1
    assert all(v >= 2 for v in n.values()), n


def test_noise_model_and_outlier_removal(env):
    sc = _scene(env, 3)
    rng = np.random.default_rng(0)
    _, depth, label = env.render_rgbd_seg("front")
    noisy = realsense_like_noise(depth, rng, sigma=0.002, dropout=0.02, quant=0.001)
    assert noisy.shape == depth.shape and noisy.dtype == np.float32
    frac_dropped = float((noisy == 0).mean())
    assert 0.01 < frac_dropped < 0.04
    valid = noisy > 0
    assert abs(float((noisy[valid] - depth[valid]).std()) - 0.002) < 0.001
    o = sc.objects[0]
    m = label == env.object_names().index(o.name)
    pts = depth_to_points(noisy, env.camera_intrinsics("front"), env.camera_extrinsics("front"), m)
    seg = segment_object_points(pts)
    assert len(seg) < len(pts)  # dropped pixels (depth 0) and table-level silhouette pixels are gone
    assert seg[:, 2].min() >= 0.004
    g = grasp_from_points(seg)
    gt_p, _ = env.grasp_point(o.name)
    assert np.linalg.norm(g["center"][:2] - gt_p[:2]) < 0.012


def test_depth_to_points_synthetic_pinhole():
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T = np.eye(4)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    pts = depth_to_points(depth, K, T, max_depth=5.0)
    assert pts.shape == (480 * 640, 3)
    assert np.allclose(pts[:, 2], -2.0)
    # top-left pixel maps to -x (left) and +y (up) in the camera frame
    assert pts[0, 0] < 0 and pts[0, 1] > 0
    assert len(depth_to_points(depth, K, T, max_depth=1.0)) == 0


def test_mask_helpers():
    m = mask_from_box([10.4, 20.6, 30, 40], (50, 60))
    assert m.shape == (50, 60) and m.dtype == bool
    assert m.sum() == (30 - 10) * (40 - 21)
    full = np.ones((50, 60), dtype=bool)
    assert intersect_mask_with_box(full, [0, 0, 5, 5]).sum() == 25
    assert mask_from_box([-5, -5, 3, 3], (10, 10)).sum() == 9


def _screwdriver_cloud(rng, center, yaw, n=1500):
    """Synthetic top-view cloud of a lying screwdriver: 6.4 cm x 2.2 cm handle plus 9 cm x 0.7 cm shaft."""
    handle = np.stack([rng.uniform(-0.062, 0.002, n), rng.uniform(-0.011, 0.011, n)], 1)
    shaft = np.stack([rng.uniform(0.0, 0.09, n // 4), rng.uniform(-0.0035, 0.0035, n // 4)], 1)
    xy = np.concatenate([handle, shaft])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = xy @ R.T + center
    z = np.where(np.arange(len(xy)) < n, rng.uniform(0.012, 0.022, len(xy)), rng.uniform(0.008, 0.0105, len(xy)))
    return np.column_stack([xy, z])


def test_largest_cluster_drops_neighbour_blob():
    from langgrasp.perception.depth_fusion import largest_cluster

    rng = np.random.default_rng(0)
    a = _screwdriver_cloud(rng, (0.0, -0.2), 0.3)
    b = _screwdriver_cloud(rng, (0.05, -0.15), 0.3)[:200]  # a stray blob of a parallel neighbour
    pts = np.concatenate([a, b])
    kept = largest_cluster(pts)
    assert len(kept) == len(a)
    g = grasp_from_points(pts)
    assert g["elongated"] and not g["fallback"]
    assert g["aspect"] > 4.0
    assert 0.015 < g["width"] < 0.03
    # handle centre is 3 cm behind the object origin along the long axis
    exp = np.array([0.0, -0.2]) + np.array([np.cos(0.3), np.sin(0.3)]) * -0.03
    assert np.linalg.norm(g["center"][:2] - exp) < 0.006
    d = (g["psi"] - (0.3 + np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2
    assert abs(np.degrees(d)) < 5


def test_compact_path_falls_back_when_width_is_implausible():
    rng = np.random.default_rng(1)
    pts = _screwdriver_cloud(rng, (0.0, -0.2), -0.9)
    # force the compact path by raising the elongation threshold above the measured aspect
    g = grasp_from_points(pts, elongated_aspect=50.0)
    assert g["fallback"] and g["elongated"]
    assert g["width"] < 0.03
    exp = np.array([0.0, -0.2]) + np.array([np.cos(-0.9), np.sin(-0.9)]) * -0.03
    assert np.linalg.norm(g["center"][:2] - exp) < 0.006
