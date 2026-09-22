import numpy as np
import pytest

from langgrasp.sim.controller import PickPlaceController
from langgrasp.sim.env import LangGraspEnv
from langgrasp.sim.kinematics import top_down_rotation
from langgrasp.sim.scenarios import make_scenario
from langgrasp.sim.scene import OBJECT_KINDS, WORKSPACE


@pytest.fixture(scope="module")
def env():
    e = LangGraspEnv(seed=0)
    yield e
    e.close()


def test_scenarios_are_deterministic_and_collision_free():
    a, b = make_scenario(42, "langvar"), make_scenario(42, "langvar")
    assert a.to_dict() == b.to_dict()
    for seed in range(30):
        sc = make_scenario(seed, ["seen", "unseen", "langvar"][seed % 3])
        assert sc.target in [o.name for o in sc.objects]
        for o in sc.objects:
            assert WORKSPACE["x"][0] - 1e-9 <= o.pos[0] <= WORKSPACE["x"][1] + 1e-9
        names = [o.name for o in sc.objects]
        assert len(names) == len(set(names))


def test_reset_places_objects(env):
    sc = make_scenario(7, "seen")
    env.reset(sc)
    for o in sc.objects:
        p, _ = env.object_pose(o.name)
        assert np.hypot(p[0] - o.pos[0], p[1] - o.pos[1]) < 0.01
        assert p[2] < 0.05


def test_ik_top_down_reaches_workspace(env):
    kin = env.kin
    for x, y in [(-0.08, -0.2), (0.0, -0.24), (0.08, -0.18)]:
        q, pe, re = kin.solve(np.array([x, y, 0.03]), 0.3, env.observe_q, iters=100)
        assert pe < 1e-3 and re < 0.02
        p, R = kin.fk(q)
        assert np.allclose(p, [x, y, 0.03], atol=1e-3)
        assert np.allclose(R, top_down_rotation(0.3), atol=0.03)


def test_top_down_rotation_is_orthonormal():
    R = top_down_rotation(0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_scripted_pick_succeeds_on_cube(env):
    successes = 0
    for seed in [1003, 1004, 1005]:
        sc = make_scenario(seed, "seen", target_kind="cube")
        env.reset(sc)
        p, psi = env.grasp_point(sc.target)
        r = PickPlaceController(env).run(np.asarray(p), psi, sc.target, width=OBJECT_KINDS["cube"]["width"])
        successes += r.placed
    assert successes >= 2


def test_camera_geometry(env):
    env.reset(make_scenario(3, "seen"))
    K = env.camera_intrinsics("front")
    assert K.shape == (3, 3) and K[0, 2] == 320 and K[1, 2] == 240
    T = env.camera_extrinsics("front")
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6)
    rgb, depth, label = env.render_rgbd_seg("front", (120, 160))
    assert rgb.shape == (120, 160, 3) and depth.shape == (120, 160) and label.shape == (120, 160)
    assert (label >= 0).sum() > 20 and np.isfinite(depth).all()
