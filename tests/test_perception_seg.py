"""Segmenter wrapper shape checks on a rendered frame with the stock yolo11n-seg.pt (skips without GPU)."""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

STOCK = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "yolo11n-seg.pt")
TUNED = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "yolo11n-seg-langgrasp.pt")
ONNX = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "yolo11n-seg-langgrasp.onnx")


def _gpu():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU")


@pytest.fixture(scope="module")
def frame():
    try:
        from langgrasp.sim.env import LangGraspEnv
        from langgrasp.sim.scenarios import make_scenario

        env = LangGraspEnv(seed=0)
        env.reset(make_scenario(2, "seen", lighting="nominal"))
        rgb = env.render("front")
        env.close()
    except Exception as ex:  # noqa: BLE001
        pytest.skip(f"MuJoCo rendering unavailable: {ex}")
    return rgb


def _check(dets, shape):
    for d in dets:
        assert set(d) == {"cls", "name", "conf", "box", "mask"}
        assert isinstance(d["cls"], int) and isinstance(d["name"], str)
        assert 0.0 <= d["conf"] <= 1.0
        x1, y1, x2, y2 = d["box"]
        assert 0 <= x1 <= x2 <= shape[1] and 0 <= y1 <= y2 <= shape[0]
        assert d["mask"].shape == shape and d["mask"].dtype == bool


def test_pt_backend_stock_weights(frame):
    _gpu()
    if not os.path.exists(STOCK):
        pytest.skip("stock checkpoint missing")
    from langgrasp.perception.segmentation import Segmenter

    seg = Segmenter(STOCK)
    seg.warmup(1)
    dets = seg.detect(frame, conf=0.05)
    _check(dets, frame.shape[:2])
    assert len(seg.names) == 80
    assert set(seg.last_timing) == {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}


def test_tuned_pt_finds_objects(frame):
    _gpu()
    if not os.path.exists(TUNED):
        pytest.skip("fine-tuned checkpoint missing")
    from langgrasp.perception.segmentation import Segmenter

    dets = Segmenter(TUNED).detect(frame, conf=0.25)
    _check(dets, frame.shape[:2])
    assert len(dets) >= 1
    assert all(d["mask"].sum() >= 60 for d in dets)


def test_onnx_backend_matches_pt(frame):
    _gpu()
    if not (os.path.exists(TUNED) and os.path.exists(ONNX)):
        pytest.skip("fine-tuned ONNX export missing")
    from langgrasp.perception.segmentation import Segmenter

    a = Segmenter(TUNED).detect(frame, conf=0.25)
    b = Segmenter(ONNX).detect(frame, conf=0.25)
    _check(b, frame.shape[:2])
    assert len(a) == len(b)
    for da in a:
        best = max((((da["mask"] & db["mask"]).sum() / max((da["mask"] | db["mask"]).sum(), 1)) for db in b if db["cls"] == da["cls"]), default=0.0)
        assert best > 0.7


def test_letterbox_and_nms():
    from langgrasp.perception.segmentation import letterbox, nms_xyxy

    img = np.zeros((480, 640, 3), dtype=np.uint8)
    lb, r, (px, py) = letterbox(img, (640, 640))
    assert lb.shape == (640, 640, 3) and r == 1.0 and (px, py) == (0, 80)
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    keep = nms_xyxy(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_yolo_masks_two_parallel_screwdrivers_regression():
    """Owner-reported case: make_scenario(7000, 'langvar') has two parallel red screwdrivers with overlapping
    boxes; the YOLO mask of each carries a blob of the other, which used to make aspect 1.8 and width 9 cm."""
    _gpu()
    if not os.path.exists(TUNED):
        pytest.skip("fine-tuned checkpoint missing")
    from langgrasp.perception.depth_fusion import depth_to_points, grasp_from_points, segment_object_points
    from langgrasp.perception.segmentation import Segmenter
    from langgrasp.sim.env import LangGraspEnv
    from langgrasp.sim.scenarios import make_scenario

    env = LangGraspEnv(seed=0)
    sc = make_scenario(7000, "langvar")
    env.reset(sc)
    rgb, depth, label = env.render_rgbd_seg("front")
    K, T = env.camera_intrinsics("front"), env.camera_extrinsics("front")
    dets = [d for d in Segmenter(TUNED).detect(rgb, conf=0.25) if d["name"] == "screwdriver"]
    env.close()
    assert len(dets) == 2
    names = env.object_names()
    for d in dets:
        # the object this mask mostly covers is its ground truth
        owner = max((o.name for o in sc.objects if o.kind == "screwdriver"), key=lambda n: (d["mask"] & (label == names.index(n))).sum())
        pts = segment_object_points(depth_to_points(depth, K, T, d["mask"]))
        g = grasp_from_points(pts)
        gt_p, gt_psi = env.grasp_point(owner)
        assert g["elongated"] and g["aspect"] > 4.0
        assert g["width"] < 0.03
        assert np.linalg.norm(g["center"][:2] - gt_p[:2]) < 0.008
        dpsi = (g["psi"] - gt_psi + np.pi / 2) % np.pi - np.pi / 2
        assert abs(np.degrees(dpsi)) < 15
