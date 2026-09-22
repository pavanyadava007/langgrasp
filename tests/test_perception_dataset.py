"""Tiny synthetic dataset build: label format and file layout."""

import os

import numpy as np
import pytest
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

from langgrasp.perception.classes import CLASS_NAMES  # noqa: E402
from langgrasp.perception.synth_dataset import SynthConfig, build_dataset, instance_polygons  # noqa: E402


def test_tiny_dataset(tmp_path):
    try:
        stats = build_dataset(str(tmp_path), SynthConfig(n_train_scenes=1, n_val_scenes=1, n_previews=1, seed=7), verbose=False)
    except Exception as ex:  # noqa: BLE001
        if "EGL" in str(ex) or "GL" in type(ex).__name__:
            pytest.skip(f"MuJoCo rendering unavailable: {ex}")
        raise
    assert stats["splits"]["train"]["images"] == 3 and stats["splits"]["val"]["images"] == 3
    d = yaml.safe_load(open(tmp_path / "data.yaml"))
    assert d["names"] == dict(enumerate(CLASS_NAMES))
    assert d["train"] == "images/train" and d["val"] == "images/val"
    for split in ("train", "val"):
        imgs = sorted(os.listdir(tmp_path / "images" / split))
        lbls = sorted(os.listdir(tmp_path / "labels" / split))
        assert len(imgs) == 3 and [i.replace(".jpg", ".txt") for i in imgs] == lbls
        for lbl in lbls:
            for line in open(tmp_path / "labels" / split / lbl):
                parts = line.split()
                assert int(parts[0]) in range(len(CLASS_NAMES))
                coords = np.array(parts[1:], dtype=float)
                assert len(coords) >= 6 and len(coords) % 2 == 0
                assert coords.min() >= 0.0 and coords.max() <= 1.0
    assert stats["splits"]["train"]["instances"] + stats["splits"]["val"]["instances"] >= 2
    assert any(f.startswith("preview_") for f in os.listdir(tmp_path))


def test_instance_polygons_skips_small_and_unseen():
    label = np.full((100, 100), -1, dtype=np.int32)
    label[10:40, 10:40] = 0  # cube_0, 900 px
    label[50:55, 50:55] = 4  # screwdriver_0, 25 px (< 60) -> skipped
    label[70:95, 70:95] = 8  # bar_0 -> unseen kind, skipped
    names = [f"{k}_{i}" for k, i in [("cube", 0), ("cube", 1), ("cube", 2), ("cube", 3), ("screwdriver", 0), ("screwdriver", 1), ("can", 0), ("can", 1), ("bar", 0), ("bar", 1)]]
    polys = instance_polygons(label, names)
    assert len(polys) == 1
    cls, poly, npx = polys[0]
    assert cls == 0 and npx == 900
    assert poly.shape[1] == 2 and poly.min() >= 0.1 - 1e-6 and poly.max() <= 0.4
