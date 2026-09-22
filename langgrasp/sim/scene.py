"""Builds the MuJoCo scene: SO-ARM100 (Apache-2.0, MuJoCo Menagerie) on a table with a pool of graspable objects.

The object pool is static (MuJoCo models cannot add bodies at runtime). Each episode places a subset of the
pool in the workspace and parks the rest out of view. Colors and light are changed at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets", "so_arm100")

# Color name -> rgba. "Seen" colors appear in demos and in the YOLO fine-tune set; "unseen" only at evaluation.
SEEN_COLORS = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.65, 0.15, 1.0),
    "blue": (0.10, 0.25, 0.90, 1.0),
    "yellow": (0.95, 0.85, 0.10, 1.0),
}
UNSEEN_COLORS = {
    "purple": (0.55, 0.15, 0.75, 1.0),
    "orange": (0.95, 0.50, 0.05, 1.0),
    "pink": (0.95, 0.45, 0.70, 1.0),
}
ALL_COLORS = {**SEEN_COLORS, **UNSEEN_COLORS}

# Object kinds. "seen" kinds are in the YOLO training set; "bar" is a held-out shape.
OBJECT_KINDS = {
    "cube": {"seen": True, "grasp_z": 0.0125, "height": 0.025, "width": 0.025},
    "screwdriver": {"seen": True, "grasp_z": 0.011, "height": 0.022, "width": 0.022},
    "can": {"seen": True, "grasp_z": 0.030, "height": 0.060, "width": 0.026},
    "bar": {"seen": False, "grasp_z": 0.0125, "height": 0.025, "width": 0.025},
}
SEEN_KINDS = [k for k, v in OBJECT_KINDS.items() if v["seen"]]
UNSEEN_KINDS = [k for k, v in OBJECT_KINDS.items() if not v["seen"]]

# Pool: (kind, slot index). Names are f"{kind}_{i}".
POOL = [("cube", i) for i in range(4)] + [("screwdriver", i) for i in range(2)] + [("can", i) for i in range(2)] + [("bar", i) for i in range(2)]

# Workspace (table frame, metres). The arm base is at the origin and reaches toward -y.
WORKSPACE = {"x": (-0.10, 0.10), "y": (-0.26, -0.17)}  # strict top-down reach verified inside r <= 0.27 m
TRAY_CENTER = (0.17, -0.16, 0.0)
TRAY_HALF = 0.065
TRAY_RIM = 0.006
PARK_POS = (1.5, 1.5)  # out of every camera's view


@dataclass
class SceneConfig:
    timestep: float = 0.002
    cam_width: int = 640
    cam_height: int = 480
    front_cam_pos: tuple = (0.0, -0.62, 0.52)
    front_cam_fovy: float = 42.0
    side_cam_pos: tuple = (0.55, -0.23, 0.30)
    side_cam_fovy: float = 45.0
    wrist_cam_fovy: float = 75.0
    n_lights: int = 2
    extra_xml: str = ""
    object_names: list = field(default_factory=lambda: [f"{k}_{i}" for k, i in POOL])


def _object_body(kind: str, idx: int) -> str:
    name = f"{kind}_{idx}"
    x0, y0 = PARK_POS
    px, py = x0 + 0.1 * idx, y0 + 0.1 * (0 if kind == "cube" else 1)
    if kind == "cube":
        return (
            f'<body name="{name}" pos="{px} {py} 0.0125"><freejoint name="{name}_free"/>'
            f'<geom name="{name}_g0" type="box" size="0.0125 0.0125 0.0125" rgba="0.8 0.1 0.1 1" mass="0.03" '
            f'friction="1 0.005 0.0001" class="obj"/></body>'
        )
    if kind == "bar":
        return (
            f'<body name="{name}" pos="{px} {py} 0.0125"><freejoint name="{name}_free"/>'
            f'<geom name="{name}_g0" type="box" size="0.03 0.0125 0.0125" rgba="0.8 0.1 0.1 1" mass="0.05" '
            f'friction="1 0.005 0.0001" class="obj"/></body>'
        )
    if kind == "can":
        return (
            f'<body name="{name}" pos="{px} {py} 0.03"><freejoint name="{name}_free"/>'
            f'<geom name="{name}_g0" type="cylinder" size="0.013 0.03" rgba="0.8 0.1 0.1 1" mass="0.04" '
            f'friction="1 0.005 0.0001" class="obj"/></body>'
        )
    if kind == "screwdriver":
        # handle (colored) + steel shaft, long axis along body x, lying on the table
        return (
            f'<body name="{name}" pos="{px} {py} 0.011"><freejoint name="{name}_free"/>'
            f'<geom name="{name}_g0" type="cylinder" size="0.011 0.032" pos="-0.03 0 0" euler="0 1.5708 0" '
            f'rgba="0.8 0.1 0.1 1" mass="0.03" friction="1 0.005 0.0001" class="obj"/>'
            f'<geom name="{name}_g1" type="cylinder" size="0.0035 0.045" pos="0.045 0 0" euler="0 1.5708 0" '
            f'rgba="0.75 0.75 0.78 1" mass="0.01" friction="1 0.005 0.0001" class="obj"/>'
            f"</body>"
        )
    raise ValueError(kind)


def build_scene_xml(cfg: SceneConfig | None = None) -> str:
    cfg = cfg or SceneConfig()
    fx, fy, fz = cfg.front_cam_pos
    sx, sy, sz = cfg.side_cam_pos
    tx, ty, tz = TRAY_CENTER
    objects = "\n".join(_object_body(k, i) for k, i in POOL)
    lights = "\n".join(
        f'<light name="light{i}" pos="{0.4 * (-1) ** i} -0.3 1.2" dir="{-0.3 * (-1) ** i} 0.15 -1" diffuse="0.7 0.7 0.7" '
        f'specular="0.2 0.2 0.2" castshadow="{"true" if i == 0 else "false"}"/>'
        for i in range(cfg.n_lights)
    )
    return f"""
<mujoco model="langgrasp">
  <option timestep="{cfg.timestep}" integrator="implicitfast"/>
  <include file="so_arm100.xml"/>
  <visual>
    <global offwidth="{cfg.cam_width}" offheight="{cfg.cam_height}"/>
    <map znear="0.01" zfar="3.0"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="table_tex" type="2d" builtin="flat" rgb1="0.62 0.60 0.56" width="64" height="64"/>
    <material name="table_mat" texture="table_tex" texrepeat="1 1" reflectance="0.0"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.6 0.7" rgb2="0.25 0.3 0.4" width="64" height="64"/>
  </asset>
  <default>
    <default class="obj">
      <geom condim="4" solimp="0.95 0.99 0.001" solref="0.005 1" group="1"/>
    </default>
  </default>
  <worldbody>
    {lights}
    <geom name="table" type="plane" size="1.0 1.0 0.02" material="table_mat" friction="1 0.005 0.0001" condim="4"/>
    <geom name="tray" type="box" pos="{tx} {ty} {tz + 0.002}" size="{TRAY_HALF} {TRAY_HALF} 0.002" rgba="0.25 0.25 0.28 1"
          contype="1" conaffinity="1"/>
    <geom name="tray_rim_n" type="box" pos="{tx} {ty + TRAY_HALF} {tz + TRAY_RIM / 2}" size="{TRAY_HALF + 0.004} 0.004 {TRAY_RIM / 2}" rgba="0.9 0.9 0.9 1"/>
    <geom name="tray_rim_s" type="box" pos="{tx} {ty - TRAY_HALF} {tz + TRAY_RIM / 2}" size="{TRAY_HALF + 0.004} 0.004 {TRAY_RIM / 2}" rgba="0.9 0.9 0.9 1"/>
    <geom name="tray_rim_e" type="box" pos="{tx + TRAY_HALF} {ty} {tz + TRAY_RIM / 2}" size="0.004 {TRAY_HALF + 0.004} {TRAY_RIM / 2}" rgba="0.9 0.9 0.9 1"/>
    <geom name="tray_rim_w" type="box" pos="{tx - TRAY_HALF} {ty} {tz + TRAY_RIM / 2}" size="0.004 {TRAY_HALF + 0.004} {TRAY_RIM / 2}" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="{fx} {fy} {fz}" mode="targetbody" target="workspace_center" fovy="{cfg.front_cam_fovy}"/>
    <camera name="side" pos="{sx} {sy} {sz}" mode="targetbody" target="workspace_center" fovy="{cfg.side_cam_fovy}"/>
    <body name="workspace_center" pos="0 -0.23 0.0" mocap="true"><site name="workspace_center" size="0.001" rgba="0 0 0 0"/></body>
    {objects}
    {cfg.extra_xml}
  </worldbody>
</mujoco>
"""


def arm_xml_with_tcp(cfg: SceneConfig | None = None) -> str:
    """Return the Menagerie arm XML with a TCP site and a wrist camera added to the Fixed_Jaw body."""
    cfg = cfg or SceneConfig()
    with open(os.path.join(ASSET_DIR, "so_arm100.xml")) as f:
        xml = f.read()
    marker = '<geom type="mesh" mesh="Fixed_Jaw" class="visual"/>'
    assert marker in xml
    extra = (
        marker
        + '\n                <site name="tcp" pos="0 -0.097 0" size="0.003" rgba="0 1 0 0"/>'
        + f'\n                <camera name="wrist" pos="0.0 0.02 0.06" xyaxes="0 -0.42 0.91 1 0 0" fovy="{cfg.wrist_cam_fovy}"/>'
    )
    return xml.replace(marker, extra)
