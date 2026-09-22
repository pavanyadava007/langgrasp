"""Episode scenarios: which objects, where, what colour, what command, which evaluation stratum."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from langgrasp.sim.scene import OBJECT_KINDS, POOL, SEEN_COLORS, SEEN_KINDS, UNSEEN_COLORS, UNSEEN_KINDS, WORKSPACE

STRATA = ("seen", "unseen", "langvar")
LIGHTING = ("nominal", "degraded")

KIND_SYNONYMS = {
    "cube": ["cube", "block"],
    "screwdriver": ["screwdriver", "driver", "tool"],
    "can": ["can", "cylinder"],
    "bar": ["bar", "brick"],
}


@dataclass
class PlacedObject:
    name: str  # pool name, e.g. cube_0
    kind: str
    color: str
    pos: tuple  # x, y (z from kind)
    yaw: float


@dataclass
class Scenario:
    stratum: str
    objects: list = field(default_factory=list)
    target: str = ""  # pool name of the target
    command: str = ""
    lang_variant: str = "plain"  # plain | synonym | attribute | spatial
    lighting: str = "nominal"
    seed: int = 0

    @property
    def target_obj(self) -> PlacedObject:
        return next(o for o in self.objects if o.name == self.target)

    def to_dict(self) -> dict:
        return {
            "stratum": self.stratum,
            "target": self.target,
            "command": self.command,
            "lang_variant": self.lang_variant,
            "lighting": self.lighting,
            "seed": self.seed,
            "objects": [o.__dict__ for o in self.objects],
        }


# footprint per kind: (radius of the round part, half-length of the long axis) used for collision-free spawning
FOOTPRINT = {"cube": (0.018, 0.0), "bar": (0.015, 0.03), "can": (0.014, 0.0), "screwdriver": (0.012, 0.062)}
SPAWN_MARGIN = 0.015
SAFE_ZONE = {"x": (-0.13, 0.105), "y": (-0.30, -0.15)}  # no part of any object outside this (tray starts at x=0.125)


def _segment_points(c: tuple, yaw: float, half_len: float, n: int = 7) -> np.ndarray:
    ts = np.linspace(-half_len, half_len, n) if half_len > 0 else np.zeros(1)
    return np.stack([c[0] + ts * np.cos(yaw), c[1] + ts * np.sin(yaw)], axis=1)


def _sample_positions(rng: np.random.Generator, kinds: list[str]) -> list[tuple[tuple[float, float], float]]:
    """Spawn (centre, yaw) per object so that oriented footprints do not overlap and stay inside SAFE_ZONE."""
    (x0, x1), (y0, y1) = WORKSPACE["x"], WORKSPACE["y"]
    (sx0, sx1), (sy0, sy1) = SAFE_ZONE["x"], SAFE_ZONE["y"]
    placed: list[tuple[tuple[float, float], float]] = []
    pts_cache: list[np.ndarray] = []
    for kind in kinds:
        r, hl = FOOTPRINT[kind]
        for _ in range(2000):
            c = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
            yaw = float(rng.uniform(-np.pi, np.pi))
            pts = _segment_points(c, yaw, hl)
            if pts[:, 0].min() - r < sx0 or pts[:, 0].max() + r > sx1 or pts[:, 1].min() - r < sy0 or pts[:, 1].max() + r > sy1:
                continue
            ok = True
            for okind, opts in zip(kinds, pts_cache, strict=False):
                orad = FOOTPRINT[okind][0]
                dmin = np.min(np.linalg.norm(pts[:, None, :] - opts[None, :, :], axis=-1))
                if dmin < r + orad + SPAWN_MARGIN:
                    ok = False
                    break
            if ok:
                placed.append((c, yaw))
                pts_cache.append(pts)
                break
        else:
            raise RuntimeError("could not place objects")
    return placed


def _slot(kind: str, used: set) -> str:
    for k, i in POOL:
        name = f"{k}_{i}"
        if k == kind and name not in used:
            used.add(name)
            return name
    raise RuntimeError(f"no free slot for {kind}")


def make_scenario(seed: int, stratum: str = "seen", n_distractors: int | None = None, lighting: str | None = None, target_kind: str | None = None, target_color: str | None = None) -> Scenario:
    """Deterministic scenario from a seed. Strata:
    seen     : target kind and colour in the training sets, plain command "pick the {color} {kind}"
    unseen   : novel colour or novel kind (bar) for the target
    langvar  : seen objects, but the command uses a synonym, an attribute-only phrase, or a spatial reference
    target_kind/target_color force the target identity (fixed-goal task for policies without language input).
    """
    assert stratum in STRATA
    rng = np.random.default_rng(seed)
    n_dis = int(rng.integers(1, 3)) if n_distractors is None else n_distractors
    lighting = lighting or ("degraded" if rng.random() < 0.25 else "nominal")
    used: set = set()
    objs: list[PlacedObject] = []
    seen_colors = list(SEEN_COLORS)
    lang_variant = "plain"

    if stratum == "unseen":
        if rng.random() < 0.5:
            t_kind, t_color = str(rng.choice(SEEN_KINDS)), str(rng.choice(list(UNSEEN_COLORS)))
        else:
            t_kind, t_color = str(rng.choice(UNSEEN_KINDS)), str(rng.choice(seen_colors))
    else:
        t_kind, t_color = str(rng.choice(SEEN_KINDS)), str(rng.choice(seen_colors))
    # fixed-goal variant (used for ACT, which is not language-conditioned): force the target identity
    t_kind = target_kind or t_kind
    t_color = target_color or t_color

    spatial_pair = stratum == "langvar" and rng.random() < 0.34
    dis: list[tuple[str, str]] = []
    for i in range(n_dis):
        if spatial_pair and i == 0:
            dis.append((t_kind, t_color))  # same object twice: only a spatial reference disambiguates
        else:
            d_kind = str(rng.choice(SEEN_KINDS))
            choices = [c for c in seen_colors if not (d_kind == t_kind and c == t_color)]
            dis.append((d_kind, str(rng.choice(choices))))
    while True:
        try:
            positions = _sample_positions(rng, [t_kind] + [k for k, _ in dis])
            break
        except RuntimeError:
            if not dis:
                raise
            dis = dis[:-1]  # crowded zone: drop the last distractor
    target_name = _slot(t_kind, used)
    objs.append(PlacedObject(target_name, t_kind, t_color, positions[0][0], positions[0][1]))
    for i, (d_kind, d_color) in enumerate(dis):
        try:
            d_name = _slot(d_kind, used)
        except RuntimeError:
            continue
        objs.append(PlacedObject(d_name, d_kind, d_color, positions[1 + i][0], positions[1 + i][1]))

    if stratum == "langvar":
        if spatial_pair:
            lang_variant = "spatial"
            # "left" is image-left in the front camera, which is world -x
            tgt, dup = objs[0], objs[1]
            side = "left" if tgt.pos[0] < dup.pos[0] else "right"
            command = f"pick the {t_color} {t_kind} on the {side}"
        elif rng.random() < 0.5:
            lang_variant = "synonym"
            syn = KIND_SYNONYMS[t_kind][1:]
            command = f"pick the {t_color} {str(rng.choice(syn))}"
        else:
            lang_variant = "attribute"
            command = f"grab the {t_color} one" if all(o.color != t_color for o in objs[1:]) else f"pick the {t_color} object"
    else:
        command = f"pick the {t_color} {t_kind}"

    return Scenario(stratum=stratum, objects=objs, target=target_name, command=command, lang_variant=lang_variant, lighting=lighting, seed=seed)


def kind_height(kind: str) -> float:
    return OBJECT_KINDS[kind]["height"]
