"""Depth fusion: back-project a segmentation mask through the depth image and derive a top-down grasp.

Camera convention is MuJoCo's: the camera looks along its -z axis, +y is image up, +x is image right. The depth
image holds the perpendicular (z-buffer) distance in metres. Pixel (u, v) is the column/row index with the
principal point at the image centre.
"""

from __future__ import annotations

import numpy as np


def depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    mask: np.ndarray | None = None,
    max_depth: float = 2.0,
) -> np.ndarray:
    """Back-project depth (HxW metres) to (N,3) world points. Pixels with depth <= 0 or > max_depth are dropped."""
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    valid = (depth > 0.0) & (depth <= max_depth) & np.isfinite(depth)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(valid)
    if len(u) == 0:
        return np.zeros((0, 3))
    d = depth[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u + 0.5 - cx) / fx * d
    y = -(v + 0.5 - cy) / fy * d
    z = -d
    p_cam = np.stack([x, y, z], axis=1)
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    return p_cam @ R.T + t


def realsense_like_noise(
    depth: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.002,
    dropout: float = 0.02,
    quant: float = 0.001,
) -> np.ndarray:
    """Simple SYNTHETIC depth-noise model: additive Gaussian noise (sigma, metres), random pixel dropout (set to
    0 = invalid) and quantisation to `quant` metres. This is a stand-in for a stereo depth camera; it is NOT a
    validated RealSense noise model (no depth-dependent sigma, no edge flying pixels, no holes at specular
    surfaces)."""
    d = np.asarray(depth, dtype=np.float32).copy()
    d += rng.normal(0.0, sigma, size=d.shape).astype(np.float32)
    if quant > 0:
        d = np.round(d / quant) * quant
    if dropout > 0:
        d[rng.random(d.shape) < dropout] = 0.0
    return d


def largest_cluster(points: np.ndarray, cell: float = 0.006) -> np.ndarray:
    """Keep only the largest 8-connected cluster of points on an xy grid of `cell` metres. A detector mask is
    only cropped to its box, so with overlapping boxes it can carry a blob of a neighbouring object; that blob
    is a separate cluster in xy and is removed here."""
    import cv2

    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    xy = pts[:, :2]
    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(np.int64)
    h, w = int(ij[:, 1].max()) + 1, int(ij[:, 0].max()) + 1
    grid = np.zeros((h, w), dtype=np.uint8)
    grid[ij[:, 1], ij[:, 0]] = 1
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    if n <= 2:
        return pts
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return pts[lab[ij[:, 1], ij[:, 0]] == best]


def segment_object_points(
    points: np.ndarray, table_z: float = 0.0, min_z: float = 0.004, z_thresh: float = 3.0, keep_largest_cluster: bool = True
) -> np.ndarray:
    """Drop table points (z < table_z + min_z), keep the largest xy cluster (see largest_cluster) and drop
    statistical outliers (robust z-score of distance to the median centre, using the median absolute deviation)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts
    pts = pts[pts[:, 2] >= table_z + min_z]
    if keep_largest_cluster:
        pts = largest_cluster(pts)
    if len(pts) < 8:
        return pts
    for _ in range(2):
        c = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - c, axis=1)
        med = np.median(dist)
        mad = np.median(np.abs(dist - med)) * 1.4826 + 1e-9
        keep = (dist - med) / mad <= z_thresh
        if keep.all():
            break
        pts = pts[keep]
    return pts


def _pca_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T) + 1e-12 * np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return c, vecs[:, order[0]], vecs[:, order[1]]


def _extent(vals: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    a, b = np.percentile(vals, [lo, hi])
    return float(a), float(b)


def _elongated_grasp(
    xy: np.ndarray, z: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray, s: np.ndarray, t: np.ndarray, bin_size: float, min_handle_width: float
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Grasp at the thickest segment along the long axis u (the handle of a screwdriver).

    Returns (center_xy, psi, width, z of the handle points). Bins the projection s along u in bin_size slices,
    measures the spread across v per bin and takes the longest run of thick bins as the handle."""
    s0, s1 = _extent(s, 5, 95)
    edges = np.arange(s0, s1 + bin_size, bin_size)
    idx = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)
    widths = np.zeros(len(edges) - 1)
    for b in range(len(widths)):
        tb = t[idx == b]
        if len(tb) >= 3:
            lo, hi = _extent(tb, 5, 95)
            widths[b] = hi - lo
    # Split bins into thin (shaft) and thick (handle) with an Otsu-style two-cluster split (minimum total
    # within-cluster variance of the sorted widths). An absolute threshold does not work: the visible top of
    # a lying cylinder of radius r is only 2 r sin(elevation) wide, so the 22 mm handle measures 10 to 18 mm
    # depending on the camera, and the bin at the handle end cap can be wider than the handle itself.
    valid = widths > 0
    srt = np.sort(widths[valid])
    thr = 0.0
    if len(srt) >= 2:
        best_cost, best_k = np.inf, 0
        for k in range(1, len(srt)):
            cost = srt[:k].var() * k + srt[k:].var() * (len(srt) - k)
            if cost < best_cost:
                best_cost, best_k = cost, k
        lo_med, hi_med = np.median(srt[:best_k]), np.median(srt[best_k:])
        if hi_med >= 1.5 * max(lo_med, 1e-6):
            thr = 0.5 * (srt[best_k - 1] + srt[best_k])
    wide = valid & (widths >= max(thr, min_handle_width))
    if not wide.any():
        wide = valid
    # largest contiguous run of wide bins = the handle
    best, cur = [], []
    for b in range(len(wide)):
        if wide[b]:
            cur.append(b)
        else:
            best, cur = (cur if len(cur) > len(best) else best), []
    best = cur if len(cur) > len(best) else best
    in_handle = np.isin(idx, best)
    hs, ht = s[in_handle], t[in_handle]
    ht0, ht1 = _extent(ht)
    t_mid = 0.5 * (ht0 + ht1)
    # points clearly off the centre line belong to the thick handle, not the thin shaft
    off = np.abs(ht - t_mid) > 0.25 * (ht1 - ht0)
    hs_ext = _extent(hs[off]) if off.sum() >= 5 else _extent(hs)
    s_mid = 0.5 * (hs_ext[0] + hs_ext[1])
    center_xy = c + u * s_mid + v * t_mid
    psi = float(np.arctan2(u[1], u[0]) + np.pi / 2)
    # jaw width: spread across the jaw axis of the points within +/- 1 cm of the grasp centre along the long axis
    near = in_handle & (np.abs(s - s_mid) <= 0.01)
    if near.sum() < 5:
        near = in_handle
    w0, w1 = _extent(t[near])
    return center_xy, psi, float(w1 - w0), z[in_handle]


def grasp_from_points(
    points: np.ndarray,
    table_z: float = 0.0,
    elongated_aspect: float = 2.0,
    bin_size: float = 0.01,
    min_handle_width: float = 0.008,
    tall_height: float = 0.045,
    max_compact_width: float = 0.05,
    max_compact_length: float = 0.08,
) -> dict:
    """Top-down grasp from segmented object points (world frame, table at z = table_z).

    Returns dict(center xyz, psi, width, long_axis xyz, aspect, n, grasp_z, elongated, fallback). psi is the
    jaw-axis yaw: the jaw closes along (cos psi, sin psi, 0). The aspect ratio is the 5th to 95th percentile
    EXTENT along the PCA major axis divided by the extent along the minor axis (an eigenvalue ratio would
    under-estimate elongation when the point density is dominated by a thick handle). Elongated objects
    (aspect >= elongated_aspect, e.g. screwdriver) are grasped at their thickest segment along the long axis
    (the handle) across the long axis; compact objects use cv2.minAreaRect and close the jaw across the
    shorter rectangle side. If the compact path yields a jaw width above max_compact_width, or a rectangle longer
    than max_compact_length (no cube, can or bar is that long), the object cannot be compact, so the elongated
    path is used as a safety net (fallback=True). Only the largest xy
    cluster of the points is used (see largest_cluster).
    """
    import cv2

    pts = largest_cluster(np.asarray(points, dtype=np.float64))
    n = len(pts)
    if n < 5:
        raise ValueError(f"too few points for a grasp: {n}")
    xy = pts[:, :2]
    c, u, v = _pca_xy(xy)
    s = (xy - c) @ u
    t = (xy - c) @ v
    s0, s1 = _extent(s, 5, 95)
    t0, t1 = _extent(t, 5, 95)
    aspect = (s1 - s0) / max(t1 - t0, 1e-6)
    elongated = aspect >= elongated_aspect
    fallback = False
    z_max = float(np.percentile(pts[:, 2], 98))

    if not elongated:
        rect = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
        (rcx, rcy), (rw, rh), ang = rect
        a = np.deg2rad(ang)
        w_dir = np.array([np.cos(a), np.sin(a)])
        h_dir = np.array([-np.sin(a), np.cos(a)])
        short_dir = w_dir if rw <= rh else h_dir
        psi = float(np.arctan2(short_dir[1], short_dir[0]))
        center_xy = np.array([rcx, rcy]) / 1000.0
        jaw = np.array([np.cos(psi), np.sin(psi)])
        proj = (xy - center_xy) @ jaw
        p0, p1 = _extent(proj)
        width = float(p1 - p0)
        z_sel = pts[:, 2]
        if width > max_compact_width or max(rw, rh) / 1000.0 > max_compact_length:
            elongated, fallback = True, True
    if elongated:
        center_xy, psi, width, z_sel = _elongated_grasp(xy, pts[:, 2], c, u, v, s, t, bin_size, min_handle_width)

    median_z = float(np.median(z_sel))
    if z_max - table_z > tall_height:
        grasp_z = z_max - 0.03
    else:
        grasp_z = 0.5 * (z_max + table_z)  # mid-height of an object resting on the table
    psi = float((psi + np.pi) % (2 * np.pi) - np.pi)
    return {
        "center": np.array([center_xy[0], center_xy[1], grasp_z]),
        "psi": psi,
        "width": float(width),
        "long_axis": np.array([u[0], u[1], 0.0]),
        "aspect": float(aspect),
        "n": int(n),
        "grasp_z": float(grasp_z),
        "median_z": median_z,
        "max_z": z_max,
        "elongated": bool(elongated),
        "fallback": fallback,
    }


def mask_from_box(box, shape: tuple[int, int]) -> np.ndarray:
    """Boolean HxW mask that is True inside the pixel box [x1, y1, x2, y2] (clipped to the image)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = [int(round(float(b))) for b in box]
    m = np.zeros((h, w), dtype=bool)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


def intersect_mask_with_box(mask: np.ndarray, box) -> np.ndarray:
    """Zero every mask pixel outside the box (removes stray mask blobs from the prototype decoding)."""
    return np.asarray(mask, dtype=bool) & mask_from_box(box, mask.shape)


def evaluate_on_sim(n_scenes: int = 60, cameras: tuple = ("front", "side"), noise: bool = False, seed: int = 0) -> dict:
    """Measure grasp_from_points against env.grasp_point on rendered scenes (ground-truth masks from the label
    map). Returns per-kind centre and psi error statistics. psi error is taken modulo pi (cube: modulo pi/2)
    and is not reported for cans (rotationally symmetric, any psi is valid)."""
    from langgrasp.sim.env import LangGraspEnv
    from langgrasp.sim.scenarios import make_scenario

    env = LangGraspEnv(seed=seed)
    rng = np.random.default_rng(seed)
    names = env.object_names()
    rows = []
    for i in range(n_scenes):
        try:
            sc = make_scenario(seed * 100_000 + i, "seen")
        except RuntimeError:  # make_scenario can fail to place objects for rare seeds; skip that seed
            continue
        env.reset(sc)
        for cam in cameras:
            _, depth, label = env.render_rgbd_seg(cam)
            if noise:
                depth = realsense_like_noise(depth, rng)
            K, T = env.camera_intrinsics(cam), env.camera_extrinsics(cam)
            for o in sc.objects:
                m = label == names.index(o.name)
                if m.sum() < 200:
                    continue
                pts = segment_object_points(depth_to_points(depth, K, T, m))
                if len(pts) < 20:
                    continue
                g = grasp_from_points(pts)
                gt_p, gt_psi = env.grasp_point(o.name)
                period = np.pi / 2 if o.kind == "cube" else np.pi
                dpsi = (g["psi"] - gt_psi + period / 2) % period - period / 2
                rows.append(
                    {
                        "kind": o.kind,
                        "camera": cam,
                        "xy_err_mm": float(np.linalg.norm(g["center"][:2] - gt_p[:2]) * 1000),
                        "z_err_mm": float(abs(g["grasp_z"] - gt_p[2]) * 1000),
                        "psi_err_deg": float(abs(np.degrees(dpsi))),
                        "width_mm": g["width"] * 1000,
                        "aspect": g["aspect"],
                        "n": g["n"],
                    }
                )
    out = {"n_scenes": n_scenes, "cameras": list(cameras), "noise": noise, "n_objects": len(rows), "per_kind": {}}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        xy = np.array([r["xy_err_mm"] for r in rs])
        z = np.array([r["z_err_mm"] for r in rs])
        psi = np.array([r["psi_err_deg"] for r in rs])
        d = {
            "n": len(rs),
            "xy_err_mm_mean": float(xy.mean()),
            "xy_err_mm_median": float(np.median(xy)),
            "xy_err_mm_p95": float(np.percentile(xy, 95)),
            "xy_err_mm_max": float(xy.max()),
            "frac_xy_err_lt_8mm": float((xy < 8.0).mean()),
            "z_err_mm_mean": float(z.mean()),
            "z_err_mm_max": float(z.max()),
            "width_mm_mean": float(np.mean([r["width_mm"] for r in rs])),
            "aspect_mean": float(np.mean([r["aspect"] for r in rs])),
        }
        if kind != "can":
            d.update(
                {
                    "psi_err_deg_mean": float(psi.mean()),
                    "psi_err_deg_median": float(np.median(psi)),
                    "psi_err_deg_p95": float(np.percentile(psi, 95)),
                    "psi_err_deg_max": float(psi.max()),
                    "frac_psi_err_lt_15deg": float((psi < 15.0).mean()),
                }
            )
        out["per_kind"][kind] = d
    return out


if __name__ == "__main__":
    import json
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    res = {"clean": evaluate_on_sim(n, noise=False), "realsense_like_noise": evaluate_on_sim(n, noise=True)}
    res["note"] = (
        "Ground-truth masks from the sim label map; errors vs env.grasp_point. z error compares grasp_z with the "
        "oracle centre height. Noise model is synthetic (see realsense_like_noise docstring), not a validated sensor model."
    )
    print(json.dumps(res, indent=2))
    with open("results/depth_fusion_accuracy.json", "w") as f:
        json.dump(res, f, indent=2)
