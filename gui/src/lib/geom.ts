// Projection from world coordinates to image pixels, matching langgrasp/perception/depth_fusion.py:
// the MuJoCo camera looks along its -z axis, +y is image up, +x is image right, and the depth image holds the
// perpendicular distance in metres. Getting this wrong would draw the grasp marker in the wrong place, so the
// inverse of the back-projection is written out rather than approximated.

export type Mat3 = number[][];
export type Mat4 = number[][];

export interface Camera {
  K: Mat3;
  T: Mat4; // T_world_cam
}

export function worldToPixel(p: readonly number[], cam: Camera): { u: number; v: number; depth: number } | null {
  const { K, T } = cam;
  const dx = p[0] - T[0][3];
  const dy = p[1] - T[1][3];
  const dz = p[2] - T[2][3];
  // p_cam = R^T * (p_world - t)
  const xc = T[0][0] * dx + T[1][0] * dy + T[2][0] * dz;
  const yc = T[0][1] * dx + T[1][1] * dy + T[2][1] * dz;
  const zc = T[0][2] * dx + T[1][2] * dy + T[2][2] * dz;
  const depth = -zc;
  if (depth <= 1e-6) return null; // behind the camera
  const fx = K[0][0];
  const fy = K[1][1];
  const cx = K[0][2];
  const cy = K[1][2];
  return { u: cx + (fx * xc) / depth - 0.5, v: cy - (fy * yc) / depth - 0.5, depth };
}

/** The jaw axis as two world points either side of the grasp centre, for the yaw and width markers. */
export function jawEndpoints(center: readonly number[], psi: number, width: number): [number[], number[]] {
  const half = Math.max(width, 0.005) / 2;
  const c = Math.cos(psi);
  const s = Math.sin(psi);
  return [
    [center[0] - half * c, center[1] - half * s, center[2]],
    [center[0] + half * c, center[1] + half * s, center[2]],
  ];
}

/** An arrow along the approach yaw, 4 cm long, for the "which way the jaw closes" marker. */
export function yawArrow(center: readonly number[], psi: number, length = 0.04): [number[], number[]] {
  return [
    [center[0], center[1], center[2]],
    [center[0] + length * Math.cos(psi), center[1] + length * Math.sin(psi), center[2]],
  ];
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export function toBox(b: readonly number[]): Box {
  return { x1: b[0], y1: b[1], x2: b[2], y2: b[3] };
}

/** The 12 edges of an axis-aligned world box, for the geofence overlay. */
export function boxEdges(lo: readonly number[], hi: readonly number[]): [number[], number[]][] {
  const c: number[][] = [];
  for (const x of [lo[0], hi[0]]) for (const y of [lo[1], hi[1]]) for (const z of [lo[2], hi[2]]) c.push([x, y, z]);
  const idx: [number, number][] = [
    [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
    [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
  ];
  return idx.map(([a, b]) => [c[a], c[b]] as [number[], number[]]);
}

// The monitor's default TCP geofence (langgrasp/safety/watchdog.py SafetyConfig). Shown only when the layer is
// on, and labelled, because it is a configuration value rather than a measurement.
export const GEOFENCE_LO = [-0.25, -0.34, 0.0];
export const GEOFENCE_HI = [0.28, 0.02, 0.25];
