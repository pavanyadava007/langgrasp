import { describe, expect, it } from "vitest";
import { boxEdges, jawEndpoints, worldToPixel, yawArrow, type Camera } from "../geom";

// Intrinsics and extrinsics of the front camera as the worker reports them (640x480, fovy 42 degrees, the
// camera aimed at the workspace centre). Kept as literals so the test does not need a running server.
// Intrinsics and extrinsics of the front camera as the worker reports them: 640x480, 42 degree vertical field
// of view, camera behind and above the table looking down at the workspace. The rotation is built from an
// angle so that it is orthonormal to double precision; worldToPixel inverts it by transposing, as MuJoCo's
// own convention allows, and a sloppy matrix here would make an exact round trip impossible.
const TILT = (50 * Math.PI) / 180;
const C = Math.cos(TILT);
const S = Math.sin(TILT);
const CAM: Camera = {
  K: [
    [624.5, 0, 320],
    [0, 624.5, 240],
    [0, 0, 1],
  ],
  T: [
    [1, 0, 0, 0],
    [0, C, -S, -0.62],
    [0, S, C, 0.52],
    [0, 0, 0, 1],
  ],
};

/** The inverse of langgrasp/perception/depth_fusion.py: depth_to_points, written out here as the reference. */
function pixelToWorld(u: number, v: number, depth: number, cam: Camera): number[] {
  const [fx, fy, cx, cy] = [cam.K[0][0], cam.K[1][1], cam.K[0][2], cam.K[1][2]];
  const x = ((u + 0.5 - cx) / fx) * depth;
  const y = (-(v + 0.5 - cy) / fy) * depth;
  const z = -depth;
  const R = cam.T;
  return [
    R[0][0] * x + R[0][1] * y + R[0][2] * z + R[0][3],
    R[1][0] * x + R[1][1] * y + R[1][2] * z + R[1][3],
    R[2][0] * x + R[2][1] * y + R[2][2] * z + R[2][3],
  ];
}

describe("worldToPixel", () => {
  it("inverts the simulator's back-projection", () => {
    // If these disagree, every overlay is drawn in the wrong place while still looking plausible.
    for (const [u, v, d] of [
      [320, 240, 0.8],
      [100, 400, 0.65],
      [560, 80, 1.1],
      [0, 0, 0.9],
      [639, 479, 0.7],
    ]) {
      const world = pixelToWorld(u, v, d, CAM);
      const back = worldToPixel(world, CAM);
      expect(back).not.toBeNull();
      expect(back!.u).toBeCloseTo(u, 6);
      expect(back!.v).toBeCloseTo(v, 6);
      expect(back!.depth).toBeCloseTo(d, 9);
    }
  });

  it("returns null for a point behind the camera instead of drawing it in front", () => {
    const behind = [0, -0.62 + 1.0, 0.52 + 1.2];
    expect(worldToPixel(behind, CAM)).toBeNull();
  });

  it("puts a point further from the camera nearer the principal point", () => {
    const near = worldToPixel(pixelToWorld(500, 300, 0.6, CAM), CAM)!;
    const far = worldToPixel(pixelToWorld(500, 300, 1.2, CAM), CAM)!;
    expect(far.depth).toBeGreaterThan(near.depth);
  });
});

describe("grasp markers", () => {
  it("places the jaw endpoints half a width either side along the yaw", () => {
    const [a, b] = jawEndpoints([0.1, -0.2, 0.015], 0, 0.04);
    expect(a[0]).toBeCloseTo(0.08, 9);
    expect(b[0]).toBeCloseTo(0.12, 9);
    expect(a[1]).toBeCloseTo(-0.2, 9);
    expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBeCloseTo(0.04, 9);
  });

  it("never collapses the jaw markers to a point for a zero width", () => {
    const [a, b] = jawEndpoints([0, 0, 0], Math.PI / 2, 0);
    expect(Math.hypot(b[0] - a[0], b[1] - a[1])).toBeGreaterThan(0.004);
  });

  it("points the yaw arrow along psi", () => {
    const [o, tip] = yawArrow([0, 0, 0], Math.PI / 2, 0.04);
    expect(tip[0] - o[0]).toBeCloseTo(0, 9);
    expect(tip[1] - o[1]).toBeCloseTo(0.04, 9);
  });
});

describe("boxEdges", () => {
  it("returns the twelve edges of a box", () => {
    const edges = boxEdges([0, 0, 0], [1, 2, 3]);
    expect(edges).toHaveLength(12);
    const lengths = edges.map(([a, b]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
    expect(lengths.filter((l) => Math.abs(l - 1) < 1e-9)).toHaveLength(4);
    expect(lengths.filter((l) => Math.abs(l - 2) < 1e-9)).toHaveLength(4);
    expect(lengths.filter((l) => Math.abs(l - 3) < 1e-9)).toHaveLength(4);
  });
});
