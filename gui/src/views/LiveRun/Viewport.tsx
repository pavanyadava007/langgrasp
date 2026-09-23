import { useEffect, useRef, useState } from "react";
import { GEOFENCE_LO, GEOFENCE_HI, boxEdges, jawEndpoints, worldToPixel, yawArrow, type Camera } from "../../lib/geom";
import { frameStore } from "../../lib/ws";
import { tickStore } from "../../lib/ticks";
import { useStore } from "../../store/store";
import type { CameraName, Candidate, GraspPose } from "../../lib/types";
import { Toggle } from "../../components/ui";

// Resting height of each object kind's centre, from langgrasp/sim/scene.py OBJECT_KINDS. Configuration, not a
// measurement: it is only used to place the optional ground-truth markers at the right height.
const GRASP_Z: Record<string, number> = { cube: 0.0125, screwdriver: 0.011, can: 0.03, bar: 0.0125 };

const CSS = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#fff";

function drawLine(ctx: CanvasRenderingContext2D, a: { u: number; v: number }, b: { u: number; v: number }, s: number) {
  ctx.beginPath();
  ctx.moveTo(a.u * s, a.v * s);
  ctx.lineTo(b.u * s, b.v * s);
  ctx.stroke();
}

/** Angles are reported in (-180, 180]: the jaw axis is symmetric under 180 degrees and best_psi can return
 *  psi + k*pi, so the raw number is often outside the range a person expects to read. */
function wrapDeg(rad: number): number {
  let d = (rad * 180) / Math.PI;
  d = ((d + 180) % 360 + 360) % 360 - 180;
  return d;
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  ctx.fillRect(x, y - 13, w + 8, 16);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 4, y - 1);
}

export function Viewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<{ url: string; img: HTMLImageElement } | null>(null);
  const camera = useStore((s) => s.camera);
  const showDepth = useStore((s) => s.showDepth);
  const layers = useStore((s) => s.layers);
  const setLayer = useStore((s) => s.setLayer);
  const setCamera = useStore((s) => s.setCamera);
  const connection = useStore((s) => s.connection);
  const [caption, setCaption] = useState("waiting for the first frame");

  useEffect(() => {
    let raf = 0;
    let lastDraw = "";
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const st = useStore.getState();
      const kind = st.showDepth ? "depth" : "rgb";
      const cam: CameraName = st.showDepth ? "front" : st.camera;
      const frame = frameStore.get(cam, kind);
      const signature = `${cam}${kind}${frameStore.version(cam, kind)}|${tickStore.version()}|${JSON.stringify(st.layers)}|${st.stages.select.payload.box ?? ""}|${st.stages.fuse.latency_ms}|${st.stages.grounding.latency_ms}`;
      if (signature === lastDraw) return;
      lastDraw = signature;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const W = frame?.meta.width ?? 640;
      const H = frame?.meta.height ?? 480;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssW = Math.max(rect.width, 1);
      const scale = cssW / W;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssW * (H / W) * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssW * (H / W));
      ctx.fillStyle = CSS("--surface-2");
      ctx.fillRect(0, 0, cssW, cssW * (H / W));
      if (frame) ctx.drawImage(frame.bitmap, 0, 0, cssW, cssW * (H / W));

      const drawn: string[] = [];
      const stages = st.stages;
      // Boxes, mask and grasp were computed from the frame captured at the start of the command. Once the arm
      // moves, the live image no longer matches them, so they are dimmed and the caption says so: otherwise a
      // viewer sees a box around an object that has already been picked up and reads it as a wrong detection.
      const moved = stages.execute.status === "running" || stages.execute.status === "ok" || stages.execute.status === "warn" || stages.execute.status === "fail";
      const perceptionAlpha = moved ? 0.5 : 1;
      const capture = stages.capture.payload as { K?: number[][]; T_world_cam?: number[][] };
      const camMatrices: Camera | null = capture.K && capture.T_world_cam ? { K: capture.K, T: capture.T_world_cam } : null;
      const projectable = camMatrices !== null && cam === "front" && !st.showDepth;

      ctx.globalAlpha = perceptionAlpha;

      // ---- grounding candidates
      if (st.layers.candidates) {
        const cands = (stages.grounding.payload.candidates as Candidate[] | undefined) ?? [];
        const winnerBox = stages.select.payload.box as number[] | undefined;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        if (cands.length) drawn.push(`candidates:${cands.length}`);
        for (const c of cands) {
          const isWinner = winnerBox && Math.abs(c.box[0] - winnerBox[0]) < 0.01;
          if (isWinner) continue;
          ctx.strokeStyle = CSS("--fg-muted");
          ctx.strokeRect(c.box[0] * scale, c.box[1] * scale, (c.box[2] - c.box[0]) * scale, (c.box[3] - c.box[1]) * scale);
          // below the box, so it cannot collide with the chosen box's label above it
          label(ctx, `${c.label} ${c.score.toFixed(2)} · colour ${c.color_frac.toFixed(2)}`, c.box[0] * scale, c.box[3] * scale + 15, CSS("--fg-muted"));
        }
        ctx.setLineDash([]);
      }

      // ---- winning box
      if (st.layers.winner) {
        const box = stages.select.payload.box as number[] | undefined;
        const score = stages.select.payload.score as number | undefined;
        const colourFrac = stages.select.payload.color_frac as number | undefined;
        if (box) {
          ctx.strokeStyle = CSS("--accent");
          ctx.lineWidth = 2.5;
          ctx.strokeRect(box[0] * scale, box[1] * scale, (box[2] - box[0]) * scale, (box[3] - box[1]) * scale);
          drawn.push("winner");
          const bits = [`${score?.toFixed(2) ?? "?"}`];
          if (colourFrac !== undefined) bits.push(`colour ${colourFrac.toFixed(2)}`);
          label(ctx, `chosen · ${bits.join(" · ")}`, box[0] * scale, box[1] * scale - 2, CSS("--accent"));
        }
      }

      // ---- YOLO mask, loaded from the recorded stage image
      if (st.layers.mask) {
        const url = stages.segment.images.mask;
        if (url && maskRef.current?.url !== url) {
          const img = new Image();
          img.src = url;
          maskRef.current = { url, img };
        }
        const m = maskRef.current;
        if (m?.img.complete && m.img.naturalWidth > 0) {
          const tint = document.createElement("canvas");
          tint.width = m.img.naturalWidth;
          tint.height = m.img.naturalHeight;
          const tctx = tint.getContext("2d");
          if (tctx) {
            tctx.drawImage(m.img, 0, 0);
            tctx.globalCompositeOperation = "multiply";
            tctx.fillStyle = CSS("--accent");
            tctx.fillRect(0, 0, tint.width, tint.height);
            tctx.globalCompositeOperation = "destination-in";
            tctx.drawImage(m.img, 0, 0);
            ctx.save();
            ctx.globalAlpha = 0.45 * perceptionAlpha;
            ctx.drawImage(tint, 0, 0, cssW, cssW * (H / W));
            ctx.restore();
            drawn.push("mask");
          }
        }
      }

      // ---- point cloud
      if (st.layers.points && projectable) {
        const pts = (stages.fuse.payload.points_xyz as number[][] | undefined) ?? [];
        ctx.fillStyle = CSS("--running");
        for (const p of pts) {
          const q = worldToPixel(p, camMatrices!);
          if (q) ctx.fillRect(q.u * scale - 1, q.v * scale - 1, 2, 2);
        }
        if (pts.length) drawn.push(`points:${pts.length}`);
      }

      // ---- grasp pose
      if (st.layers.grasp && projectable) {
        const g = (stages.fuse.payload.grasp as GraspPose | null) ?? null;
        const psiUsed = (stages.execute.payload.psi_used as number | undefined) ?? g?.psi;
        const centerFromExecute = stages.execute.payload.center as number[] | undefined;
        const center = g?.center ?? centerFromExecute;
        if (center && psiUsed !== undefined) {
          const width = (g?.width as number | undefined) ?? (stages.execute.payload.width as number | undefined) ?? 0.025;
          const c = worldToPixel(center, camMatrices!);
          if (c) {
            const ok = CSS("--ok");
            ctx.strokeStyle = ok;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(c.u * scale - 9, c.v * scale);
            ctx.lineTo(c.u * scale + 9, c.v * scale);
            ctx.moveTo(c.u * scale, c.v * scale - 9);
            ctx.lineTo(c.u * scale, c.v * scale + 9);
            ctx.stroke();
            const [a, b] = jawEndpoints(center, psiUsed, width);
            const pa = worldToPixel(a, camMatrices!);
            const pb = worldToPixel(b, camMatrices!);
            if (pa && pb) {
              ctx.lineWidth = 3;
              drawLine(ctx, pa, pb, scale);
              label(ctx, `grasp ${(width * 1000).toFixed(0)} mm · yaw ${wrapDeg(psiUsed).toFixed(0)}°`, c.u * scale + 14, c.v * scale + 26, ok);
              drawn.push("grasp");
            }
            const [o, tip] = yawArrow(center, psiUsed);
            const po = worldToPixel(o, camMatrices!);
            const pt = worldToPixel(tip, camMatrices!);
            if (po && pt) {
              ctx.lineWidth = 1.5;
              ctx.setLineDash([3, 3]);
              drawLine(ctx, po, pt, scale);
              ctx.setLineDash([]);
            }
          }
        }
      }

      ctx.globalAlpha = 1;

      // ---- executed fingertip path (live, not from the capture)
      if (st.layers.waypoints && projectable) {
        const trail = tickStore.trail();
        ctx.strokeStyle = CSS("--warn");
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (const p of trail) {
          const q = worldToPixel(p, camMatrices!);
          if (!q) continue;
          if (!started) {
            ctx.moveTo(q.u * scale, q.v * scale);
            started = true;
          } else ctx.lineTo(q.u * scale, q.v * scale);
        }
        if (started) {
          ctx.stroke();
          drawn.push(`path:${trail.length}`);
        }
      }

      // ---- geofence
      if (st.layers.geofence && projectable) {
        ctx.strokeStyle = CSS("--fail");
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 4]);
        for (const [a, b] of boxEdges(GEOFENCE_LO, GEOFENCE_HI)) {
          const pa = worldToPixel(a, camMatrices!);
          const pb = worldToPixel(b, camMatrices!);
          if (pa && pb) drawLine(ctx, pa, pb, scale);
        }
        ctx.setLineDash([]);
        label(ctx, "TCP geofence (configuration, not a measurement)", 8, 18, CSS("--fail"));
        drawn.push("geofence");
      }

      // ---- ground truth, always dashed, always tagged
      if (st.layers.groundTruth && projectable && st.scene) {
        const gt = CSS("--gt");
        ctx.strokeStyle = gt;
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        for (const o of st.scene.objects) {
          const p = [o.pos[0], o.pos[1], GRASP_Z[o.kind] ?? 0.0125];
          const q = worldToPixel(p, camMatrices!);
          if (!q) continue;
          ctx.beginPath();
          ctx.arc(q.u * scale, q.v * scale, 12, 0, Math.PI * 2);
          ctx.stroke();
          label(ctx, `GT ${o.name === st.scene!.target ? "target " : ""}${o.color} ${o.kind}`, q.u * scale + 14, q.v * scale, gt);
          drawn.push("groundTruth");
        }
        ctx.setLineDash([]);
      }

      // What was actually painted, for the canvas description and for the end-to-end test, which otherwise
      // could only check that a canvas exists.
      canvas.dataset.overlays = drawn.join(",");
      canvas.dataset.tick = String(frame?.meta.tick ?? "");
      const fps = frameStore.fps();
      const age = frame ? ((performance.now() - frame.received) / 1000).toFixed(1) : null;
      const overlayNote = moved && (st.layers.candidates || st.layers.winner || st.layers.mask || st.layers.grasp)
        ? " · boxes, mask and grasp are from the frame captured when the command started, so they do not follow the arm"
        : "";
      setCaption(
        frame
          ? `${cam} camera, ${kind}, ${W}x${H}, ${fps.toFixed(1)} fps, newest frame ${age} s old${projectable ? "" : " · 3D overlays need the front camera"}${overlayNote}`
          : "waiting for the first frame",
      );
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  const cameraButtons: { id: CameraName | "depth"; label: string }[] = [
    { id: "front", label: "Front" },
    { id: "wrist", label: "Wrist" },
    { id: "side", label: "Side" },
    { id: "depth", label: "Depth" },
  ];

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Camera" className="flex overflow-hidden rounded-m border border-edge">
          {cameraButtons.map((c) => {
            const active = c.id === "depth" ? showDepth : !showDepth && camera === c.id;
            return (
              <button
                key={c.id}
                type="button"
                aria-pressed={active}
                onClick={() => (c.id === "depth" ? setCamera("front", true) : setCamera(c.id as CameraName, false))}
                className={`px-3 py-1.5 text-sm ${active ? "bg-accent text-[color:var(--accent-fg)]" : "bg-surface-2 text-fg-muted hover:text-fg"}`}
              >
                {c.label}
              </button>
            );
          })}
        </div>
        {connection !== "open" && (
          <span className="text-xs text-warn">
            stream {connection}
            {useStore.getState().connectionDetail ? `: ${useStore.getState().connectionDetail}` : ""}
          </span>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="w-full rounded-m border border-edge bg-surface-2"
        aria-label={`Live ${camera} camera view of the simulated workspace. ${caption}`}
        data-overlays=""
        role="img"
      />
      <p className="num mt-1 text-xs text-fg-muted">{caption}</p>

      <fieldset className="mt-3 rounded-m border border-edge p-3">
        <legend className="px-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Overlay layers</legend>
        <div className="grid grid-cols-2 gap-x-4 sm:grid-cols-4">
          <Toggle checked={layers.candidates} onChange={(v) => setLayer("candidates", v)} label="All candidates" />
          <Toggle checked={layers.winner} onChange={(v) => setLayer("winner", v)} label="Chosen box" />
          <Toggle checked={layers.mask} onChange={(v) => setLayer("mask", v)} label="Segmentation mask" />
          <Toggle checked={layers.points} onChange={(v) => setLayer("points", v)} label="Point cloud" />
          <Toggle checked={layers.grasp} onChange={(v) => setLayer("grasp", v)} label="Grasp, yaw, width" />
          <Toggle checked={layers.waypoints} onChange={(v) => setLayer("waypoints", v)} label="Fingertip path" />
          <Toggle checked={layers.geofence} onChange={(v) => setLayer("geofence", v)} label="Geofence" />
          <Toggle
            checked={layers.groundTruth}
            onChange={(v) => setLayer("groundTruth", v)}
            label={<span className="text-gt">Ground truth (GT)</span>}
            hint="From the simulator, never from perception"
          />
        </div>
      </fieldset>
    </div>
  );
}
