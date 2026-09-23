import { StatusIcon, STATUS_COLOR } from "../../components/Status";
import { Button, Term } from "../../components/ui";
import { latencyLabel, metres, ms, num, degrees, xyz } from "../../lib/format";
import { useStore } from "../../store/store";
import type { Candidate, GraspPose, StageName } from "../../lib/types";

// Everything a stage reported, in the order a person debugging would want it: the decision, then why, then the
// numbers, then the images. Nothing here is computed: every value is a field of the stage payload.

function Row({ k, v, title }: { k: string; v: React.ReactNode; title?: string }) {
  return (
    <div className="flex gap-3 border-b border-edge py-1.5 last:border-0">
      <dt className="w-44 flex-none text-xs uppercase tracking-wide text-fg-muted" title={title}>
        {k}
      </dt>
      <dd className="num min-w-0 flex-1 break-words text-sm">{v}</dd>
    </div>
  );
}

function CandidateTable({ cands, winner }: { cands: Candidate[]; winner: number | null }) {
  return (
    <table className="w-full text-xs">
      <caption className="mb-1 text-left text-xs text-fg-muted">
        Every box the grounder returned, in its own order. The colour fraction is the share of saturated pixels in the box matching the colour word.
      </caption>
      <thead>
        <tr className="text-left text-fg-muted">
          <th className="py-1 pr-2">#</th>
          <th className="py-1 pr-2">label</th>
          <th className="py-1 pr-2">score</th>
          <th className="py-1 pr-2">colour</th>
          <th className="py-1 pr-2">area px</th>
          <th className="py-1">chosen</th>
        </tr>
      </thead>
      <tbody className="num">
        {cands.map((c, i) => (
          <tr key={i} className={i === winner ? "text-accent" : ""}>
            <td className="py-1 pr-2">{i + 1}</td>
            <td className="py-1 pr-2">{c.label}</td>
            <td className="py-1 pr-2">{c.score.toFixed(3)}</td>
            <td className="py-1 pr-2">{c.color_frac.toFixed(2)}</td>
            <td className="py-1 pr-2">{Math.round(c.area_px)}</td>
            <td className="py-1">{i === winner ? "yes" : ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StageBody({ stage }: { stage: StageName }) {
  const st = useStore((s) => s.stages[stage]);
  const p = st.payload;
  switch (stage) {
    case "command":
      return (
        <dl>
          <Row k="text" v={String(p.text ?? "")} />
          <Row k="source" v={String(p.source ?? "")} />
          <Row k="controller" v={String(p.controller ?? "")} />
          {p.stt_latency_ms ? <Row k="transcription" v={ms(Number(p.stt_latency_ms))} /> : null}
        </dl>
      );
    case "parse":
      return (
        <dl>
          <Row k="action" v={String(p.action ?? "")} />
          <Row k="colour word" v={p.color ? String(p.color) : "none"} />
          <Row k="head noun" v={p.noun ? String(p.noun) : "none"} />
          <Row k="spatial reference" v={p.spatial ? String(p.spatial) : "none"} />
          <Row k="phrase sent to the grounder" v={String(p.phrase ?? "")} />
          <Row k="attribute only" v={p.generic ? "yes, no object noun" : "no"} />
          {Array.isArray(p.notes) && p.notes.length ? <Row k="parser notes" v={(p.notes as string[]).join("; ")} /> : null}
        </dl>
      );
    case "capture":
      return (
        <dl>
          <Row k="camera" v={String(p.camera ?? "")} />
          <Row k="size" v={Array.isArray(p.size) ? `${(p.size as number[])[1]} x ${(p.size as number[])[0]} px` : "?"} />
          <Row k="focal length" v={`${num(p.fx_px as number, 1)} px`} />
          <Row k="depth range" v={`${num(p.depth_min_m as number, 3)} to ${num(p.depth_max_m as number, 3)} m`} />
          <Row k="invalid depth pixels" v={String(p.depth_invalid_px ?? "?")} title="Dropped before back-projection" />
          <Row k="synthetic depth noise" v={p.depth_noise ? "on (sigma 2 mm, 2% dropout, 1 mm quantisation)" : "off"} />
        </dl>
      );
    case "grounding": {
      const cands = (p.candidates as Candidate[] | undefined) ?? [];
      return (
        <>
          <dl>
            <Row k="query sent" v={String(p.query ?? "")} />
            <Row
              k="colour fallback"
              v={p.fallback ? <Term term="fallback">yes, re-grounded as "object"</Term> : "no"}
              title="A colour word with no matching candidate re-grounds the generic word 'object'"
            />
            <Row k="grounder time" v={ms(p.grounder_latency_ms as number)} title="Sum over the model calls this command made" />
            <Row k="box threshold" v={num(p.box_threshold as number, 2)} />
            <Row k="text threshold" v={num(p.text_threshold as number, 2)} />
          </dl>
          <div className="mt-3">{cands.length ? <CandidateTable cands={cands} winner={null} /> : <p className="text-sm text-fg-muted">No candidates.</p>}</div>
        </>
      );
    }
    case "select":
      return (
        <dl>
          <Row k="rule that decided" v={String(p.rule ?? "")} />
          <Row k="candidates" v={String(p.n_candidates ?? "?")} />
          <Row k="dropped by colour" v={String(p.color_filtered ?? 0)} />
          <Row k="spatial reference used" v={p.spatial_used ? `yes, ${p.spatial}` : "no"} />
          <Row k="flagged ambiguous" v={p.ambiguous ? <span className="text-warn">yes, top two are close</span> : "no"} title="Top-1 and top-2 within 0.05, with similar colour fractions" />
          <Row k="chosen box" v={Array.isArray(p.box) ? (p.box as number[]).map((v) => v.toFixed(1)).join(", ") : "none"} />
          <Row k="score" v={num(p.score as number, 3)} />
        </dl>
      );
    case "gate":
      return (
        <dl>
          <Row k="decision" v={String(p.decision ?? "")} />
          <Row k="reason" v={String(p.reason ?? "")} />
          <Row k="score" v={num(p.score as number, 3)} />
          <Row k="threshold" v={num(p.threshold as number, 2)} />
          {p.ambiguity_margin !== undefined ? <Row k="ambiguity margin" v={<Term term="ambiguity margin">{num(p.ambiguity_margin as number, 2)}</Term>} /> : null}
          {p.select_flagged_ambiguous !== undefined ? <Row k="selector flagged ambiguous" v={p.select_flagged_ambiguous ? "yes" : "no"} /> : null}
          <Row k="what the gate was given" v={<span className="text-fg-muted">{String(p.gate_input ?? "")}</span>} />
        </dl>
      );
    case "segment":
      return (
        <>
          <dl>
            <Row k="mask source" v={String(p.mask_source ?? "")} />
            <Row k="mask pixels" v={String(p.mask_px ?? "?")} />
            <Row k="detections" v={String(p.n_detections ?? "0")} />
            <Row k="best IoU with the box" v={p.best_iou == null ? "none above 0.3" : num(p.best_iou as number, 2)} />
            {p.timing ? <Row k="backend timing" v={Object.entries(p.timing as Record<string, number>).map(([k, v]) => `${k.replace("_ms", "")} ${v.toFixed(1)}`).join(" · ")} /> : null}
          </dl>
          {Array.isArray(p.detections) && (p.detections as unknown[]).length ? (
            <table className="mt-3 w-full text-xs">
              <thead>
                <tr className="text-left text-fg-muted">
                  <th className="py-1 pr-2">class</th>
                  <th className="py-1 pr-2">confidence</th>
                  <th className="py-1">IoU with the chosen box</th>
                </tr>
              </thead>
              <tbody className="num">
                {(p.detections as { name: string; conf: number; iou_with_box: number }[]).map((d, i) => (
                  <tr key={i}>
                    <td className="py-1 pr-2">{d.name}</td>
                    <td className="py-1 pr-2">{d.conf.toFixed(2)}</td>
                    <td className="py-1">{d.iou_with_box.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </>
      );
    case "fuse": {
      const g = (p.grasp as GraspPose | null) ?? null;
      return (
        <dl>
          <Row k="back-projected points" v={String(p.n_points_backprojected ?? "?")} />
          <Row k="after table removal" v={`${p.n_points_object ?? "?"} (minimum ${p.min_points ?? 20})`} />
          {g ? (
            <>
              <Row k="grasp centre" v={`${xyz(g.center)} m`} />
              <Row k="jaw yaw" v={<Term term="jaw yaw">{degrees(g.psi)}</Term>} />
              <Row k="jaw width" v={metres(g.width)} />
              <Row k="aspect ratio" v={num(g.aspect, 2)} />
              <Row k="elongated" v={g.elongated ? "yes, handle slice used" : "no"} />
              <Row k="fallback fit" v={g.fallback ? "yes" : "no"} />
              <Row k="points in the fit" v={String(g.n)} />
            </>
          ) : (
            <Row k="grasp" v={<span className="text-fail">none</span>} />
          )}
          {p.max_grasp_width !== undefined ? <Row k="jaw maximum" v={metres(p.max_grasp_width as number)} /> : null}
        </dl>
      );
    }
    case "execute":
      return (
        <dl>
          <Row k="target" v={String(p.target ?? "")} />
          <Row k="commanded centre" v={Array.isArray(p.center) ? `${xyz(p.center as number[])} m` : "?"} />
          <Row k="yaw actually used" v={degrees(p.psi_used as number)} title="best_psi picks the reachable representative of psi + k*pi" />
          <Row k="ticks" v={String(p.steps ?? "?")} />
          <Row k="IK converged at the last waypoint" v={p.ik_ok === undefined ? "?" : p.ik_ok ? "yes" : "no"} />
          <Row k="worst position error" v={metres(p.max_pos_err_m as number)} />
          <Row k="orientation error" v={degrees(p.rot_err_rad as number)} />
          <Row k="grasped / lifted / placed" v={`${p.grasped ? "yes" : "no"} / ${p.lifted ? "yes" : "no"} / ${p.placed ? "yes" : "no"}`} />
          {p.ground_truth ? <Row k="pose source" v={<span className="text-gt">ground truth from the simulator (oracle)</span>} /> : null}
        </dl>
      );
    default:
      return null;
  }
}

export function StageDrawer() {
  const stage = useStore((s) => s.drawerStage);
  const openDrawer = useStore((s) => s.openDrawer);
  const st = useStore((s) => (s.drawerStage ? s.stages[s.drawerStage] : null));
  const meta = useStore((s) => s.system?.stages.find((x) => x.name === s.drawerStage));
  if (!stage || !st) return null;
  const images = Object.entries(st.images);
  return (
    <div className="mt-3 rounded-l border border-accent bg-surface" style={{ boxShadow: "var(--shadow)" }}>
      <header className="flex items-start justify-between gap-3 border-b border-edge px-4 py-3">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <span className={STATUS_COLOR[st.status]}>
              <StatusIcon status={st.status} />
            </span>
            {meta?.title ?? stage}
            <span className="num text-xs font-normal text-fg-muted">
              {st.latency_ms === null ? "" : `${ms(st.latency_ms, 1)} (${latencyLabel(st.latency_kind)})`}
            </span>
          </h2>
          {meta?.help && <p className="mt-1 max-w-3xl text-xs text-fg-muted">{meta.help}</p>}
          {st.message && <p className={`mt-2 max-w-3xl text-sm ${st.status === "fail" ? "text-fail" : "text-warn"}`}>{st.message}</p>}
        </div>
        <Button variant="ghost" onClick={() => openDrawer(null)} title="Close (Escape)">
          Close
        </Button>
      </header>
      <div className="grid gap-4 px-4 py-3 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div>
          <StageBody stage={stage} />
        </div>
        {images.length > 0 && (
          <div>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-fg-muted">What this stage saw</h3>
            <div className="space-y-2">
              {images.map(([role, url]) => (
                <figure key={role} className="m-0">
                  <img src={url} alt={`${role} image from the ${stage} stage`} className="w-full rounded-m border border-edge" loading="lazy" />
                  <figcaption className="mt-1 text-xs text-fg-muted">{role}</figcaption>
                </figure>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
