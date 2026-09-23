import { StatusIcon, STATUS_COLOR, statusWord } from "../../components/Status";
import { ms } from "../../lib/format";
import { useStore } from "../../store/store";
import { STAGES, type StageName } from "../../lib/types";

// The nine stages as a horizontal timeline. Each cell is a button: status word, latency, and the one fact that
// matters about that stage. Clicking opens the drawer with everything the stage reported.

function oneLiner(stage: StageName, payload: Record<string, unknown>, running: boolean, tick: { tick: number; phase: string } | null): string {
  switch (stage) {
    case "command":
      return String(payload.source ?? "");
    case "parse": {
      const bits = [payload.action, payload.color, payload.noun ?? (payload.generic ? "generic" : "")].filter(Boolean);
      return bits.join(" ");
    }
    case "capture":
      return payload.size ? `${(payload.size as number[])[1]}x${(payload.size as number[])[0]}${payload.depth_noise ? ", noise on" : ""}` : "";
    case "grounding": {
      const n = (payload.candidates as unknown[] | undefined)?.length;
      return n === undefined ? "" : `${n} candidate${n === 1 ? "" : "s"}${payload.fallback ? ", fallback" : ""}`;
    }
    case "select":
      return payload.rule ? String(payload.rule) : "";
    case "gate":
      return payload.score !== undefined ? `${Number(payload.score).toFixed(2)} vs ${Number(payload.threshold).toFixed(2)}` : "";
    case "segment":
      return payload.mask_source ? String(payload.mask_source) : "";
    case "fuse": {
      const g = payload.grasp as { center?: number[]; width?: number } | null | undefined;
      if (g?.center) return `${g.center.map((v) => v.toFixed(2)).join(", ")} m`;
      return payload.n_points_object !== undefined ? `${payload.n_points_object} points` : "";
    }
    case "execute":
      if (running && tick) return `${tick.phase}, tick ${tick.tick}`;
      return payload.steps !== undefined ? `${payload.steps} ticks, IK ${payload.ik_ok ? "ok" : "off target"}` : "";
    default:
      return "";
  }
}

export function Stepper() {
  const stages = useStore((s) => s.stages);
  const openDrawer = useStore((s) => s.openDrawer);
  const drawerStage = useStore((s) => s.drawerStage);
  const system = useStore((s) => s.system);
  const lastTick = useStore((s) => s.lastTick);
  const running = useStore((s) => s.running);
  const titles = new Map((system?.stages ?? []).map((s) => [s.name, s]));

  return (
    <ol className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-5 2xl:grid-cols-9" aria-label="Pipeline stages">
      {STAGES.map((name, i) => {
        const st = stages[name];
        const meta = titles.get(name);
        const active = drawerStage === name;
        const spinning = st.status === "running";
        return (
          <li key={name}>
            <button
              type="button"
              onClick={() => openDrawer(active ? null : name)}
              aria-current={active ? "true" : undefined}
              title={meta?.help}
              className={`h-full w-full rounded-m border px-2 py-2 text-left transition-colors ${active ? "border-accent bg-surface-2" : "border-edge bg-surface hover:border-accent"}`}
            >
              <span className="flex items-center gap-1.5">
                <span className="num text-[11px] text-fg-muted">{i + 1}</span>
                <span className={STATUS_COLOR[st.status] ?? "text-pending"}>
                  <StatusIcon status={st.status} spin={spinning} />
                </span>
                <span className="truncate text-xs font-semibold">{meta?.title ?? name}</span>
              </span>
              <span className="num mt-1 block text-[11px] text-fg-muted">
                {st.status === "pending" ? "pending" : st.status === "running" ? "running" : st.latency_ms === null ? statusWord(st.status) : ms(st.latency_ms, 1)}
              </span>
              <span className="mt-0.5 block truncate text-[11px] text-fg-muted" title={oneLiner(name, st.payload, running, lastTick)}>
                {oneLiner(name, st.payload, running, lastTick) || " "}
              </span>
              {st.status === "warn" || st.status === "fail" ? (
                <span className={`mt-1 block text-[11px] ${st.status === "fail" ? "text-fail" : "text-warn"}`}>{(st.message ?? "").slice(0, 60)}</span>
              ) : null}
            </button>
          </li>
        );
      })}
    </ol>
  );
}
