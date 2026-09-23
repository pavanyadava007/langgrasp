import { useEffect, useState } from "react";
import { Card, NotRun, Skeleton, Term } from "../../components/ui";
import { api } from "../../lib/api";
import { StatusIcon } from "../../components/Status";
import { ago, ms, num } from "../../lib/format";
import { useStore } from "../../store/store";

// The hazard table, the mitigations, and what is only simulated. The status column is the point of this view:
// a reader has to be able to tell what was exercised from what needs hardware that does not exist here.

interface Mitigation {
  what: string;
  where: string;
  status?: string;
}
interface Hazard {
  id: string;
  title: string;
  standard: string;
  trigger: string;
  hazard: string;
  sec: string;
  mitigations: Mitigation[];
  tests: string[];
  status: "sim" | "partial" | "hardware";
  residual: string;
  measured?: string;
  gui_note?: string;
}
interface Fmea {
  scope: string;
  document: string;
  source: string;
  checked_by: string;
  mtime: number;
  scales: Record<string, string>;
  statuses: Record<string, string>;
  hazards: Hazard[];
  cross_cutting: { what: string; where: string; tests: string[] }[];
  not_addressed: string[];
}

const STATUS_STYLE: Record<string, { icon: string; className: string; word: string }> = {
  sim: { icon: "ok", className: "text-ok", word: "tested in simulation" },
  partial: { icon: "warn", className: "text-warn", word: "software half tested" },
  hardware: { icon: "pending", className: "text-pending", word: "hardware only, not exercised" },
};

function HazardRow({ h }: { h: Hazard }) {
  const [open, setOpen] = useState(false);
  const st = STATUS_STYLE[h.status];
  return (
    <>
      <tr className="border-t border-edge align-top">
        <td className="py-2 pr-3">
          <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} className="text-left">
            <span className="num font-semibold">{h.id}</span> <span className="font-medium">{h.title}</span>
            <span className="block text-xs text-fg-muted">{open ? "hide detail" : "show detail"}</span>
          </button>
        </td>
        <td className="py-2 pr-3 text-xs">{h.standard}</td>
        <td className="num py-2 pr-3 text-xs">{h.sec}</td>
        <td className="py-2 pr-3 text-xs">
          <ul className="space-y-1">
            {h.mitigations.map((m, i) => (
              <li key={i} className={m.status === "hardware" ? "text-pending" : ""}>
                {m.what}
                <span className="num block text-[11px] text-fg-muted">{m.where}</span>
              </li>
            ))}
          </ul>
        </td>
        <td className="py-2 pr-3">
          <span className={`flex items-center gap-1.5 text-xs ${st.className}`}>
            <StatusIcon status={st.icon} />
            {st.word}
          </span>
          <ul className="num mt-1 space-y-0.5 text-[11px] text-fg-muted">
            {h.tests.map((t) => (
              <li key={t}>{t.replace("tests/", "")}</li>
            ))}
          </ul>
        </td>
      </tr>
      {open && (
        <tr className="border-t border-edge bg-surface-2">
          <td colSpan={5} className="px-3 py-3 text-sm">
            <p>
              <span className="text-xs uppercase tracking-wide text-fg-muted">Triggering condition:</span> {h.trigger}
            </p>
            <p className="mt-1">
              <span className="text-xs uppercase tracking-wide text-fg-muted">Hazardous behaviour:</span> {h.hazard}
            </p>
            <p className="mt-1 text-warn">
              <span className="text-xs uppercase tracking-wide text-fg-muted">Residual risk:</span> {h.residual}
            </p>
            {h.measured && (
              <p className="num mt-1 text-xs">
                <span className="uppercase tracking-wide text-fg-muted">Measured:</span> {h.measured}
              </p>
            )}
            {h.gui_note && (
              <p className="mt-1 text-xs text-fg-muted">
                <span className="uppercase tracking-wide">In this GUI:</span> {h.gui_note}
              </p>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

function SystemCard() {
  const system = useStore((s) => s.system);
  const refresh = useStore((s) => s.refreshSystem);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  if (!system) return <Skeleton h={160} />;
  const models = Object.entries(system.models);
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-3">
      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Machine</h3>
        <dl className="text-sm">
          <div className="flex gap-2 border-b border-edge py-1">
            <dt className="w-24 flex-none text-xs uppercase tracking-wide text-fg-muted">GPU</dt>
            <dd className="num">{system.gpu ?? <NotRun />}</dd>
          </div>
          <div className="flex gap-2 border-b border-edge py-1">
            <dt className="w-24 flex-none text-xs uppercase tracking-wide text-fg-muted">label</dt>
            <dd className="num text-xs">{system.hardware_label ?? <NotRun />}</dd>
          </div>
          <div className="flex gap-2 border-b border-edge py-1">
            <dt className="w-24 flex-none text-xs uppercase tracking-wide text-fg-muted">python</dt>
            <dd className="num">{system.python}</dd>
          </div>
          <div className="flex gap-2 py-1">
            <dt className="w-24 flex-none text-xs uppercase tracking-wide text-fg-muted">worker</dt>
            <dd className="num">
              pid {system.worker.pid ?? "?"}, {system.worker.alive ? "alive" : "not running"}
              {system.worker.busy ? ", busy" : ""}
            </dd>
          </div>
        </dl>
        <h3 className="mb-1 mt-3 text-xs font-medium uppercase tracking-wide text-fg-muted">Versions</h3>
        <ul className="num text-xs">
          {Object.entries(system.versions).map(([k, v]) => (
            <li key={k} className="flex justify-between gap-2 border-b border-edge py-0.5">
              <span>{k}</span>
              <span className={v ? "" : "text-fg-muted"}>{v ?? "not installed"}</span>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Models in this worker</h3>
        {models.length === 0 ? (
          <NotRun />
        ) : (
          <ul className="text-sm">
            {models.map(([name, m]) => (
              <li key={name} className="border-b border-edge py-1">
                <span className="flex items-center gap-1.5">
                  <span className={m.state === "warm" ? "text-ok" : m.state === "loading" ? "text-running" : "text-warn"}>
                    <StatusIcon status={m.state === "warm" ? "ok" : m.state === "loading" ? "running" : "warn"} />
                  </span>
                  <span className="num">{name}</span>
                  <span className="text-xs text-fg-muted">{m.state}</span>
                  {m.load_ms !== null && <span className="num text-xs text-fg-muted">{ms(m.load_ms, 0)} to load</span>}
                </span>
                {m.detail && <span className="num block pl-5 text-[11px] text-fg-muted">{m.detail}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Weights and engines on disk</h3>
        <ul className="text-sm">
          {system.engines.map((e) => (
            <li key={e.path} className="flex items-baseline justify-between gap-2 border-b border-edge py-1">
              <span className="num text-xs">{e.path.replace("checkpoints/", "")}</span>
              <span className={`num text-xs ${e.present ? "" : "text-fg-muted"}`}>{e.present ? `${num(e.size_mb, 1)} MB · ${ago(e.mtime)}` : "not built"}</span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-fg-muted">
          TensorRT engines are built for one GPU and are not portable: these are <Term term="tensorrt">L4 engines</Term> and would have to be rebuilt on a Jetson.
        </p>
      </div>
    </div>
  );
}

export function Safety() {
  const [fmea, setFmea] = useState<Fmea | null>(null);
  const [error, setError] = useState<string | null>(null);
  const safety = useStore((s) => s.safety);
  const system = useStore((s) => s.system);

  useEffect(() => {
    api
      .fmea()
      .then((body) => {
        if (!Array.isArray((body as { hazards?: unknown[] }).hazards)) throw new Error("The hazard table came back in a shape this page does not understand.");
        return body as unknown as Fmea;
      })
      .then(setFmea)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="min-w-0 space-y-4">
      <Card title="System" subtitle="What this worker actually loaded, and what is on disk.">
        <SystemCard />
      </Card>

      <Card title="Safety monitor, right now" subtitle={system?.safety_note}>
        {safety ? (
          <div className="grid grid-cols-[minmax(0,1fr)] gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <h3 className="text-xs uppercase tracking-wide text-fg-muted">State</h3>
              <p className="text-lg font-semibold">{safety.state}</p>
              <p className="text-xs text-fg-muted">{safety.reason || "ok"}</p>
            </div>
            <div>
              <h3 className="text-xs uppercase tracking-wide text-fg-muted">Mode</h3>
              <p className="text-lg font-semibold">{safety.mode}</p>
              <p className="text-xs text-fg-muted">{safety.mode === "monitor" ? "clipping is reported, not applied" : "clipping is applied, which changes the trajectory"}</p>
            </div>
            <div>
              <h3 className="text-xs uppercase tracking-wide text-fg-muted">Watchdog ages</h3>
              <ul className="num text-xs">
                {Object.entries(safety.watchdog).map(([k, v]) => (
                  <li key={k}>
                    {k}: {v === null ? "never sampled" : `${num(v, 2)} s`}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="text-xs uppercase tracking-wide text-fg-muted">Clipping</h3>
              <p className="num text-xs">
                applied {safety.clips.velocity ?? 0} velocity, {safety.clips.limit ?? 0} limit
              </p>
              <p className="num text-xs">
                would clip {safety.would_clip.velocity ?? 0} velocity, {safety.would_clip.limit ?? 0} limit
              </p>
            </div>
          </div>
        ) : (
          <NotRun />
        )}
        {(system as unknown as { watchdog_note?: string })?.watchdog_note && (
          <p className="mt-3 max-w-4xl text-xs text-fg-muted">{(system as unknown as { watchdog_note?: string }).watchdog_note}</p>
        )}
      </Card>

      {error && (
        <Card title="Hazard analysis">
          <p className="text-sm text-fail">{error}</p>
          <p className="mt-1 text-xs text-fg-muted">The table lives in langgrasp/gui/fmea.yaml and mirrors docs/FMEA.md.</p>
        </Card>
      )}
      {error ? null : !fmea ? (
        <Skeleton h={300} />
      ) : (
        <>
          <Card
            title="Hazard analysis"
            subtitle={fmea.scope}
            footer={
              <span className="num">
                source: {fmea.source} · mirrors {fmea.document} · kept in step by {fmea.checked_by} · {ago(fmea.mtime)}
              </span>
            }
          >
            <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Hazard table, scrolls sideways">
              <table className="w-full min-w-[60rem] text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
                    <th className="py-1 pr-3">hazard</th>
                    <th className="py-1 pr-3">standard</th>
                    <th className="py-1 pr-3" title={Object.values(fmea.scales).join("\n")}>
                      S / E / C
                    </th>
                    <th className="py-1 pr-3">mitigation, and where it lives</th>
                    <th className="py-1">status and evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {fmea.hazards.map((h) => (
                    <HazardRow key={h.id} h={h} />
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs">
              {Object.entries(fmea.statuses).map(([k, v]) => (
                <li key={k} className={`flex items-center gap-1.5 ${STATUS_STYLE[k].className}`}>
                  <StatusIcon status={STATUS_STYLE[k].icon} />
                  <span>
                    {STATUS_STYLE[k].word}: <span className="text-fg-muted">{v}</span>
                  </span>
                </li>
              ))}
            </ul>
          </Card>

          <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-2">
            <Card title="Mechanisms that cut across every hazard">
              <ul className="space-y-2 text-sm">
                {fmea.cross_cutting.map((c, i) => (
                  <li key={i}>
                    {c.what}
                    <span className="num block text-xs text-fg-muted">{c.where}</span>
                    <span className="num block text-[11px] text-fg-muted">{c.tests.map((t) => t.replace("tests/", "")).join(", ")}</span>
                  </li>
                ))}
              </ul>
            </Card>
            <Card title="Not addressed at all" subtitle="Stated here because a hazard analysis that lists only what it covers is misleading.">
              <ul className="list-inside list-disc space-y-1 text-sm">
                {fmea.not_addressed.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
