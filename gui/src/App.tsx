import { useEffect } from "react";
import { Button, Kbd, Term } from "./components/ui";
import { StatusIcon } from "./components/Status";
import { Inspector } from "./views/Inspector";
import { LiveRun } from "./views/LiveRun";
import { Placeholder } from "./views/Placeholder";
import { useStore, type ViewName } from "./store/store";

const NAV: { id: ViewName; label: string; key: string }[] = [
  { id: "live", label: "Live Run", key: "1" },
  { id: "inspector", label: "Pipeline Inspector", key: "2" },
  { id: "results", label: "Results", key: "3" },
  { id: "batch", label: "Batch Evaluate", key: "4" },
  { id: "safety", label: "Safety & System", key: "5" },
];

function ModelStatus() {
  const system = useStore((s) => s.system);
  const models = system?.models ?? {};
  const entries = Object.entries(models);
  const loading = entries.filter(([, m]) => m.state === "loading").map(([k]) => k);
  const bad = entries.filter(([, m]) => m.state === "error" || m.state === "missing");
  if (entries.length === 0) return <span className="text-xs text-fg-muted">connecting to the worker</span>;
  if (loading.length) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-running">
        <StatusIcon status="running" spin /> loading {loading.join(", ")}
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs" title={entries.map(([k, m]) => `${k}: ${m.state}${m.load_ms ? ` in ${(m.load_ms / 1000).toFixed(1)} s` : ""}${m.detail ? ` (${m.detail})` : ""}`).join("\n")}>
      <span className={bad.length ? "text-warn" : "text-ok"}>
        <StatusIcon status={bad.length ? "warn" : "ok"} />
      </span>
      models {bad.length ? `warm, ${bad.length} unavailable` : "warm"}
    </span>
  );
}

function Notices() {
  const notices = useStore((s) => s.notices);
  const dismiss = useStore((s) => s.dismiss);
  if (!notices.length) return null;
  return (
    <div className="fixed bottom-28 right-4 z-50 w-[min(22rem,calc(100vw-2rem))] space-y-2 xl:bottom-4">
      {notices.map((n) => (
        <div
          key={n.id}
          className={`rounded-m border bg-surface p-3 text-sm ${n.level === "error" ? "border-fail" : n.level === "warn" ? "border-warn" : "border-edge"}`}
          style={{ boxShadow: "var(--shadow)" }}
        >
          <div className="flex items-start gap-2">
            <span className={n.level === "error" ? "text-fail" : n.level === "warn" ? "text-warn" : "text-fg-muted"}>
              <StatusIcon status={n.level === "error" ? "fail" : n.level === "warn" ? "warn" : "ok"} />
            </span>
            <p className="min-w-0 flex-1 break-words">{n.message}</p>
            <button type="button" onClick={() => dismiss(n.id)} aria-label="Dismiss" className="text-fg-muted hover:text-fg">
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}

function Shortcuts() {
  const open = useStore((s) => s.shortcutsOpen);
  const setState = useStore((s) => s.setState);
  if (!open) return null;
  const rows: [string, string][] = [
    ["Enter", "run the command in the bar"],
    ["Space", "pause or resume the executor; Step advances one tick"],
    ["E", "e-stop, no confirmation"],
    ["R", "rebuild this seed and run it again"],
    ["1 to 5", "switch view"],
    ["Escape", "close the stage drawer or this sheet"],
    ["?", "show this sheet"],
  ];
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={() => setState({ shortcutsOpen: false })}>
      <div role="dialog" aria-label="Keyboard shortcuts" className="w-full max-w-md rounded-l border border-edge bg-surface p-4" onClick={(e) => e.stopPropagation()}>
        <h2 className="mb-3 text-lg font-semibold">Keyboard</h2>
        <dl className="space-y-2">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-baseline gap-3">
              <dt className="w-20 flex-none">
                <Kbd>{k}</Kbd>
              </dt>
              <dd className="text-sm">{v}</dd>
            </div>
          ))}
        </dl>
        <div className="mt-4 text-right">
          <Button onClick={() => setState({ shortcutsOpen: false })}>Close</Button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  const banner = useStore((s) => s.banner);
  const announce = useStore((s) => s.announce);
  const init = useStore((s) => s.init);
  const hardware = useStore((s) => s.system?.hardware_label);

  useEffect(() => {
    void init();
  }, [init]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
      const s = useStore.getState();
      if (e.key === "Escape") {
        if (s.shortcutsOpen) s.setState({ shortcutsOpen: false });
        else if (s.drawerStage) s.openDrawer(null);
        return;
      }
      if (typing) return;
      if (e.key === "?") {
        s.setState({ shortcutsOpen: true });
        return;
      }
      if (e.key.toLowerCase() === "e") {
        e.preventDefault();
        void s.estop();
        return;
      }
      if (e.key.toLowerCase() === "r") {
        e.preventDefault();
        void s.replay();
        return;
      }
      if (e.key === " " && t?.tagName !== "BUTTON") {
        e.preventDefault();
        void (s.paused ? s.stepOnce() : s.togglePause());
        return;
      }
      const idx = ["1", "2", "3", "4", "5"].indexOf(e.key);
      if (idx >= 0) setView(NAV[idx].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setView]);

  return (
    <div className="min-h-full">
      <a href="#main" className="sr-only focus:not-sr-only">
        Skip to the main content
      </a>
      <header className="sticky top-0 z-20 border-b border-edge bg-surface">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2">
          <span className="text-lg font-semibold tracking-tight">LangGrasp</span>
          <span className="rounded-s border border-warn px-2 py-0.5 text-xs text-warn" title={hardware ?? undefined}>
            {banner}
          </span>
          <div className="ml-auto flex items-center gap-4">
            <ModelStatus />
            <Button variant="ghost" onClick={() => useStore.getState().setState({ shortcutsOpen: true })} title="Keyboard shortcuts (?)">
              <Kbd>?</Kbd>
            </Button>
            <Button variant="ghost" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} title="Switch theme">
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </Button>
          </div>
        </div>
      </header>

      <nav aria-label="Views" className="overflow-x-auto border-b border-edge p-2 lg:hidden">
        <ul className="flex gap-1">
          {NAV.map((n) => (
            <li key={n.id}>
              <button
                type="button"
                onClick={() => setView(n.id)}
                aria-current={view === n.id ? "page" : undefined}
                className={`whitespace-nowrap rounded-m px-3 py-2 text-sm ${view === n.id ? "bg-surface-2 font-semibold" : "text-fg-muted"}`}
              >
                {n.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="flex items-start">
        <nav aria-label="Views" className="sticky top-[3.25rem] hidden w-[13.5rem] flex-none self-start border-r border-edge p-3 lg:block">
          <ul className="space-y-1">
            {NAV.map((n) => (
              <li key={n.id}>
                <button
                  type="button"
                  onClick={() => setView(n.id)}
                  aria-current={view === n.id ? "page" : undefined}
                  className={`flex w-full items-center gap-2 rounded-m px-3 py-2 text-left text-sm ${view === n.id ? "bg-surface-2 font-semibold text-fg" : "text-fg-muted hover:text-fg"}`}
                >
                  <Kbd>{n.key}</Kbd>
                  {n.label}
                </button>
              </li>
            ))}
          </ul>
          <p className="mt-4 px-1 text-xs text-fg-muted">
            Every number in this app comes from a file in <span className="num">results/</span> or from the run you are watching. Anything not measured says{" "}
            <span className="num">not run</span>.
          </p>
        </nav>

        <main id="main" className="min-w-0 flex-1 p-4">
        {view === "live" && <LiveRun />}
        {view === "inspector" && <Inspector />}
        {view === "results" && (
          <Placeholder
            title="Results Dashboard"
            phase="phase 5"
            what="The measured protocol, its ablations and the per-object and per-lighting breakdowns, each widget linked to the JSON file it came from and that file's timestamp."
            sources={["results/oracle_protocol.json", "results/modular_protocol.json", "results/modular_nocolor_protocol.json", "results/yolo_latency_l4.json", "results/ppo_sim2sim_gap.json"]}
          />
        )}
        {view === "batch" && (
          <Placeholder
            title="Batch Evaluate"
            phase="phase 5"
            what="Run the protocol or a subset as a background job, with a live per-scene grid; clicking a cell replays that seed here in Live Run. It writes through the existing harness and never overwrites a results file without being told to."
            sources={["langgrasp/eval/harness.py", "results/"]}
          />
        )}
        {view === "safety" && (
          <Placeholder
            title="Safety & System"
            phase="phase 6"
            what="The FMEA table, hazard by hazard, with the code that implements each mitigation, the test that exercises it, and whether it is tested in simulation or needs hardware. Plus versions, engine files and model load state."
            sources={["docs/FMEA.md", "results/safety_clip_audit.json", "/api/system"]}
          />
        )}
        </main>
      </div>

      <div aria-live="polite" className="sr-only">
        {announce}
      </div>
      <Notices />
      <Shortcuts />
      <footer className="px-4 pb-6 pt-2 text-xs text-fg-muted lg:pl-[14.5rem]">
        <Term term="monitor mode">Safety monitor in observe mode by default</Term>. MuJoCo simulation of an SO-ARM100, no robot, no Jetson, nothing here is a hardware
        measurement.
      </footer>
    </div>
  );
}
