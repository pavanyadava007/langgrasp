import { useEffect, useMemo, useState } from "react";
import { Button, Card, Field, NumberInput, Select, Term, Toggle } from "../../components/ui";
import { StatusIcon } from "../../components/Status";
import { ApiError, api } from "../../lib/api";
import { num } from "../../lib/format";
import { clearResultsCache } from "../../lib/results";
import { useStore, type JobState } from "../../store/store";

// Running the protocol from the browser. Two rules the form enforces, because a results file is the record of
// what this project measured: a job writes to a timestamped file unless someone types a name, and overwriting
// an existing file takes a second, deliberate confirmation.

const STRATA: { id: "seen" | "unseen" | "langvar"; label: string }[] = [
  { id: "seen", label: "seen" },
  { id: "unseen", label: "unseen" },
  { id: "langvar", label: "language variation" },
];

function cellStyle(t: Record<string, unknown>): { className: string; title: string; symbol: string } {
  if (t.placed) return { className: "bg-ok", title: "placed in the tray", symbol: "placed" };
  if (t.lifted) return { className: "bg-warn", title: "lifted but not placed", symbol: "lifted" };
  if (t.aborted) return { className: "bg-pending", title: `aborted: ${t.aborted}`, symbol: "aborted" };
  return { className: "bg-fail", title: "not lifted", symbol: "failed" };
}

function Grid({ job }: { job: JobState }) {
  const setSceneForm = useStore((s) => s.setSceneForm);
  const setState = useStore((s) => s.setState);
  const cells = job.trials;
  return (
    <div>
      <div className="flex flex-wrap gap-1">
        {cells.map((t, i) => {
          const st = cellStyle(t);
          return (
            <button
              key={i}
              type="button"
              title={`seed ${t.seed} · ${t.stratum} · ${t.color} ${t.kind} · ${st.title}. Click to open this scene in Live Run.`}
              onClick={() => {
                setSceneForm({ seed: Number(t.seed), stratum: (t.stratum as "seen") ?? "seen" });
                setState({ command: String(t.command ?? ""), view: "live" });
              }}
              className={`h-5 w-5 rounded-s ${st.className}`}
            >
              <span className="sr-only">
                seed {String(t.seed)}, {st.symbol}
              </span>
            </button>
          );
        })}
        {Array.from({ length: Math.max(0, job.total - cells.length) }).map((_, i) => (
          <span key={`p${i}`} className="h-5 w-5 rounded-s border border-edge" aria-hidden="true" />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-s bg-ok" /> placed
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-s bg-warn" /> lifted only
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-s bg-fail" /> not lifted
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-s bg-pending" /> refused before moving
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-s border border-edge" /> not run yet
        </li>
      </ul>
    </div>
  );
}

export function Batch() {
  const jobs = useStore((s) => s.jobs);
  const config = useStore((s) => s.config);
  const setConfig = useStore((s) => s.setConfig);
  const busy = useStore((s) => s.system?.worker.busy ?? false);
  const notify = useStore((s) => s.notify);
  const refreshSystem = useStore((s) => s.refreshSystem);

  const [controller, setController] = useState<"pipeline" | "oracle">("pipeline");
  const [counts, setCounts] = useState<Record<string, number | "">>({ seen: 10, unseen: 10, langvar: 10 });
  const [fixedGoal, setFixedGoal] = useState(false);
  const [baseSeed, setBaseSeed] = useState<number | "">(5000);
  const [outPath, setOutPath] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [starting, setStarting] = useState(false);
  const [conflict, setConflict] = useState<string | null>(null);

  const total = useMemo(() => STRATA.reduce((a, s) => a + (Number(counts[s.id]) || 0), 0), [counts]);
  const active = Object.values(jobs).find((j) => j.state === "running" || j.state === "queued");

  useEffect(() => {
    const id = window.setInterval(() => void refreshSystem(), 4000);
    return () => window.clearInterval(id);
  }, [refreshSystem]);

  async function start() {
    setStarting(true);
    setConflict(null);
    try {
      const strata = Object.fromEntries(STRATA.map((s) => [s.id, Number(counts[s.id]) || 0]).filter(([, v]) => (v as number) > 0));
      const r = await api.newJob({ controller, strata: strata as Record<string, number>, config, fixed_goal: fixedGoal, base_seed: Number(baseSeed) || 5000, out_path: outPath || null, overwrite });
      notify("info", `Started ${r.total} trials, writing to ${r.out_path}.`);
      clearResultsCache();
    } catch (e) {
      if (e instanceof ApiError && e.code === "file_exists") setConflict(e.message);
      else notify("error", e instanceof ApiError ? e.message : String(e));
    } finally {
      setStarting(false);
      void refreshSystem();
    }
  }

  return (
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
      <Card title="Run the protocol" subtitle="The same harness the evaluation scripts use, so the file it writes has the same shape as the ones in results/.">
        <Field label="Approach" htmlFor="b-controller">
          <Select
            id="b-controller"
            value={controller}
            onChange={(v) => setController(v as "pipeline" | "oracle")}
            options={[
              { value: "pipeline", label: "Modular pipeline" },
              { value: "oracle", label: "Oracle, ground-truth grasp pose" },
            ]}
          />
        </Field>
        <fieldset className="mb-3">
          <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Trials per stratum</legend>
          {STRATA.map((s) => (
            <div key={s.id} className="mb-2 flex items-center gap-2">
              <label htmlFor={`n-${s.id}`} className="w-40 text-sm">
                {s.label}
              </label>
              <NumberInput id={`n-${s.id}`} value={counts[s.id]} onChange={(v) => setCounts((c) => ({ ...c, [s.id]: v }))} min={0} max={200} />
            </div>
          ))}
          <p className="num text-xs text-fg-muted">{total} trials in total. The published protocol is 40 per stratum.</p>
        </fieldset>
        <Field label="First seed" htmlFor="b-seed" hint="Seeds run from here, so the same numbers reproduce the same scenes.">
          <NumberInput id="b-seed" value={baseSeed} onChange={setBaseSeed} min={0} />
        </Field>
        <Toggle checked={fixedGoal} onChange={setFixedGoal} label="Fixed goal: red cube" hint="The 100 scenes shared with ACT" />
        <Toggle checked={config.use_color_check} onChange={(v) => setConfig({ use_color_check: v })} label={<Term term="colour check">Colour check</Term>} />
        <Toggle checked={config.use_yolo_mask} onChange={(v) => setConfig({ use_yolo_mask: v })} label="YOLO11-seg mask" />
        <Toggle checked={config.depth_noise} onChange={(v) => setConfig({ depth_noise: v })} label="Synthetic depth noise" />
        <Field label="Write to" htmlFor="b-out" hint="Blank writes results/gui_<approach>_<timestamp>.json, which never collides with a published file.">
          <input
            id="b-out"
            value={outPath}
            onChange={(e) => {
              setOutPath(e.target.value);
              setOverwrite(false);
              setConflict(null);
            }}
            placeholder="gui_modular_<timestamp>.json"
            className="num w-full rounded-m border border-edge bg-surface-2 px-2 py-[7px] text-sm"
          />
        </Field>
        {conflict && (
          <div className="mb-3 rounded-m border border-warn p-2">
            <p className="text-xs text-warn">{conflict}</p>
            <Toggle checked={overwrite} onChange={setOverwrite} label="Overwrite that file" hint="The previous measurement is lost" />
          </div>
        )}
        <Button variant="primary" full onClick={start} disabled={starting || total === 0 || busy || !!active}>
          {busy || active ? "The simulator is busy" : `Start ${total} trials`}
        </Button>
        <p className="mt-2 text-xs text-fg-muted">
          Batch trials are not paced and record no frames, so they run at the same speed as the evaluation scripts. Watch the scenes change in Live Run.
        </p>
      </Card>

      <div className="min-w-0 space-y-4">
        {Object.values(jobs).length === 0 ? (
          <Card title="Jobs">
            <p className="text-sm text-fg-muted">No job has run in this session.</p>
          </Card>
        ) : (
          Object.values(jobs)
            .slice()
            .reverse()
            .map((job) => {
              const placed = job.trials.filter((t) => t.placed).length;
              const grounded = job.trials.filter((t) => t.grounding_correct === true).length;
              const scored = job.trials.filter((t) => t.grounding_correct !== null).length;
              return (
                <Card
                  key={job.job_id}
                  title={
                    <span className="flex items-center gap-2">
                      <span className={job.state === "done" ? "text-ok" : job.state === "running" ? "text-running" : "text-warn"}>
                        <StatusIcon status={job.state === "done" ? "ok" : job.state === "running" ? "running" : "warn"} spin />
                      </span>
                      {job.job_id}
                    </span>
                  }
                  subtitle={job.message ?? undefined}
                  right={
                    job.state === "running" ? (
                      <Button
                        onClick={async () => {
                          await api.cancelJob(job.job_id);
                          notify("warn", "Cancelling after the current trial. No file will be written.");
                        }}
                      >
                        Cancel
                      </Button>
                    ) : job.out_path ? (
                      <a className="text-xs text-accent underline" href={`/api/results/${job.out_path.split("/").pop()}`} target="_blank" rel="noreferrer">
                        open the JSON
                      </a>
                    ) : null
                  }
                  footer={job.out_path ? <span className="num">{job.out_path}</span> : undefined}
                >
                  <div className="mb-2 h-2 w-full overflow-hidden rounded-s bg-surface-2">
                    <div className="h-2 bg-accent" style={{ width: `${job.total ? (100 * job.done) / job.total : 0}%` }} />
                  </div>
                  <p className="num mb-3 text-sm">
                    {job.done} of {job.total} · placed {placed}
                    {job.done ? ` (${num((100 * placed) / job.done, 1)}%)` : ""} · grounding correct {grounded}
                    {scored ? ` of ${scored}` : ""}
                  </p>
                  <Grid job={job} />
                </Card>
              );
            })
        )}
      </div>
    </div>
  );
}
