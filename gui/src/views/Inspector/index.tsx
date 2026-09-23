import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { Button, Card, NotRun, Term } from "../../components/ui";
import { StatusIcon, STATUS_COLOR } from "../../components/Status";
import { api, type RunRow } from "../../lib/api";
import { ago, latencyLabel, metres, ms, num } from "../../lib/format";
import { frameUrlForTick, loadRun, waterfall, type LoadedRun } from "../../lib/run";
import { useStore } from "../../store/store";

const JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"];
// CSS variables, so the chart follows the theme and the labels stay readable on both backgrounds.
const JOINT_COLOURS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

function Waterfall({ run }: { run: LoadedRun }) {
  const rows = waterfall(run);
  const total = Math.max(...rows.map((r) => r.startMs + r.ms), 1);
  return (
    <div>
      <p className="mb-2 text-xs text-fg-muted">
        Where the time went in this command. Bars are each stage's own timing laid out against the run clock; a stage with no bar was not timed separately.
      </p>
      <ul className="space-y-1">
        {rows.map((r) => (
          <li key={r.stage} className="flex items-center gap-2">
            <span className="w-24 flex-none truncate text-xs">{r.stage}</span>
            <span className="relative h-4 flex-1 rounded-s bg-surface-2">
              {r.ms > 0 && (
                <span
                  className="absolute top-0 h-4 rounded-s"
                  style={{
                    left: `${(100 * r.startMs) / total}%`,
                    width: `${Math.max(0.6, (100 * r.ms) / total)}%`,
                    background: r.status === "fail" ? "var(--fail)" : r.status === "warn" ? "var(--warn)" : "var(--accent)",
                  }}
                  title={`${r.stage}: ${ms(r.ms, 1)} (${latencyLabel(r.kind)})`}
                />
              )}
            </span>
            <span className="num w-24 flex-none text-right text-xs text-fg-muted">{r.ms > 0 ? ms(r.ms, 1) : "not timed"}</span>
          </li>
        ))}
      </ul>
      <p className="num mt-2 text-xs text-fg-muted">total {ms(total)}</p>
    </div>
  );
}

function JointChart({ run, tick, joints }: { run: LoadedRun; tick: number; joints: boolean[] }) {
  const data = useMemo(
    () =>
      run.ticks.map((t) => {
        const row: Record<string, number> = { tick: t.tick };
        JOINTS.forEach((name, i) => {
          row[name] = t.q[i];
          row[`${name}_target`] = t.q_target[i];
        });
        return row;
      }),
    [run],
  );
  return (
    <ResponsiveContainer width="100%" height={230}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
        <CartesianGrid stroke="var(--edge)" strokeDasharray="2 3" />
        <XAxis dataKey="tick" stroke="var(--fg-muted)" fontSize={11} />
        <YAxis stroke="var(--fg-muted)" fontSize={11} tickFormatter={(v: number) => v.toFixed(1)} />
        <RTooltip
          contentStyle={{ background: "var(--surface)", border: "1px solid var(--edge)", borderRadius: 8, fontSize: 12 }}
          formatter={(v: number, name: string) => [`${Number(v).toFixed(3)} rad`, String(name).replace("_target", " target")]}
        />
        <ReferenceLine x={tick} stroke="var(--focus)" />
        {JOINTS.map((name, i) => (joints[i] ? <Line key={name} type="monotone" dataKey={name} stroke={JOINT_COLOURS[i]} dot={false} strokeWidth={1.8} isAnimationActive={false} /> : null))}
        {JOINTS.map((name, i) =>
          joints[i] ? (
            <Line key={`${name}t`} type="monotone" dataKey={`${name}_target`} stroke={JOINT_COLOURS[i]} dot={false} strokeWidth={1} strokeDasharray="3 3" isAnimationActive={false} />
          ) : null,
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

function TrajectoryChart({ run, tick }: { run: LoadedRun; tick: number }) {
  const centre = run.stages.execute?.payload?.center as number[] | undefined;
  const data = useMemo(
    () =>
      run.ticks.map((t) => ({
        tick: t.tick,
        z: t.tcp[2] * 1000,
        err: centre ? 1000 * Math.hypot(t.tcp[0] - centre[0], t.tcp[1] - centre[1]) : null,
      })),
    [run, centre],
  );
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
        <CartesianGrid stroke="var(--edge)" strokeDasharray="2 3" />
        <XAxis dataKey="tick" stroke="var(--fg-muted)" fontSize={11} />
        <YAxis stroke="var(--fg-muted)" fontSize={11} />
        <RTooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--edge)", borderRadius: 8, fontSize: 12 }} />
        <Legend verticalAlign="top" height={24} wrapperStyle={{ fontSize: 11, color: "var(--fg-muted)" }} />
        <ReferenceLine x={tick} stroke="var(--focus)" />
        <Line type="monotone" dataKey="z" name="fingertip height (mm)" stroke="var(--accent)" dot={false} strokeWidth={1.8} isAnimationActive={false} />
        <Line type="monotone" dataKey="err" name="distance to the commanded grasp point (mm)" stroke="var(--warn)" dot={false} strokeWidth={1.4} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function Inspector() {
  const [rows, setRows] = useState<RunRow[] | null>(null);
  const [run, setRun] = useState<LoadedRun | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [joints, setJoints] = useState([true, true, true, true, true]);
  const setSceneForm = useStore((s) => s.setSceneForm);
  const setState = useStore((s) => s.setState);
  const timer = useRef<number | null>(null);

  const open = useCallback(async (id: string) => {
    setPlaying(false);
    try {
      const loaded = await loadRun(id);
      setRun(loaded);
      setIndex(0);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const { runs } = await api.runs();
      setRows(runs);
      return runs;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return [];
    }
  }, []);

  useEffect(() => {
    void (async () => {
      const runs = await refresh();
      if (runs.length) void open(runs[0].id);
    })();
  }, [refresh, open]);

  useEffect(() => {
    if (!playing || !run) return;
    timer.current = window.setInterval(() => {
      setIndex((i) => {
        if (i + 1 >= run.ticks.length) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 100);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [playing, run]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!run) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowRight") setIndex((i) => Math.min(i + 1, run.ticks.length - 1));
      if (e.key === "ArrowLeft") setIndex((i) => Math.max(i - 1, 0));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run]);

  const tick = run?.ticks[index];

  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[260px_minmax(0,1fr)]">
      <div className="min-w-0">
        <Card
          title="Recorded runs"
          subtitle="Every live run is written to runs/gui, newest first."
          right={
            <Button variant="ghost" onClick={() => void refresh()}>
              Refresh
            </Button>
          }
        >
          {rows === null ? (
            <p className="text-sm text-fg-muted">Loading.</p>
          ) : rows.length === 0 ? (
            <p className="text-sm text-fg-muted">
              No runs recorded yet. Run a command in Live Run, unless the server was started with <span className="num">--no-record</span>.
            </p>
          ) : (
            <ul className="max-h-[70vh] space-y-1 overflow-y-auto">
              {rows.map((r) => (
                <li key={r.id}>
                  <button
                    type="button"
                    onClick={() => void open(r.id)}
                    aria-current={run?.id === r.id ? "true" : undefined}
                    className={`w-full rounded-m border px-2 py-2 text-left text-xs ${run?.id === r.id ? "border-accent bg-surface-2" : "border-edge hover:border-accent"}`}
                  >
                    <span className="block truncate font-medium">{r.command ?? r.id}</span>
                    <span className="num block text-fg-muted">
                      seed {r.seed ?? "?"} · {r.controller} · {r.n_events} events
                    </span>
                    <span className="num block text-fg-muted">{ago(r.mtime)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="min-w-0 space-y-4">
        {error && <p className="rounded-m border border-fail px-3 py-2 text-sm text-fail">{error}</p>}
        {!run ? (
          <Card title="Pipeline Inspector">
            <p className="text-sm text-fg-muted">Pick a run on the left to step through it.</p>
          </Card>
        ) : (
          <>
            <Card
              title={run.meta.command ?? run.id}
              subtitle={`${run.id} · ${run.meta.controller} · safety ${run.meta.safety_mode} · ${run.ticks.length} ticks`}
              right={
                <Button
                  onClick={() => {
                    setSceneForm({ seed: run.meta.seed ?? 5000, stratum: (run.meta.scenario?.stratum as "seen" | "unseen" | "langvar") ?? "seen" });
                    setState({ command: run.meta.command ?? "", view: "live" });
                  }}
                >
                  Replay this seed in Live Run
                </Button>
              }
              footer={run.meta.note}
            >
              <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
                {/* min-w-0: a grid item's automatic minimum is its content's min-content width, and a 640 px
                    frame would otherwise widen the whole page on a narrow screen however wide the box is. */}
                <div className="min-w-0">
                  {tick ? (
                    <img
                      src={frameUrlForTick(run, tick.tick)}
                      alt={`Front camera at tick ${tick.tick}, phase ${tick.phase}`}
                      className="w-full max-w-full rounded-m border border-edge"
                    />
                  ) : (
                    <p className="text-sm text-fg-muted">This run recorded no ticks.</p>
                  )}
                  <div className="mt-2 flex items-center gap-2">
                    <Button onClick={() => setPlaying((p) => !p)} disabled={!run.ticks.length}>
                      {playing ? "Pause" : "Play"}
                    </Button>
                    <input
                      type="range"
                      min={0}
                      max={Math.max(0, run.ticks.length - 1)}
                      value={index}
                      onChange={(e) => setIndex(Number(e.target.value))}
                      aria-label="Tick"
                      className="flex-1 accent-[color:var(--accent)]"
                    />
                  </div>
                  {tick && (
                    <p className="num mt-1 text-xs text-fg-muted">
                      tick {tick.tick} of {run.ticks[run.ticks.length - 1].tick} · phase {tick.phase} · simulated t {num(tick.sim_t ?? 0, 2)} s · fingertip{" "}
                      {tick.tcp.map((v) => v.toFixed(3)).join(", ")} m · jaw {num(tick.jaw, 3)} rad
                    </p>
                  )}
                  <p className="mt-1 text-xs text-fg-muted">Arrow keys step one 10 Hz tick.</p>
                </div>
                <div>
                  <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Outcome</h3>
                  {run.outcome ? (
                    <p className="text-sm">
                      <span className={run.outcome.placed ? "text-ok" : "text-warn"}>{run.outcome.message}</span>
                      <span className="num block text-xs text-fg-muted">
                        grounding {run.outcome.grounding_correct === null ? "not scored" : run.outcome.grounding_correct ? "correct" : "wrong"} · lifted{" "}
                        {String(run.outcome.lifted)} · placed {String(run.outcome.placed)}
                      </span>
                    </p>
                  ) : (
                    <NotRun />
                  )}
                  <h3 className="mb-1 mt-3 text-xs font-medium uppercase tracking-wide text-fg-muted">Stages</h3>
                  <ul className="space-y-0.5">
                    {run.stageOrder.map((s) => {
                      const e = run.stages[s]!;
                      return (
                        <li key={s} className="flex items-center gap-2 text-xs">
                          <span className={STATUS_COLOR[e.status]}>
                            <StatusIcon status={e.status} />
                          </span>
                          <span className="w-20 flex-none">{s}</span>
                          <span className="num text-fg-muted">{e.latency_ms === null ? "not timed" : ms(e.latency_ms, 1)}</span>
                          {e.message && <span className="truncate text-fg-muted">{e.message}</span>}
                        </li>
                      );
                    })}
                  </ul>
                  {run.stages.execute?.payload?.max_pos_err_m !== undefined && (
                    <p className="num mt-2 text-xs text-fg-muted">worst IK position error {metres(run.stages.execute.payload.max_pos_err_m as number)}</p>
                  )}
                </div>
              </div>
            </Card>

            <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-2">
              <Card title="Joint angles against their targets" subtitle="Solid: measured. Dashed: the target commanded that tick. The gap is servo lag.">
                <div className="mb-2 flex flex-wrap gap-3">
                  {JOINTS.map((name, i) => (
                    <label key={name} className="flex items-center gap-1 text-xs">
                      <input
                        type="checkbox"
                        checked={joints[i]}
                        onChange={(e) => setJoints((j) => j.map((v, k) => (k === i ? e.target.checked : v)))}
                        className="accent-[color:var(--accent)]"
                      />
                      <span className="inline-block h-2.5 w-2.5 flex-none rounded-s" style={{ background: JOINT_COLOURS[i] }} aria-hidden="true" />
                      <span>{name}</span>
                    </label>
                  ))}
                </div>
                {run.ticks.length ? <JointChart run={run} tick={tick?.tick ?? 0} joints={joints} /> : <NotRun />}
              </Card>
              <Card title="Fingertip" subtitle="Height above the table, and horizontal distance to the commanded grasp point.">
                {run.ticks.length ? <TrajectoryChart run={run} tick={tick?.tick ?? 0} /> : <NotRun />}
              </Card>
            </div>

            <Card title="Latency waterfall" subtitle={<Term term="monitor mode">Measured during this run, on a GPU shared with other processes</Term>}>
              <Waterfall run={run} />
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
