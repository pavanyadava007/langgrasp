import { useEffect, useState } from "react";
import { Card, NotRun, Skeleton, Term } from "../../components/ui";
import type { ManifestSection } from "../../lib/api";
import { ago, ms, num, pct } from "../../lib/format";
import { clearResultsCache, readManifest, readResults, type GroupRates, type LatencySummary, type ProtocolFile, type Rate } from "../../lib/results";
import { BreakdownTable, Delta, Glossary, KeyValue, LatencyBar, LatencyTable, RateBar, Source } from "./parts";

const STAGE_COLOURS: Record<string, string> = {
  parse: "#8b95a3",
  capture: "#4ade80",
  grounding: "#5b96ff",
  segmentation: "#fbbf24",
  depth_fusion: "#d0a3ff",
  execute: "#ff7b72",
};

function useManifest() {
  const [sections, setSections] = useState<Record<string, ManifestSection> | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    readManifest().then(setSections).catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  return { sections, error };
}

function useFile<T>(name: string | undefined, present: boolean) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!name || !present) return;
    setLoading(true);
    readResults<T>(name)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [name, present]);
  return { data, loading };
}

function ProtocolCard({ section }: { section: ManifestSection }) {
  const [files, setFiles] = useState<Record<string, ProtocolFile | null> | null>(null);
  useEffect(() => {
    void (async () => {
      const entries = await Promise.all(
        Object.entries(section.files).map(async ([label, info]) => {
          if (!info.present) return [label, null] as const;
          try {
            return [label, await readResults<ProtocolFile>(info.file)] as const;
          } catch {
            return [label, null] as const;
          }
        }),
      );
      setFiles(Object.fromEntries(entries));
    })();
  }, [section]);

  if (!files) return <Skeleton h={180} />;
  const strata = ["seen", "unseen", "langvar"];
  const baseline = files["Modular: Grounding DINO + colour check + YOLO11-seg + depth fusion"];
  return (
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Protocol by approach and stratum, scrolls sideways">
      <table className="w-full min-w-[54rem] text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
            <th className="py-1 pr-3">approach</th>
            {strata.map((s) => (
              <th key={s} className="py-1 pr-3">
                {s === "langvar" ? "language variation" : s} placed
              </th>
            ))}
            <th className="py-1 pr-3">grounding, all</th>
            <th className="py-1 pr-3">end to end</th>
            <th className="py-1">n</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(section.files).map(([label, info]) => {
            const f = files[label];
            return (
              <tr key={label} className="border-t border-edge align-top">
                <td className="py-2 pr-3">
                  <span className="block max-w-[18rem]">{label}</span>
                  <span className="num block text-xs text-fg-muted">
                    results/{info.file}
                    {info.mtime ? ` · ${ago(info.mtime)}` : ""}
                  </span>
                </td>
                {strata.map((s) => (
                  <td key={s} className="py-2 pr-3">
                    <RateBar rate={f?.summary.strata[s]?.place} label={`${label} ${s}`} />
                  </td>
                ))}
                <td className="py-2 pr-3">
                  <RateBar rate={f?.summary.strata.all?.grounding} label="grounding" />
                </td>
                <td className="num py-2 pr-3 text-xs">{f?.summary.latency_ms?.end_to_end?.median_ms ? `${ms(f.summary.latency_ms.end_to_end.median_ms)} median` : <NotRun />}</td>
                <td className="num py-2 text-xs">{f?.summary.n_trials ?? <NotRun />}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {baseline && (
        <div className="mt-4">
          <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">What each ablation costs, placed rate against the full pipeline</h3>
          <ul className="space-y-1 text-sm">
            {Object.entries(files)
              .filter(([label, f]) => f && label.startsWith("Modular,"))
              .map(([label, f]) => (
                <li key={label} className="flex flex-wrap items-baseline gap-2">
                  <span className="text-fg-muted">{label.replace("Modular, ", "")}:</span>
                  <Delta a={baseline.summary.strata.all?.place} b={f!.summary.strata.all?.place} />
                </li>
              ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function FixedGoalCard({ section }: { section: ManifestSection }) {
  const [files, setFiles] = useState<Record<string, ProtocolFile | null> | null>(null);
  useEffect(() => {
    void (async () => {
      const entries = await Promise.all(
        Object.entries(section.files).map(async ([label, info]) => {
          if (!info.present) return [label, null] as const;
          try {
            return [label, await readResults<ProtocolFile>(info.file)] as const;
          } catch {
            return [label, null] as const;
          }
        }),
      );
      setFiles(Object.fromEntries(entries));
    })();
  }, [section]);
  if (!files) return <Skeleton h={140} />;
  return (
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Fixed-goal comparison, scrolls sideways">
    <table className="w-full min-w-[34rem] text-sm">
      <thead>
        <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
          <th className="py-1 pr-3">approach</th>
          <th className="py-1 pr-3">grasped and lifted</th>
          <th className="py-1 pr-3">placed</th>
          <th className="py-1">n</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(section.files).map(([label, info]) => {
          const f = files[label];
          const all: GroupRates | undefined = f?.summary.strata.all;
          return (
            <tr key={label} className="border-t border-edge align-top">
              <td className="py-2 pr-3">
                <span className="block max-w-[22rem]">{label}</span>
                <span className="num block text-xs text-fg-muted">results/{info.file}</span>
              </td>
              <td className="py-2 pr-3">
                <RateBar rate={all?.grasp} label="grasp" />
              </td>
              <td className="py-2 pr-3">
                <RateBar rate={all?.place} label="place" />
              </td>
              <td className="num py-2 text-xs">{f?.summary.n_trials ?? <NotRun />}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
    </div>
  );
}

interface YoloLatency {
  hardware: string;
  backends: Record<string, { inference_only?: LatencySummary; end_to_end?: LatencySummary; path?: string }>;
}
interface YoloMap {
  backends: Record<string, { box_map50?: number; box_map50_95?: number; mask_map50?: number; mask_map50_95?: number }>;
  val_settings?: string;
}
interface YoloTrain {
  train_images: number;
  val_images: number;
  epochs: number;
  minutes: number;
  box_map50: number;
  box_map50_95: number;
  mask_map50: number;
  mask_map50_95?: number;
  hardware: string;
}

function PerceptionCard({ section }: { section: ManifestSection }) {
  const train = useFile<YoloTrain>(section.files["Fine-tune"]?.file, !!section.files["Fine-tune"]?.present);
  const lat = useFile<YoloLatency>(section.files["Latency"]?.file, !!section.files["Latency"]?.present);
  const maps = useFile<YoloMap>(section.files["mAP per export"]?.file, !!section.files["mAP per export"]?.present);
  const backends = Object.keys(lat.data?.backends ?? {});
  return (
    <div>
      {train.data ? (
        <KeyValue
          rows={[
            ["training images", `${train.data.train_images} train / ${train.data.val_images} val, ${train.data.epochs} epochs, ${num(train.data.minutes, 0)} min`],
            [<Term key="m" term="map50-95">box mAP50 / mAP50-95</Term>, `${num(train.data.box_map50)} / ${num(train.data.box_map50_95)}`],
            ["mask mAP50 / mAP50-95", `${num(train.data.mask_map50)} / ${train.data.mask_map50_95 === undefined ? "not run" : num(train.data.mask_map50_95)}`],
            ["hardware", train.data.hardware],
          ]}
        />
      ) : (
        <NotRun />
      )}
      <h3 className="mb-1 mt-4 text-xs font-medium uppercase tracking-wide text-fg-muted">Latency and accuracy by backend</h3>
      {backends.length ? (
        <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="YOLO backends, scrolls sideways">
          <table className="w-full min-w-[36rem] text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
                <th className="py-1 pr-3">backend</th>
                <th className="py-1 pr-3">inference median</th>
                <th className="py-1 pr-3">end to end median</th>
                <th className="py-1 pr-3">box mAP50</th>
                <th className="py-1">mask mAP50</th>
              </tr>
            </thead>
            <tbody className="num">
              {backends.map((b) => (
                <tr key={b} className="border-t border-edge">
                  <td className="py-1 pr-3 font-sans">{b}</td>
                  <td className="py-1 pr-3">{ms(lat.data!.backends[b].inference_only?.median_ms, 2)}</td>
                  <td className="py-1 pr-3">{ms(lat.data!.backends[b].end_to_end?.median_ms, 2)}</td>
                  <td className="py-1 pr-3">{maps.data?.backends?.[b]?.box_map50 === undefined ? "not run" : num(maps.data.backends[b].box_map50)}</td>
                  <td className="py-1">{maps.data?.backends?.[b]?.mask_map50 === undefined ? "not run" : num(maps.data.backends[b].mask_map50)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <NotRun />
      )}
      {lat.data?.hardware && <p className="num mt-2 text-xs text-fg-muted">{lat.data.hardware}</p>}
    </div>
  );
}

interface DepthFusion {
  clean: { n_objects: number; per_kind: Record<string, Record<string, number>> };
  realsense_like_noise?: { n_objects: number; per_kind: Record<string, Record<string, number>> };
  note?: string;
}

function DepthCard({ file, present }: { file?: string; present: boolean }) {
  const { data } = useFile<DepthFusion>(file, present);
  if (!data) return <NotRun />;
  const kinds = Object.keys(data.clean.per_kind);
  return (
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Depth fusion error by object, scrolls sideways">
      <table className="w-full min-w-[40rem] text-sm">
        <caption className="mb-2 text-left text-xs text-fg-muted">
          Grasp centre and yaw against the simulator's own pose, with ground-truth masks. Clean depth first, then the synthetic noise model.
        </caption>
        <thead>
          <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
            <th className="py-1 pr-3">object</th>
            <th className="py-1 pr-3">n</th>
            <th className="py-1 pr-3">median centre error</th>
            <th className="py-1 pr-3">p95</th>
            <th className="py-1 pr-3">with noise, median</th>
            <th className="py-1">yaw p95</th>
          </tr>
        </thead>
        <tbody className="num">
          {kinds.map((k) => {
            const c = data.clean.per_kind[k];
            const n = data.realsense_like_noise?.per_kind?.[k];
            return (
              <tr key={k} className="border-t border-edge">
                <td className="py-1 pr-3 font-sans">{k}</td>
                <td className="py-1 pr-3">{c.n}</td>
                <td className="py-1 pr-3">{num(c.xy_err_mm_median, 2)} mm</td>
                <td className="py-1 pr-3">{num(c.xy_err_mm_p95, 2)} mm</td>
                <td className="py-1 pr-3">{n ? `${num(n.xy_err_mm_median, 2)} mm` : "not run"}</td>
                <td className="py-1">{c.psi_err_deg_p95 === undefined ? "not measured for this shape" : `${num(c.psi_err_deg_p95, 1)}°`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {data.note && <p className="mt-2 max-w-3xl text-xs text-fg-muted">{data.note}</p>}
    </div>
  );
}

interface GapFile {
  label: string;
  note: string;
  episodes_per_cell: number;
  rows: { checkpoint: string; condition: string; success: number; ci95: [number, number]; k: number; n: number }[];
}

function GapCard({ file, present, task }: { file?: string; present: boolean; task: string }) {
  const { data } = useFile<GapFile>(file, present);
  if (!data) return <NotRun what={task} />;
  return (
    <div>
      <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="PPO sim to sim gap, scrolls sideways">
      <table className="w-full min-w-[26rem] text-sm">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
            <th className="py-1 pr-3">checkpoint</th>
            <th className="py-1 pr-3">condition</th>
            <th className="py-1">success</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((r, i) => (
            <tr key={i} className="border-t border-edge">
              <td className="num py-1 pr-3">{r.checkpoint.split("/").pop()}</td>
              <td className="py-1 pr-3">{r.condition}</td>
              <td className="py-1">
                <RateBar rate={{ k: r.k, n: r.n, p: r.success, lo: r.ci95[0], hi: r.ci95[1] } as Rate} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <p className="mt-2 max-w-3xl text-xs text-fg-muted">{data.note}</p>
    </div>
  );
}

interface PipelineBench {
  hardware: string;
  note: string;
  n_commands: number;
  backends: Record<string, Record<string, LatencySummary | string>>;
}

function LatencyCard({ section }: { section: ManifestSection }) {
  const bench = useFile<PipelineBench>(section.files["Clean bench, no other GPU job"]?.file, !!section.files["Clean bench, no other GPU job"]?.present);
  const protocol = useFile<ProtocolFile>(section.files["During the protocol runs"]?.file, !!section.files["During the protocol runs"]?.present);
  const backend = bench.data ? Object.keys(bench.data.backends)[0] : null;
  const stagesOf = (b: string) =>
    ["parse", "capture", "grounding", "segmentation", "depth_fusion"]
      .map((k) => ({ name: k, ms: ((bench.data!.backends[b] as Record<string, LatencySummary>)[k]?.median_ms ?? 0) as number, colour: STAGE_COLOURS[k] }))
      .filter((s) => s.ms > 0);
  return (
    <div>
      {bench.data && backend ? (
        <>
          <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Perception budget per command, clean bench, {backend} backend</h3>
          <LatencyBar stages={stagesOf(backend)} total={stagesOf(backend).reduce((a, s) => a + s.ms, 0)} />
          <p className="mt-2 max-w-3xl text-xs text-fg-muted">{bench.data.note}</p>
          <div className="mt-3 overflow-x-auto" tabIndex={0} role="region" aria-label="Latency by segmenter backend, scrolls sideways">
            <table className="w-full min-w-[32rem] text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
                  <th className="py-1 pr-3">segmenter backend</th>
                  <th className="py-1 pr-3">grounding</th>
                  <th className="py-1 pr-3">segmentation</th>
                  <th className="py-1 pr-3">depth fusion</th>
                  <th className="py-1">perception total</th>
                </tr>
              </thead>
              <tbody className="num">
                {Object.entries(bench.data.backends).map(([b, v]) => (
                  <tr key={b} className="border-t border-edge">
                    <td className="py-1 pr-3 font-sans">{b}</td>
                    {["grounding", "segmentation", "depth_fusion", "perception_total"].map((k) => (
                      <td key={k} className="py-1 pr-3">
                        {ms((v as Record<string, LatencySummary>)[k]?.median_ms, 1)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <NotRun what="the clean latency bench" />
      )}
      <h3 className="mb-1 mt-4 text-xs font-medium uppercase tracking-wide text-fg-muted">Recorded during the protocol run, with other jobs sharing the GPU</h3>
      <LatencyTable latency={protocol.data?.summary.latency_ms} keys={["parse", "capture", "grounding", "segmentation", "depth_fusion", "execute", "end_to_end"]} />
    </div>
  );
}

interface SttFile {
  hardware: string;
  dataset: string;
  note: string;
  results: Record<string, { median_ms: number; p90_ms: number; wer_on_samples: number; device: string }>;
}
interface RosFile {
  hardware: string;
  rmw: string;
  latency: {
    end_to_end_ms?: LatencySummary;
    since_capture_ms?: Record<string, LatencySummary>;
    per_stage_ms?: Record<string, LatencySummary>;
    n_messages?: Record<string, number>;
  };
  note?: string;
  safety_last?: { message?: string };
}
interface ClipAudit {
  question: string;
  max_joint_vel_rad_per_s: number;
  conclusion: string;
  modes: Record<string, { summary: Record<string, number | string[]> }>;
}

function SpeechRosSafetyCard({ section }: { section: ManifestSection }) {
  const stt = useFile<SttFile>(section.files["faster-whisper latency"]?.file, !!section.files["faster-whisper latency"]?.present);
  const ros = useFile<RosFile>(section.files["ROS 2 Humble pipeline smoke"]?.file, !!section.files["ROS 2 Humble pipeline smoke"]?.present);
  const clip = useFile<ClipAudit>(section.files["Safety clip audit"]?.file, !!section.files["Safety clip audit"]?.present);
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-3">
      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Speech to text</h3>
        {stt.data ? (
          <>
            <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Speech to text latency, scrolls sideways">
            <table className="w-full min-w-[16rem] text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
                  <th className="py-1 pr-2">model</th>
                  <th className="py-1 pr-2">median</th>
                  <th className="py-1">WER</th>
                </tr>
              </thead>
              <tbody className="num">
                {Object.entries(stt.data.results).map(([size, v]) => (
                  <tr key={size} className="border-t border-edge">
                    <td className="py-1 pr-2 font-sans">{size}</td>
                    <td className="py-1 pr-2">{ms(v.median_ms)}</td>
                    <td className="py-1">{pct(v.wer_on_samples)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
            <p className="mt-2 text-xs text-fg-muted">{stt.data.note}</p>
          </>
        ) : (
          <NotRun />
        )}
      </div>
      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">ROS 2 pipeline in Docker</h3>
        {ros.data ? (
          <KeyValue
            rows={[
              ["middleware", ros.data.rmw],
              ["camera frame to a safe joint command", ms(ros.data.latency.end_to_end_ms?.median_ms, 1)],
              ["safety node, added delay", ms(ros.data.latency.per_stage_ms?.["joint_command->joint_command_safe"]?.median_ms, 2)],
              ["commands through the safety node", ros.data.latency.n_messages?.joint_command_safe ?? "not run"],
              ["hardware", ros.data.hardware],
            ]}
          />
        ) : (
          <NotRun />
        )}
      </div>
      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Safety clip audit</h3>
        {clip.data ? (
          <>
            <KeyValue
              rows={[
                ["velocity limit", `${clip.data.max_joint_vel_rad_per_s} rad/s per joint`],
                ["placed, observing", `${clip.data.modes.monitor.summary.placed} of ${clip.data.modes.monitor.summary.n_trials}`],
                ["placed, enforcing", `${clip.data.modes.enforce.summary.placed} of ${clip.data.modes.enforce.summary.n_trials}`],
                ["ticks that would be clipped", pct(clip.data.modes.monitor.summary.clipped_tick_fraction as number)],
                ["peak commanded speed", `${num(clip.data.modes.monitor.summary.commanded_peak_rad_per_s_max as number, 1)} rad/s`],
              ]}
            />
            <p className="mt-2 text-xs text-fg-muted">{clip.data.conclusion}</p>
          </>
        ) : (
          <NotRun />
        )}
      </div>
    </div>
  );
}

export function Results() {
  const { sections, error } = useManifest();
  const [protocolFile, setProtocolFile] = useState<ProtocolFile | null>(null);

  useEffect(() => {
    readResults<ProtocolFile>("modular_protocol.json").then(setProtocolFile).catch(() => setProtocolFile(null));
  }, []);

  if (error) return <Card title="Results">{<p className="text-sm text-fail">{error}</p>}</Card>;
  if (!sections)
    return (
      <div className="space-y-4">
        <Skeleton h={220} />
        <Skeleton h={180} />
      </div>
    );

  const s = sections;
  const srcOf = (section: ManifestSection) => Object.values(section.files).map((f) => f.file);
  const mtimeOf = (section: ManifestSection) => Math.max(0, ...Object.values(section.files).map((f) => f.mtime ?? 0)) || null;

  return (
    <div className="min-w-0 space-y-4">
      <Card title="Measured results" subtitle="Everything below is read from results/*.json at the moment you opened this page. Nothing on this page is computed by the browser except differences between two measured rates.">
        <Glossary />
        <p className="mt-2 text-xs text-fg-muted">
          A rate over zero trials is written as NaN in these files, which is not valid JSON, so the server replaces those with null before the browser sees them
          and a card shows "not run" instead of a number. Add <span className="num">?raw=1</span> to any file link for the bytes on disk.
        </p>
        <button type="button" className="mt-2 text-xs text-accent underline" onClick={() => { clearResultsCache(); location.reload(); }}>
          Re-read the files
        </button>
      </Card>

      <Card title={s.protocol.title} footer={<Source file={srcOf(s.protocol)} mtime={mtimeOf(s.protocol)} />}>
        <ProtocolCard section={s.protocol} />
      </Card>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-3">
        <Card title="By object kind" footer={<Source file="modular_protocol.json" />}>
          <BreakdownTable groups={protocolFile?.summary.kinds} caption="Modular pipeline, all strata." />
        </Card>
        <Card title="By lighting" footer={<Source file="modular_protocol.json" />}>
          <BreakdownTable groups={protocolFile?.summary.lighting} caption="Degraded lighting is 25 to 45% of nominal brightness." />
        </Card>
        <Card title="By language variant" footer={<Source file="modular_protocol.json" />}>
          <BreakdownTable groups={protocolFile?.summary.lang_variants} caption="Plain, synonym, attribute only, spatial reference." />
        </Card>
      </div>

      {protocolFile && Object.keys(protocolFile.summary.aborts ?? {}).length > 0 && (
        <Card title="Why the pipeline refused to act" footer={<Source file="modular_protocol.json" />}>
          <ul className="num text-sm">
            {Object.entries(protocolFile.summary.aborts).map(([reason, n]) => (
              <li key={reason}>
                {n} × {reason}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card title={s.fixed_goal.title} subtitle="The same 100 scenes for every approach, so ACT and the pipeline are compared on identical scenes." footer={<Source file={srcOf(s.fixed_goal)} mtime={mtimeOf(s.fixed_goal)} />}>
        <FixedGoalCard section={s.fixed_goal} />
      </Card>

      <Card title={s.latency.title} footer={<Source file={srcOf(s.latency)} mtime={mtimeOf(s.latency)} />}>
        <LatencyCard section={s.latency} />
      </Card>

      <Card title={s.perception.title} footer={<Source file={srcOf(s.perception)} mtime={mtimeOf(s.perception)} />}>
        <PerceptionCard section={s.perception} />
      </Card>

      <Card title="Depth fusion against the simulator's own poses" footer={<Source file={s.perception.files["Depth fusion vs ground truth"]?.file ?? "depth_fusion_accuracy.json"} />}>
        <DepthCard file={s.perception.files["Depth fusion vs ground truth"]?.file} present={!!s.perception.files["Depth fusion vs ground truth"]?.present} />
      </Card>

      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-2">
        <Card title="PPO reach, sim to sim" subtitle={<Term term="sim-to-sim gap">nominal against a deliberately shifted simulator</Term>} footer={<Source file={s.rl.files.Reach?.file ?? "ppo_sim2sim_gap.json"} />}>
          <GapCard file={s.rl.files.Reach?.file} present={!!s.rl.files.Reach?.present} task="PPO reach" />
        </Card>
        <Card title="PPO lift, sim to sim" footer={<Source file={s.rl.files.Lift?.file ?? "ppo_sim2sim_gap_lift.json"} />}>
          <GapCard file={s.rl.files.Lift?.file} present={!!s.rl.files.Lift?.present} task="PPO lift" />
        </Card>
      </div>

      <Card title={s.speech_ros_safety.title} footer={<Source file={srcOf(s.speech_ros_safety)} mtime={mtimeOf(s.speech_ros_safety)} />}>
        <SpeechRosSafetyCard section={s.speech_ros_safety} />
      </Card>
    </div>
  );
}
