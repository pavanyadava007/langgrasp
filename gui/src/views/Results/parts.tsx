import type { ReactNode } from "react";
import { NotRun, Term } from "../../components/ui";
import { ago, ms, num } from "../../lib/format";
import { isRate, type GroupRates, type LatencySummary, type Rate } from "../../lib/results";

/** A rate with its Wilson interval, drawn to scale. The bar is the interval, the tick is the point estimate. */
export function RateBar({ rate, label }: { rate: Rate | undefined | null; label?: string }) {
  if (!isRate(rate)) return <NotRun what={label} />;
  const pct = (v: number) => `${(100 * v).toFixed(0)}%`;
  return (
    <div className="min-w-[9rem]">
      <div className="num text-xs">
        {(100 * rate.p).toFixed(1)}% <span className="text-fg-muted">[{pct(rate.lo)}, {pct(rate.hi)}]</span> <span className="text-fg-muted">{rate.k}/{rate.n}</span>
      </div>
      <div className="relative mt-1 h-2 rounded-s bg-surface-2" title={`${rate.k} of ${rate.n}, Wilson 95% interval ${pct(rate.lo)} to ${pct(rate.hi)}`}>
        <div className="absolute h-2 rounded-s bg-accent opacity-40" style={{ left: `${100 * rate.lo}%`, width: `${Math.max(1, 100 * (rate.hi - rate.lo))}%` }} />
        <div className="absolute h-2 w-[2px] bg-accent" style={{ left: `calc(${100 * rate.p}% - 1px)` }} />
      </div>
    </div>
  );
}

export function Source({ file, mtime, extra }: { file: string | string[]; mtime?: number | null; extra?: ReactNode }) {
  const files = Array.isArray(file) ? file : [file];
  return (
    <span className="num">
      source: {files.map((f) => `results/${f}`).join(", ")}
      {mtime ? ` · ${ago(mtime)}` : ""}
      {extra ? <> · {extra}</> : null}
    </span>
  );
}

export function RatesRow({ rates }: { rates: GroupRates | undefined }) {
  return (
    <>
      <td className="py-1 pr-3">
        <RateBar rate={rates?.grasp} label="grasp" />
      </td>
      <td className="py-1 pr-3">
        <RateBar rate={rates?.place} label="place" />
      </td>
      <td className="py-1">
        <RateBar rate={rates?.grounding} label="grounding" />
      </td>
    </>
  );
}

export function BreakdownTable({ groups, caption }: { groups: Record<string, GroupRates> | undefined; caption: string }) {
  const entries = Object.entries(groups ?? {});
  if (!entries.length) return <NotRun />;
  return (
    <table className="w-full text-sm">
      <caption className="mb-2 text-left text-xs text-fg-muted">{caption}</caption>
      <thead>
        <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
          <th className="py-1 pr-3">group</th>
          <th className="py-1 pr-3">grasp</th>
          <th className="py-1 pr-3">place</th>
          <th className="py-1">grounding</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([name, rates]) => (
          <tr key={name} className="border-t border-edge">
            <td className="py-1 pr-3">{name}</td>
            <RatesRow rates={rates} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A stacked bar of the per-stage latency budget. Stage widths are the medians the file records. */
export function LatencyBar({ stages, total }: { stages: { name: string; ms: number; colour: string }[]; total: number }) {
  return (
    <div>
      <div className="flex h-5 w-full overflow-hidden rounded-s bg-surface-2">
        {stages.map((s) => (
          <div key={s.name} className="h-5" style={{ width: `${(100 * s.ms) / total}%`, background: s.colour }} title={`${s.name}: ${ms(s.ms, 1)}`} />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {stages.map((s) => (
          <li key={s.name} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-s" style={{ background: s.colour }} />
            <span>{s.name}</span>
            <span className="num text-fg-muted">{ms(s.ms, 1)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function LatencyTable({ latency, keys }: { latency: Record<string, LatencySummary> | undefined; keys: string[] }) {
  if (!latency) return <NotRun />;
  const rows = keys.filter((k) => latency[k]?.n);
  if (!rows.length) return <NotRun />;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs uppercase tracking-wide text-fg-muted">
          <th className="py-1 pr-3">stage</th>
          <th className="py-1 pr-3">median</th>
          <th className="py-1 pr-3">p90</th>
          <th className="py-1 pr-3">p99</th>
          <th className="py-1">n</th>
        </tr>
      </thead>
      <tbody className="num">
        {rows.map((k) => (
          <tr key={k} className="border-t border-edge">
            <td className="py-1 pr-3 font-sans">{k}</td>
            <td className="py-1 pr-3">{ms(latency[k].median_ms, 1)}</td>
            <td className="py-1 pr-3">{ms(latency[k].p90_ms, 1)}</td>
            <td className="py-1 pr-3">{ms(latency[k].p99_ms, 1)}</td>
            <td className="py-1">{latency[k].n}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Delta({ a, b }: { a: Rate | undefined; b: Rate | undefined }) {
  if (!isRate(a) || !isRate(b)) return <NotRun />;
  const d = 100 * (b.p - a.p);
  const colour = d > 0.05 ? "text-ok" : d < -0.05 ? "text-fail" : "text-fg-muted";
  return (
    <span className={`num ${colour}`} title={`${b.k}/${b.n} against ${a.k}/${a.n}: the difference of two measured rates`}>
      {d > 0 ? "+" : ""}
      {d.toFixed(1)} points ({b.k - a.k} placements)
    </span>
  );
}

export function KeyValue({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="text-sm">
      {rows.map(([k, v], i) => (
        <div key={i} className="flex gap-3 border-b border-edge py-1 last:border-0">
          <dt className="w-52 flex-none text-xs uppercase tracking-wide text-fg-muted">{k}</dt>
          <dd className="num min-w-0 flex-1 break-words">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Glossary() {
  return (
    <p className="text-xs text-fg-muted">
      Rates carry <Term term="wilson interval">Wilson 95% intervals</Term>. <Term term="stratum">Strata</Term> are seen, unseen and language variation.{" "}
      <Term term="ablation">Ablations</Term> switch one part off. Nothing here is a hardware measurement, and every card names the file it came from.
    </p>
  );
}

export function fmtPct(r: Rate | undefined): string {
  return isRate(r) ? `${(100 * r.p).toFixed(1)}%` : "not run";
}

export function fmtNum(v: unknown, digits = 3): string {
  return typeof v === "number" ? num(v, digits) : "not run";
}
