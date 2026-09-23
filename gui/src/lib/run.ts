// Loading a recorded run and indexing it for the Inspector. Everything here comes out of
// runs/gui/<id>/events.jsonl, which the worker wrote while the run happened.

import { api } from "./api";
import type { OutcomeEvent, SafetyEvent, StageFinishedEvent, StageName, StageStartedEvent, TickEvent } from "./types";

export interface RunMeta {
  run_id?: string;
  seed?: number;
  command?: string;
  controller?: string;
  config?: Record<string, unknown>;
  safety_mode?: string;
  scenario?: { stratum?: string; target?: string; command?: string; lighting?: string; objects?: unknown[] } | null;
  hardware?: string;
  note?: string;
  started?: string;
}

export interface LoadedRun {
  id: string;
  meta: RunMeta;
  ticks: TickEvent[];
  stages: Partial<Record<StageName, StageFinishedEvent>>;
  stageOrder: StageName[];
  outcome: OutcomeEvent | null;
  safety: SafetyEvent[];
  framesUrl: string;
  t0: number;
}

export async function loadRun(id: string): Promise<LoadedRun> {
  const { meta, events, frames_url } = await api.runDetail(id);
  const ticks: TickEvent[] = [];
  const stages: Partial<Record<StageName, StageFinishedEvent>> = {};
  const started: Partial<Record<StageName, Record<string, unknown>>> = {};
  const stageOrder: StageName[] = [];
  const safety: SafetyEvent[] = [];
  let outcome: OutcomeEvent | null = null;
  let t0 = Infinity;
  for (const raw of events as { type: string; t: number }[]) {
    t0 = Math.min(t0, raw.t);
    switch (raw.type) {
      case "tick":
        ticks.push(raw as unknown as TickEvent);
        break;
      case "stage_started": {
        const e = raw as unknown as StageStartedEvent;
        started[e.stage] = { ...(started[e.stage] ?? {}), ...e.payload };
        break;
      }
      case "stage_finished": {
        const e = raw as unknown as StageFinishedEvent;
        if (!(e.stage in stages)) stageOrder.push(e.stage);
        // The started payload says what the stage was asked to do; the finished payload says what happened.
        stages[e.stage] = { ...e, payload: { ...(started[e.stage] ?? {}), ...e.payload } };
        break;
      }
      case "outcome":
        outcome = raw as unknown as OutcomeEvent;
        break;
      case "safety":
        safety.push(raw as unknown as SafetyEvent);
        break;
      default:
        break;
    }
  }
  ticks.sort((a, b) => a.tick - b.tick);
  return { id, meta: meta as RunMeta, ticks, stages, stageOrder, outcome, safety, framesUrl: frames_url, t0: Number.isFinite(t0) ? t0 : 0 };
}

export function frameUrlForTick(run: LoadedRun, tick: number): string {
  return `${run.framesUrl}/tick_${String(tick).padStart(5, "0")}_front.jpg`;
}

/** Stage bars for the waterfall: start offset and duration in milliseconds, relative to the run's first event. */
export function waterfall(run: LoadedRun): { stage: StageName; startMs: number; ms: number; status: string; kind: string }[] {
  const rows: { stage: StageName; startMs: number; ms: number; status: string; kind: string }[] = [];
  for (const stage of run.stageOrder) {
    const e = run.stages[stage]!;
    const ms = e.latency_ms ?? 0;
    const endMs = (e.t - run.t0) * 1000;
    rows.push({ stage, startMs: Math.max(0, endMs - ms), ms, status: e.status, kind: e.latency_kind });
  }
  return rows;
}
