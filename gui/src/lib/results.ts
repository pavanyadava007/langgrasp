// Reading results/*.json for the dashboard. Nothing is computed here that is not arithmetic on numbers the
// files already contain, and a file that is absent stays absent: it never becomes a zero.

import { api, type ManifestSection } from "./api";

export interface Rate {
  k: number;
  n: number;
  p: number;
  lo: number;
  hi: number;
}

export interface GroupRates {
  grasp?: Rate;
  place?: Rate;
  grounding?: Rate;
}

export interface LatencySummary {
  n: number;
  median_ms?: number;
  p90_ms?: number;
  p99_ms?: number;
  mean_ms?: number;
  max_ms?: number;
}

export interface ProtocolFile {
  approach: string;
  notes?: string;
  hardware: string;
  minutes?: number;
  summary: {
    n_trials: number;
    strata: Record<string, GroupRates>;
    lang_variants: Record<string, GroupRates>;
    lighting: Record<string, GroupRates>;
    kinds: Record<string, GroupRates>;
    aborts: Record<string, number>;
    latency_ms: Record<string, LatencySummary>;
  };
  trials: {
    seed: number;
    stratum: string;
    lang_variant: string;
    lighting: string;
    command: string;
    target: string;
    kind: string;
    color: string;
    grounding_correct: boolean | null;
    grasped: boolean;
    lifted: boolean;
    placed: boolean;
    aborted: string | null;
    latency_ms: Record<string, number>;
    extra: Record<string, unknown>;
  }[];
}

const cache = new Map<string, Promise<unknown>>();

export function readResults<T>(name: string): Promise<T> {
  if (!cache.has(name)) cache.set(name, api.resultsFile<T>(name));
  return cache.get(name) as Promise<T>;
}

export function clearResultsCache() {
  cache.clear();
}

export type Manifest = Record<string, ManifestSection>;

export async function readManifest(): Promise<Manifest> {
  const { sections } = await api.manifest();
  return sections;
}

/** Load every present file of a section, keyed by its label. Absent files map to null, never to a zero. */
export async function readSection<T>(section: ManifestSection): Promise<Record<string, T | null>> {
  const entries = await Promise.all(
    Object.entries(section.files).map(async ([label, info]) => {
      if (!info.present || !info.parses) return [label, null] as const;
      try {
        return [label, await readResults<T>(info.file)] as const;
      } catch {
        return [label, null] as const;
      }
    }),
  );
  return Object.fromEntries(entries);
}

export function isRate(r: Rate | undefined | null): r is Rate {
  return !!r && typeof r.n === "number" && r.n > 0 && Number.isFinite(r.p);
}
