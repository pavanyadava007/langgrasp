// Thin typed wrappers over the REST API. Errors carry the server's own sentence, which is written to be shown
// to a person, so components can render err.message directly instead of inventing their own wording.

import type { ExampleCommand, RunConfig, Scenario, SystemInfo } from "./types";

export class ApiError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  } catch (e) {
    throw new ApiError(0, "offline", `Cannot reach the server at ${location.host}. Is the port forward still open?`);
  }
  if (!res.ok) {
    let code = String(res.status);
    let message = `${init?.method ?? "GET"} ${path} failed with ${res.status}.`;
    try {
      const body = await res.json();
      if (body?.error) {
        code = body.error.code ?? code;
        message = body.error.message ?? message;
      } else if (body?.detail) {
        message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* the body was not JSON; the generic sentence above stands */
    }
    throw new ApiError(res.status, code, message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const post = <T,>(path: string, body?: unknown) => call<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  system: () => call<SystemInfo>("/api/system"),
  scene: () => call<{ scene: Scenario | null; examples: ExampleCommand[] }>("/api/scene"),
  newScene: (body: { seed: number; stratum: string; lighting?: string | null; n_distractors?: number | null; fixed_goal?: boolean }) =>
    post<{ scene: Scenario }>("/api/scene", body),
  run: (body: { command?: string | null; controller: string; config: RunConfig; source?: string; stt_latency_ms?: number | null }) =>
    post<{ run_id: string }>("/api/run", body),
  estop: () => post<{ state: string; estop_latency_ms: number | null; note: string }>("/api/estop", { client_ts_ms: performance.now() }),
  reset: () => post<{ state: string; reason: string }>("/api/reset"),
  pause: (body: { paused?: boolean; step?: boolean }) => post<{ paused: boolean; stepped: boolean }>("/api/pause", body),
  confirm: (decision: "confirm" | "reject") => post<{ accepted: boolean }>("/api/confirm", { decision }),
  results: () => call<{ files: ResultsFileRow[]; note: string }>("/api/results"),
  manifest: () => call<{ sections: Record<string, ManifestSection> }>("/api/results/manifest"),
  resultsFile: <T = unknown,>(name: string) => call<T>(`/api/results/${name}`),
  runs: () => call<{ runs: RunRow[] }>("/api/runs"),
  runDetail: (id: string) => call<{ meta: Record<string, unknown>; events: unknown[]; frames_url: string }>(`/api/runs/${id}`),
  stt: async (blob: Blob, modelSize?: string) => {
    const form = new FormData();
    form.append("audio", blob, "clip.webm");
    const res = await fetch(`/api/stt${modelSize ? `?model_size=${modelSize}` : ""}`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      throw new ApiError(res.status, body?.error?.code ?? String(res.status), body?.error?.message ?? "Transcription failed.");
    }
    return (await res.json()) as { text: string; normalized: string; latency_ms: number; model_size: string; device: string; note: string };
  },
};

export interface ResultsFileRow {
  file: string;
  size: number;
  mtime: number;
  parses: boolean;
  approach: string | null;
  n_trials: number | null;
  hardware: string | null;
}

export interface ManifestSection {
  title: string;
  files: Record<string, { file: string; present: boolean; mtime: number | null; parses: boolean }>;
}

export interface RunRow {
  id: string;
  seed: number | null;
  command: string | null;
  controller: string | null;
  started: string | null;
  safety_mode: string | null;
  mtime: number;
  n_events: number;
}
