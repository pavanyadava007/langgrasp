// Formatting rules for the whole UI. Two of them matter for honesty:
// a number that was never measured reads "not run", and a latency always says what kind of time it is.

export const NOT_RUN = "not run";

export function ms(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NOT_RUN;
  if (value < 1) return `${value.toFixed(2)} ms`;
  if (value < 10) return `${value.toFixed(Math.max(digits, 1))} ms`;
  if (value >= 10_000) return `${(value / 1000).toFixed(1)} s`;
  return `${value.toFixed(digits)} ms`;
}

export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NOT_RUN;
  return value.toFixed(digits);
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NOT_RUN;
  return `${(100 * value).toFixed(digits)}%`;
}

export function metres(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return NOT_RUN;
  return `${(value * 1000).toFixed(1)} mm`;
}

export function degrees(radians: number | null | undefined): string {
  if (radians === null || radians === undefined || !Number.isFinite(radians)) return NOT_RUN;
  return `${((radians * 180) / Math.PI).toFixed(1)}°`;
}

export function xyz(v: number[] | null | undefined): string {
  if (!v || v.length < 3) return NOT_RUN;
  return v.slice(0, 3).map((x) => x.toFixed(3)).join(", ");
}

export function latencyLabel(kind: string): string {
  if (kind === "wall_paced") return "wall clock, includes pacing and rendering";
  if (kind === "none") return "not timed separately";
  return "compute, the same region the evaluation scripts time";
}

export function ago(mtime: number | null | undefined): string {
  if (!mtime) return NOT_RUN;
  const d = new Date(mtime * 1000);
  return d.toISOString().slice(0, 16).replace("T", " ");
}

export function titleCase(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
