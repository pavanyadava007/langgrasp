import type { StageStatus } from "../lib/types";

// Status is never colour alone: every state has an icon and a word. The icons are inline SVG so they inherit
// currentColor and stay crisp in both themes.

const ICONS: Record<string, JSX.Element> = {
  ok: (
    <path d="M3 8.5l3.2 3.2L13 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  ),
  warn: (
    <>
      <path d="M8 2.5l5.8 10.2H2.2L8 2.5z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M8 6.4v3.1M8 11.2v.9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </>
  ),
  fail: <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
  pending: <circle cx="8" cy="8" r="4.6" fill="none" stroke="currentColor" strokeWidth="1.6" strokeDasharray="2 2" />,
  running: <path d="M8 2.2a5.8 5.8 0 1 1-5.8 5.8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
  skipped: <path d="M3 8h10M9.5 4.5L13 8l-3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />,
  estop: <path d="M5.2 2.4h5.6L13.6 5.2v5.6L10.8 13.6H5.2L2.4 10.8V5.2L5.2 2.4z" fill="none" stroke="currentColor" strokeWidth="1.6" />,
  hold: <path d="M6 4v8M10 4v8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />,
  reduced: <path d="M8 3a5 5 0 0 1 0 10z" fill="currentColor" />,
};

export const STATUS_COLOR: Record<string, string> = {
  ok: "text-ok",
  warn: "text-warn",
  fail: "text-fail",
  pending: "text-pending",
  running: "text-running",
  skipped: "text-pending",
  estop: "text-estop",
  hold: "text-warn",
  reduced: "text-warn",
};

export function StatusIcon({ status, spin = false, className = "" }: { status: string; spin?: boolean; className?: string }) {
  const icon = ICONS[status] ?? ICONS.pending;
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" className={`${spin && status === "running" ? "spin" : ""} ${className}`} style={{ flex: "0 0 auto" }}>
      {icon}
    </svg>
  );
}

export function statusWord(status: StageStatus): string {
  switch (status) {
    case "ok":
      return "ok";
    case "warn":
      return "warning";
    case "fail":
      return "failed";
    case "running":
      return "running";
    case "skipped":
      return "skipped";
    default:
      return "pending";
  }
}
