import { StatusIcon } from "../../components/Status";
import { Button, Term } from "../../components/ui";
import { ms, num } from "../../lib/format";
import { useStore } from "../../store/store";
import type { SafetyStateName } from "../../lib/types";

// Always visible, because a safety reviewer should never have to go looking for the state. Everything here is
// the monitor's own report: state, the reason in its own words, the watchdog ages it measured, the gate value
// against its threshold, and what it clipped or would have clipped.

const STATE_STYLE: Record<SafetyStateName, { icon: string; className: string; word: string }> = {
  RUN: { icon: "ok", className: "text-ok border-ok", word: "RUN" },
  REDUCED_SPEED: { icon: "reduced", className: "text-warn border-warn", word: "REDUCED SPEED" },
  HOLD: { icon: "hold", className: "text-warn border-warn", word: "HOLD" },
  ESTOP: { icon: "estop", className: "text-estop border-estop", word: "E-STOP" },
};

function Watchdog({ topic, age, timeout }: { topic: string; age: number | null; timeout: number }) {
  const stale = age === null || age > timeout;
  return (
    <div className="flex items-baseline justify-between gap-2 py-0.5">
      <span className="text-xs text-fg-muted">{topic}</span>
      <span className={`num text-xs ${stale ? "text-warn" : ""}`}>
        {age === null ? "never sampled" : `${age.toFixed(2)} s`}
        <span className="text-fg-muted"> / {timeout.toFixed(2)} s</span>
      </span>
    </div>
  );
}

export function SafetyRail({ compact = false }: { compact?: boolean }) {
  const safety = useStore((s) => s.safety);
  const gate = useStore((s) => s.gate);
  const estopLatency = useStore((s) => s.estopLatencyMs);
  const estop = useStore((s) => s.estop);
  const reset = useStore((s) => s.reset);
  const confirmGate = useStore((s) => s.confirmGate);
  const paused = useStore((s) => s.paused);
  const togglePause = useStore((s) => s.togglePause);
  const stepOnce = useStore((s) => s.stepOnce);

  const state: SafetyStateName = safety?.state ?? "HOLD";
  const style = STATE_STYLE[state];
  const latched = state === "ESTOP";

  const watchdogs = (
    <div className="mt-3">
      <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">
        <Term term="watchdog">Watchdogs</Term>
      </h3>
      <Watchdog topic="camera" age={safety?.watchdog.camera ?? null} timeout={0.5} />
      <Watchdog topic="joint states" age={safety?.watchdog.joint_states ?? null} timeout={0.2} />
      <Watchdog topic="command" age={safety?.watchdog.command ?? null} timeout={30} />
    </div>
  );

  const clipping = (
    <div className="mt-3">
      <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Clipping</h3>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-fg-muted">applied</span>
        <span className="num">
          {safety?.clips.velocity ?? 0} velocity, {safety?.clips.limit ?? 0} joint limit
        </span>
      </div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-fg-muted">would clip</span>
        <span className="num">
          {safety?.would_clip.velocity ?? 0} velocity, {safety?.would_clip.limit ?? 0} joint limit
        </span>
      </div>
      {safety?.mode === "monitor" && (safety.would_clip.velocity ?? 0) > 0 && (
        <p className="mt-1 text-xs text-fg-muted">
          Reported, not applied. Enforcing the 1.5 rad/s limit changes the trajectory: 120/120 placed observing against 73/120 enforcing, measured in
          results/safety_clip_audit.json.
        </p>
      )}
    </div>
  );

  const gateBlock = gate ? (
    <div className="mt-3 rounded-m border border-edge bg-surface-2 p-3">
      <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-fg-muted">Grounding gate</h3>
      <p className="num text-sm">
        {num(gate.score, 3)} <span className="text-fg-muted">vs threshold</span> {num(gate.threshold, 2)}
      </p>
      {gate.top2.length > 1 && (
        <p className="num mt-1 text-xs text-fg-muted">
          top two {gate.top2.map((v) => v.toFixed(3)).join(" / ")}
          {gate.ambiguous ? <span className="text-warn"> within the ambiguity margin</span> : null}
        </p>
      )}
      {gate.needs_confirmation && (
        <div className="mt-2 flex gap-2">
          <Button variant="primary" onClick={() => confirmGate("confirm")}>
            Confirm
          </Button>
          <Button onClick={() => confirmGate("reject")}>Reject</Button>
        </div>
      )}
    </div>
  ) : null;

  const controls = (
    <>
      <Button variant="danger" full onClick={estop} title="Latch the e-stop (keyboard: E)">
        E-STOP
      </Button>
      <Button full onClick={reset} disabled={!latched} title={latched ? "Release the latch, the arm can move again" : "Nothing is latched"}>
        Reset {latched ? "" : "(nothing latched)"}
      </Button>
      <div className="flex gap-2">
        <Button full onClick={togglePause} title="Pause or resume the executor (keyboard: Space)">
          {paused ? "Resume" : "Pause"}
        </Button>
        <Button full onClick={stepOnce} disabled={!paused} title="Advance one 10 Hz tick">
          Step
        </Button>
      </div>
    </>
  );

  const latency = (
    <p className="num text-xs text-fg-muted">
      e-stop latency: {estopLatency === null ? "not measured yet" : ms(estopLatency, 1)}
      {estopLatency !== null && <span className="block">click to the arm being held, this machine's monotonic clock</span>}
    </p>
  );

  if (compact) {
    // A slim bar: state, reason and the two controls that must always be reachable. Everything else is one
    // tap away, so the bar never eats the viewport on a narrow screen.
    return (
      <div className="px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`flex items-center gap-1.5 rounded-m border px-2 py-1 text-sm font-semibold ${style.className}`} role="status" aria-live="assertive">
            <StatusIcon status={style.icon} />
            {style.word}
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-fg-muted">{safety?.reason || "ok"}</span>
          {gate && (
            <span className="num text-xs">
              gate {num(gate.score, 2)}/{num(gate.threshold, 2)}
            </span>
          )}
          <Button variant="danger" onClick={estop}>
            E-STOP
          </Button>
          <Button onClick={reset} disabled={!latched}>
            Reset
          </Button>
        </div>
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-fg-muted">Watchdogs, clipping and pacing</summary>
          <div className="pb-2">
            {gateBlock}
            {watchdogs}
            {clipping}
            <div className="mt-3 flex gap-2">
              <Button full onClick={togglePause}>
                {paused ? "Resume" : "Pause"}
              </Button>
              <Button full onClick={stepOnce} disabled={!paused}>
                Step
              </Button>
            </div>
            <div className="mt-2">{latency}</div>
          </div>
        </details>
      </div>
    );
  }

  return (
    <aside className="rounded-l border border-edge bg-surface p-3" aria-label="Safety" style={{ boxShadow: "var(--shadow)" }}>
      <h2 className="mb-2 text-sm font-semibold">Safety</h2>
      <div className={`flex items-center gap-2 rounded-m border px-3 py-2 ${style.className}`} role="status" aria-live="assertive">
        <StatusIcon status={style.icon} />
        <span className="font-semibold tracking-wide">{style.word}</span>
        {safety?.mode ? (
          <span className="ml-auto text-xs text-fg-muted">
            <Term term="monitor mode">{safety.mode}</Term>
          </span>
        ) : null}
      </div>
      <p className="mt-2 text-xs text-fg-muted">{safety ? safety.reason || "ok" : "waiting for the monitor's first report"}</p>
      {gateBlock}
      {watchdogs}
      {clipping}
      <div className="mt-4 space-y-2">
        {controls}
        {latency}
      </div>
    </aside>
  );
}
