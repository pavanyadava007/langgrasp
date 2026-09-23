import { useEffect, useRef } from "react";
import { Button, Kbd } from "../../components/ui";
import { useStore } from "../../store/store";

// The command and the example chips. Chips come from /api/scene, which builds them from the scenario
// generator's own grammar, so they are the three strata rather than a hand-picked list.

export function CommandBar() {
  const command = useStore((s) => s.command);
  const setState = useStore((s) => s.setState);
  const examples = useStore((s) => s.examples);
  const startRun = useStore((s) => s.startRun);
  const running = useStore((s) => s.running);
  const scene = useStore((s) => s.scene);
  const safety = useStore((s) => s.safety);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onFocusCommand = () => inputRef.current?.focus();
    window.addEventListener("langgrasp:focus-command", onFocusCommand);
    return () => window.removeEventListener("langgrasp:focus-command", onFocusCommand);
  }, []);

  const latched = safety?.state === "ESTOP";
  const custom = scene ? command.trim().toLowerCase() !== scene.command.trim().toLowerCase() : false;

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!running && !latched) void startRun("typed");
        }}
        className="flex flex-wrap items-center gap-2"
      >
        <label htmlFor="command" className="sr-only">
          Command for the robot
        </label>
        <input
          id="command"
          ref={inputRef}
          value={command}
          onChange={(e) => setState({ command: e.target.value })}
          placeholder="pick the blue screwdriver"
          autoComplete="off"
          spellCheck={false}
          className="min-w-[16rem] flex-1 rounded-m border border-edge bg-surface-2 px-3 py-2 text-base"
        />
        <Button type="submit" variant="primary" disabled={running || latched} title={latched ? "Reset the e-stop first" : "Run this command (Enter)"}>
          {running ? "Running" : "Run"} <Kbd>⏎</Kbd>
        </Button>
      </form>
      <div className="mt-2 flex flex-wrap items-baseline gap-1.5">
        <span className="mr-1 text-xs text-fg-muted">examples:</span>
        {examples.map((ex) => (
          <button
            key={ex.command}
            type="button"
            title={`${ex.stratum}: ${ex.why}`}
            onClick={() => setState({ command: ex.command })}
            className="rounded-s border border-edge bg-surface-2 px-2 py-1 text-xs text-fg-muted hover:border-accent hover:text-fg"
          >
            <span className="text-[10px] uppercase tracking-wide text-fg-muted">{ex.stratum}</span> {ex.command}
          </button>
        ))}
        {scene && (
          <button
            type="button"
            onClick={() => setState({ command: scene.command })}
            title="The command this scene was generated for, the only one the ground truth can grade"
            className="rounded-s border border-accent px-2 py-1 text-xs text-accent"
          >
            this scene: {scene.command}
          </button>
        )}
      </div>
      {custom && (
        <p className="mt-2 text-xs text-warn">
          This is not the command this scene was generated for, so grounding cannot be graded: the simulator only knows that {scene?.target} is this scene's
          target. The run still happens, and grasp and place still refer to {scene?.target}.
        </p>
      )}
    </div>
  );
}
