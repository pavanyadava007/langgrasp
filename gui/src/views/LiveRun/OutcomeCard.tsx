import { StatusIcon } from "../../components/Status";
import { Term } from "../../components/ui";
import { ms } from "../../lib/format";
import { useStore } from "../../store/store";

// Three graded levels and the reason, with one rule: if the ground truth cannot grade this command, the card
// says so rather than showing a failure that is not one.

function Level({ label, value, help }: { label: string; value: boolean | null; help?: string }) {
  const status = value === null ? "pending" : value ? "ok" : "fail";
  const colour = value === null ? "text-fg-muted" : value ? "text-ok" : "text-fail";
  return (
    <div className="flex items-center gap-2" title={help}>
      <span className={colour}>
        <StatusIcon status={status} />
      </span>
      <span className="text-sm">
        {label}
        <span className={`ml-1 ${colour}`}>{value === null ? "not scored" : value ? "yes" : "no"}</span>
      </span>
    </div>
  );
}

export function OutcomeCard() {
  const outcome = useStore((s) => s.outcome);
  const running = useStore((s) => s.running);
  const scene = useStore((s) => s.scene);

  if (!outcome) {
    return (
      <div className="rounded-l border border-edge bg-surface p-3">
        <h2 className="text-sm font-semibold">Outcome</h2>
        <p className="mt-1 text-sm text-fg-muted">{running ? "Run in progress." : "No run yet. Press Run, or pick an example command."}</p>
      </div>
    );
  }

  const unscored = outcome.scored_by === "none";
  return (
    <div className={`rounded-l border bg-surface p-3 ${outcome.placed ? "border-ok" : outcome.aborted ? "border-warn" : "border-edge"}`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-semibold">
          Outcome{" "}
          <span className="text-xs font-normal text-fg-muted">
            {unscored ? "not graded" : <>graded against the simulator's ground truth, which the pipeline never sees</>}
          </span>
        </h2>
        <span className="num text-xs text-fg-muted">
          {outcome.latency_ms.end_to_end ? `end to end ${ms(outcome.latency_ms.end_to_end)}` : ""}
          {outcome.sim_compute_ms ? ` · simulator compute ${ms(outcome.sim_compute_ms)}` : ""}
          {outcome.wall_ms ? ` · wall ${ms(outcome.wall_ms)}` : ""}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1">
        <Level label="grounding correct:" value={unscored ? null : outcome.grounding_correct} help="The chosen box centre lies on the target's pixels in the simulator's label image" />
        <Level label="grasped and lifted:" value={outcome.lifted} help="Held by both jaw pads and raised more than 5 cm" />
        <Level label="placed in the tray:" value={outcome.placed} />
      </div>
      {outcome.aborted && (
        <p className="mt-2 text-sm text-warn">
          Aborted: <span className="num">{outcome.aborted}</span>
        </p>
      )}
      <p className="mt-2 text-sm text-fg-muted">{outcome.message}</p>
      {scene && !unscored && (
        <p className="mt-1 text-xs text-fg-muted">
          This scene is <span className="num">seed {scene.seed}</span>, stratum <Term term="stratum">{scene.stratum}</Term>, target{" "}
          <span className="num">{scene.target}</span>.
        </p>
      )}
    </div>
  );
}
