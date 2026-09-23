import { useMediaQuery } from "../../lib/media";
import { CommandBar } from "./CommandBar";
import { Controls } from "./Controls";
import { OutcomeCard } from "./OutcomeCard";
import { SafetyRail } from "./SafetyRail";
import { StageDrawer } from "./StageDrawer";
import { Stepper } from "./Stepper";
import { Viewport } from "./Viewport";

// Three columns above 1280 px: controls, viewport with the stepper, safety. Below that the controls collapse
// into a details element and the safety rail moves to a fixed bar, so nothing about safety is ever hidden.
export function LiveRun() {
  // The controls exist once in the DOM, not once per breakpoint: two copies would mean two elements with the
  // same id, which breaks every label association and is invalid HTML.
  const wide = useMediaQuery("(min-width: 1280px)");
  return (
    <div className="grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[280px_minmax(0,1fr)_280px]">
      <div className="hidden xl:block">
        {wide && (
          <div className="sticky top-4 max-h-[calc(100vh-2rem)] overflow-y-auto rounded-l border border-edge bg-surface p-3">
            <Controls />
          </div>
        )}
      </div>

      <div className="min-w-0 space-y-4">
        <div className="rounded-l border border-edge bg-surface p-3">
          <CommandBar />
        </div>
        {!wide && (
          <details className="rounded-l border border-edge bg-surface p-3 xl:hidden">
            <summary className="cursor-pointer text-sm font-semibold">Scene and controls</summary>
            <div className="mt-3">
              <Controls />
            </div>
          </details>
        )}
        <div className="rounded-l border border-edge bg-surface p-3">
          <Viewport />
        </div>
        <div className="rounded-l border border-edge bg-surface p-3">
          <h2 className="mb-2 text-sm font-semibold">Pipeline</h2>
          <Stepper />
        </div>
        <StageDrawer />
        <OutcomeCard />
        <div className="h-24 xl:hidden" />
      </div>

      <div className="hidden xl:block">
        {wide && (
          <div className="sticky top-4 max-h-[calc(100vh-2rem)] overflow-y-auto">
            <SafetyRail />
          </div>
        )}
      </div>

      {!wide && (
        <div className="fixed inset-x-0 bottom-0 z-30 border-t border-edge bg-surface xl:hidden" style={{ boxShadow: "var(--shadow)" }}>
          <SafetyRail compact />
        </div>
      )}
    </div>
  );
}
