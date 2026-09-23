// Tick history outside React. The executor runs at 10 Hz and the TCP trail needs every sample, but none of it
// should re-render the tree, so it lives here and the canvas reads it on its own animation frame.

import type { TickEvent } from "./types";

const MAX = 4000;
let ticks: TickEvent[] = [];
let version = 0;

export const tickStore = {
  push(tick: TickEvent) {
    ticks.push(tick);
    if (ticks.length > MAX) ticks = ticks.slice(-MAX);
    version += 1;
  },
  clear() {
    ticks = [];
    version += 1;
  },
  all(): readonly TickEvent[] {
    return ticks;
  },
  version(): number {
    return version;
  },
  /** World-space fingertip trail, oldest first. */
  trail(): number[][] {
    return ticks.map((t) => t.tcp);
  },
};
