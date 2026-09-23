import { describe, expect, it } from "vitest";
import { frameUrlForTick, waterfall, type LoadedRun } from "../run";
import type { StageFinishedEvent, TickEvent } from "../types";

function stage(stageName: string, t: number, latency: number | null, status = "ok"): StageFinishedEvent {
  return {
    type: "stage_finished",
    t,
    run_id: "r1",
    stage: stageName as StageFinishedEvent["stage"],
    status: status as StageFinishedEvent["status"],
    latency_ms: latency,
    latency_kind: latency === null ? "none" : "compute",
    message: null,
    payload: {},
    images: {},
  };
}

const run: LoadedRun = {
  id: "2026-09-23T13-40-04_5000",
  meta: { seed: 5000, command: "pick the blue cube" },
  ticks: [{ tick: 1 } as TickEvent, { tick: 2 } as TickEvent],
  stages: { parse: stage("parse", 100.1, 0.1), grounding: stage("grounding", 100.4, 273), select: stage("select", 100.4, null), execute: stage("execute", 106.4, 5923) },
  stageOrder: ["parse", "grounding", "select", "execute"],
  outcome: null,
  safety: [],
  framesUrl: "/runs/2026-09-23T13-40-04_5000/frames",
  t0: 100,
};

describe("waterfall", () => {
  it("places each bar so that it ends when the stage ended", () => {
    const rows = waterfall(run);
    expect(rows.map((r) => r.stage)).toEqual(["parse", "grounding", "select", "execute"]);
    const grounding = rows.find((r) => r.stage === "grounding")!;
    // finished 400 ms into the run after 273 ms of work, so it started at 127 ms
    expect(grounding.startMs).toBeCloseTo(127, 6);
    expect(grounding.ms).toBe(273);
  });

  it("gives an untimed stage no bar rather than a zero-length one at the origin", () => {
    const select = waterfall(run).find((r) => r.stage === "select")!;
    expect(select.ms).toBe(0);
    expect(select.kind).toBe("none");
  });

  it("never places a bar before the start of the run", () => {
    const odd: LoadedRun = { ...run, stages: { parse: stage("parse", 100.05, 500) }, stageOrder: ["parse"] };
    expect(waterfall(odd)[0].startMs).toBe(0);
  });
});

describe("frameUrlForTick", () => {
  it("pads the tick the way the worker names the file", () => {
    expect(frameUrlForTick(run, 7)).toBe("/runs/2026-09-23T13-40-04_5000/frames/tick_00007_front.jpg");
    expect(frameUrlForTick(run, 1234)).toBe("/runs/2026-09-23T13-40-04_5000/frames/tick_01234_front.jpg");
  });
});
