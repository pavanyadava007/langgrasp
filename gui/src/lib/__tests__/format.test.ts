import { describe, expect, it } from "vitest";
import { ago, degrees, latencyLabel, metres, ms, NOT_RUN, num, pct, xyz } from "../format";

// The rule these guard: a number that was never measured must read "not run", never 0, never a dash.
describe("formatting refuses to invent a number", () => {
  it("renders missing values as not run", () => {
    for (const f of [ms, num, pct, metres, degrees]) {
      expect(f(null)).toBe(NOT_RUN);
      expect(f(undefined)).toBe(NOT_RUN);
      expect(f(NaN)).toBe(NOT_RUN);
    }
    expect(xyz(null)).toBe(NOT_RUN);
    expect(xyz([1, 2])).toBe(NOT_RUN);
    expect(ago(null)).toBe(NOT_RUN);
  });

  it("keeps a real zero", () => {
    expect(ms(0)).toBe("0.00 ms");
    expect(num(0)).toBe("0.00");
    expect(pct(0)).toBe("0.0%");
  });

  it("scales units the way a reader expects", () => {
    expect(ms(0.42)).toBe("0.42 ms");
    expect(ms(3.456, 1)).toBe("3.5 ms");
    expect(ms(273.4)).toBe("273 ms");
    expect(ms(5923.5)).toBe("5924 ms");
    expect(ms(12345)).toBe("12.3 s");
    expect(metres(0.0248)).toBe("24.8 mm");
    expect(degrees(Math.PI / 2)).toBe("90.0°");
    expect(xyz([-0.0644, -0.2151, 0.0141])).toBe("-0.064, -0.215, 0.014");
  });

  it("says what kind of time a latency is", () => {
    expect(latencyLabel("compute")).toContain("evaluation scripts");
    expect(latencyLabel("wall_paced")).toContain("pacing");
    expect(latencyLabel("none")).toContain("not timed");
  });
});
