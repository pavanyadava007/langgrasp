import { defineConfig, devices } from "@playwright/test";

// The end-to-end suite starts the real thing: the API, the simulation worker and the built frontend, on a
// port of its own so it never collides with a GUI someone is already using. It uses the oracle grounder by
// default so it does not need the 660 MB Grounding DINO checkpoint; set E2E_GROUNDER=gdino for the full path.
const PORT = Number(process.env.E2E_PORT ?? 8011);
const GROUNDER = process.env.E2E_GROUNDER ?? "oracle";
const PYTHON = process.env.E2E_PYTHON ?? ".venv/bin/python"; // relative to cwd below, which is the repo root

export default defineConfig({
  testDir: "./e2e",
  testMatch: /.*\.spec\.ts/,
  timeout: 120_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["json", { outputFile: "e2e-results.json" }]] : [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 1000 },
    permissions: ["microphone"],
  },
  webServer: {
    command: `MUJOCO_GL=egl ${PYTHON} -m langgrasp.gui --port ${PORT} --grounder ${GROUNDER} --log-level warning`,
    url: `http://127.0.0.1:${PORT}/api/system`,
    cwd: "..",
    timeout: 180_000,
    reuseExistingServer: false,
    stdout: "pipe",
    stderr: "pipe",
  },
});
