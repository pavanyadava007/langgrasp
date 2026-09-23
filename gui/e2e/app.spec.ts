import { expect, test, type Page } from "@playwright/test";

// Each test starts from a simulator that is free and not latched: an e-stop is deliberately sticky, so one
// test leaving it closed would fail every test after it for the wrong reason.
test.beforeEach(async ({ request }) => {
  await request.post("/api/reset");
  for (let i = 0; i < 60; i++) {
    const system = await (await request.get("/api/system")).json();
    if (!system.worker.busy) return;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("the worker stayed busy between tests");
});

// End to end against the real stack: the built frontend, the API, and a simulation worker with the arm in it.
// These check the things a screenshot cannot: that a command runs through all nine stages, that the overlays
// are actually painted, that the outcome is graded against ground truth, and that the e-stop stops the arm.

async function waitForWarm(page: Page) {
  await page.goto("/");
  await expect(page.locator("#command")).toBeVisible();
  await expect(page.getByText(/models warm|models .*unavailable/)).toBeVisible({ timeout: 60_000 });
}

async function newScene(page: Page, seed: number, stratum: "seen" | "unseen" | "langvar" = "seen") {
  await page.locator("#seed").fill(String(seed));
  await page.locator("#stratum").selectOption(stratum);
  // Wait for the scene the server actually built. Waiting for the chip to appear is not enough: it is already
  // on screen from the previous scene, so a command typed straight afterwards is overwritten when the new
  // scene lands.
  const [response] = await Promise.all([
    page.waitForResponse((r) => r.url().includes("/api/scene") && r.request().method() === "POST"),
    page.getByRole("button", { name: "New scene" }).click(),
  ]);
  const scene = (await response.json()).scene as { command: string; seed: number; target: string };
  await expect(page.getByRole("button", { name: `this scene: ${scene.command}` })).toBeVisible();
  await expect(page.locator("#command")).toHaveValue(scene.command);
  return scene;
}

async function runCurrentCommand(page: Page) {
  const run = page.getByRole("button", { name: /^Run/ });
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await run.click();
  await expect(page.getByText(/Outcome (graded|not graded)/)).toBeVisible({ timeout: 90_000 });
}

test.describe("Live Run", () => {
  test("the page states what it is before anything else", async ({ page }) => {
    await waitForWarm(page);
    // the banner names whatever machine the worker found, and always says it is not real hardware
    await expect(page.getByText(/^Simulation · .* · not real hardware$/)).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Views" }).first()).toBeVisible();
    // the safety state is shown before a run, not after
    await expect(page.getByRole("complementary", { name: "Safety" })).toBeVisible();
  });

  test("a command runs through all nine stages, paints the overlays and is graded", async ({ page }) => {
    await waitForWarm(page);
    await newScene(page, 5000, "seen");
    // the command this scene was generated for: the only one ground truth can grade
    await page.getByRole("button", { name: /^this scene:/ }).click();
    await page.locator("#speed").selectOption("0");
    await runCurrentCommand(page);

    const stages = page.locator('ol[aria-label="Pipeline stages"] > li');
    await expect(stages).toHaveCount(9);
    const text = await stages.allInnerTexts();
    for (const name of ["Command", "Parse", "Capture", "Ground", "Select", "Safety gate", "Segment", "Depth fusion", "Execute"]) {
      expect(text.join(" ")).toContain(name);
    }
    expect(text.join(" ")).not.toContain("pending");

    // a camera frame was decoded and drawn, not just requested: the caption reports the newest frame's age
    await expect(page.locator("canvas + p")).toContainText(/front camera, rgb, \d+x\d+/);
    await expect(page.locator("canvas + p")).not.toContainText("waiting for the first frame");

    // the overlays were painted, not merely enabled
    const overlays = await page.locator("canvas").first().getAttribute("data-overlays");
    expect(overlays).toContain("winner");
    expect(overlays).toContain("grasp");
    expect(overlays).toMatch(/candidates:[1-9]/);

    // and the outcome is graded against the simulator, which the pipeline never sees
    await expect(page.getByText("graded against the simulator's ground truth")).toBeVisible();
    await expect(page.getByText(/placed in the tray:\s*yes/i)).toBeVisible();
  });

  test("a stage drawer shows what that stage decided", async ({ page }) => {
    await waitForWarm(page);
    await newScene(page, 5000, "seen");
    await page.getByRole("button", { name: /^this scene:/ }).click();
    await page.locator("#speed").selectOption("0");
    await runCurrentCommand(page);
    await page.locator('ol[aria-label="Pipeline stages"] > li').nth(3).getByRole("button").click();
    await expect(page.getByText("QUERY SENT")).toBeVisible();
    await expect(page.getByText("COLOUR FALLBACK")).toBeVisible();
    await page.getByRole("button", { name: "Close" }).click();
    await expect(page.getByText("QUERY SENT")).toHaveCount(0);
  });

  test("a command the scene did not ask for is reported as not graded", async ({ page }) => {
    await waitForWarm(page);
    await newScene(page, 5000, "seen");
    await page.locator("#command").fill("pick the green can");
    await expect(page.getByText(/grounding cannot be graded/)).toBeVisible();
    await page.locator("#speed").selectOption("0");
    await runCurrentCommand(page);
    // twice on the page: in the outcome card, and in the polite live region a screen reader reads
    await expect(page.getByText(/Not scored: this scene's own command is/).first()).toBeVisible();
    await expect(page.getByRole("heading", { name: /^Outcome/ })).toContainText("not graded");
  });

  test("E stops the arm and Reset releases the latch", async ({ page }) => {
    await waitForWarm(page);
    await newScene(page, 5000, "seen");
    await page.getByRole("button", { name: /^this scene:/ }).click();
    await page.locator("#speed").selectOption("1");
    const run = page.getByRole("button", { name: /^Run/ });
    await expect(run).toBeEnabled({ timeout: 30_000 });
    await run.click();
    // wait until the arm is actually moving, then press the key
    await expect(page.locator('ol[aria-label="Pipeline stages"] > li').nth(8)).toContainText(/tick \d+/, { timeout: 60_000 });
    await page.keyboard.press("e");

    const rail = page.getByRole("complementary", { name: "Safety" });
    const state = rail.getByRole("status");
    await expect(state).toContainText("E-STOP", { timeout: 10_000 });
    const latency = await rail.getByText(/e-stop latency:/).innerText();
    const ms = Number(latency.match(/([\d.]+) ms/)?.[1]);
    expect(ms).toBeGreaterThan(0);
    expect(ms).toBeLessThan(100); // the acceptance criterion, measured rather than asserted in prose

    await expect(page.getByText(/Aborted:\s*safety:estop/).first()).toBeVisible({ timeout: 30_000 });
    // while the latch is closed there is nothing to press: Run is disabled and says why
    const runAgain = page.getByRole("button", { name: /^Run/ });
    await expect(runAgain).toBeDisabled();
    await expect(runAgain).toHaveAttribute("title", /Reset the e-stop first/);

    await rail.getByRole("button", { name: /^Reset/ }).click();
    await expect(state).not.toContainText("E-STOP", { timeout: 15_000 });
  });
});

test.describe("the other views", () => {
  test("Results shows measured rates with their source files", async ({ page }) => {
    await waitForWarm(page);
    await page.keyboard.press("3");
    await expect(page.getByText("Stratified protocol, place rate with Wilson 95% intervals")).toBeVisible();
    await expect(page.getByText("results/modular_protocol.json").first()).toBeVisible();
    // Wilson intervals, not bare percentages
    await expect(page.getByText(/%\s*\[\d+%, \d+%\]/).first()).toBeVisible();
    // Five of the results files contain a bare NaN, which the browser cannot parse. It must never reach a
    // table: the only place the word may appear is the sentence explaining that the server replaced it.
    const tables = await page.locator("main table").allInnerTexts();
    expect(tables.join(" ")).not.toContain("NaN");
    expect(tables.join(" ")).toContain("100.0%");
  });

  test("Safety lists every hazard with its status and evidence", async ({ page }) => {
    await waitForWarm(page);
    await page.keyboard.press("5");
    await expect(page.getByRole("heading", { name: "Hazard analysis" })).toBeVisible();
    for (const id of ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"]) {
      await expect(page.getByText(id, { exact: true }).first()).toBeVisible();
    }
    await expect(page.getByText(/hardware only, not exercised/).first()).toBeVisible();
    await expect(page.getByText(/test_safety.py::test_estop_latch_persists_until_reset/).first()).toBeVisible();
  });

  test("Batch refuses to overwrite a published results file", async ({ page }) => {
    await waitForWarm(page);
    await page.keyboard.press("4");
    await page.locator("#n-seen").fill("1");
    await page.locator("#n-unseen").fill("0");
    await page.locator("#n-langvar").fill("0");
    await page.locator("#b-out").fill("modular_protocol.json");
    const start = page.getByRole("button", { name: /^Start/ });
    await expect(start).toBeEnabled({ timeout: 20_000 });
    await start.click();
    await expect(page.getByText(/already exists/)).toBeVisible({ timeout: 20_000 });
    await expect(page.getByLabel(/Overwrite that file/)).not.toBeChecked();
  });

  test("the Inspector replays a recorded run tick by tick", async ({ page }) => {
    await waitForWarm(page);
    await newScene(page, 5000, "seen");
    await page.getByRole("button", { name: /^this scene:/ }).click();
    await page.locator("#speed").selectOption("0");
    await runCurrentCommand(page);

    await page.keyboard.press("2");
    await expect(page.getByText("Recorded runs")).toBeVisible();
    await expect(page.getByText(/tick \d+ of \d+/)).toBeVisible({ timeout: 30_000 });
    const before = await page.getByText(/tick \d+ of \d+/).innerText();
    await page.locator('input[aria-label="Tick"]').focus();
    for (let i = 0; i < 5; i++) await page.keyboard.press("ArrowRight");
    await expect(page.getByText(/tick \d+ of \d+/)).not.toHaveText(before);
    // the recorded frame for that tick is served
    await expect(page.locator("main img").first()).toBeVisible();
    expect(await page.locator("main img").first().evaluate((i: HTMLImageElement) => i.naturalWidth)).toBeGreaterThan(100);
  });
});
