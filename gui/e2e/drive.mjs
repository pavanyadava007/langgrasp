// Drive the real UI in headless Chromium: load it, run a pick, screenshot each state, report console errors.
import { chromium } from "@playwright/test";
import { writeFileSync } from "node:fs";

const OUT = process.env.OUT ?? "/tmp/shots";
const BASE = "http://127.0.0.1:8010";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const errors = [];
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(BASE, { waitUntil: "domcontentloaded" });
await page.waitForSelector("text=LangGrasp", { timeout: 15000 });
await page.waitForTimeout(2500); // let the first frames and the system call land
await page.screenshot({ path: `${OUT}/01-idle.png`, fullPage: false });

const report = {};
report.banner = await page.locator("header span", { hasText: "Simulation" }).first().innerText();
report.models = await page.locator("header").getByText(/models/).first().innerText().catch(() => "not shown");
report.stageCells = await page.locator('ol[aria-label="Pipeline stages"] > li').count();
report.safetyState = await page.locator('[aria-label="Safety"] [role="status"]').first().innerText();
report.command = await page.locator("#command").inputValue();
report.captionBefore = await page.locator("canvas + p").first().innerText();

// Run the scene's own command
await page.getByRole("button", { name: /^Run/ }).click();
await page.waitForTimeout(1200);
await page.screenshot({ path: `${OUT}/02-running.png` });
report.duringRun = await page.locator('ol[aria-label="Pipeline stages"]').innerText();

await page.waitForFunction(() => !!document.body.innerText.match(/Outcome (graded|not graded)/), null, { timeout: 60000 });
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/03-outcome.png` });
report.outcome = await page.locator("h2", { hasText: "Outcome" }).locator("..").innerText();
report.stagesAfter = await page.locator('ol[aria-label="Pipeline stages"]').innerText();
report.caption = await page.locator("canvas + p").first().innerText();

// Open the grounding drawer
await page.locator('ol[aria-label="Pipeline stages"] > li').nth(3).locator("button").click();
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/04-drawer.png`, fullPage: true });
report.drawer = (await page.locator('h2:has-text("Ground")').first().locator("../..").innerText()).slice(0, 900);

// E-stop then reset
await page.keyboard.press("e");
await page.waitForTimeout(900);
report.afterEstop = await page.locator('[aria-label="Safety"] [role="status"]').first().innerText();
report.estopLatency = await page.locator("text=/e-stop latency/").first().innerText();
await page.screenshot({ path: `${OUT}/05-estop.png` });
await page.getByRole("button", { name: /^Reset/ }).click();
await page.waitForTimeout(900);
report.afterReset = await page.locator('[aria-label="Safety"] [role="status"]').first().innerText();

// Light theme + narrow layout
await page.getByRole("button", { name: /Light theme/ }).click();
await page.waitForTimeout(300);
await page.screenshot({ path: `${OUT}/06-light.png` });
await page.setViewportSize({ width: 1024, height: 900 });
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/07-1024.png` });
await page.setViewportSize({ width: 420, height: 900 });
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/08-phone.png` });
report.horizontalScrollAt420 = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);

report.consoleErrors = errors;
writeFileSync(`${OUT}/report.json`, JSON.stringify(report, null, 1));
console.log(JSON.stringify(report, null, 1));
await browser.close();
