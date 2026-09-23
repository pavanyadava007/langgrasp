// Screenshots for the README and the demo guide, taken from the running GUI so they cannot go stale silently.
// Run: node e2e/screenshots.mjs   (GUI_URL and OUT override the address and the output directory)
import { chromium } from "@playwright/test";
import { mkdirSync } from "node:fs";

const BASE = process.env.GUI_URL ?? "http://127.0.0.1:8010";
const OUT = process.env.OUT ?? "../media/gui";
mkdirSync(OUT, { recursive: true });

const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
await page.goto(BASE, { waitUntil: "domcontentloaded" });
await page.waitForSelector("#command");
await page.waitForTimeout(2500);

// 1. Live Run, mid pick, at 1x so the arm is in the middle of the motion
await page.locator("#seed").fill("5002");
const [resp] = await Promise.all([
  page.waitForResponse((r) => r.url().includes("/api/scene") && r.request().method() === "POST"),
  page.getByRole("button", { name: "New scene" }).click(),
]);
await resp.json();
await page.locator("#speed").selectOption("1");
await page.getByRole("button", { name: /^Run/ }).click();
await page.waitForFunction(() => /tick \d+/.test(document.body.innerText), null, { timeout: 60000 });
await page.waitForTimeout(2600);
await page.screenshot({ path: `${OUT}/live-run.png` });
await page.waitForFunction(() => /Outcome (graded|not graded)/.test(document.body.innerText), null, { timeout: 90000 });
await page.waitForTimeout(500);

// 2. a stage drawer
await page.locator('ol[aria-label="Pipeline stages"] > li').nth(3).getByRole("button").click();
await page.waitForTimeout(600);
await page.locator('ol[aria-label="Pipeline stages"]').scrollIntoViewIfNeeded();
await page.screenshot({ path: `${OUT}/stage-drawer.png` });
await page.getByRole("button", { name: "Close" }).click();

// 3. the other views
for (const [key, name] of [["2", "inspector"], ["3", "results"], ["4", "batch"], ["5", "safety"]]) {
  await page.keyboard.press(key);
  await page.waitForTimeout(2200);
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: name === "results" || name === "safety" });
}
console.log("wrote screenshots to", OUT);
await b.close();
