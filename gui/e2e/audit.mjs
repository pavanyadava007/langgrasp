// Accessibility and responsive audit of every view, against the running server.
// Run: node e2e/audit.mjs   (the GUI must be up; GUI_URL overrides the address)
import { chromium } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { writeFileSync } from "node:fs";

const BASE = process.env.GUI_URL ?? "http://127.0.0.1:8010";
const OUT = process.env.OUT ?? "/tmp/shots";
const VIEWS = [
  { key: "1", name: "Live Run" },
  { key: "2", name: "Pipeline Inspector" },
  { key: "3", name: "Results" },
  { key: "4", name: "Batch Evaluate" },
  { key: "5", name: "Safety & System" },
];
const WIDTHS = [1440, 1024, 420];

const browser = await chromium.launch();
const report = { base: BASE, views: {}, responsive: {}, consoleErrors: [] };

for (const theme of ["dark", "light"]) {
  for (const view of VIEWS) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.on("pageerror", (e) => report.consoleErrors.push(`${view.name}/${theme}: ${e.message}`));
    await page.goto(BASE, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("#command");
    if (theme === "light") await page.getByRole("button", { name: /Light theme/ }).click();
    await page.keyboard.press(view.key);
    await page.waitForTimeout(2200);
    const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze();
    const key = `${view.name} (${theme})`;
    report.views[key] = results.violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      nodes: v.nodes.slice(0, 4).map((n) => ({ target: n.target, summary: (n.failureSummary ?? "").split("\n").slice(0, 3).join(" | ") })),
      count: v.nodes.length,
    }));
    if (theme === "dark") await page.screenshot({ path: `${OUT}/view-${view.key}.png`, fullPage: true });
    await context.close();
    writeFileSync(`${OUT}/audit.json`, JSON.stringify(report, null, 1));
    console.log(`${key}: ${report.views[key].length} violations`);
  }
}

// Responsive: no horizontal scrolling, and the safety controls stay reachable
for (const width of WIDTHS) {
  const context = await browser.newContext({ viewport: { width, height: 900 } });
  const page = await context.newPage();
  await page.goto(BASE, { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#command");
  const perView = {};
  for (const view of VIEWS) {
    await page.keyboard.press(view.key);
    await page.waitForTimeout(900);
    perView[view.name] = {
      overflow: await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth),
      estopVisible: await page.getByRole("button", { name: "E-STOP" }).first().isVisible().catch(() => false),
    };
  }
  report.responsive[width] = perView;
  await context.close();
}

writeFileSync(`${OUT}/audit.json`, JSON.stringify(report, null, 1));
const total = Object.values(report.views).reduce((a, v) => a + v.length, 0);
console.log(`violations: ${total}`);
for (const [k, v] of Object.entries(report.views)) if (v.length) console.log(k, JSON.stringify(v, null, 1));
console.log("responsive:", JSON.stringify(report.responsive, null, 1));
if (report.consoleErrors.length) console.log("page errors:", report.consoleErrors);
