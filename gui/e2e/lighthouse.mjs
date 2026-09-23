// Lighthouse accessibility score for the app, measured rather than asserted.
// Run: node e2e/lighthouse.mjs   (the GUI must be running; GUI_URL overrides the address)
import { launch } from "chrome-launcher";
import lighthouse from "lighthouse";
import { writeFileSync } from "node:fs";

const URL = process.env.GUI_URL ?? "http://127.0.0.1:8010";
const OUT = process.env.OUT ?? "/tmp/shots";
const chromePath = process.env.CHROME_PATH ?? (await import("@playwright/test")).chromium.executablePath();

const chrome = await launch({ chromePath, chromeFlags: ["--headless=new", "--no-sandbox", "--disable-gpu"] });
try {
  const run = await lighthouse(URL, { port: chrome.port, output: "json", logLevel: "error", onlyCategories: ["accessibility", "best-practices", "performance"] });
  const cats = run.lhr.categories;
  const scores = Object.fromEntries(Object.entries(cats).map(([k, v]) => [k, Math.round((v.score ?? 0) * 100)]));
  const failed = Object.values(run.lhr.audits)
    .filter((a) => a.score !== null && a.score < 1 && (cats.accessibility.auditRefs ?? []).some((r) => r.id === a.id))
    .map((a) => ({ id: a.id, title: a.title, score: a.score }));
  writeFileSync(`${OUT}/lighthouse.json`, JSON.stringify({ url: URL, scores, failedAccessibilityAudits: failed }, null, 1));
  console.log(JSON.stringify({ scores, failedAccessibilityAudits: failed }, null, 1));
} finally {
  await chrome.kill();
}
