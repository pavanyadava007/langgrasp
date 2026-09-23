import { chromium } from "@playwright/test";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
await p.goto("http://127.0.0.1:8010", { waitUntil: "domcontentloaded" });
await p.waitForSelector("#command");
await p.waitForTimeout(1500);
const idle = await p.locator("canvas + p").first().innerText();
await p.getByRole("button", { name: /^Run/ }).click();
// sample the caption once the whole 2 s window sits inside the paced execution
const samples = [];
for (let i = 0; i < 14; i++) {
  await p.waitForTimeout(600);
  const txt = await p.locator("canvas + p").first().innerText();
  const m = txt.match(/([\d.]+) fps/);
  const phase = await p.locator('ol[aria-label="Pipeline stages"] > li').nth(8).innerText();
  samples.push({ t: (i + 1) * 0.6, fps: m ? Number(m[1]) : null, executing: /running/.test(phase) });
}
console.log("idle caption:", idle);
console.log("during the run:", JSON.stringify(samples.filter((s) => s.executing)));
const mid = samples.filter((s) => s.executing).slice(2, -1).map((s) => s.fps);
console.log("front camera fps while executing, windows fully inside the run:", mid, "max", Math.max(...mid));
await b.close();
