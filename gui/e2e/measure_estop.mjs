// Measure the e-stop from the browser in both conditions: while the arm is moving, and while it is idle.
import { chromium } from "@playwright/test";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 1000 } });
await p.goto("http://127.0.0.1:8010", { waitUntil: "domcontentloaded" });
await p.waitForSelector("#command");
await p.waitForTimeout(1500);

async function latency() {
  const txt = await p.locator("text=/e-stop latency/").first().innerText();
  return Number(txt.match(/([\d.]+) ms/)?.[1] ?? NaN);
}

const idle = [];
for (let i = 0; i < 5; i++) {
  await p.keyboard.press("e");
  await p.waitForTimeout(500);
  idle.push(await latency());
  await p.getByRole("button", { name: /^Reset/ }).click();
  await p.waitForTimeout(400);
}

const moving = [];
for (let i = 0; i < 5; i++) {
  await p.getByRole("button", { name: /^Run/ }).click();
  await p.waitForTimeout(1600); // into the executor
  await p.keyboard.press("e");
  await p.waitForTimeout(600);
  moving.push(await latency());
  await p.getByRole("button", { name: /^Reset/ }).click();
  await p.waitForTimeout(500);
}
const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
console.log(JSON.stringify({ idle, idle_median: med(idle), moving, moving_median: med(moving), max: Math.max(...idle, ...moving) }, null, 1));
await b.close();
