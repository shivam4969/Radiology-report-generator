#!/usr/bin/env node
// End-to-end smoke test with the mock engine. Starts the API and the Vite dev
// server, drives the UI with playwright-core and writes screenshots to
// docs/screens/. No API key needed.
//
//   npm run smoke
//   CHROME=/path/to/chrome npm run smoke

import { spawn } from 'node:child_process';
import { mkdirSync, existsSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const ROOT = resolve(import.meta.dirname, '..');
const OUT = resolve(ROOT, 'docs/screens');
const API_PORT = 8790;
const WEB_PORT = 5190;
const CHROME = process.env.CHROME || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const DB = resolve(ROOT, 'data/smoke.db');

mkdirSync(OUT, { recursive: true });
for (const f of [DB, `${DB}-wal`, `${DB}-shm`]) if (existsSync(f)) rmSync(f);

const env = { ...process.env, ENGINE: 'mock', MODEL: 'mock-cxr-v1', MOCK_DELAY_MS: '500', PORT: String(API_PORT), DB_PATH: DB };
const procs = [];
const start = (cmd, args, name) => {
  const p = spawn(cmd, args, { cwd: ROOT, env, stdio: ['ignore', 'pipe', 'pipe'] });
  p.stdout.on('data', (d) => process.env.SMOKE_VERBOSE && process.stdout.write(`[${name}] ${d}`));
  p.stderr.on('data', (d) => process.stderr.write(`[${name}] ${d}`));
  procs.push(p);
  return p;
};
const stop = () => procs.forEach((p) => !p.killed && p.kill('SIGTERM'));
process.on('exit', stop);
process.on('SIGINT', () => process.exit(130));

async function waitFor(url, ms = 20000) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`timeout waiting for ${url}`);
}

const fail = (msg) => {
  console.error(`SMOKE FAIL: ${msg}`);
  process.exit(1);
};

try {
  start('node', ['server/index.js'], 'api');
  start('npx', ['vite', '--config', 'web/vite.config.js', '--port', String(WEB_PORT), '--strictPort'], 'web');
  await waitFor(`http://localhost:${API_PORT}/api/config`);
  await waitFor(`http://localhost:${WEB_PORT}/`);

  const browser = await chromium.launch({ executablePath: CHROME, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  page.on('pageerror', (e) => fail(`page error: ${e.message}`));
  page.on('console', (m) => m.type() === 'error' && console.error('[console]', m.text()));

  await page.goto(`http://localhost:${WEB_PORT}/`);
  await page.getByTestId('sample').click();
  await page.getByTestId('drop').locator('img').waitFor();
  await page.getByTestId('history').fill('54F, cough and fever for 3 days');
  await page.screenshot({ path: resolve(OUT, 'intake.png') });

  await page.getByTestId('analyze').click();
  // wait until at least two regions have arrived, then catch the cloud mid-read
  await page.locator('[data-testid="path-step"]').nth(1).waitFor({ timeout: 20000 });
  await page.waitForTimeout(700);
  const cloudOn = await page.getByTestId('cloud').evaluate((el) => el.classList.contains('on'));
  if (!cloudOn) fail('cloud not shown while reading');
  await page.screenshot({ path: resolve(OUT, 'reading.png') });

  await page.getByTestId('report').waitFor({ timeout: 30000 });
  const impression = await page.getByLabel('IMPRESSION').inputValue();
  if (!/pneumonia/i.test(impression)) fail(`unexpected impression: ${impression}`);
  const steps = await page.locator('.mini-path span').count();
  if (steps < 4) fail(`expected reading path in report view, got ${steps}`);
  await page.screenshot({ path: resolve(OUT, 'report.png') });

  await page.getByTestId('ddx-btn').click();
  await page.getByTestId('ddx-panel').waitFor({ timeout: 15000 });
  const ddxText = await page.getByTestId('ddx-panel').innerText();
  if (!/pneumonia/i.test(ddxText) || !/guideline/i.test(ddxText)) fail('ddx panel missing content');
  await page.getByTestId('ddx-panel').scrollIntoViewIfNeeded();
  await page.screenshot({ path: resolve(OUT, 'ddx.png'), fullPage: true });

  // rating round trip
  await page.getByTestId('rate-ok').click();
  await page.getByText('Saved. Thank you.').waitFor({ timeout: 5000 });
  const summary = await (await fetch(`http://localhost:${API_PORT}/api/benchmark/summary`)).json();
  if (!summary.ratings.length || summary.ratings[0].acceptable !== 1) fail(`rating not stored: ${JSON.stringify(summary)}`);
  const csv = await (await fetch(`http://localhost:${API_PORT}/api/benchmark.csv`)).text();
  if (!/mock-cxr-v1/.test(csv)) fail('csv missing model');

  // replay and new
  await page.getByTestId('replay').click();
  await page.locator('[data-testid="path-step"][data-state="now"]').first().waitFor({ timeout: 5000 });
  await page.getByText('Skip replay').click();
  await page.getByTestId('report').waitFor();
  await page.getByTestId('new').click();
  await page.getByTestId('analyze').waitFor();
  const hasImg = await page.getByTestId('drop').locator('img').count();
  if (hasImg) fail('New did not clear the image');

  await browser.close();
  console.log(`SMOKE OK  screenshots in ${OUT}: intake.png reading.png report.png ddx.png`);
  process.exit(0);
} catch (e) {
  fail(e.stack || e.message);
}
