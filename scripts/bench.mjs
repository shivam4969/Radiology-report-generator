#!/usr/bin/env node
// Run every image in samples/ through the configured engine N times and store
// the reads in data/openread.db without ratings, so humans can rate them later.
//
//   npm run bench                 # N = 1
//   npm run bench -- --n 3        # three reads per image
//   npm run bench -- --dir ./my-images --history "cough and fever"
//
// Compare two models: run once with MODEL=a, once with MODEL=b, rate the reads
// (GET /api/benchmark/next-unrated in the UI, or your own tool), then look at
// GET /api/benchmark/summary or /api/benchmark.csv.

import { readdirSync, readFileSync } from 'node:fs';
import { resolve, extname, basename } from 'node:path';
import { engineConfig, loadEngine } from '../server/engines/index.js';
import { insertRead, hashReport, hashBytes } from '../server/lib/db.js';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : def;
};
const N = Number(opt('n', 1));
const DIR = resolve(opt('dir', 'samples'));
const HISTORY = opt('history', '');
const MIME = { '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp' };

const { engine: engineName, model } = engineConfig();
const engine = await loadEngine(engineName);
const files = readdirSync(DIR).filter((f) => MIME[extname(f).toLowerCase()]);
if (!files.length) {
  console.error(`No images in ${DIR}`);
  process.exit(1);
}
console.log(`bench: engine=${engineName} model=${model} images=${files.length} n=${N}`);

let ok = 0;
let failed = 0;
for (const f of files) {
  const buf = readFileSync(resolve(DIR, f));
  const imageBase64 = buf.toString('base64');
  const mime = MIME[extname(f).toLowerCase()];
  for (let i = 0; i < N; i++) {
    const t0 = Date.now();
    const path = [];
    let report = null;
    let err = null;
    for await (const ev of engine.analyze({ imageBase64, mime, history: HISTORY, model })) {
      if (ev.type === 'path') path.push(ev.item);
      else if (ev.type === 'report') report = ev.report;
      else if (ev.type === 'error') err = ev.message;
    }
    const ms = Date.now() - t0;
    if (!report) {
      failed++;
      console.log(`  ${f} #${i + 1}  FAILED ${ms}ms  ${err || ''}`);
      continue;
    }
    const id = insertRead({
      engine: engineName,
      model,
      source: 'bench',
      imageName: basename(f),
      imageHash: hashBytes(buf),
      history: HISTORY,
      reportHash: hashReport(report),
      report,
      path,
      durationMs: ms,
    });
    ok++;
    console.log(`  ${f} #${i + 1}  read_id=${id}  ${ms}ms  regions=${path.length}  "${report.impression.split('\n')[0].slice(0, 70)}"`);
  }
}
console.log(`done: ${ok} stored, ${failed} failed. Rate them in the UI or query data/openread.db.`);
