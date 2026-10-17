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
//
// Label agreement (local engine): if the image directory has a labels.csv
// (columns: filename,label; see samples/README.md), the classifier's top
// finding is compared with the label and a per-label table is printed at the
// end. A label matches when it equals the top finding (case-insensitive,
// "Pneumonia" == "pneumonia"), when it is one of the synonyms below, or when
// the label is "No Finding" / "Normal" and nothing reaches the threshold.

import { readdirSync, readFileSync, existsSync } from 'node:fs';
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

const THRESHOLD = Number(process.env.CXR_THRESHOLD || 0.5);
const SYNONYMS = {
  pneumonia: ['pneumonia', 'consolidation', 'lung opacity', 'infiltration'],
  'lung opacity': ['lung opacity', 'consolidation', 'pneumonia', 'infiltration'],
  consolidation: ['consolidation', 'pneumonia', 'lung opacity'],
  infiltration: ['infiltration', 'lung opacity', 'consolidation', 'pneumonia'],
  effusion: ['effusion'],
  'pleural effusion': ['effusion'],
  mass: ['mass', 'nodule', 'lung lesion'],
  nodule: ['nodule', 'mass', 'lung lesion'],
  cardiomegaly: ['cardiomegaly', 'enlarged cardiomediastinum'],
  'no finding': [],
  normal: [],
};

function readLabels(dir) {
  const f = resolve(dir, 'labels.csv');
  if (!existsSync(f)) return null;
  const rows = readFileSync(f, 'utf8').split(/\r?\n/).filter((l) => l.trim());
  const header = rows.shift().split(',').map((h) => h.trim().toLowerCase());
  const fi = header.indexOf('filename');
  const li = header.indexOf('label');
  if (fi < 0 || li < 0) {
    console.error(`labels.csv needs "filename" and "label" columns; got ${header.join(',')}`);
    return null;
  }
  const map = new Map();
  for (const r of rows) {
    const cells = r.split(',');
    map.set(cells[fi].trim(), cells[li].trim());
  }
  return map;
}

function agrees(label, probs) {
  const top = Object.entries(probs || {}).sort((a, b) => b[1] - a[1])[0];
  const l = label.toLowerCase().split('|')[0].trim(); // NIH multi-labels: keep the first
  const negative = l === 'no finding' || l === 'normal';
  if (!top) return { agree: false, top: null };
  const [name, p] = top;
  if (negative) return { agree: p < THRESHOLD, top: name };
  const accepted = SYNONYMS[l] || [l];
  return { agree: p >= THRESHOLD && accepted.includes(name.toLowerCase()), top: name };
}

const { engine: engineName, model } = engineConfig();
const labels = readLabels(DIR);
const agreement = new Map(); // label -> { n, agree, tops: Map }
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
    let probs = null;
    for await (const ev of engine.analyze({ imageBase64, mime, history: HISTORY, model })) {
      if (ev.type === 'path') path.push(ev.item);
      else if (ev.type === 'report') report = ev.report;
      else if (ev.type === 'findings') probs = ev.probs;
      else if (ev.type === 'error') err = ev.message;
    }
    let agreeNote = '';
    if (labels && probs && i === 0 && labels.has(f)) {
      const label = labels.get(f);
      const { agree, top } = agrees(label, probs);
      const key = label.split('|')[0].trim();
      const a = agreement.get(key) || { n: 0, agree: 0, tops: new Map() };
      a.n++;
      if (agree) a.agree++;
      a.tops.set(top, (a.tops.get(top) || 0) + 1);
      agreement.set(key, a);
      agreeNote = `  label=${label} top=${top} ${agree ? 'AGREE' : 'differ'}`;
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
    console.log(`  ${f} #${i + 1}  read_id=${id}  ${ms}ms  regions=${path.length}  "${report.impression.split('\n')[0].slice(0, 70)}"${agreeNote}`);
  }
}
console.log(`done: ${ok} stored, ${failed} failed. Rate them in the UI or query data/openread.db.`);

if (labels && agreement.size) {
  console.log(`\nlabel agreement (classifier top finding vs labels.csv, threshold ${THRESHOLD}):`);
  console.log('  label                        n   agree   most common top finding');
  let N = 0;
  let A = 0;
  for (const [label, a] of [...agreement.entries()].sort((x, y) => y[1].n - x[1].n)) {
    N += a.n;
    A += a.agree;
    const top = [...a.tops.entries()].sort((x, y) => y[1] - x[1])[0];
    console.log(`  ${label.padEnd(28)} ${String(a.n).padStart(3)}   ${String(Math.round((100 * a.agree) / a.n) + '%').padStart(5)}   ${top ? `${top[0]} (${top[1]})` : ''}`);
  }
  console.log(`  ${'all'.padEnd(28)} ${String(N).padStart(3)}   ${String(Math.round((100 * A) / N) + '%').padStart(5)}`);
  console.log('  (a research classifier scored against noisy public labels; this is a sanity check, not a validation)');
} else if (labels) {
  console.log('\nlabels.csv found but no image matched a row, or the engine emitted no findings event (only ENGINE=local does).');
}
