// The local (on-device) engine, exercised through the mock Python worker so no
// PyTorch, ExecuTorch or weights are needed. Python 3 must be on PATH.

import { test, after } from 'node:test';
import assert from 'node:assert/strict';

process.env.CXR_WORKER = 'py/cxr_worker_mock.py';
process.env.LOCAL_LLM = 'template';
process.env.LOCAL_PATH_DELAY_MS = '1';
process.env.DB_PATH = ':memory:';

const { default: local } = await import('../server/engines/local.js');
const { buildReadingPath, templateReport, templateDdx, describeBox, reportWriterUser } = await import('../server/lib/findings.js');
const { loadEngine, engineConfig } = await import('../server/engines/index.js');

after(() => local.stopWorker());

const PROBS = { Pneumonia: 0.83, Consolidation: 0.74, Effusion: 0.21, Cardiomegaly: 0.12, Fracture: 0.03, Pneumothorax: 0.02 };
const CAMS = { Pneumonia: [0.12, 0.52, 0.32, 0.3], Consolidation: [0.14, 0.55, 0.28, 0.26] };

test('local engine: findings -> path -> report -> done, in that order', async () => {
  const evs = [];
  for await (const ev of local.analyze({ imageBase64: 'AAAA', mime: 'image/jpeg', history: 'cough and fever' })) evs.push(ev);
  const types = evs.map((e) => e.type);
  assert.equal(types[0], 'findings');
  assert.ok(types.filter((t) => t === 'path').length >= 5);
  assert.deepEqual(types.slice(-2), ['report', 'done']);
  assert.ok(types.lastIndexOf('path') < types.indexOf('report'));
  assert.ok(!types.includes('error'), JSON.stringify(evs.find((e) => e.type === 'error')));

  const findings = evs[0];
  assert.equal(findings.backend, 'mock');
  assert.equal(findings.probs.Pneumonia, 0.83);

  const path = evs.filter((e) => e.type === 'path').map((e) => e.item);
  // anatomic order: lungs overview, positives (zoomed), heart, pleura, bones
  assert.match(path[0].region, /Both lungs, overview/);
  assert.equal(path[0].zoom, false);
  const zoomed = path.filter((p) => p.zoom);
  assert.equal(zoomed.length, 4, 'Pneumonia, Consolidation, Lung Opacity, Infiltration are >= 0.5 in the mock');
  assert.match(zoomed[0].region, /^Pneumonia, right lower zone/);
  assert.deepEqual(zoomed[0].box, [0.12, 0.52, 0.32, 0.3]);
  assert.match(zoomed[0].note, /83%/);
  assert.match(zoomed[1].region, /^Consolidation/);
  assert.match(path.at(-3).region, /Heart and mediastinum/);
  assert.match(path.at(-2).region, /Pleura and costophrenic angles/);
  assert.match(path.at(-1).region, /Bones and soft tissues/);
  assert.equal(path.at(-1).zoom, false);
  path.forEach((p, i) => assert.equal(p.index, i));

  const { report } = evs.find((e) => e.type === 'report');
  assert.equal(report.exam, 'Chest radiograph, one view.');
  assert.match(report.findings, /^Lungs: Pneumonia \(right lower zone\), classifier probability 83%/m);
  assert.match(report.findings, /Pleura: No pleural effusion or pneumothorax flagged/);
  assert.match(report.impression, /^1\. Pneumonia, right lower zone \(83%\)\./);
  assert.match(report.impression, /Radiologist review required/);
  assert.match(report.impression, /cough and fever/);
});

test('local engine: the worker is reused across calls and answers the ddx with a rule-based differential', async () => {
  let report;
  for await (const ev of local.analyze({ imageBase64: 'BBBB' })) if (ev.type === 'report') report = ev.report;
  const txt = await local.complete({ system: 'ignored', text: `${report.findings}\n${report.impression}` });
  const o = JSON.parse(txt);
  assert.ok(Array.isArray(o.items));
  assert.match(o.items[0].finding, /consolidation/i);
  assert.match(o.items[0].differentials[0].dx, /pneumonia/i);
  assert.ok(o.items[0].management.guideline);
});

test('registry loads the local engine and gives it a pipeline model name', async () => {
  const e = await loadEngine('local');
  assert.equal(e.name, 'local');
  const prev = process.env.ENGINE;
  process.env.ENGINE = 'local';
  delete process.env.MODEL;
  assert.equal(engineConfig().model, 'densenet121-res224-all+template');
  if (prev === undefined) delete process.env.ENGINE;
  else process.env.ENGINE = prev;
});

test('buildReadingPath with nothing positive still visits every region', () => {
  const items = buildReadingPath({ probs: { Pneumonia: 0.2, Effusion: 0.1 }, cams: {} });
  assert.deepEqual(
    items.map((i) => i.region),
    ['Both lungs, overview', 'Heart and mediastinum', 'Pleura and costophrenic angles', 'Bones and soft tissues'],
  );
  assert.match(items[0].note, /highest is Pneumonia at 20%/);
  assert.ok(items.every((i) => !i.zoom));
});

test('buildReadingPath respects the threshold and the max findings', () => {
  const items = buildReadingPath({ probs: PROBS, cams: CAMS }, { threshold: 0.8, maxFindings: 4 });
  assert.equal(items.filter((i) => i.zoom).length, 1);
  const all = buildReadingPath({ probs: PROBS, cams: CAMS }, { threshold: 0.1, maxFindings: 2 });
  assert.equal(all.filter((i) => i.zoom).length, 2);
});

test('templateReport on a normal study says so without claiming to have seen the image', () => {
  const r = templateReport({ probs: { Pneumonia: 0.1, Cardiomegaly: 0.3 }, cams: {} });
  assert.match(r.findings, /not flagged|No .* flagged/);
  assert.match(r.impression, /No finding reaches the classifier's 50% threshold; highest is Cardiomegaly at 30%/);
  assert.match(r.findings, /Lines and tubes: Not assessed/);
});

test('templateDdx keys on the top finding and ignores negated lines', () => {
  assert.match(templateDdx('Pleura: Effusion (right lower zone), classifier probability 70%.').items[0].finding, /effusion/i);
  assert.match(templateDdx('1. Pneumothorax, left upper zone (91%).').items[0].finding, /pneumothorax/i);
  assert.match(templateDdx('Heart and mediastinum: Cardiomegaly, classifier probability 77%.').items[0].finding, /cardiomegaly/i);
  assert.match(templateDdx('Lungs: Nodule (right upper zone), classifier probability 64%.').items[0].management.guideline, /Fleischner/);
  const normal = templateDdx('Lungs: No consolidation, opacity, mass or nodule flagged by the classifier.\nPleura: No pleural effusion or pneumothorax flagged.');
  assert.equal(normal.items[0].finding, 'No acute abnormality');
});

test('describeBox follows radiology convention (patient right on the left)', () => {
  assert.equal(describeBox([0.1, 0.6, 0.3, 0.3]), 'right lower zone');
  assert.equal(describeBox([0.6, 0.1, 0.3, 0.2]), 'left upper zone');
  assert.equal(describeBox([0.35, 0.4, 0.3, 0.3]), 'mid zone, central');
});

test('report writer prompt carries history, positives with percentages and the template', () => {
  const u = reportWriterUser({ probs: PROBS, cams: CAMS, history: '54F cough' });
  assert.match(u, /Clinical history: 54F cough/);
  assert.match(u, /- Pneumonia: 83%, right lower zone/);
  assert.match(u, /Below threshold:[\s\S]*Effusion 21%/);
  assert.match(u, /Lines and tubes: Not assessed/);
});
