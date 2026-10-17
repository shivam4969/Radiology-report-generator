// OpenRead API. ESM, Express, Server-Sent Events over POST.

import express from 'express';
import { existsSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { engineConfig, loadEngine } from './engines/index.js';
import { DDX_SYSTEM_PROMPT } from './lib/prompt.js';
import { parseModelJson } from './lib/json.js';
import { getDb, hashReport, hashBytes, insertRead, insertRating, benchmarkSummary, benchmarkRows, toCsv, nextUnrated } from './lib/db.js';

const PORT = Number(process.env.PORT || 8787);
const ROOT = resolve(import.meta.dirname, '..');
const SAMPLES = resolve(ROOT, 'samples');
const { engine: ENGINE, model: MODEL } = engineConfig();

const app = express();
app.use(express.json({ limit: '30mb' }));
app.disable('x-powered-by');

let engine;
async function getEngine() {
  if (!engine) engine = await loadEngine(ENGINE);
  return engine;
}

app.get('/api/config', (_req, res) => {
  res.json({ engine: ENGINE, model: MODEL });
});

app.get('/api/samples', (_req, res) => {
  const files = existsSync(SAMPLES) ? readdirSync(SAMPLES).filter((f) => /\.(jpe?g|png|webp)$/i.test(f)) : [];
  res.json(files.map((f) => ({ name: f, url: `/samples/${f}` })));
});
app.use('/samples', express.static(SAMPLES, { maxAge: '1d' }));

// POST /api/analyze  { imageBase64, mime, history, imageName? }
// Response: text/event-stream with events path | report | done | error
app.post('/api/analyze', async (req, res) => {
  const { imageBase64, mime = 'image/jpeg', history = '', imageName = null } = req.body || {};
  if (!imageBase64 || typeof imageBase64 !== 'string') {
    return res.status(400).json({ error: 'imageBase64 is required' });
  }
  if (!MODEL && ENGINE !== 'mock') {
    return res.status(500).json({ error: 'MODEL is not set. Put ENGINE and MODEL in .env.' });
  }
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache, no-transform',
    connection: 'keep-alive',
    'x-accel-buffering': 'no',
  });
  const send = (type, data) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`);
  };
  let closed = false;
  res.on('close', () => (closed = true));

  const started = Date.now();
  const path = [];
  let report = null;
  try {
    const eng = await getEngine();
    send('start', { engine: ENGINE, model: MODEL });
    for await (const ev of eng.analyze({ imageBase64, mime, history, model: MODEL })) {
      if (closed) break;
      if (ev.type === 'path') {
        path.push(ev.item);
        send('path', { item: ev.item });
      } else if (ev.type === 'report') {
        report = ev.report;
        send('report', { report });
      } else if (ev.type === 'error') {
        send('error', { message: ev.message, raw: ev.raw?.slice?.(0, 2000) });
      } else if (ev.type === 'done') {
        let readId = null;
        if (report) {
          const reportHash = hashReport(report);
          readId = insertRead({
            engine: ENGINE,
            model: MODEL,
            source: 'ui',
            imageName,
            imageHash: hashBytes(Buffer.from(imageBase64.slice(0, 4096), 'base64')),
            history,
            reportHash,
            report,
            path,
            durationMs: Date.now() - started,
          });
          send('done', { readId, reportHash, durationMs: Date.now() - started });
        } else {
          send('done', { readId: null, durationMs: Date.now() - started });
        }
      }
    }
  } catch (err) {
    send('error', { message: err?.message || String(err) });
  }
  res.end();
});

// POST /api/ddx  { report: { exam, findings, impression } | text }
app.post('/api/ddx', async (req, res) => {
  const body = req.body || {};
  const text =
    typeof body.text === 'string'
      ? body.text
      : body.report
        ? `EXAM: ${body.report.exam ?? ''}\n\nFINDINGS:\n${body.report.findings ?? ''}\n\nIMPRESSION:\n${body.report.impression ?? ''}`
        : '';
  if (!text.trim()) return res.status(400).json({ error: 'report text is required' });
  try {
    const eng = await getEngine();
    const raw = await eng.complete({ system: DDX_SYSTEM_PROMPT, text: `Draft report:\n\n${text}`, model: MODEL });
    const obj = parseModelJson(raw);
    if (!obj || !Array.isArray(obj.items)) {
      return res.status(502).json({ error: 'Model did not return parseable JSON', raw: raw.slice(0, 2000) });
    }
    res.json({ items: obj.items, model: MODEL });
  } catch (err) {
    res.status(502).json({ error: err?.message || String(err) });
  }
});

// POST /api/rate  { readId, acceptable, reason, studyType, icu, modality, report?, model? }
app.post('/api/rate', (req, res) => {
  const b = req.body || {};
  if (typeof b.acceptable !== 'boolean') return res.status(400).json({ error: 'acceptable (boolean) is required' });
  let reportHash = b.reportHash || (b.report ? hashReport(b.report) : null);
  let engineName = ENGINE;
  let model = b.model || MODEL;
  if (b.readId) {
    const row = getDb().prepare('SELECT engine, model, report_hash FROM reads WHERE id = ?').get(b.readId);
    if (row) {
      engineName = row.engine;
      model = row.model;
      reportHash = reportHash || row.report_hash;
    }
  }
  if (!reportHash) return res.status(400).json({ error: 'readId, reportHash or report is required' });
  const id = insertRating({
    readId: b.readId ?? null,
    engine: engineName,
    model,
    reportHash,
    acceptable: b.acceptable,
    reason: b.reason,
    studyType: b.studyType,
    icu: b.icu,
    modality: b.modality,
  });
  res.json({ ok: true, id });
});

app.get('/api/benchmark/summary', (_req, res) => {
  res.json(benchmarkSummary());
});

app.get('/api/benchmark.csv', (_req, res) => {
  res.setHeader('content-type', 'text/csv; charset=utf-8');
  res.setHeader('content-disposition', 'attachment; filename="openread-benchmark.csv"');
  res.send(toCsv(benchmarkRows()));
});

// Oldest unrated read (bench results waiting for a human verdict).
app.get('/api/benchmark/next-unrated', (_req, res) => {
  const row = nextUnrated();
  if (!row) return res.json(null);
  res.json({ ...row, report: JSON.parse(row.report_json), path: JSON.parse(row.path_json || '[]') });
});

// Production: serve the built front end.
const DIST = resolve(ROOT, 'web/dist');
if (existsSync(DIST)) {
  app.use(express.static(DIST));
  app.get(/^(?!\/api\/).*/, (_req, res) => res.sendFile(resolve(DIST, 'index.html')));
}

getDb();
app.listen(PORT, () => {
  console.log(`OpenRead API  http://localhost:${PORT}  engine=${ENGINE} model=${MODEL || '(unset)'}`);
});
