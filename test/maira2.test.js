import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';

import { findingsToRead, xyxyToXywh } from '../server/lib/maira2.js';

test('xyxyToXywh converts corners and drops empty boxes', () => {
  assert.deepEqual(xyxyToXywh([0.1, 0.2, 0.5, 0.6]), [0.1, 0.2, 0.4, 0.4]);
  assert.deepEqual(xyxyToXywh([0.5, 0.6, 0.1, 0.2]), [0.1, 0.2, 0.4, 0.4]);
  assert.equal(xyxyToXywh([0.2, 0.2, 0.2, 0.4]), null);
  assert.equal(xyxyToXywh([1, 2]), null);
});

test('findingsToRead keeps ungrounded sentences in the report only', () => {
  const read = findingsToRead({
    lateral: false,
    findings: [
      { text: 'Patchy opacity at the right base.', boxes: [[0.1, 0.5, 0.4, 0.8]] },
      { text: 'The heart size is normal.', boxes: [] },
    ],
  });
  assert.equal(read.reading_path.length, 1);
  assert.equal(read.reading_path[0].region, 'Patchy opacity at the right base.');
  assert.deepEqual(read.reading_path[0].box, [0.1, 0.5, 0.3, 0.3]);
  assert.equal(read.reading_path[0].zoom, true);
  assert.match(read.report.findings, /heart size is normal/);
  assert.match(read.report.impression, /^1\. Patchy opacity/);
  assert.match(read.report.impression, /2\. The heart size is normal/);
  assert.equal(read.report.exam, 'Chest radiograph, frontal view.');
});

test('findingsToRead emits one path item per box', () => {
  const read = findingsToRead({
    lateral: true,
    findings: [{ text: 'Bilateral effusions.', boxes: [[0.1, 0.6, 0.4, 0.9], [0.6, 0.6, 0.9, 0.9]] }],
  });
  assert.equal(read.reading_path.length, 2);
  assert.match(read.reading_path[1].region, /\(2\)$/);
  assert.equal(read.reading_path[0].note, 'Bilateral effusions.');
  assert.equal(read.report.exam, 'Chest radiograph, frontal and lateral views.');
});

test('maira2 engine turns a sidecar response into path events then a report', async () => {
  process.env.MAIRA2_PACE_MS = '1';
  let seen = null;
  const server = createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      seen = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      const payload = JSON.stringify({
        findings: [
          { text: 'Right lower lobe opacity.', boxes: [[0.12, 0.52, 0.44, 0.82]] },
          { text: 'No pleural effusion.', boxes: [] },
        ],
      });
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) });
      res.end(payload);
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  process.env.MAIRA2_BASE_URL = `http://127.0.0.1:${server.address().port}`;
  try {
    const { default: engine } = await import('../server/engines/maira2.js');
    const events = [];
    for await (const ev of engine.analyze({
      imageBase64: 'aGVsbG8=',
      mime: 'image/png',
      history: 'cough and fever',
      model: 'microsoft/maira-2',
    })) {
      events.push(ev);
    }
    assert.equal(seen.history, 'cough and fever');
    assert.equal(seen.grounding, true);
    assert.equal(seen.imageBase64, 'aGVsbG8=');
    const path = events.filter((ev) => ev.type === 'path');
    assert.equal(path.length, 1);
    assert.deepEqual(path[0].item.box, [0.12, 0.52, 0.32, 0.3]);
    assert.equal(path[0].item.zoom, true);
    const report = events.find((ev) => ev.type === 'report').report;
    assert.match(report.findings, /No pleural effusion/);
    assert.deepEqual(events.slice(-2).map((ev) => ev.type), ['report', 'done']);
    await assert.rejects(
      engine.complete({ system: 's', text: 't', model: 'microsoft/maira-2' }),
      /DDX_ENGINE/,
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
