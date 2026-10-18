import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';

test('chexagent engine puts sidecar findings into the report', async () => {
  let seen = null;
  const server = createServer((req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      seen = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      const payload = JSON.stringify({
        findings: '[Breathing: Lungs] The lungs are clear. [Cardiac: Heart Size] The heart size is normal.',
      });
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) });
      res.end(payload);
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  process.env.CHEXAGENT_BASE_URL = `http://127.0.0.1:${server.address().port}`;
  try {
    const { default: engine } = await import('../server/engines/chexagent.js');
    const events = [];
    for await (const ev of engine.analyze({
      imageBase64: 'aGVsbG8=',
      mime: 'image/jpeg',
      history: 'cough',
      model: 'StanfordAIMI/CheXagent-2-3b',
    })) {
      events.push(ev);
    }
    assert.equal(seen.history, 'cough');
    assert.equal(seen.imageBase64, 'aGVsbG8=');
    assert.equal(events.filter((ev) => ev.type === 'path').length, 0);
    const report = events.find((ev) => ev.type === 'report').report;
    assert.match(report.findings, /lungs are clear/);
    assert.equal(report.exam, 'Chest radiograph, frontal view.');
    assert.deepEqual(events.slice(-2).map((ev) => ev.type), ['report', 'done']);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});
