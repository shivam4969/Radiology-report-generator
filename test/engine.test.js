import { test } from 'node:test';
import assert from 'node:assert/strict';

process.env.MOCK_DELAY_MS = '1';
process.env.DB_PATH = ':memory:';

const { default: mock, MOCK_RESULT } = await import('../server/engines/mock.js');
const { loadEngine, ENGINE_NAMES } = await import('../server/engines/index.js');
const { makeEngine } = await import('../server/engines/base.js');
const db = await import('../server/lib/db.js');

test('mock engine streams path events, then report, then done', async () => {
  const types = [];
  const path = [];
  let report;
  for await (const ev of mock.analyze({ imageBase64: 'AAAA', mime: 'image/jpeg', history: 'cough', model: 'mock' })) {
    types.push(ev.type);
    if (ev.type === 'path') path.push(ev.item);
    if (ev.type === 'report') report = ev.report;
  }
  assert.equal(path.length, MOCK_RESULT.reading_path.length);
  assert.equal(types.at(-1), 'done');
  assert.equal(types.at(-2), 'report');
  assert.ok(types.indexOf('path') < types.indexOf('report'));
  assert.equal(report.exam, MOCK_RESULT.report.exam);
  assert.ok(path.some((p) => p.zoom));
});

test('mock engine answers the ddx prompt with parseable JSON', async () => {
  const txt = await mock.complete({ system: 's', text: 't', model: 'mock' });
  const o = JSON.parse(txt);
  assert.ok(Array.isArray(o.items));
});

test('registry knows every engine', async () => {
  assert.deepEqual(ENGINE_NAMES.sort(), ['anthropic', 'bedrock', 'chexagent', 'local', 'maira2', 'mock', 'openai-compatible']);
  const e = await loadEngine('mock');
  assert.equal(e.name, 'mock');
  await assert.rejects(loadEngine('nope'), /Unknown ENGINE/);
});

test('a non-streaming provider still yields path items before the report', async () => {
  const whole = JSON.stringify(MOCK_RESULT);
  const eng = makeEngine('once', async function* () {
    yield { type: 'delta', text: whole };
  });
  const types = [];
  for await (const ev of eng.analyze({ imageBase64: 'x', model: 'm' })) types.push(ev.type);
  assert.equal(types.filter((t) => t === 'path').length, MOCK_RESULT.reading_path.length);
  assert.deepEqual(types.slice(-2), ['report', 'done']);
});

test('fenced output with prose is handled end to end', async () => {
  const eng = makeEngine('fenced', async function* () {
    yield { type: 'delta', text: 'Sure, here it is:\n```json\n' };
    yield { type: 'delta', text: JSON.stringify(MOCK_RESULT) };
    yield { type: 'delta', text: '\n```\nHope this helps.' };
  });
  const evs = [];
  for await (const ev of eng.analyze({ imageBase64: 'x', model: 'm' })) evs.push(ev);
  assert.equal(evs.find((e) => e.type === 'report').report.impression, MOCK_RESULT.report.impression);
});

test('unparseable output becomes an error event', async () => {
  const eng = makeEngine('bad', async function* () {
    yield { type: 'delta', text: 'I cannot help with that.' };
  });
  const evs = [];
  for await (const ev of eng.analyze({ imageBase64: 'x', model: 'm' })) evs.push(ev);
  assert.equal(evs.at(-1).type, 'error');
});

test('db stores reads and ratings and summarises them', () => {
  const rh = db.hashReport(MOCK_RESULT.report);
  const id = db.insertRead({ engine: 'mock', model: 'm1', source: 'bench', imageName: 'a.jpg', reportHash: rh, report: MOCK_RESULT.report, path: [] });
  db.insertRead({ engine: 'mock', model: 'm1', source: 'bench', imageName: 'a.jpg', reportHash: rh, report: MOCK_RESULT.report, path: [] });
  db.insertRating({ readId: id, engine: 'mock', model: 'm1', reportHash: rh, acceptable: true, studyType: 'Chest 1 view', icu: false, modality: 'XR' });
  db.insertRating({ readId: null, engine: 'mock', model: 'm2', reportHash: rh, acceptable: false, reason: 'missed effusion', studyType: 'Chest 1 view', icu: true, modality: 'XR' });
  const s = db.benchmarkSummary();
  const m1 = s.ratings.find((r) => r.model === 'm1');
  const m2 = s.ratings.find((r) => r.model === 'm2');
  assert.equal(m1.acceptable_pct, 100);
  assert.equal(m2.acceptable_pct, 0);
  assert.equal(s.unrated.find((u) => u.model === 'm1').unrated, 1);
  const csv = db.toCsv(db.benchmarkRows());
  assert.match(csv.split('\n')[0], /^read_id,read_at,engine,model/);
  assert.equal(csv.trim().split('\n').length, 3);
  assert.equal(db.hashReport('a'), db.hashReport('a '));
  assert.notEqual(db.hashReport('a'), db.hashReport('b'));
});
