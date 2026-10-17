import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseModelJson, extractObject, PathScanner, normalizePathItem } from '../server/lib/json.js';

const SAMPLE = {
  reading_path: [
    { region: 'Both lungs', box: [0.1, 0.1, 0.8, 0.7], note: 'Overview, with "quotes" and a } brace', zoom: false },
    { region: 'Right lower lobe', box: [0.12, 0.52, 0.32, 0.3], note: 'Dense. Looking closer.', zoom: true },
    { region: 'Heart', box: [0.33, 0.35, 0.36, 0.38], note: 'Normal size', zoom: false },
  ],
  report: { exam: 'Chest radiograph, one view.', findings: 'Lungs: opacity RLL.\nPleura: clear.', impression: '1. RLL pneumonia.' },
};
const TEXT = JSON.stringify(SAMPLE, null, 2);

test('parses a bare JSON object', () => {
  assert.deepEqual(parseModelJson(TEXT), SAMPLE);
});

test('parses JSON inside ```json fences with prose around it', () => {
  const s = 'Here is the reading:\n\n```json\n' + TEXT + '\n```\n\nLet me know if you need anything else.';
  assert.deepEqual(parseModelJson(s), SAMPLE);
});

test('parses JSON inside bare ``` fences', () => {
  assert.deepEqual(parseModelJson('```\n' + TEXT + '\n```'), SAMPLE);
});

test('ignores trailing text after the object', () => {
  assert.deepEqual(parseModelJson(TEXT + '\n\nNote: this is a draft. } ] extra braces'), SAMPLE);
});

test('ignores leading prose before the object', () => {
  assert.deepEqual(parseModelJson('Sure! The JSON object is:\n' + TEXT), SAMPLE);
});

test('tolerates trailing commas', () => {
  const s = '{"reading_path": [{"region": "A", "box": [0,0,1,1], "note": "n", "zoom": false,},], "report": {"exam": "x",},}';
  const o = parseModelJson(s);
  assert.equal(o.reading_path[0].region, 'A');
  assert.equal(o.report.exam, 'x');
});

test('closes a truncated (partial stream) object', () => {
  const cut = TEXT.slice(0, TEXT.indexOf('"Heart"') + 8); // ends inside the third item
  const o = parseModelJson(cut);
  assert.ok(o, 'should parse something');
  assert.equal(o.reading_path.length, 3);
  assert.equal(o.reading_path[0].region, 'Both lungs');
  assert.equal(o.reading_path[2].region, 'Heart');
  assert.equal(extractObject(cut).complete, false);
});

test('closes a truncation that ends mid-string', () => {
  const cut = TEXT.slice(0, TEXT.indexOf('Dense. Look') + 6);
  const o = parseModelJson(cut);
  assert.ok(o);
  assert.equal(o.reading_path.length, 2);
  assert.match(o.reading_path[1].note, /^Dense/);
});

test('closes a truncation that ends after a key with no value', () => {
  const cut = TEXT.slice(0, TEXT.indexOf('"findings"') + 12); // '"findings": '
  const o = parseModelJson(cut);
  assert.ok(o);
  assert.equal(o.report.exam, 'Chest radiograph, one view.');
});

test('returns null when there is no object', () => {
  assert.equal(parseModelJson('I cannot read this image.'), null);
  assert.equal(parseModelJson(''), null);
});

test('PathScanner emits each item as soon as it closes, across arbitrary chunk boundaries', () => {
  for (const size of [1, 3, 7, 16, 50, 1000]) {
    const sc = new PathScanner();
    const got = [];
    const emittedAt = [];
    for (let i = 0; i < TEXT.length; i += size) {
      const items = sc.push(TEXT.slice(i, i + size));
      for (const it of items) {
        got.push(it);
        emittedAt.push(i + size);
      }
    }
    assert.equal(got.length, 3, `chunk size ${size}`);
    assert.deepEqual(
      got.map((g) => g.region),
      ['Both lungs', 'Right lower lobe', 'Heart'],
    );
    assert.equal(got[1].zoom, true);
    assert.deepEqual(got[1].box, [0.12, 0.52, 0.32, 0.3]);
    assert.ok(sc.done);
    // First item must be emitted before the report section even starts arriving.
    assert.ok(emittedAt[0] <= TEXT.indexOf('"report"') + size, `first item emitted early (chunk ${size})`);
  }
});

test('PathScanner handles braces and brackets inside strings', () => {
  const sc = new PathScanner();
  const items = sc.push('{"reading_path":[{"region":"a } ] {","box":[0,0,1,1],"note":"x","zoom":false},{"region":"b","box":[0,0,1,1],"note":"\\"q\\"","zoom":true}],"report":{}}');
  assert.equal(items.length, 2);
  assert.equal(items[0].region, 'a } ] {');
  assert.equal(items[1].note, '"q"');
});

test('PathScanner tolerates code fences and prose before the key', () => {
  const sc = new PathScanner();
  const a = sc.push('Here you go:\n```json\n{\n  "reading_pa');
  assert.equal(a.length, 0);
  const b = sc.push('th": [ {"region":"R","box":[0.5,0.5,0.1,0.1],"note":"","zoom":false} ');
  assert.equal(b.length, 1);
  assert.equal(b[0].index, 0);
  const c = sc.push('], "report": {"exam":"e"} }\n```');
  assert.equal(c.length, 0);
  assert.ok(sc.done);
});

test('PathScanner skips a malformed item and continues', () => {
  const sc = new PathScanner();
  const items = sc.push('{"reading_path":[{"region": nope},{"region":"ok","box":[0,0,1,1],"note":"","zoom":false}]}');
  assert.equal(items.length, 1);
  assert.equal(items[0].region, 'ok');
});

test('normalizePathItem clamps boxes and rescales percent boxes', () => {
  assert.deepEqual(normalizePathItem({ region: 'x', box: [-0.2, 0.5, 1.4, 0.2] }).box, [0, 0.5, 1, 0.2]);
  assert.deepEqual(normalizePathItem({ region: 'x', box: [10, 20, 50, 25] }).box, [0.1, 0.2, 0.5, 0.25]);
  assert.deepEqual(normalizePathItem({ region: 'x' }).box, [0.1, 0.1, 0.8, 0.8]);
  assert.equal(normalizePathItem({}, 4).region, 'Region 5');
});
