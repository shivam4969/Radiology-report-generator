// MAIRA-2 engine. The weights run in scripts/maira2_server.py. This module
// asks that sidecar for grounded findings and yields the OpenRead JSON.
// MAIRA-2 has its own reporting prompt, so the shared system prompt is not sent.
//
//   ENGINE=maira2
//   MODEL=microsoft/maira-2
//   MAIRA2_BASE_URL=http://127.0.0.1:8008
//   MAIRA2_PACE_MS=600     pause between regions so the reading cloud can move

import { makeEngine, readError } from './base.js';
import { findingsToRead } from '../lib/maira2.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function baseUrl() {
  return (process.env.MAIRA2_BASE_URL || 'http://127.0.0.1:8008').replace(/\/$/, '');
}

function indicationFrom(history, text) {
  const direct = (history || '').trim();
  if (direct) return direct;
  const match = String(text || '').match(/^Clinical history:\s*([\s\S]*?)\n\nRead the chest X-ray/);
  return match ? match[1].trim() : '';
}

async function* generate({ text, history, image }) {
  if (!image) {
    yield {
      type: 'error',
      message:
        'MAIRA-2 drafts grounded findings from a chest X-ray. Differential diagnosis needs a text model: set DDX_ENGINE (for example openai-compatible) and DDX_MODEL.',
    };
    return;
  }

  const url = `${baseUrl()}/read`;
  let res;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        imageBase64: image.base64,
        mime: image.mime,
        history: indicationFrom(history, text),
        grounding: true,
      }),
    });
  } catch (err) {
    yield {
      type: 'error',
      message: `maira2: cannot reach ${baseUrl()} (${err?.message || err}). Start it with: python3 scripts/maira2_server.py`,
    };
    return;
  }
  if (!res.ok) {
    yield { type: 'error', message: `maira2: ${await readError(res)}` };
    return;
  }

  const data = await res.json();
  const read = findingsToRead(data);
  if (!read.report.findings) {
    yield {
      type: 'error',
      message: data?.error || 'MAIRA-2 returned no findings.',
      raw: typeof data?.raw === 'string' ? data.raw : undefined,
    };
    return;
  }

  const pace = Number(process.env.MAIRA2_PACE_MS ?? 600);
  const path = read.reading_path;
  if (!path.length || !(pace > 0)) {
    yield { type: 'delta', text: JSON.stringify(read) };
    return;
  }

  yield { type: 'delta', text: '{"reading_path":[' };
  for (let i = 0; i < path.length; i++) {
    yield { type: 'delta', text: (i ? ',' : '') + JSON.stringify(path[i]) };
    await sleep(pace);
  }
  yield { type: 'delta', text: `],"report":${JSON.stringify(read.report)}}` };
}

export default makeEngine('maira2', generate);
