// Shared plumbing for every engine.
//
// An engine module exports `generate({ system, text, image, model })`, an async
// generator of `{ type: 'delta', text }` chunks (one chunk for a provider that
// does not stream). `makeEngine` wraps it into the public interface:
//
//   analyze({ imageBase64, mime, history, model }) -> async iterator of events
//     { type: 'path',   item }      as soon as each reading_path item is parseable
//     { type: 'report', report }    once the full object has arrived
//     { type: 'done',   raw, usage } at the end
//     { type: 'error',  message }
//
//   complete({ system, text, model }) -> string  (text-only helper, used by /api/ddx)

import { SYSTEM_PROMPT, userMessage } from '../lib/prompt.js';
import { PathScanner, parseModelJson, normalizePathItem, normalizeReport } from '../lib/json.js';

export function makeEngine(name, generate) {
  async function* analyze({ imageBase64, mime = 'image/jpeg', history = '', model }) {
    const scanner = new PathScanner();
    let raw = '';
    let emittedPath = 0;
    try {
      for await (const ev of generate({
        system: SYSTEM_PROMPT,
        text: userMessage(history),
        history,
        image: { base64: imageBase64, mime },
        model,
      })) {
        if (ev.type === 'delta' && ev.text) {
          raw += ev.text;
          for (const item of scanner.push(ev.text)) {
            emittedPath++;
            yield { type: 'path', item };
          }
        } else if (ev.type === 'error') {
          yield ev;
          return;
        }
      }
    } catch (err) {
      yield { type: 'error', message: err?.message || String(err) };
      return;
    }
    const obj = parseModelJson(raw);
    if (!obj) {
      yield { type: 'error', message: 'Model did not return parseable JSON.', raw };
      return;
    }
    // If the scanner missed items (for example a non-streaming provider or odd
    // whitespace), emit whatever the final parse has that was not sent yet.
    const path = Array.isArray(obj.reading_path) ? obj.reading_path : [];
    for (let i = emittedPath; i < path.length; i++) {
      yield { type: 'path', item: normalizePathItem(path[i], i) };
    }
    yield { type: 'report', report: normalizeReport(obj.report) };
    yield { type: 'done', raw };
  }

  async function complete({ system, text, model }) {
    let out = '';
    for await (const ev of generate({ system, text, image: null, model })) {
      if (ev.type === 'delta') out += ev.text || '';
      else if (ev.type === 'error') throw new Error(ev.message);
    }
    return out;
  }

  return { name, generate, analyze, complete };
}

/** Parse a text/event-stream body from fetch into `data:` payload strings. */
export async function* sseLines(response) {
  const reader = response.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = frame
        .split('\n')
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trimStart())
        .join('\n');
      if (data) yield data;
    }
  }
  if (buf.trim()) {
    const data = buf.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trimStart()).join('\n');
    if (data) yield data;
  }
}

export async function readError(res) {
  let body = '';
  try {
    body = await res.text();
  } catch {}
  return `${res.status} ${res.statusText}${body ? `: ${body.slice(0, 400)}` : ''}`;
}
