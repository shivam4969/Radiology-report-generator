// Thin client for the OpenRead API. SSE over POST via fetch + ReadableStream.

export async function getConfig() {
  const r = await fetch('/api/config');
  return r.json();
}

export async function getSamples() {
  const r = await fetch('/api/samples');
  return r.json();
}

/**
 * POST /api/analyze and call onEvent for each server event:
 *   { type: 'start' | 'path' | 'report' | 'done' | 'error', ... }
 * Resolves when the stream ends. Pass an AbortSignal to cancel.
 */
export async function analyze({ imageBase64, mime, history, imageName }, onEvent, signal) {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
    body: JSON.stringify({ imageBase64, mime, history, imageName }),
    signal,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.error) msg = j.error;
    } catch {}
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const handle = (frame) => {
    const data = frame
      .split('\n')
      .filter((l) => l.startsWith('data:'))
      .map((l) => l.slice(5).trimStart())
      .join('\n');
    if (!data) return;
    try {
      onEvent(JSON.parse(data));
    } catch (e) {
      console.warn('bad SSE frame', data, e);
    }
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      handle(buf.slice(0, i));
      buf = buf.slice(i + 2);
    }
  }
  if (buf.trim()) handle(buf);
}

export async function ddx(report) {
  const r = await fetch('/api/ddx', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ report }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || `${r.status}`);
  return j;
}

export async function rate(payload) {
  const r = await fetch('/api/rate', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || `${r.status}`);
  return j;
}

/** File or Blob -> { dataUrl, base64, mime } */
export function readImage(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onerror = () => reject(fr.error);
    fr.onload = () => {
      const dataUrl = String(fr.result);
      const m = dataUrl.match(/^data:([^;]+);base64,(.*)$/s);
      if (!m) return reject(new Error('Unsupported image'));
      resolve({ dataUrl, mime: m[1], base64: m[2] });
    };
    fr.readAsDataURL(blob);
  });
}
