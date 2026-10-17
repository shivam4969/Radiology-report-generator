// OpenAI-compatible chat completions: OpenAI, OpenRouter, vLLM, Ollama (/v1),
// LM Studio, Astra and anything else that speaks the same API.
//
//   OPENAI_BASE_URL   default https://api.openai.com/v1
//   OPENAI_API_KEY    optional for local servers
//   MODEL             e.g. gpt-4o, qwen3-vl, google/medgemma-4b-it

import { makeEngine, sseLines, readError } from './base.js';

const BASE = (process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');
const KEY = process.env.OPENAI_API_KEY || '';
const NO_STREAM = process.env.OPENAI_NO_STREAM === '1';

async function* generate({ system, text, image, model }) {
  const content = image
    ? [
        { type: 'text', text },
        { type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.base64}` } },
      ]
    : text;
  const body = {
    model,
    stream: !NO_STREAM,
    temperature: 0.2,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content },
    ],
  };
  const headers = { 'content-type': 'application/json' };
  if (KEY) headers.authorization = `Bearer ${KEY}`;
  if (BASE.includes('openrouter.ai')) headers['HTTP-Referer'] = 'https://github.com/openread';

  const res = await fetch(`${BASE}/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
  if (!res.ok) {
    yield { type: 'error', message: `openai-compatible: ${await readError(res)}` };
    return;
  }
  if (NO_STREAM || !res.headers.get('content-type')?.includes('text/event-stream')) {
    const json = await res.json();
    yield { type: 'delta', text: json?.choices?.[0]?.message?.content ?? '' };
    return;
  }
  for await (const data of sseLines(res)) {
    if (data === '[DONE]') break;
    let ev;
    try {
      ev = JSON.parse(data);
    } catch {
      continue;
    }
    const delta = ev?.choices?.[0]?.delta?.content;
    if (typeof delta === 'string' && delta) yield { type: 'delta', text: delta };
  }
}

export default makeEngine('openai-compatible', generate);
