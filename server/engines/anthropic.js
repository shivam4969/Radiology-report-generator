// Anthropic Messages API with image blocks, streamed.
//
//   ANTHROPIC_API_KEY
//   ANTHROPIC_BASE_URL  default https://api.anthropic.com
//   MODEL               e.g. claude-sonnet-4-5

import { makeEngine, sseLines, readError } from './base.js';

const BASE = (process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com').replace(/\/$/, '');
const KEY = process.env.ANTHROPIC_API_KEY || '';
const MAX_TOKENS = Number(process.env.MAX_TOKENS || 2048);

async function* generate({ system, text, image, model }) {
  const content = image
    ? [
        { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.base64 } },
        { type: 'text', text },
      ]
    : [{ type: 'text', text }];
  const body = {
    model,
    max_tokens: MAX_TOKENS,
    temperature: 0.2,
    stream: true,
    system,
    messages: [{ role: 'user', content }],
  };
  const res = await fetch(`${BASE}/v1/messages`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    yield { type: 'error', message: `anthropic: ${await readError(res)}` };
    return;
  }
  for await (const data of sseLines(res)) {
    let ev;
    try {
      ev = JSON.parse(data);
    } catch {
      continue;
    }
    if (ev.type === 'content_block_delta' && ev.delta?.type === 'text_delta') {
      yield { type: 'delta', text: ev.delta.text };
    } else if (ev.type === 'error') {
      yield { type: 'error', message: `anthropic: ${ev.error?.message || 'stream error'}` };
      return;
    } else if (ev.type === 'message_stop') {
      break;
    }
  }
}

export default makeEngine('anthropic', generate);
