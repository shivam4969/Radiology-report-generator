// CheXagent engine. The weights run in scripts/chexagent_server.py.
// CheXagent writes a findings section. It does not emit grounded boxes, so the
// reading path stays empty and the text lands in the report.
//
//   ENGINE=chexagent
//   MODEL=StanfordAIMI/CheXagent-2-3b
//   CHEXAGENT_BASE_URL=http://127.0.0.1:8009

import { request as httpRequest } from 'node:http';
import { makeEngine } from './base.js';

function postJson(url, body) {
  const payload = Buffer.from(JSON.stringify(body));
  const target = new URL(url);
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        protocol: target.protocol,
        hostname: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'content-length': String(payload.length),
        },
      },
      (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          resolve({ status: res.statusCode || 0, text: Buffer.concat(chunks).toString('utf8') });
        });
      },
    );
    req.setTimeout(0);
    req.on('error', reject);
    req.end(payload);
  });
}

function baseUrl() {
  return (process.env.CHEXAGENT_BASE_URL || 'http://127.0.0.1:8009').replace(/\/$/, '');
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
        'CheXagent drafts findings from a chest X-ray. Differential diagnosis needs a text model: set DDX_ENGINE and DDX_MODEL.',
    };
    return;
  }

  const url = `${baseUrl()}/read`;
  let res;
  try {
    res = await postJson(url, {
      imageBase64: image.base64,
      mime: image.mime,
      history: indicationFrom(history, text),
    });
  } catch (err) {
    yield {
      type: 'error',
      message: `chexagent: cannot reach ${baseUrl()} (${err?.message || err}). Start it with: npm run chexagent`,
    };
    return;
  }
  let data = {};
  try {
    data = res.text ? JSON.parse(res.text) : {};
  } catch {
    data = { error: res.text };
  }
  if (res.status < 200 || res.status >= 300) {
    yield { type: 'error', message: `chexagent: ${data.error || res.text || res.status}` };
    return;
  }
  const findings = String(data?.findings || '').trim();
  if (!findings) {
    yield { type: 'error', message: data?.error || 'CheXagent returned no findings.' };
    return;
  }

  yield {
    type: 'delta',
    text: JSON.stringify({
      reading_path: [],
      report: {
        exam: String(data?.exam || '').trim() || 'Chest radiograph, frontal view.',
        findings,
        impression: findings,
      },
    }),
  };
}

export default makeEngine('chexagent', generate);
