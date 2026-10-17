// Local engine: a fully offline pipeline of small specialist models.
//
//   ENGINE=local
//
//   image ──► findings model (py/cxr_worker.py: ExecuTorch .pte of TorchXRayVision
//             DenseNet-121, 18 probabilities + CAM boxes)
//         ──► reading path (built here, fixed anatomic order, streamed with delays)
//         ──► report writer (small local LLM, or a template)
//
//   CXR_PYTHON            python executable (default: python3; python on Windows)
//   CXR_WORKER            worker script (default: py/cxr_worker.py; tests use py/cxr_worker_mock.py)
//   CXR_PTE               path to the exported classifier (default: models/cxr-densenet121.pte)
//   CXR_THRESHOLD         probability at which a finding is "positive" (default 0.5)
//   CXR_MAX_FINDINGS      max positive findings shown on the path (default 4)
//   LOCAL_PATH_DELAY_MS   pause between path items so the UI animates (default 600)
//
//   LOCAL_LLM             template | executorch | openai-compatible   (default template)
//   LLM_RUNNER            executorch: path to the llama runner binary (llama_main / llama_main.exe)
//   LLM_MODEL             executorch: the LLM .pte
//   LLM_TOKENIZER         executorch: tokenizer file (tokenizer.model or tokenizer.json)
//   LLM_RUNNER_ARGS       executorch: extra args appended verbatim (e.g. "--cpu_threads 8")
//   LLM_MAX_TOKENS_FLAG   executorch: flag for the token budget; --max_new_tokens (default) or --seq_len on older runners
//   LLM_CHAT_TEMPLATE     llama3 | chatml | none   (default llama3)
//   LOCAL_LLM_URL         openai-compatible: e.g. http://localhost:11434/v1 (Ollama)
//   LOCAL_LLM_MODEL       openai-compatible: e.g. llama3.2:3b, qwen3:4b
//   LLM_MAX_NEW_TOKENS    default 700
//
// Events yielded by analyze(), in order:
//   findings  { probs, cams, backend, ms }   (the server forwards it; bench.mjs uses it)
//   path      { item } ... one per region
//   token     { text }  while the LLM streams
//   report    { report }
//   done      { raw }

import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { parseModelJson, normalizeReport } from '../lib/json.js';
import { sseLines, readError } from './base.js';
import { buildReadingPath, templateReport, templateDdx, REPORT_WRITER_SYSTEM, reportWriterUser, topFinding } from '../lib/findings.js';

const ROOT = resolve(import.meta.dirname, '../..');
const env = (k, d) => (process.env[k] === undefined || process.env[k] === '' ? d : process.env[k]);

const PYTHON = env('CXR_PYTHON', process.platform === 'win32' ? 'python' : 'python3');
const WORKER = resolve(ROOT, env('CXR_WORKER', 'py/cxr_worker.py'));
const PTE = resolve(ROOT, env('CXR_PTE', 'models/cxr-densenet121.pte'));
const THRESHOLD = Number(env('CXR_THRESHOLD', 0.5));
const MAX_FINDINGS = Number(env('CXR_MAX_FINDINGS', 4));
const PATH_DELAY = Number(env('LOCAL_PATH_DELAY_MS', 600));
const LLM = env('LOCAL_LLM', 'template').toLowerCase();
const MAX_NEW_TOKENS = Number(env('LLM_MAX_NEW_TOKENS', 700));

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.error('[local]', ...a);

// ------------------------------------------------------------ findings -----

let worker = null; // { proc, pending: Map<id, {resolve,reject}>, ready: Promise, backend }
let nextId = 1;

function startWorker() {
  if (!existsSync(WORKER)) throw new Error(`CXR_WORKER not found: ${WORKER}`);
  const proc = spawn(PYTHON, [WORKER], {
    cwd: ROOT,
    env: { ...process.env, CXR_PTE: PTE, PYTHONUNBUFFERED: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  const w = { proc, pending: new Map(), backend: null, ready: null, failed: null };
  // Hold the event loop open only while a request is in flight, so scripts
  // like bench.mjs exit when they are done instead of waiting on the worker.
  const refAll = (on) => {
    for (const s of [proc, proc.stdin, proc.stdout, proc.stderr]) (on ? s.ref : s.unref).call(s);
  };
  w.setBusy = () => refAll(true);
  w.setIdle = () => w.pending.size === 0 && refAll(false);
  let readyResolve, readyReject;
  w.ready = new Promise((res, rej) => ((readyResolve = res), (readyReject = rej)));
  w.ready.catch(() => {}); // avoid an unhandled rejection; callers await it themselves

  createInterface({ input: proc.stdout }).on('line', (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      log('worker said:', line);
      return;
    }
    if (msg.event === 'ready') {
      w.backend = msg.backend;
      log(`findings worker ready, backend=${msg.backend}, ${msg.pathologies?.length ?? '?'} pathologies`);
      readyResolve();
      return;
    }
    const p = w.pending.get(msg.id);
    if (!p) return;
    w.pending.delete(msg.id);
    w.setIdle();
    if (msg.error) p.reject(new Error(`findings worker: ${msg.error}`));
    else p.resolve(msg);
  });
  createInterface({ input: proc.stderr }).on('line', (line) => log(line));
  proc.on('error', (err) => {
    w.failed = new Error(`could not start ${PYTHON} ${WORKER}: ${err.message}`);
    readyReject(w.failed);
    for (const p of w.pending.values()) p.reject(w.failed);
    w.pending.clear();
    if (worker === w) worker = null;
  });
  proc.on('exit', (code, signal) => {
    const err = new Error(`findings worker exited (code ${code}${signal ? `, ${signal}` : ''}). See [local] log lines above; run \`${PYTHON} ${WORKER} --once samples/cxr-pneumonia-cc0.jpg\` to debug.`);
    w.failed = err;
    readyReject(err);
    for (const p of w.pending.values()) p.reject(err);
    w.pending.clear();
    if (worker === w) worker = null;
  });
  return w;
}

export async function classify(imageBase64, mime = 'image/jpeg') {
  if (!worker) worker = startWorker();
  const w = worker;
  w.setBusy();
  try {
    await w.ready;
  } catch (err) {
    w.setIdle();
    throw err;
  }
  const id = nextId++;
  const result = new Promise((resolve, reject) => w.pending.set(id, { resolve, reject }));
  w.proc.stdin.write(JSON.stringify({ id, image_b64: imageBase64, mime }) + '\n');
  return result;
}

export function stopWorker() {
  if (worker?.proc && !worker.proc.killed) worker.proc.kill('SIGTERM');
  worker = null;
}
process.on('exit', stopWorker);

// ------------------------------------------------------------------ LLM -----

function chatPrompt(system, user) {
  const tmpl = env('LLM_CHAT_TEMPLATE', 'llama3').toLowerCase();
  if (tmpl === 'chatml') {
    return `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n`;
  }
  if (tmpl === 'none') return `${system}\n\n${user}\n\n`;
  return (
    `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n${system}<|eot_id|>` +
    `<|start_header_id|>user<|end_header_id|>\n\n${user}<|eot_id|>` +
    `<|start_header_id|>assistant<|end_header_id|>\n\n`
  );
}

/** ExecuTorch llama runner: spawn, stream stdout, strip the echoed prompt and the stats lines. */
async function* llmExecutorch(system, user) {
  const runner = env('LLM_RUNNER', '');
  const model = env('LLM_MODEL', '');
  const tok = env('LLM_TOKENIZER', '');
  if (!runner || !model || !tok) {
    throw new Error('LOCAL_LLM=executorch needs LLM_RUNNER, LLM_MODEL and LLM_TOKENIZER (see scripts/export_llm.md)');
  }
  const prompt = chatPrompt(system, user);
  const flag = env('LLM_MAX_TOKENS_FLAG', '--max_new_tokens');
  const args = ['--model_path', model, '--tokenizer_path', tok, '--prompt', prompt, flag, String(MAX_NEW_TOKENS), '--temperature', '0'];
  const extra = env('LLM_RUNNER_ARGS', '').trim();
  if (extra) args.push(...extra.split(/\s+/));
  const proc = spawn(runner, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  createInterface({ input: proc.stderr }).on('line', (l) => log('runner:', l));

  // The runner echoes the prompt before generating. Everything up to and
  // including the tail of the prompt is dropped; the rest is streamed.
  const tail = prompt.slice(-60);
  let raw = '';
  let started = false;
  let sent = 0;
  const strip = (s) => s.replace(/^PyTorchObserver .*$/gm, '').replace(/^[IEW] \d{2}:\d{2}:\d{2}\.\d+ .*$/gm, '');
  const exit = new Promise((res) => proc.on('exit', res));
  for await (const chunk of proc.stdout) {
    raw += chunk.toString();
    if (!started) {
      const i = raw.indexOf(tail);
      if (i >= 0) {
        raw = raw.slice(i + tail.length);
        started = true;
      } else if (raw.length > prompt.length + 400) {
        started = true; // no echo (newer runner); keep everything
      } else continue;
    }
    const clean = strip(raw);
    if (clean.length > sent) {
      yield { type: 'delta', text: clean.slice(sent) };
      sent = clean.length;
    }
  }
  const code = await exit;
  if (!started) {
    const clean = strip(raw);
    if (clean) yield { type: 'delta', text: clean };
  }
  if (code !== 0 && !raw.trim()) throw new Error(`llama runner exited with code ${code}`);
}

/** Any OpenAI-compatible local server (Ollama, vLLM, LM Studio, llama.cpp server). */
async function* llmOpenAICompatible(system, user) {
  const base = env('LOCAL_LLM_URL', 'http://localhost:11434/v1').replace(/\/$/, '');
  const model = env('LOCAL_LLM_MODEL', '');
  if (!model) throw new Error('LOCAL_LLM=openai-compatible needs LOCAL_LLM_MODEL (e.g. llama3.2:3b)');
  const body = {
    model,
    stream: true,
    temperature: 0,
    messages: [
      { role: 'system', content: system },
      { role: 'user', content: user },
    ],
  };
  const res = await fetch(`${base}/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`local LLM at ${base}: ${await readError(res)}`);
  if (!res.headers.get('content-type')?.includes('text/event-stream')) {
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

function llmGenerator(system, user) {
  if (LLM === 'executorch') return llmExecutorch(system, user);
  if (LLM === 'openai-compatible' || LLM === 'openai' || LLM === 'ollama') return llmOpenAICompatible(system, user);
  throw new Error(`LOCAL_LLM="${LLM}" is not one of template, executorch, openai-compatible`);
}

export function llmLabel() {
  if (LLM === 'template') return 'template';
  if (LLM === 'executorch') return `executorch:${env('LLM_MODEL', '').split(/[\\/]/).pop() || '?'}`;
  return `${LLM}:${env('LOCAL_LLM_MODEL', '?')}`;
}

// -------------------------------------------------------------- engine -----

async function* analyze({ imageBase64, mime = 'image/jpeg', history = '' }) {
  let findings;
  try {
    findings = await classify(imageBase64, mime);
  } catch (err) {
    yield { type: 'error', message: err.message };
    return;
  }
  yield { type: 'findings', probs: findings.probs, cams: findings.cams, backend: findings.backend, ms: findings.ms };

  const items = buildReadingPath(findings, { threshold: THRESHOLD, maxFindings: MAX_FINDINGS });
  for (const item of items) {
    yield { type: 'path', item };
    await sleep(PATH_DELAY);
  }

  let report;
  let raw = '';
  if (LLM === 'template') {
    report = templateReport(findings, { threshold: THRESHOLD, history });
    raw = JSON.stringify({ report });
  } else {
    const user = reportWriterUser({ probs: findings.probs, cams: findings.cams, history, threshold: THRESHOLD });
    try {
      for await (const ev of llmGenerator(REPORT_WRITER_SYSTEM, user)) {
        if (ev.type === 'delta' && ev.text) {
          raw += ev.text;
          yield { type: 'token', text: ev.text };
        }
      }
      const obj = parseModelJson(raw);
      const r = obj?.report ?? (obj?.findings || obj?.impression ? obj : null);
      if (!r) throw new Error('report writer did not return parseable JSON');
      report = normalizeReport(r);
    } catch (err) {
      log(`report writer failed (${err.message}); using the templated report instead`);
      report = templateReport(findings, { threshold: THRESHOLD, history });
      raw = raw || JSON.stringify({ report });
    }
  }
  yield { type: 'report', report };
  yield { type: 'done', raw, findings: { top: topFinding(findings.probs), backend: findings.backend } };
}

/** Text-only completion, used by /api/ddx. */
async function complete({ system, text }) {
  if (LLM === 'template') return JSON.stringify(templateDdx(text));
  let out = '';
  for await (const ev of llmGenerator(system, text)) if (ev.type === 'delta') out += ev.text || '';
  return out;
}

export default { name: 'local', analyze, complete, classify, stopWorker };
