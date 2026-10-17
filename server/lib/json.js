// Tolerant JSON helpers for model output.
//
// parseModelJson(text)  -> object | null
//   Accepts a bare JSON object, an object wrapped in ``` or ```json fences,
//   prose before or after it, and a truncated (partial stream) object, which
//   it closes as best it can.
//
// PathScanner            -> incremental scanner for reading_path items
//   feed it chunks of a streamed response; it returns each reading_path item
//   as soon as that item's object is complete, without waiting for the rest.

export function stripFences(text) {
  if (!text) return '';
  let t = String(text);
  // ```json ... ``` or ``` ... ``` anywhere in the text
  const fence = t.match(/```(?:json|JSON)?\s*([\s\S]*?)```/);
  if (fence) t = fence[1];
  else t = t.replace(/^\s*```(?:json|JSON)?\s*/, '').replace(/\s*```\s*$/, '');
  return t;
}

// Scan from the first "{" and return the balanced object text, or, if the
// text ends mid-object, the text from "{" to the end plus the closers needed.
export function extractObject(text) {
  const t = stripFences(text);
  const start = t.indexOf('{');
  if (start < 0) return null;
  const stack = [];
  let inStr = false;
  let esc = false;
  for (let i = start; i < t.length; i++) {
    const c = t[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === '{') stack.push('}');
    else if (c === '[') stack.push(']');
    else if (c === '}' || c === ']') {
      stack.pop();
      if (stack.length === 0) return { text: t.slice(start, i + 1), complete: true };
    }
  }
  // Truncated. Close an open string, drop a dangling comma or key, then close containers.
  let body = t.slice(start);
  if (inStr) body += '"';
  body = body.replace(/,\s*$/, '');
  body = body.replace(/,?\s*"[^"]*"\s*:\s*$/, ''); // dangling key with no value
  body = body.replace(/,\s*$/, '');
  return { text: body + stack.reverse().join(''), complete: false };
}

export function parseModelJson(text) {
  const ex = extractObject(text);
  if (!ex) return null;
  try {
    return JSON.parse(ex.text);
  } catch {
    // Last resort: strip trailing commas before closers, which some models emit.
    try {
      return JSON.parse(ex.text.replace(/,(\s*[}\]])/g, '$1'));
    } catch {
      return null;
    }
  }
}

const KEY_RE = /"reading_path"\s*:\s*\[/;

export class PathScanner {
  constructor() {
    this.buf = '';
    this.pos = 0; // next char to examine
    this.phase = 'seek'; // seek | array | item | done
    this.depth = 0;
    this.inStr = false;
    this.esc = false;
    this.itemStart = -1;
    this.count = 0;
  }

  get done() {
    return this.phase === 'done';
  }

  /** Feed a chunk. Returns an array of newly completed reading_path items. */
  push(chunk) {
    this.buf += chunk;
    const out = [];
    if (this.phase === 'seek') {
      const m = KEY_RE.exec(this.buf);
      if (!m) return out;
      this.pos = m.index + m[0].length;
      this.phase = 'array';
    }
    while (this.pos < this.buf.length && this.phase !== 'done') {
      const c = this.buf[this.pos];
      if (this.phase === 'array') {
        if (c === '{') {
          this.phase = 'item';
          this.itemStart = this.pos;
          this.depth = 1;
          this.inStr = false;
          this.esc = false;
        } else if (c === ']') {
          this.phase = 'done';
        }
        this.pos++;
        continue;
      }
      // phase === 'item'
      if (this.inStr) {
        if (this.esc) this.esc = false;
        else if (c === '\\') this.esc = true;
        else if (c === '"') this.inStr = false;
      } else if (c === '"') this.inStr = true;
      else if (c === '{' || c === '[') this.depth++;
      else if (c === '}' || c === ']') {
        this.depth--;
        if (this.depth === 0) {
          const raw = this.buf.slice(this.itemStart, this.pos + 1);
          try {
            const item = JSON.parse(raw);
            out.push(normalizePathItem(item, this.count));
            this.count++;
          } catch {
            // malformed item: skip it rather than stall the stream
          }
          this.phase = 'array';
        }
      }
      this.pos++;
    }
    return out;
  }
}

export function normalizePathItem(item, index = 0) {
  const box = Array.isArray(item?.box) && item.box.length === 4
    ? item.box.map((v) => clamp01(Number(v)))
    : [0.1, 0.1, 0.8, 0.8];
  // Some models answer in pixels or percent; bring anything > 1 back to 0..1.
  const big = item?.box?.some?.((v) => Number(v) > 1.5);
  const norm = big ? item.box.map((v) => clamp01(Number(v) / 100)) : box;
  return {
    index,
    region: String(item?.region ?? `Region ${index + 1}`),
    box: norm,
    note: String(item?.note ?? ''),
    zoom: Boolean(item?.zoom),
  };
}

export function normalizeReport(r) {
  return {
    exam: String(r?.exam ?? ''),
    findings: String(r?.findings ?? ''),
    impression: String(r?.impression ?? ''),
  };
}

function clamp01(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.min(1, Math.max(0, v));
}
