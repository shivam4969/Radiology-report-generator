// Map a MAIRA-2 grounded report onto the OpenRead JSON contract.
//
// The sidecar returns findings as { text, boxes }, where each box is
// [x1, y1, x2, y2] in fractions of the original image. A finding with no
// box stays in the report and is left off the reading path: MAIRA-2 omits
// boxes for normal anatomy and for observations it cannot localise.

const REGION_MAX = 72;

export function xyxyToXywh(box) {
  if (!Array.isArray(box) || box.length < 4) return null;
  const [a, b, c, d] = box.slice(0, 4).map(Number);
  if (![a, b, c, d].every(Number.isFinite)) return null;
  const x = Math.min(a, c);
  const y = Math.min(b, d);
  const w = Math.abs(c - a);
  const h = Math.abs(d - b);
  if (w <= 0 || h <= 0) return null;
  const r = (v) => Math.round(v * 10000) / 10000;
  return [r(x), r(y), r(w), r(h)];
}

function regionLabel(text) {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= REGION_MAX) return clean;
  return `${clean.slice(0, REGION_MAX - 1).trimEnd()}…`;
}

export function findingsToRead(data) {
  const src = Array.isArray(data) ? data : data?.findings;
  const findings = Array.isArray(src) ? src : [];
  const lateral = Boolean(data?.lateral);
  const readingPath = [];
  const lines = [];

  for (const finding of findings) {
    const text = String(finding?.text ?? finding?.finding ?? '').trim();
    if (!text) continue;
    lines.push(text);
    const boxes = (Array.isArray(finding?.boxes) ? finding.boxes : [])
      .map(xyxyToXywh)
      .filter(Boolean);
    const label = regionLabel(text);
    boxes.forEach((box, j) => {
      readingPath.push({
        region: boxes.length > 1 ? `${label} (${j + 1})` : label,
        box,
        note: text,
        zoom: true,
      });
    });
  }

  const impression = lines.length
    ? lines.map((line, i) => `${i + 1}. ${line}`).join('\n')
    : '';

  return {
    reading_path: readingPath,
    report: {
      exam: lateral
        ? 'Chest radiograph, frontal and lateral views.'
        : 'Chest radiograph, frontal view.',
      findings: lines.join('\n'),
      impression,
    },
  };
}
