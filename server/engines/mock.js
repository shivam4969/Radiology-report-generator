// Mock engine: streams a canned reading path and report for the sample
// pneumonia image, so the UI, tests and smoke run with no API key.
//
//   ENGINE=mock
//   MOCK_DELAY_MS   pause between streamed chunks (default 700)

import { makeEngine } from './base.js';

const DELAY = Number(process.env.MOCK_DELAY_MS ?? 700);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export const MOCK_RESULT = {
  reading_path: [
    { region: 'Both lungs, overview', box: [0.08, 0.12, 0.84, 0.72], note: 'Checking the air spaces top to bottom. The right base looks denser than the left.', zoom: false },
    { region: 'Right lower lobe', box: [0.12, 0.52, 0.32, 0.3], note: 'Patchy airspace opacity with air bronchograms, the pattern of lobar pneumonia.', zoom: true },
    { region: 'Left lung', box: [0.55, 0.12, 0.36, 0.7], note: 'Clear. No consolidation, nodule or pneumothorax.', zoom: false },
    { region: 'Heart and mediastinum', box: [0.33, 0.35, 0.36, 0.38], note: 'Normal heart size. Mediastinal contours are normal.', zoom: false },
    { region: 'Diaphragm and costophrenic angles', box: [0.08, 0.68, 0.84, 0.22], note: 'Angles are sharp. No pleural effusion.', zoom: false },
    { region: 'Bones and soft tissues', box: [0.04, 0.04, 0.92, 0.92], note: 'No fracture. Soft tissues unremarkable.', zoom: false },
  ],
  report: {
    exam: 'Chest radiograph, one view.',
    findings:
      'Lungs: Patchy airspace opacity in the right lower lobe with air bronchograms. Left lung clear.\n' +
      'Pleura: No pleural effusion or pneumothorax.\n' +
      'Heart and mediastinum: Normal heart size. Mediastinal contours normal.\n' +
      'Bones and soft tissues: No acute abnormality.\n' +
      'Lines and tubes: None.',
    impression: '1. Right lower lobe airspace opacity, most consistent with pneumonia.\n2. No pleural effusion. Heart size normal.',
  },
};

export const MOCK_DDX = {
  items: [
    {
      finding: 'Right lower lobe airspace opacity',
      question: 'The haze at the base of the right lung. What could it be?',
      differentials: [
        { dx: 'Bacterial pneumonia', why: 'lobar consolidation with air bronchograms; most likely with cough and fever' },
        { dx: 'Aspiration pneumonitis', why: 'dependent lower lobe location' },
        { dx: 'Post-obstructive pneumonia', why: 'unlikely, but the reason to confirm clearance' },
      ],
      management: {
        next_step: 'Treat clinically; repeat chest radiograph in 6 to 8 weeks to confirm resolution.',
        guideline: 'ACR Appropriateness Criteria, Acute Respiratory Illness in Immunocompetent Patients',
      },
    },
  ],
};

// Pretty-print, then split into chunks that break mid-item so the incremental
// scanner is exercised the way a real stream would.
function chunksOf(obj) {
  const text = JSON.stringify(obj, null, 2);
  const parts = [];
  let i = 0;
  while (i < text.length) {
    const n = 40 + Math.floor(Math.random() * 120);
    parts.push(text.slice(i, i + n));
    i += n;
  }
  return parts;
}

async function* generate({ system, image }) {
  const payload = image ? MOCK_RESULT : MOCK_DDX;
  const parts = chunksOf(payload);
  // The reading path accounts for most of the text; slow those chunks down so
  // the cloud has time to travel, then speed up for the report.
  const pathEnd = parts.findIndex((p) => p.includes('"report"'));
  for (let i = 0; i < parts.length; i++) {
    yield { type: 'delta', text: parts[i] };
    const inPath = pathEnd < 0 || i < pathEnd;
    await sleep(image ? (inPath ? DELAY / 2 : DELAY / 8) : DELAY / 10);
  }
}

export default makeEngine('mock', generate);
