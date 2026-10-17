// Turning classifier output into the things OpenRead shows: a reading path in
// a fixed anatomic order, a deterministic templated report, a rule-based
// differential, and the prompt the local report-writer LLM gets.
//
// Everything here is pure (no I/O) so it is unit-testable without Python.
//
// The classifier is TorchXRayVision densenet121-res224-all: 18 outputs, each a
// probability on the model's own operating point (0.5 = its decision
// threshold). It is a research model. The numbers are not diagnoses and the
// boxes show where the classifier's activation was strongest, not a lesion.

export const GROUPS = {
  lung: ['Pneumonia', 'Consolidation', 'Lung Opacity', 'Infiltration', 'Atelectasis', 'Edema', 'Mass', 'Nodule', 'Lung Lesion', 'Emphysema', 'Fibrosis'],
  pleura: ['Effusion', 'Pneumothorax', 'Pleural_Thickening'],
  heart: ['Cardiomegaly', 'Enlarged Cardiomediastinum'],
  bones: ['Fracture', 'Hernia'],
};

const LABEL = {
  Pleural_Thickening: 'Pleural thickening',
  'Lung Opacity': 'Lung opacity',
  'Lung Lesion': 'Lung lesion',
  'Enlarged Cardiomediastinum': 'Enlarged cardiomediastinum',
};
export const label = (p) => LABEL[p] || p;
export const pct = (p) => `${Math.round(p * 100)}%`;

const BOX = {
  lungs: [0.08, 0.12, 0.84, 0.72],
  heart: [0.33, 0.35, 0.36, 0.38],
  pleura: [0.08, 0.6, 0.84, 0.3],
  bones: [0.04, 0.04, 0.92, 0.92],
};

/** Sorted [{ name, p, box }] of findings at or above the threshold. */
export function positives(probs, cams, threshold = 0.5, max = 4) {
  return Object.entries(probs || {})
    .filter(([, p]) => Number(p) >= threshold)
    .sort((a, b) => b[1] - a[1])
    .slice(0, max)
    .map(([name, p]) => ({ name, p: Number(p), box: cams?.[name] || BOX.lungs }));
}

export function topFinding(probs) {
  const e = Object.entries(probs || {}).sort((a, b) => b[1] - a[1])[0];
  return e ? { name: e[0], p: Number(e[1]) } : null;
}

/** Plain-language location from a box, radiology convention (patient right = viewer left). */
export function describeBox(box) {
  if (!Array.isArray(box) || box.length !== 4) return '';
  const cx = box[0] + box[2] / 2;
  const cy = box[1] + box[3] / 2;
  const side = cx < 0.45 ? 'right' : cx > 0.55 ? 'left' : 'central';
  const zone = cy < 0.4 ? 'upper' : cy < 0.65 ? 'mid' : 'lower';
  return side === 'central' ? `${zone} zone, central` : `${side} ${zone} zone`;
}

function groupNote(probs, names, threshold, normalText) {
  const pos = names.filter((n) => Number(probs[n]) >= threshold).sort((a, b) => probs[b] - probs[a]);
  if (pos.length) return pos.map((n) => `${label(n)} ${pct(probs[n])}`).join(', ') + ' flagged by the classifier.';
  const hi = names.filter((n) => probs[n] != null).sort((a, b) => probs[b] - probs[a])[0];
  return `${normalText}${hi ? ` Highest here: ${label(hi)} ${pct(probs[hi])}.` : ''}`;
}

/**
 * Build the reading path. Fixed anatomic order:
 *   1. both lungs overview
 *   2. each positive finding at or above threshold, with its CAM box, zoom: true
 *   3. heart and mediastinum
 *   4. pleura and costophrenic angles
 *   5. bones and soft tissues
 */
export function buildReadingPath({ probs, cams }, { threshold = 0.5, maxFindings = 4 } = {}) {
  const pos = positives(probs, cams, threshold, maxFindings);
  const items = [];
  const top = topFinding(probs);
  items.push({
    region: 'Both lungs, overview',
    box: BOX.lungs,
    note: pos.length
      ? `Classifier pass over the whole film. Strongest activations: ${pos.map((f) => `${label(f.name)} ${pct(f.p)}`).join(', ')}.`
      : `Classifier pass over the whole film. Nothing reaches the ${pct(threshold)} threshold${top ? `; highest is ${label(top.name)} at ${pct(top.p)}` : ''}.`,
    zoom: false,
  });
  for (const f of pos) {
    items.push({
      region: `${label(f.name)}, ${describeBox(f.box)}`,
      box: f.box,
      note: `${label(f.name)}: ${pct(f.p)} probability. This box is where the classifier's activation for it was strongest.`,
      zoom: true,
    });
  }
  items.push({
    region: 'Heart and mediastinum',
    box: BOX.heart,
    note: groupNote(probs, GROUPS.heart, threshold, 'Heart size and mediastinal contours not flagged.'),
    zoom: false,
  });
  items.push({
    region: 'Pleura and costophrenic angles',
    box: BOX.pleura,
    note: groupNote(probs, GROUPS.pleura, threshold, 'No effusion or pneumothorax flagged.'),
    zoom: false,
  });
  items.push({
    region: 'Bones and soft tissues',
    box: BOX.bones,
    note: groupNote(probs, GROUPS.bones, threshold, 'No fracture flagged.'),
    zoom: false,
  });
  return items.map((it, index) => ({ index, ...it }));
}

/** Deterministic report straight from the probabilities (LOCAL_LLM=template). */
export function templateReport({ probs, cams }, { threshold = 0.5, history = '' } = {}) {
  const line = (names, normal) => {
    const pos = names.filter((n) => Number(probs[n]) >= threshold).sort((a, b) => probs[b] - probs[a]);
    if (!pos.length) return normal;
    return pos
      .map((n) => {
        const where = cams?.[n] ? ` (${describeBox(cams[n])})` : '';
        return `${label(n)}${where}, classifier probability ${pct(probs[n])}`;
      })
      .join('. ') + '.';
  };
  const findings = [
    `Lungs: ${line(GROUPS.lung, 'No consolidation, opacity, mass or nodule flagged by the classifier.')}`,
    `Pleura: ${line(GROUPS.pleura, 'No pleural effusion or pneumothorax flagged.')}`,
    `Heart and mediastinum: ${line(GROUPS.heart, 'Heart size and mediastinal contours not flagged as abnormal.')}`,
    `Bones and soft tissues: ${line(GROUPS.bones, 'No fracture flagged.')}`,
    'Lines and tubes: Not assessed by the classifier.',
  ].join('\n');

  const pos = positives(probs, cams, threshold, 6);
  const imp = [];
  if (pos.length) {
    pos.forEach((f, i) => imp.push(`${i + 1}. ${label(f.name)}${cams?.[f.name] ? `, ${describeBox(cams[f.name])}` : ''} (${pct(f.p)}).`));
  } else {
    const top = topFinding(probs);
    imp.push(`1. No finding reaches the classifier's ${pct(threshold)} threshold${top ? `; highest is ${label(top.name)} at ${pct(top.p)}` : ''}.`);
  }
  imp.push(`${imp.length + 1}. Automated draft from classifier probabilities${history ? ` (history: ${history.trim()})` : ''}. Radiologist review required.`);
  return { exam: 'Chest radiograph, one view.', findings, impression: imp.join('\n') };
}

// ----------------------------------------------------------------- DDX ------

const DDX_RULES = [
  {
    key: 'pneumothorax',
    match: /pneumothorax/i,
    item: {
      finding: 'Pneumothorax',
      question: 'Air in the pleural space. How big is it and is the patient stable?',
      differentials: [
        { dx: 'Spontaneous pneumothorax', why: 'tall thin patient or underlying bullous disease' },
        { dx: 'Traumatic or iatrogenic pneumothorax', why: 'recent line placement, biopsy or trauma' },
        { dx: 'Skin fold or bulla mimic', why: 'a line that does not follow the chest wall contour; confirm on expiratory or lateral view' },
      ],
      management: {
        next_step: 'Confirm with clinical assessment; small and stable may be observed, large or symptomatic needs aspiration or chest tube.',
        guideline: 'BTS Pleural Disease Guideline 2023',
      },
    },
  },
  {
    key: 'effusion',
    match: /effusion/i,
    item: {
      finding: 'Pleural effusion',
      question: 'Fluid blunting the costophrenic angle. Transudate or exudate?',
      differentials: [
        { dx: 'Heart failure', why: 'bilateral effusions with cardiomegaly and vascular congestion' },
        { dx: 'Parapneumonic effusion', why: 'adjacent consolidation, fever' },
        { dx: 'Malignant effusion', why: 'unilateral, large, known malignancy' },
      ],
      management: {
        next_step: 'Lateral decubitus view or ultrasound to confirm and size; thoracentesis if new and unexplained.',
        guideline: 'ACR Appropriateness Criteria, Pleural Effusion',
      },
    },
  },
  {
    key: 'consolidation',
    match: /consolidation|pneumonia|airspace|opacity|infiltrat/i,
    item: {
      finding: 'Airspace opacity / consolidation',
      question: 'Dense lung with air bronchograms. Infection or something else?',
      differentials: [
        { dx: 'Bacterial pneumonia', why: 'lobar consolidation with air bronchograms, cough and fever' },
        { dx: 'Aspiration pneumonitis', why: 'dependent lower lobe location, risk factors' },
        { dx: 'Atelectasis', why: 'volume loss with shift of fissures or mediastinum' },
        { dx: 'Post-obstructive pneumonia or mass', why: 'reason to confirm clearance on follow-up' },
      ],
      management: {
        next_step: 'Treat clinically; repeat chest radiograph in 6 to 8 weeks to confirm resolution.',
        guideline: 'ACR Appropriateness Criteria, Acute Respiratory Illness in Immunocompetent Patients',
      },
    },
  },
  {
    key: 'cardiomegaly',
    match: /cardiomegaly|enlarged cardiomediastinum|heart size.*(enlarged|increased)/i,
    item: {
      finding: 'Cardiomegaly',
      question: 'Cardiothoracic ratio above 0.5. Real, or AP projection?',
      differentials: [
        { dx: 'Cardiomyopathy or chamber enlargement', why: 'globular heart, pulmonary venous congestion' },
        { dx: 'Pericardial effusion', why: 'water-bottle silhouette with clear lungs' },
        { dx: 'AP or portable projection magnification', why: 'technique, not pathology' },
      ],
      management: {
        next_step: 'Correlate with projection and prior films; echocardiography if new or unexplained.',
        guideline: 'ACR Appropriateness Criteria, Dyspnea Suspected Cardiac Origin',
      },
    },
  },
  {
    key: 'nodule',
    match: /nodule|mass|lung lesion/i,
    item: {
      finding: 'Pulmonary nodule or mass',
      question: 'A focal rounded density. Benign or needs work-up?',
      differentials: [
        { dx: 'Granuloma', why: 'calcified, stable over years' },
        { dx: 'Primary lung carcinoma', why: 'spiculated, growing, smoker' },
        { dx: 'Metastasis', why: 'multiple, known primary' },
        { dx: 'Nipple shadow or overlapping bone', why: 'mimic; nipple markers or a repeat view resolve it' },
      ],
      management: {
        next_step: 'Compare with priors; non-contrast chest CT if new or not clearly benign.',
        guideline: 'Fleischner Society 2017',
      },
    },
  },
];

const DDX_NORMAL = {
  finding: 'No acute abnormality',
  question: 'The classifier flagged nothing above threshold. Anything to do?',
  differentials: [
    { dx: 'Normal chest radiograph', why: 'no finding reached the operating point' },
    { dx: 'Subtle finding below the classifier threshold', why: 'a research classifier misses small or atypical findings' },
  ],
  management: {
    next_step: 'Radiologist review of the image; correlate with symptoms and prior imaging.',
    guideline: 'ACR Appropriateness Criteria, Routine Chest Radiography',
  },
};

/** Rule-based differential keyed on findings mentioned in the report text (LOCAL_LLM=template). */
export function templateDdx(reportText) {
  const text = String(reportText || '');
  // Only look at lines that are not "No ... flagged" style negatives.
  const positiveText = text
    .split('\n')
    .filter((l) => !/^\s*(lungs|pleura|heart and mediastinum|bones and soft tissues|lines and tubes)\s*:\s*(no |not |heart size and)/i.test(l))
    .filter((l) => !/^\s*\d+\.\s*(no finding|automated draft)/i.test(l))
    .join('\n');
  const items = DDX_RULES.filter((r) => r.match.test(positiveText)).map((r) => r.item);
  return { items: items.length ? items : [DDX_NORMAL] };
}

// --------------------------------------------------------- LLM prompt -------

export const REPORT_WRITER_SYSTEM = `You are OpenRead's report writer, running fully on-device. A chest X-ray classifier has already looked at the image; you have NOT seen the image. You turn the classifier's findings into a draft chest radiograph report for a radiologist to review.
Answer with ONE JSON object and nothing else. No prose, no markdown fences:
{
  "report": {
    "exam": "Chest radiograph, one view.",
    "findings": "one line per system: Lungs, Pleura, Heart and mediastinum, Bones and soft tissues, Lines and tubes",
    "impression": "numbered, most important first"
  }
}
Rules:
- Report only the classifier's findings. Quote each positive finding's probability as a percentage, with its location.
- Negatives: say "not flagged by the classifier", never "normal". You cannot see the image.
- Lines and tubes: write "Not assessed."
- Use the clinical history only for context; do not invent symptoms, priors or measurements.
- Keep it short. No disclaimers inside the fields.`;

export function reportWriterUser({ probs, cams, history, threshold = 0.5 }) {
  const pos = positives(probs, cams, threshold, 8);
  const neg = Object.entries(probs || {})
    .filter(([, p]) => Number(p) < threshold)
    .sort((a, b) => b[1] - a[1]);
  const lines = [];
  lines.push(`Clinical history: ${(history || '').trim() || 'none provided'}`);
  lines.push('');
  lines.push(`Classifier: TorchXRayVision DenseNet-121 (densenet121-res224-all). Threshold ${pct(threshold)}.`);
  lines.push('');
  lines.push(pos.length ? 'Positive findings (probability, location of strongest activation):' : 'Positive findings: none at or above threshold.');
  for (const f of pos) lines.push(`- ${label(f.name)}: ${pct(f.p)}, ${describeBox(f.box)}`);
  lines.push('');
  lines.push('Below threshold:');
  lines.push(neg.map(([n, p]) => `${label(n)} ${pct(p)}`).join(', '));
  lines.push('');
  lines.push('Template:');
  lines.push('Lungs: ...\nPleura: ...\nHeart and mediastinum: ...\nBones and soft tissues: ...\nLines and tubes: Not assessed.');
  lines.push('');
  lines.push('Write the JSON object now.');
  return lines.join('\n');
}
