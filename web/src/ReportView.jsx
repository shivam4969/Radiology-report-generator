import { useState } from 'react';
import Film from './Film.jsx';
import { ddx as fetchDdx, rate as postRate } from './api.js';

const STUDY_TYPES = ['Chest 1 view', 'Chest 2 view', 'Portable'];
const MODALITIES = ['XR', 'CT', 'Other'];

export default function ReportView({ image, path, report, setReport, readId, reportHash, model, onReplay, onNew }) {
  const [copied, setCopied] = useState(false);
  const [ddx, setDdx] = useState(null);
  const [ddxBusy, setDdxBusy] = useState(false);
  const [ddxErr, setDdxErr] = useState('');
  const [hover, setHover] = useState(null);

  const text = `EXAM:\n${report.exam}\n\nFINDINGS:\n${report.findings}\n\nIMPRESSION:\n${report.impression}`;

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function showDdx() {
    setDdxBusy(true);
    setDdxErr('');
    try {
      const j = await fetchDdx(report);
      setDdx(j.items);
    } catch (e) {
      setDdxErr(e.message);
    } finally {
      setDdxBusy(false);
    }
  }

  const abnormal = path.find((p) => p.zoom) || null;

  return (
    <section className="report-view">
      <div className="toolbar">
          <button className="btn" onClick={copy} data-testid="copy">
            {copied ? 'Copied' : 'Copy report'}
          </button>
          <button className="btn" onClick={onReplay} data-testid="replay">
            Replay read
          </button>
          <button className="btn outline-orange" onClick={showDdx} disabled={ddxBusy || !!ddx} data-testid="ddx-btn">
            {ddxBusy ? 'Thinking…' : 'Show me differential diagnosis / management'}
          </button>
          <span className="sp" />
        <button className="btn orange" onClick={onNew} data-testid="new">
          New
        </button>
      </div>
      <div>
        <div className="rep" data-testid="report">
          <Section label="EXAM" value={report.exam} onChange={(v) => setReport({ ...report, exam: v })} rows={1} />
          <Section label="FINDINGS" value={report.findings} onChange={(v) => setReport({ ...report, findings: v })} rows={6} />
          <Section label="IMPRESSION" value={report.impression} onChange={(v) => setReport({ ...report, impression: v })} rows={3} />
          <p className="hint">Draft, editable. Click any field to change it. {model && <>Model: <code>{model}</code>.</>}</p>
        </div>
        {ddxErr && <p className="error">{ddxErr}</p>}
      </div>

      <div className="side">
        <Film src={image.dataUrl} item={hover != null ? path[hover] : abnormal} tag={image.name} />
        <div className="mini-path">
          {path.map((p, i) => (
            <span key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              {i + 1}. {p.region}
            </span>
          ))}
        </div>
      </div>

      {ddx && <DdxPanel items={ddx} />}

      <Rating readId={readId} reportHash={reportHash} report={report} model={model} />
    </section>
  );
}

function Section({ label, value, onChange, rows }) {
  return (
    <div className="sec">
      <div className="h">{label}</div>
      <textarea
        value={value}
        rows={Math.max(rows, value.split('\n').length)}
        onChange={(e) => onChange(e.target.value)}
        aria-label={label}
        spellCheck={false}
      />
    </div>
  );
}

function DdxPanel({ items }) {
  return (
    <div className="ddx" data-testid="ddx-panel">
      <h3>DIFFERENTIAL DIAGNOSIS AND MANAGEMENT</h3>
      {items.map((it, i) => (
        <div className="item" key={i}>
          <div className="q">
            <small>{it.finding}</small>
            {it.question}
          </div>
          <ul>
            {(it.differentials || []).map((d, j) => (
              <li key={j}>
                <b>{d.dx}</b>
                {d.why && <span>{d.why}</span>}
              </li>
            ))}
          </ul>
          {it.management && (
            <div className="m">
              <b>What to do next:</b> {it.management.next_step}
              {it.management.guideline && <small>Guideline: {it.management.guideline}</small>}
            </div>
          )}
        </div>
      ))}
      <p className="fine">Shown on request, from the report text only. Not part of the report. The radiologist decides.</p>
    </div>
  );
}

function Rating({ readId, reportHash, report, model }) {
  const [verdict, setVerdict] = useState(null); // true | false | null
  const [reason, setReason] = useState('');
  const [studyType, setStudyType] = useState('Chest 1 view');
  const [icu, setIcu] = useState(false);
  const [modality, setModality] = useState('XR');
  const [sent, setSent] = useState(false);
  const [err, setErr] = useState('');

  async function submit(acceptable) {
    setVerdict(acceptable);
    if (!acceptable && !reason.trim()) return; // wait for a reason
    await send(acceptable);
  }

  async function send(acceptable) {
    setErr('');
    try {
      await postRate({ readId, reportHash, report, model, acceptable, reason: acceptable ? null : reason, studyType, icu, modality });
      setSent(true);
    } catch (e) {
      setErr(e.message);
    }
  }

  return (
    <div className="rating" data-testid="rating">
      <div className="row">
        <span className="label">VERDICT</span>
        <button className={`btn orange${verdict === true ? ' on' : ''}`} onClick={() => submit(true)} disabled={sent} data-testid="rate-ok">
          Acceptable
        </button>
        <button className={`btn outline-red${verdict === false ? ' on' : ''}`} onClick={() => submit(false)} disabled={sent} data-testid="rate-bad">
          Not acceptable
        </button>
        {sent && <span className="thanks">Saved. Thank you.</span>}
        {err && <span className="error">{err}</span>}
      </div>
      {verdict === false && !sent && (
        <div className="row">
          <span className="label">REASON</span>
          <textarea value={reason} onChange={(e) => setReason(e.target.value)} placeholder="What is wrong with the draft?" data-testid="reason" />
          <button className="btn outline-red" disabled={!reason.trim()} onClick={() => send(false)}>
            Save
          </button>
        </div>
      )}
      <div className="row">
        <span className="label">STUDY</span>
        {STUDY_TYPES.map((s) => (
          <button key={s} className={`chip${studyType === s ? ' on' : ''}`} onClick={() => setStudyType(s)} disabled={sent}>
            {s}
          </button>
        ))}
        <button className={`chip${icu ? ' on' : ''}`} onClick={() => setIcu(!icu)} disabled={sent}>
          ICU
        </button>
        <span className="label" style={{ minWidth: 0, marginLeft: 8 }}>
          MODALITY
        </span>
        {MODALITIES.map((m) => (
          <button key={m} className={`chip${modality === m ? ' on' : ''}`} onClick={() => setModality(m)} disabled={sent}>
            {m}
          </button>
        ))}
      </div>
    </div>
  );
}
