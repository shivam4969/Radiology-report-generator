import { useCallback, useEffect, useRef, useState } from 'react';
import Film from './Film.jsx';
import PathList from './PathList.jsx';
import ReportView from './ReportView.jsx';
import { analyze, getConfig, getSamples, readImage } from './api.js';

const REDUCED = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const REPLAY_STEP_MS = REDUCED ? 400 : 1400;

export default function App() {
  const [config, setConfig] = useState({ engine: '…', model: '' });
  const [image, setImage] = useState(null); // { dataUrl, base64, mime, name }
  const [history, setHistory] = useState('');
  const [phase, setPhase] = useState('intake'); // intake | reading | report
  const [path, setPath] = useState([]);
  const [current, setCurrent] = useState(-1);
  const [report, setReport] = useState(null);
  const [readMeta, setReadMeta] = useState({ readId: null, reportHash: null });
  const [error, setError] = useState('');
  const [replaying, setReplaying] = useState(false);
  const abortRef = useRef(null);
  const replayRef = useRef(null);

  useEffect(() => {
    getConfig().then(setConfig).catch(() => {});
  }, []);

  const loadBlob = useCallback(async (blob, name) => {
    if (!blob || !/^image\//.test(blob.type)) {
      setError('That is not an image. Paste or drop a JPEG or PNG.');
      return;
    }
    setError('');
    const img = await readImage(blob);
    setImage({ ...img, name: name || blob.name || 'pasted image' });
  }, []);

  // Ctrl+V anywhere on the page
  useEffect(() => {
    if (phase !== 'intake') return;
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
      if (item) {
        e.preventDefault();
        loadBlob(item.getAsFile(), 'pasted image');
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [phase, loadBlob]);

  async function loadSample() {
    try {
      const list = await getSamples();
      if (!list.length) return setError('No samples found in samples/.');
      const r = await fetch(list[0].url);
      await loadBlob(await r.blob(), list[0].name);
    } catch (e) {
      setError(e.message);
    }
  }

  async function run() {
    if (!image) return;
    setError('');
    setPath([]);
    setCurrent(-1);
    setReport(null);
    setReadMeta({ readId: null, reportHash: null });
    setPhase('reading');
    const ac = new AbortController();
    abortRef.current = ac;
    let items = [];
    let gotReport = null;
    try {
      await analyze(
        { imageBase64: image.base64, mime: image.mime, history, imageName: image.name },
        (ev) => {
          if (ev.type === 'path') {
            items = [...items, ev.item];
            setPath(items);
            setCurrent(items.length - 1);
          } else if (ev.type === 'report') {
            gotReport = ev.report;
            setReport(ev.report);
          } else if (ev.type === 'done') {
            setReadMeta({ readId: ev.readId ?? null, reportHash: ev.reportHash ?? null });
          } else if (ev.type === 'error') {
            setError(ev.message + (ev.raw ? `\n\nModel output:\n${ev.raw}` : ''));
          }
        },
        ac.signal,
      );
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message);
    }
    abortRef.current = null;
    if (gotReport) {
      // Let the last region land before switching to the report.
      await new Promise((r) => setTimeout(r, REDUCED ? 0 : 900));
      setPhase('report');
    } else {
      setPhase(items.length ? 'reading' : 'intake');
    }
  }

  function replay() {
    if (!path.length) return;
    clearInterval(replayRef.current);
    setReplaying(true);
    setPhase('reading');
    let i = 0;
    setCurrent(0);
    replayRef.current = setInterval(() => {
      i++;
      if (i >= path.length) {
        clearInterval(replayRef.current);
        setTimeout(() => {
          setReplaying(false);
          setPhase('report');
        }, REPLAY_STEP_MS);
        return;
      }
      setCurrent(i);
    }, REPLAY_STEP_MS);
  }

  function skipReplay() {
    clearInterval(replayRef.current);
    setReplaying(false);
    setPhase('report');
  }

  // Benchmark flow: load the oldest unrated read (from scripts/bench.mjs) for a verdict.
  async function loadUnrated() {
    setError('');
    try {
      const r = await fetch('/api/benchmark/next-unrated');
      const row = await r.json();
      if (!row) return setError('No unrated reads. Run `npm run bench` first.');
      let img = { dataUrl: '', base64: '', mime: 'image/jpeg', name: row.image_name || `read #${row.id}` };
      if (row.image_name) {
        const ir = await fetch(`/samples/${encodeURIComponent(row.image_name)}`);
        if (ir.ok) img = { ...(await readImage(await ir.blob())), name: row.image_name };
      }
      setImage(img);
      setHistory(row.history || '');
      setPath(row.path || []);
      setCurrent(-1);
      setReport(row.report);
      setReadMeta({ readId: row.id, reportHash: row.report_hash, bench: true }); // model hidden: rate blind
      setPhase('report');
    } catch (e) {
      setError(e.message);
    }
  }

  function reset() {
    abortRef.current?.abort();
    clearInterval(replayRef.current);
    setImage(null);
    setHistory('');
    setPath([]);
    setCurrent(-1);
    setReport(null);
    setReadMeta({ readId: null, reportHash: null });
    setError('');
    setReplaying(false);
    setPhase('intake');
  }

  return (
    <div className="app">
      <header className="bar">
        <span className="brand">
          OPEN<span className="o">READ</span>
        </span>
        <span className="engine" data-testid="engine">
          {config.engine}
          {config.model ? ` · ${config.model}` : ''}
        </span>
        <span className="sp" />
        {phase === 'reading' && !replaying && (
          <span>
            Analyzing · chest X-ray{history ? ` · ${history.slice(0, 40)}` : ''}
          </span>
        )}
        {phase === 'reading' && replaying && (
          <button className="btn small" onClick={skipReplay}>
            Skip replay
          </button>
        )}
        {phase !== 'intake' && (
          <button className="btn small" onClick={reset}>
            New
          </button>
        )}
      </header>

      <main>
        {phase === 'intake' && (
          <Intake image={image} history={history} setHistory={setHistory} onBlob={loadBlob} onSample={loadSample} onRun={run} error={error} />
        )}

        {phase === 'reading' && image && (
          <section className="reading">
            <Film src={image.dataUrl} item={current >= 0 ? path[current] : null} live={!replaying} tag={image.name} />
            <PathList items={path} current={current} live={!replaying} onPick={(i) => setCurrent(i)} />
            {error && <p className="error" style={{ gridColumn: '1 / -1' }}>{error}</p>}
            {error && !report && (
              <div style={{ gridColumn: '1 / -1' }}>
                <button className="btn" onClick={() => setPhase('intake')}>
                  Back
                </button>
              </div>
            )}
          </section>
        )}

        {phase === 'report' && report && (
          <ReportView
            image={image}
            path={path}
            report={report}
            setReport={setReport}
            readId={readMeta.readId}
            reportHash={readMeta.reportHash}
            model={readMeta.bench ? '' : config.model}
            onReplay={replay}
            onNew={reset}
          />
        )}
      </main>

      <footer>
        Research prototype. Not for clinical use. The radiologist decides.
        {phase === 'intake' && (
          <>
            {' · '}
            <a href="#" onClick={(e) => { e.preventDefault(); loadUnrated(); }} data-testid="rate-bench">
              Rate bench reads
            </a>
            {' · '}
            <a href="/api/benchmark/summary">Benchmark</a>
          </>
        )}
      </footer>
    </div>
  );
}

function Intake({ image, history, setHistory, onBlob, onSample, onRun, error }) {
  const [over, setOver] = useState(false);
  const fileRef = useRef(null);

  function onDrop(e) {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer?.files?.[0];
    if (f) onBlob(f, f.name);
  }

  return (
    <section className="intake">
      <div
        className={`drop${over ? ' over' : ''}`}
        tabIndex={0}
        role="button"
        aria-label="Paste, drop or choose a chest X-ray image"
        onClick={() => fileRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && fileRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        data-testid="drop"
      >
        {image ? (
          <>
            <img src={image.dataUrl} alt="Selected chest X-ray" />
            <div className="overlay">
              <span>{image.name}</span>
              <span>Click to replace</span>
            </div>
          </>
        ) : (
          <>
            <h2>Paste a chest X-ray</h2>
            <p>
              <span className="kbd">Ctrl</span> + <span className="kbd">V</span>, drop a file here, or{' '}
              <u>choose a file</u>
            </p>
            <p style={{ fontSize: 13 }}>JPEG or PNG. The image goes only to the model you configured.</p>
          </>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          hidden
          data-testid="file"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onBlob(f, f.name);
            e.target.value = '';
          }}
        />
      </div>

      <div className="field">
        <label htmlFor="history">HISTORY</label>
        <input id="history" value={history} onChange={(e) => setHistory(e.target.value)} placeholder="e.g. 54F, cough and fever for 3 days" data-testid="history" />
      </div>

      {error && <p className="error">{error}</p>}

      <div className="actions">
        <button className="btn orange" disabled={!image} onClick={onRun} data-testid="analyze">
          Analyze
        </button>
        <button className="btn" onClick={onSample} data-testid="sample">
          Try the sample image
        </button>
        <span className="sample-link">Sample: lobar pneumonia, CC0, Mikael Häggström.</span>
      </div>
    </section>
  );
}
