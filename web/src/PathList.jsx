/**
 * Regions in the order the model reported them.
 *   items     reading_path items so far
 *   current   index of the region the cloud is on
 *   onPick    click a region to move the cloud there
 */
export default function PathList({ items, current, live, onPick }) {
  return (
    <aside className="path" aria-live="polite">
      <h3>READING PATH</h3>
      <ol data-testid="path-list">
        {items.length === 0 && <li className="waiting">Waiting for the model to report its first region…</li>}
        {items.map((it, i) => {
          const state = i === current ? 'now' : i < current || !live ? 'done' : '';
          return (
            <li key={i} className={`step ${state}`} onClick={() => onPick?.(i)} data-testid="path-step" data-state={state}>
              <span className="mark">{state === 'now' ? <span className="dot" /> : state === 'done' ? '✓' : ''}</span>
              <span>
                {it.region}
                {it.zoom && <span className="zoom">zoom</span>}
              </span>
              {it.note && <small>{it.note}</small>}
            </li>
          );
        })}
      </ol>
      <p className="caption">Reading path: the regions the model reports, in its order. Not eye tracking.</p>
    </aside>
  );
}
