import { useEffect, useRef, useState } from 'react';

/**
 * The X-ray with the reading cloud on top.
 *   src       image data URL
 *   item      current reading_path item { box:[x,y,w,h], zoom } or null
 *   live      show the blinking "Reading" badge
 *   tag       small caption bottom-left
 */
export default function Film({ src, item, live, tag, onClick }) {
  const ref = useRef(null);
  const [box, setBox] = useState(null); // rendered image rect inside the container
  const [nat, setNat] = useState(null); // natural size

  // Track container size and the natural image size so boxes in 0..1 map to
  // the drawn image (object-fit: contain), not the black letterbox.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    measure();
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nat]);

  function measure() {
    const el = ref.current;
    if (!el || !nat) return;
    const cw = el.clientWidth;
    const ch = el.clientHeight;
    const s = Math.min(cw / nat.w, ch / nat.h);
    const w = nat.w * s;
    const h = nat.h * s;
    setBox({ x: (cw - w) / 2, y: (ch - h) / 2, w, h });
  }

  let cloud = null;
  let ring = null;
  let inset = null;
  if (box && item) {
    const [bx, by, bw, bh] = item.box;
    const cx = box.x + (bx + bw / 2) * box.w;
    const cy = box.y + (by + bh / 2) * box.h;
    const span = Math.max(bw * box.w, bh * box.h);
    // Loose cloud for an overview, tightened when the model zooms in.
    const d = item.zoom ? Math.max(90, span * 1.15) : Math.max(140, span * 1.35);
    cloud = { left: cx, top: cy, width: d, height: d };
    if (item.zoom) {
      const r = Math.max(70, span * 0.8);
      ring = { left: cx, top: cy, width: r, height: r };
      // Magnified inset: scale the whole image so the box fills the inset.
      const S = Math.round(box.w * 0.38);
      const k = S / Math.max(bw * nat.w, bh * nat.h, 1);
      inset = {
        size: S,
        img: {
          width: nat.w * k,
          height: nat.h * k,
          left: -(bx * nat.w * k) + (S - bw * nat.w * k) / 2,
          top: -(by * nat.h * k) + (S - bh * nat.h * k) / 2,
        },
      };
    }
  }

  return (
    <div className="film" ref={ref} onClick={onClick} data-testid="film">
      <img
        className="base"
        src={src}
        alt="Chest radiograph being read"
        draggable={false}
        onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      <div className={`cloud${cloud ? ' on' : ''}`} style={cloud || undefined} data-testid="cloud" />
      <div className={`ring${ring ? ' on' : ''}`} style={ring || undefined} />
      <div className={`inset${inset ? ' on' : ''}`} style={inset ? { width: inset.size, height: inset.size } : undefined} data-testid="inset">
        {inset && <img src={src} alt="" draggable={false} style={{ position: 'absolute', ...inset.img, maxWidth: 'none' }} />}
        <span className="label">ZOOM</span>
      </div>
      {tag && <span className="tag">{tag}</span>}
      {live && <span className="rd">Reading</span>}
    </div>
  );
}
