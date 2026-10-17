#!/usr/bin/env python3
"""Long-running chest X-ray findings worker. JSON lines over stdin/stdout.

Spawned once by server/engines/local.js. Protocol:

  stdout, first line:   {"event": "ready", "backend": "executorch"|"torch", "pathologies": [...]}
  stdin, one per line:  {"id": 1, "image_b64": "<base64>", "mime": "image/jpeg"}
  stdout, one per line: {"id": 1, "probs": {"Pneumonia": 0.71, ...},
                                  "cams": {"Pneumonia": [x, y, w, h], ...},   (fractions 0..1 of the full image)
                                  "backend": "executorch", "ms": 48}
                 or     {"id": 1, "error": "..."}

Backends, in order of preference:
  1. ExecuTorch runtime running models/cxr-densenet121.pte  (--pte PATH or CXR_PTE)
  2. PyTorch + torchxrayvision directly, if the .pte is missing (logged clearly)
  3. Otherwise a helpful error naming scripts/export_cxr.py.

The box for each pathology is a Grad-CAM style saliency box: the class
activation map (classifier weights x final 7x7 feature map) is upsampled,
thresholded at 60% of its max, and the bounding box of the largest connected
region above that threshold is returned. It shows where the classifier's
activation was strongest, which is not the same thing as a lesion outline.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cxr_model import INPUT_SIZE, PATHOLOGIES, preprocess  # noqa: E402

CAM_UPSAMPLE = 56  # CAMs are 7x7; upsample to this before thresholding
CAM_THRESHOLD = 0.60  # fraction of the max activation


def log(msg: str) -> None:
    print(f"[cxr_worker] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- backends ---


class ExecuTorchBackend:
    name = "executorch"

    def __init__(self, pte: Path):
        import torch  # the ExecuTorch pybindings take/return torch tensors

        self.torch = torch
        self.method = None
        try:
            from executorch.runtime import Runtime, Verification

            runtime = Runtime.get()
            program = runtime.load_program(pte, verification=Verification.Minimal)
            self.method = program.load_method("forward")
            self._call = lambda x: self.method.execute([x])
        except ImportError:
            # Older wheels only ship the pybindings module.
            from executorch.extension.pybindings.portable_lib import _load_for_executorch

            module = _load_for_executorch(str(pte))
            self._call = lambda x: module.forward([x])
        meta_path = pte.with_suffix(".meta.json")
        self.pathologies = PATHOLOGIES
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.pathologies = meta.get("pathologies", PATHOLOGIES)

    def run(self, x: np.ndarray):
        out = self._call(self.torch.from_numpy(x))
        probs, cams = out[0], out[1]
        return probs.detach().numpy()[0], cams.detach().numpy()[0]


class TorchBackend:
    name = "torch"

    def __init__(self):
        from cxr_model import build_wrapper

        import torch

        self.torch = torch
        self.model, self.pathologies = build_wrapper()
        self.model.eval()

    def run(self, x: np.ndarray):
        with self.torch.no_grad():
            probs, cams = self.model(self.torch.from_numpy(x))
        return probs.numpy()[0], cams.numpy()[0]


def pick_backend(pte: Path | None):
    if pte and pte.exists():
        try:
            b = ExecuTorchBackend(pte)
            log(f"ExecuTorch runtime loaded {pte}")
            return b
        except Exception as e:  # pragma: no cover - depends on local install
            log(f"could not load {pte} with the ExecuTorch runtime ({e!r}); falling back to PyTorch")
    else:
        log(f"no .pte at {pte} ; falling back to PyTorch + torchxrayvision (slower, CPU only)")
    try:
        return TorchBackend()
    except ImportError as e:
        raise SystemExit(
            "Neither an ExecuTorch .pte nor the PyTorch fallback is available "
            f"({e}). Run `python scripts/export_cxr.py` to produce models/cxr-densenet121.pte, "
            "or `pip install -r py/requirements.txt` for the PyTorch fallback."
        )


# -------------------------------------------------------------------- CAM ---


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    cam = np.maximum(cam, 0.0).astype(np.float32)
    img = Image.fromarray(cam, mode="F").resize((size, size), getattr(Image, "Resampling", Image).BILINEAR)
    return np.asarray(img, dtype=np.float32)


def largest_region_bbox(mask: np.ndarray):
    """Bounding box (x0, y0, x1, y1) of the largest 4-connected True region."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = None
    best_n = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            n = 0
            x0 = x1 = sx
            y0 = y1 = sy
            while q:
                y, x = q.popleft()
                n += 1
                x0, x1 = min(x0, x), max(x1, x)
                y0, y1 = min(y0, y), max(y1, y)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if n > best_n:
                best_n = n
                best = (x0, y0, x1, y1)
    return best


def cam_box(cam: np.ndarray, crop, upsample_to: int = CAM_UPSAMPLE, thr: float = CAM_THRESHOLD):
    """7x7 CAM -> [x, y, w, h] as fractions of the full original image."""
    up = upsample(cam, upsample_to)
    mx = float(up.max())
    if mx <= 0:
        return [0.1, 0.1, 0.8, 0.8]
    bb = largest_region_bbox(up >= thr * mx)
    if bb is None:
        return [0.1, 0.1, 0.8, 0.8]
    x0, y0, x1, y1 = bb
    # cell -> fraction of the square crop (inclusive cells, so +1)
    fx0, fy0 = x0 / upsample_to, y0 / upsample_to
    fx1, fy1 = (x1 + 1) / upsample_to, (y1 + 1) / upsample_to
    cx0, cy0, side, W, H = crop
    # crop fraction -> full-image fraction
    X0 = (cx0 + fx0 * side) / W
    Y0 = (cy0 + fy0 * side) / H
    X1 = (cx0 + fx1 * side) / W
    Y1 = (cy0 + fy1 * side) / H
    clamp = lambda v: max(0.0, min(1.0, v))
    return [round(clamp(X0), 4), round(clamp(Y0), 4), round(clamp(X1 - X0), 4), round(clamp(Y1 - Y0), 4)]


# ------------------------------------------------------------------- main ---


def decode_image(b64: str):
    from PIL import Image

    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


def handle(backend, req: dict) -> dict:
    t0 = time.time()
    img = decode_image(req["image_b64"])
    x, crop = preprocess(img, INPUT_SIZE)
    probs, cams = backend.run(x)
    names = backend.pathologies
    out_probs = {names[i]: round(float(probs[i]), 4) for i in range(len(names))}
    out_cams = {names[i]: cam_box(cams[i], crop) for i in range(len(names))}
    return {
        "id": req.get("id"),
        "probs": out_probs,
        "cams": out_cams,
        "backend": backend.name,
        "ms": int((time.time() - t0) * 1000),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pte", default=os.environ.get("CXR_PTE", str(HERE.parent / "models" / "cxr-densenet121.pte")))
    ap.add_argument("--once", metavar="IMAGE", help="run one image file and print the result (debugging)")
    args = ap.parse_args()

    backend = pick_backend(Path(args.pte) if args.pte else None)

    if args.once:
        b64 = base64.b64encode(Path(args.once).read_bytes()).decode()
        print(json.dumps(handle(backend, {"id": 0, "image_b64": b64}), indent=2))
        return

    print(json.dumps({"event": "ready", "backend": backend.name, "pathologies": list(backend.pathologies)}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle(backend, req)
        except Exception as e:  # keep serving after a bad image
            rid = None
            try:
                rid = json.loads(line).get("id")
            except Exception:
                pass
            res = {"id": rid, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
