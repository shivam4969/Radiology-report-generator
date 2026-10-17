#!/usr/bin/env python3
"""Export the chest X-ray classifier to ExecuTorch.

  python scripts/export_cxr.py                      # XNNPACK (CPU), writes models/cxr-densenet121.pte
  python scripts/export_cxr.py --backend qnn        # Qualcomm HTP (NPU); needs QNN_SDK_ROOT
  python scripts/export_cxr.py --backend portable   # no partitioner, reference ops only (slow, always works)
  python scripts/export_cxr.py --no-verify          # skip the sample-image check

What it does:
  1. loads TorchXRayVision DenseNet-121 "densenet121-res224-all" (downloads ~30 MB of weights
     to ~/.torchxrayvision on first run)
  2. wraps it so the graph also emits the class activation maps (py/cxr_model.py)
  3. torch.export -> to_edge_transform_and_lower(partitioner) -> to_executorch -> .pte
  4. writes models/cxr-densenet121.meta.json (pathology order, input size, backend)
  5. runs the .pte through the ExecuTorch Python runtime on samples/cxr-pneumonia-cc0.jpg and
     prints the top findings next to the eager PyTorch result, so you can see they agree

Backend notes:
  xnnpack   Works on every host that has an executorch wheel (Linux x86_64 / arm64, macOS arm64).
            On the Snapdragon laptop this is the CPU fallback.
  qnn       Requires the Qualcomm AI Engine Direct SDK (QNN) and an executorch build with the
            Qualcomm backend (see scripts/setup-snapdragon.md). The compile step runs on a Linux
            x86_64 host; the resulting .pte then runs on the device's HTP/NPU. Pick the SoC with
            --soc (default from QNN_SOC). Check the chipset names available in your executorch
            version: `python -c "from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset; print(list(QcomChipset))"`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from cxr_model import INPUT_SIZE, build_wrapper, preprocess  # noqa: E402


def export(backend: str, out: Path, soc: str | None, fp16: bool) -> list[str]:
    import torch
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    model, pathologies = build_wrapper()
    example = (torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32),)

    print(f"[export] tracing with torch.export ... (torch {torch.__version__})")
    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(model, example)
    print(f"[export] traced in {time.time() - t0:.1f}s, {len(list(ep.graph.nodes))} graph nodes")

    partitioners = []
    if backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        partitioners = [XnnpackPartitioner()]
    elif backend == "qnn":
        partitioners = [qnn_partitioner(soc, fp16)]
    elif backend == "portable":
        partitioners = []
    else:
        raise SystemExit(f"unknown backend {backend}")

    print(f"[export] lowering to edge dialect, backend={backend}")
    t0 = time.time()
    edge = to_edge_transform_and_lower(
        ep,
        partitioner=partitioners,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    et = edge.to_executorch()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(et.buffer)
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

    meta = {
        "model": "torchxrayvision densenet121-res224-all",
        "pathologies": pathologies,
        "input": [1, 1, INPUT_SIZE, INPUT_SIZE],
        "outputs": ["probs[1,18]", "cams[1,18,7,7]"],
        "backend": backend,
        "soc": soc if backend == "qnn" else None,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
    }
    try:
        import executorch

        meta["executorch"] = getattr(executorch, "__version__", "unknown")
    except Exception:
        pass
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] wrote {out.with_suffix('.meta.json')}")
    return pathologies


def qnn_partitioner(soc: str | None, fp16: bool):
    """Qualcomm HTP partitioner. Module paths are those of executorch 0.6 to 1.0;
    double-check against executorch/backends/qualcomm/ in your checkout."""
    if not os.environ.get("QNN_SDK_ROOT"):
        raise SystemExit(
            "QNN_SDK_ROOT is not set. Install the Qualcomm AI Engine Direct SDK and export "
            "QNN_SDK_ROOT=/path/to/qairt/<version> (see scripts/setup-snapdragon.md)."
        )
    try:
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
        )
    except ImportError as e:
        raise SystemExit(
            f"executorch was built without the Qualcomm backend ({e}). Build it with "
            "EXECUTORCH_BUILD_QNN=ON (backends/qualcomm/scripts/build.sh) before exporting for qnn."
        )
    soc = soc or os.environ.get("QNN_SOC") or "SM8650"
    chipset = getattr(QcomChipset, soc, None)
    if chipset is None:
        raise SystemExit(f"unknown --soc {soc}. Available: {[c.name for c in QcomChipset]}")
    backend_options = generate_htp_compiler_spec(use_fp16=fp16)
    compiler_specs = generate_qnn_executorch_compiler_spec(soc_model=chipset, backend_options=backend_options)
    return QnnPartitioner(compiler_specs)


def verify(pte: Path, image: Path, pathologies: list[str]) -> None:
    import torch
    from PIL import Image

    x, _crop = preprocess(Image.open(image), INPUT_SIZE)
    xt = torch.from_numpy(x)

    model, _ = build_wrapper()
    with torch.no_grad():
        eager_probs, _ = model(xt)
    eager = eager_probs[0].numpy()

    try:
        from executorch.runtime import Runtime, Verification

        program = Runtime.get().load_program(pte, verification=Verification.Minimal)
        method = program.load_method("forward")
        outs = method.execute([xt])
    except ImportError:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch

        outs = _load_for_executorch(str(pte)).forward([xt])
    et_probs = outs[0].numpy()[0]
    cams = outs[1].numpy()[0]

    print(f"\n[verify] {image.name}  (pathology: eager PyTorch  vs  ExecuTorch .pte)")
    order = sorted(range(len(pathologies)), key=lambda i: -et_probs[i])
    for i in order[:6]:
        print(f"   {pathologies[i]:<28} {eager[i]:.3f}   {et_probs[i]:.3f}")
    diff = float(abs(eager - et_probs).max())
    print(f"[verify] max |diff| over 18 outputs = {diff:.4f}   cams shape {tuple(cams.shape)}")
    if diff > 0.05:
        print("[verify] WARNING: eager and exported outputs differ by more than 0.05; check the backend/quantisation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["xnnpack", "qnn", "portable"], default="xnnpack")
    ap.add_argument("--out", default=str(ROOT / "models" / "cxr-densenet121.pte"))
    ap.add_argument("--soc", default=None, help="Qualcomm chipset name for --backend qnn (QcomChipset member)")
    ap.add_argument("--no-fp16", action="store_true", help="qnn: keep fp32 on HTP instead of fp16")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--image", default=str(ROOT / "samples" / "cxr-pneumonia-cc0.jpg"))
    args = ap.parse_args()

    out = Path(args.out)
    pathologies = export(args.backend, out, args.soc, not args.no_fp16)
    if args.no_verify:
        return
    if args.backend == "qnn":
        print("[verify] skipped: a QNN .pte only runs on the Qualcomm device, not on the export host.")
        return
    verify(out, Path(args.image), pathologies)


if __name__ == "__main__":
    main()
