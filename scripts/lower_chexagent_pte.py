#!/usr/bin/env python3
"""Lower exported CheXagent graphs to ExecuTorch .pte files.

Run with the environment that has ExecuTorch installed:

  .venv/bin/python scripts/lower_chexagent_pte.py
  CHEXAGENT_BACKEND=metal CHEXAGENT_PTE_NAMES=vision_encoder,text_prefill,text_decode \\
    .venv/bin/python scripts/lower_chexagent_pte.py

metal writes chexagent-executorch/metal/<name>.pte and does not replace the
CPU .pte files. The Metal backend is experimental.
"""

from __future__ import annotations

import os
import sys

import torch
from executorch.exir import to_edge
from executorch.exir.program._program import EdgeCompileConfig

# INT4 unpack uses bitwise ops that have portable kernels but sit outside Core ATen.
EDGE_CONFIG = EdgeCompileConfig(
    _core_aten_ops_exception_list=[
        torch.ops.aten.bitwise_and.Scalar,
        torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
    ]
)

IN_DIR = os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch")
NAMES = tuple(
    name.strip()
    for name in os.environ.get("CHEXAGENT_PTE_NAMES", "vision_encoder,text_decoder").split(",")
    if name.strip()
)


def lower_cpu(name: str, program) -> None:
    print(f"lowering {name} for cpu", flush=True)
    manager = to_edge(program, compile_config=EDGE_CONFIG)
    executorch_program = manager.to_executorch()
    dst = os.path.join(IN_DIR, f"{name}.pte")
    with open(dst, "wb") as handle:
        executorch_program.write_to_file(handle)
    print(f"wrote {dst} ({os.path.getsize(dst)} bytes)", flush=True)


def _metal_options_without_torchao(cls, compile_specs):
    """Metal's option builder imports a TorchAO MPS library this install lacks.

    CheXagent unpacks INT4 inside the graph, so that library is unused.
    """
    _ = compile_specs
    options = {
        "aot_inductor.link_libtorch": False,
        "aot_inductor.package": True,
        "aot_inductor.package_constants_in_so": False,
        "aot_inductor.package_constants_on_disk_format": "binary_blob",
        "max_autotune": True,
        "padding_stride_threshold": float("inf"),
        "aot_inductor.custom_ops_to_c_shims": {},
    }
    return options


def lower_metal(name: str, program) -> None:
    from executorch.backends.apple.metal.metal_backend import MetalBackend
    from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
    from executorch.exir import to_edge_transform_and_lower

    MetalBackend.get_aoti_compile_options = classmethod(_metal_options_without_torchao)
    print(f"lowering {name} for metal", flush=True)
    specs = [MetalBackend.generate_method_name_compile_spec("forward")]
    manager = to_edge_transform_and_lower(
        program,
        partitioner=[MetalPartitioner(specs)],
        compile_config=EDGE_CONFIG,
    )
    executorch_program = manager.to_executorch()
    out_dir = os.path.join(IN_DIR, "metal")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{name}.pte")
    with open(dst, "wb") as handle:
        executorch_program.write_to_file(handle)
    print(f"wrote {dst} ({os.path.getsize(dst)} bytes)", flush=True)


def lower(name: str) -> None:
    src = os.path.join(IN_DIR, f"{name}.pt2")
    print(f"loading {src}", flush=True)
    program = torch.export.load(src)
    backend = os.environ.get("CHEXAGENT_BACKEND", "cpu").strip().lower()
    if backend in {"metal", "mps"}:
        lower_metal(name, program)
        return
    lower_cpu(name, program)


def main() -> int:
    missing = [name for name in NAMES if not os.path.exists(os.path.join(IN_DIR, f"{name}.pt2"))]
    if missing:
        print(f"missing exports: {missing}. Run scripts/export_chexagent_pte.py first.", flush=True)
        return 1
    for name in NAMES:
        lower(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
