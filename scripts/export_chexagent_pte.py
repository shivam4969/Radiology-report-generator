#!/usr/bin/env python3
"""Export CheXagent as tensor-only graphs for ExecuTorch.

The stock forward opens an image path hidden in the token ids. That cannot be
recorded. This script moves the file open outside the graph:

  pixels = load_pixels(path)          # PIL, stays in the runner
  image_embeds = vision_encoder(pixels)
  logits = text_decoder(embeddings)   # embeddings already contain image_embeds

Run with the CheXagent environment (Transformers 4.40):

  .venv-chexagent/bin/python scripts/export_chexagent_pte.py
  .venv/bin/python scripts/lower_chexagent_pte.py

Linear weights are INT4, group size 32, packed two values per byte. The
activations, LayerNorm, and the Phi attention score stay in floating point.
Set CHEXAGENT_QUANT=fp16 to export float16 weights instead. The decoder graph
is one prefill of length CHEXAGENT_DECODER_LEN (default 8). A findings prompt
with one frontal film is about 1079 tokens, and generation still needs a KV
cache, which this prefill graph does not include.
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from torch import nn

MODEL_DIR = os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b")
OUT_DIR = os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch")
DECODER_LEN = int(os.environ.get("CHEXAGENT_DECODER_LEN", "8"))
ONLY = os.environ.get("CHEXAGENT_ONLY", "")
QUANT = os.environ.get("CHEXAGENT_QUANT", "int4")
GROUP_SIZE = int(os.environ.get("CHEXAGENT_GROUP_SIZE", "32"))


def load_pixels(visual, path: str) -> torch.Tensor:
    """Open the film here, before any exported function runs."""
    return visual.load_image(path, training=False).unsqueeze(0)


class Float32LayerNorm(nn.Module):
    """LayerNorm in float32. The portable FP16 kernel drifts badly."""

    def __init__(self, source: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = source.normalized_shape
        self.eps = source.eps
        self.weight = nn.Parameter(source.weight.detach().float())
        self.bias = nn.Parameter(source.bias.detach().float()) if source.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return y.to(dtype=x.dtype)


def promote_layer_norms(root: nn.Module) -> int:
    replaced = 0
    for path, child in list(root.named_modules()):
        if not isinstance(child, nn.LayerNorm) or not path:
            continue
        parent_path, _, attr = path.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        setattr(parent, attr, Float32LayerNorm(child))
        replaced += 1
    return replaced


def quantize_weight(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group INT4. Packed uint8 holds two values in each byte."""
    out, in_features = weight.shape
    grouped = weight.detach().float().reshape(out, in_features // group_size, group_size)
    scale = grouped.abs().amax(dim=-1).clamp(min=1e-8) / 7
    codes = (grouped / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32) + 8
    codes = codes.to(torch.uint8).reshape(out, in_features)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low | (high << 4)).contiguous()
    return packed, scale.half().contiguous()


def dequantize_weight(qdata: torch.Tensor, scale: torch.Tensor, group_size: int, dtype: torch.dtype) -> torch.Tensor:
    low = torch.bitwise_and(qdata, 0x0F).to(torch.int32)
    high = torch.bitwise_right_shift(qdata, 4).to(torch.int32)
    codes = torch.stack([low, high], dim=-1).reshape(qdata.shape[0], -1)
    expanded = scale.to(torch.float32).repeat_interleave(group_size, dim=-1)
    return ((codes.float() - 8) * expanded).to(dtype)


class Int4Linear(nn.Module):
    def __init__(self, source: nn.Linear, group_size: int):
        super().__init__()
        packed, scale = quantize_weight(source.weight, group_size)
        self.register_buffer("qdata", packed)
        self.register_buffer("scale", scale)
        self.group_size = group_size
        if source.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_weight(self.qdata, self.scale, self.group_size, x.dtype)
        return nn.functional.linear(x, weight, self.bias)


def quantize_linears(root: nn.Module, group_size: int) -> tuple[int, int]:
    replaced = 0
    skipped = 0
    for path, child in list(root.named_modules()):
        if not isinstance(child, nn.Linear) or not path:
            continue
        parent_path, _, attr = path.rpartition(".")
        parent = root.get_submodule(parent_path) if parent_path else root
        if child.in_features % group_size != 0 or isinstance(parent, nn.MultiheadAttention):
            skipped += 1
            print(f"skip {path} in_features={child.in_features}", flush=True)
            continue
        setattr(parent, attr, Int4Linear(child, group_size))
        replaced += 1
    return replaced, skipped


class VisionEncoder(nn.Module):
    """visual.forward: pixels in, 1024 embeddings out. No path, no PIL."""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.visual(pixels)


class TextDecoder(nn.Module):
    """Phi stack on embeddings that already include the image vectors.

    Cache and the image-path branch are absent, so the graph is one prefill.
    """

    def __init__(self, body, lm_head):
        super().__init__()
        self.layers = body.layers
        self.final_layernorm = body.final_layernorm
        self.lm_head = lm_head

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = inputs_embeds.shape
        position_ids = torch.arange(seq, device=inputs_embeds.device).unsqueeze(0)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=inputs_embeds.device).triu(1)
        neg = torch.finfo(inputs_embeds.dtype).min
        attention_mask = torch.zeros(
            batch, 1, seq, seq, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        attention_mask = attention_mask.masked_fill(causal, neg)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )[0]
        return self.lm_head(self.final_layernorm(hidden))


def export_one(name: str, module: nn.Module, example: tuple) -> str:
    print(f"export {name} {tuple(example[0].shape)}", flush=True)
    module.eval()
    try:
        program = torch.export.export(module, example)
    except Exception as exc:
        traceback.print_exc()
        print(f"{name} export failed: {type(exc).__name__}: {exc}", flush=True)
        return ""
    path = os.path.join(OUT_DIR, f"{name}.pt2")
    torch.export.save(program, path)
    print(f"wrote {path}", flush=True)
    return path


def main() -> None:
    from transformers import AutoModelForCausalLM

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading {MODEL_DIR}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.half().eval()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"weights {next(model.parameters()).dtype} {weight_bytes} bytes", flush=True)
    visual = model.model.visual

    sample = "samples/cxr-pneumonia-cc0.jpg"
    if os.path.exists(sample):
        pixels = load_pixels(visual, sample).half()
        print(f"pixels outside forward {tuple(pixels.shape)} {pixels.dtype}", flush=True)
    else:
        pixels = torch.zeros(1, 3, 512, 512, dtype=torch.float16)

    if ONLY in ("", "vision"):
        vision = VisionEncoder(visual).eval()
        print(f"vision layer norms in fp32: {promote_layer_norms(vision)}", flush=True)
        if QUANT == "int4":
            replaced, skipped = quantize_linears(vision, GROUP_SIZE)
            print(f"vision int4 linears {replaced} skipped {skipped}", flush=True)
        export_one("vision_encoder", vision, (pixels,))

    if ONLY not in ("", "text"):
        return

    hidden = model.config.hidden_size
    decoder = TextDecoder(model.model, model.lm_head).eval()
    print(f"decoder layer norms in fp32: {promote_layer_norms(decoder)}", flush=True)
    embeds = torch.zeros(1, DECODER_LEN, hidden, dtype=torch.float16)
    if QUANT == "int4":
        with torch.inference_mode():
            before = decoder(embeds)
        replaced, skipped = quantize_linears(decoder, GROUP_SIZE)
        with torch.inference_mode():
            after = decoder(embeds)
        same = torch.equal(before.argmax(-1), after.argmax(-1))
        drift = (before.float() - after.float()).abs()
        print(
            f"decoder int4 linears {replaced} skipped {skipped} "
            f"argmax_equal {bool(same)} mean_abs {float(drift.mean())} max_abs {float(drift.max())}",
            flush=True,
        )
    export_one("text_decoder", decoder, (embeds,))


if __name__ == "__main__":
    sys.exit(main())
