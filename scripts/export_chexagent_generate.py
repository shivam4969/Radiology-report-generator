#!/usr/bin/env python3
"""Export a CheXagent text graph that can write a findings section.

The checked-in text_decoder.pte is one forward of 8 tokens. A real prompt,
with the film spliced in, is about 1085 tokens, and the report is generated
one token at a time. Linear weights are INT4, group size 32. This script exports:

  text_prefill.pt2   embeddings [1, 1536, 2560] -> logits and a KV cache
  text_decode.pt2    one new embedding plus that cache -> the next logits

The vision graph is unchanged. Image pixels stay outside the graph, and the
runner splices the 1024 image embeddings into the prompt before prefill.

Run with the CheXagent environment (Transformers 4.40):

  .venv-chexagent/bin/python scripts/export_chexagent_generate.py

Research weights only. Not for clinical use.
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from torch import nn

MODEL_DIR = os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b")
OUT_DIR = os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch")
PREFILL = int(os.environ.get("CHEXAGENT_PREFILL", "1536"))
MAX_SEQ = int(os.environ.get("CHEXAGENT_MAX_SEQ", "2048"))
HEADS = 32
HEAD_DIM = 80
HIDDEN = 2560


class Float32LayerNorm(nn.Module):
    """LayerNorm in float32. The portable FP16 kernel drifts."""

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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class DecodeLayer(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        attn = layer.self_attn
        self.input_layernorm = Float32LayerNorm(layer.input_layernorm)
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.dense = attn.dense
        self.mlp = layer.mlp
        rotary = attn.rotary_emb.dim
        self.rotary_dim = rotary
        self.register_buffer("cos", attn.rotary_emb.cos_cached.detach().float())
        self.register_buffer("sin", attn.rotary_emb.sin_cached.detach().float())

    def forward(self, hidden, position_ids, cache_k, cache_v, attn_mask):
        residual = hidden
        normed = self.input_layernorm(hidden)
        query = self.q_proj(normed).view(1, -1, HEADS, HEAD_DIM).transpose(1, 2)
        key = self.k_proj(normed).view(1, -1, HEADS, HEAD_DIM).transpose(1, 2)
        value = self.v_proj(normed).view(1, -1, HEADS, HEAD_DIM).transpose(1, 2)

        rot = self.rotary_dim
        cos = self.cos[position_ids].unsqueeze(1).to(dtype=query.dtype)
        sin = self.sin[position_ids].unsqueeze(1).to(dtype=query.dtype)
        query_rot, query_pass = query[..., :rot], query[..., rot:]
        key_rot, key_pass = key[..., :rot], key[..., rot:]
        query_rot = query_rot * cos + rotate_half(query_rot) * sin
        key_rot = key_rot * cos + rotate_half(key_rot) * sin
        query = torch.cat((query_rot, query_pass), dim=-1)
        key = torch.cat((key_rot, key_pass), dim=-1)

        index = position_ids.reshape(-1)
        cache_k = torch.index_copy(cache_k, 2, index, key)
        cache_v = torch.index_copy(cache_v, 2, index, value)
        scores = torch.matmul(query.float(), cache_k.float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        scores = scores + attn_mask.float()
        probs = torch.softmax(scores, dim=-1).to(dtype=cache_v.dtype)
        mixed = torch.matmul(probs, cache_v).transpose(1, 2).reshape(1, -1, HIDDEN)
        hidden = self.dense(mixed) + self.mlp(normed) + residual
        return hidden, cache_k, cache_v


class GenerateDecoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        body = model.model
        self.layers = nn.ModuleList(DecodeLayer(layer) for layer in body.layers)
        self.final_layernorm = Float32LayerNorm(body.final_layernorm)
        self.lm_head = model.lm_head

    def forward(self, embeds, position_ids, cache_k, cache_v, attn_mask):
        hidden = embeds
        keys = []
        values = []
        for index, layer in enumerate(self.layers):
            hidden, key, value = layer(hidden, position_ids, cache_k[index], cache_v[index], attn_mask)
            keys.append(key)
            values.append(value)
        logits = self.lm_head(self.final_layernorm(hidden))
        return logits, torch.stack(keys), torch.stack(values)


def prompt_ids(tokenizer, image_path: str, indication: str) -> torch.Tensor:
    prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
    query = tokenizer.from_list_format([{"image": image_path}, {"text": prompt}])
    conversation = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    return tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")


def splice_image(embeds: torch.Tensor, input_ids: torch.Tensor, image: torch.Tensor, tokenizer) -> torch.Tensor:
    ids = input_ids[0].tolist()
    start = ids.index(tokenizer.img_start_id)
    end = ids.index(tokenizer.img_end_id)
    spliced = embeds.clone()
    spliced[0, start + 1 : end] = image[0].to(dtype=spliced.dtype)
    return spliced


def prefill_mask(real_len: int) -> torch.Tensor:
    query = torch.arange(PREFILL).view(PREFILL, 1)
    key = torch.arange(MAX_SEQ).view(1, MAX_SEQ)
    allowed = (key <= query) & (key < real_len)
    mask = torch.zeros(1, 1, PREFILL, MAX_SEQ, dtype=torch.float32)
    return mask.masked_fill(~allowed, torch.finfo(torch.float32).min)


def decode_mask(position: int) -> torch.Tensor:
    allowed = torch.arange(MAX_SEQ) <= position
    mask = torch.zeros(1, 1, 1, MAX_SEQ, dtype=torch.float32)
    return mask.masked_fill(~allowed, torch.finfo(torch.float32).min)


def empty_cache(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.zeros(32, 1, HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16, device=device)


def pad_embeds(embeds: torch.Tensor) -> torch.Tensor:
    padded = torch.zeros(1, PREFILL, HIDDEN, dtype=embeds.dtype, device=embeds.device)
    padded[:, : embeds.shape[1]] = embeds
    return padded


def check(model, decoder: GenerateDecoder, tokenizer, image_path: str) -> int:
    indication = "54F, cough and fever for 3 days"
    input_ids = prompt_ids(tokenizer, image_path, indication)
    real_len = int(input_ids.shape[1])
    print(f"prompt tokens {real_len}", flush=True)
    if real_len > PREFILL:
        raise SystemExit(f"prompt is {real_len} tokens, prefill is {PREFILL}")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    visual = model.model.visual
    pixels = visual.load_image(image_path, training=False).unsqueeze(0).to(device=device, dtype=torch.float16)
    with torch.inference_mode():
        reference = model(input_ids)
        image = visual(pixels)
        embeds = splice_image(model.model.embed_tokens(input_ids), input_ids, image, tokenizer)
        logits, cache_k, cache_v = decoder(
            pad_embeds(embeds),
            torch.arange(PREFILL, device=device).view(1, -1),
            empty_cache(device),
            empty_cache(device),
            prefill_mask(real_len).to(device),
        )
    ref_id = int(reference.logits[0, -1].argmax())
    got_id = int(logits[0, real_len - 1].argmax())
    print(f"prefill next token  reference={ref_id}  graph={got_id}", flush=True)

    position = real_len
    with torch.inference_mode():
        cached = model(input_ids, use_cache=True)
        nxt = torch.tensor([[ref_id]], dtype=torch.long, device=device)
        follow = model(nxt, past_key_values=cached.past_key_values, use_cache=True)
        step_logits, _, _ = decoder(
            model.model.embed_tokens(nxt),
            torch.tensor([[position]], dtype=torch.long, device=device),
            cache_k,
            cache_v,
            decode_mask(position).to(device),
        )
    ref_second = int(follow.logits[0, -1].argmax())
    got_second = int(step_logits[0, -1].argmax())
    print(f"decode next token   reference={ref_second}  graph={got_second}", flush=True)
    if got_id != ref_id or got_second != ref_second:
        return 1
    return 0


def export_one(name: str, module: nn.Module, example: tuple) -> None:
    print(f"export {name}", flush=True)
    program = torch.export.export(module, example)
    path = os.path.join(OUT_DIR, f"{name}.pt2")
    torch.export.save(program, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)", flush=True)


def main() -> int:
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if PREFILL > MAX_SEQ or MAX_SEQ > 2048:
        raise SystemExit(f"need prefill <= max seq <= 2048, got {PREFILL} and {MAX_SEQ}")
    os.makedirs(OUT_DIR, exist_ok=True)
    image_path = os.path.join(OUT_DIR, "blank.jpg")
    Image.new("RGB", (64, 64), (140, 140, 140)).save(image_path, format="JPEG")

    print(f"loading {MODEL_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device=device, dtype=torch.float16).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    decoder = GenerateDecoder(model).to(device).eval()
    print(f"checking the graph against the full model on {device}", flush=True)
    if check(model, decoder, tokenizer, image_path) != 0:
        print("graph does not match the full model; not exporting", flush=True)
        return 1

    sys.path.insert(0, os.path.dirname(__file__))
    from export_chexagent_pte import quantize_linears

    replaced, skipped = quantize_linears(decoder, 32)
    print(f"int4 linears {replaced} skipped {skipped}", flush=True)
    print("moving the graph to cpu for export", flush=True)
    decoder = decoder.to("cpu")
    embeds = torch.zeros(1, PREFILL, HIDDEN, dtype=torch.float16)
    position = torch.arange(PREFILL).view(1, -1)
    cache = empty_cache()
    export_one("text_prefill", decoder, (embeds, position, cache, cache.clone(), prefill_mask(PREFILL)))
    export_one(
        "text_decode",
        decoder,
        (
            torch.zeros(1, 1, HIDDEN, dtype=torch.float16),
            torch.zeros(1, 1, dtype=torch.long),
            cache,
            cache.clone(),
            decode_mask(0),
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
