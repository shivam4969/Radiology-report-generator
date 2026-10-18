#!/usr/bin/env python3
"""On-device vision-language reader. JSON lines over stdin/stdout, same shape as cxr_worker.py.

Spawned by server/engines/local.js when LOCAL_READER is internvl3 or smolvlm. The model LOOKS AT THE FILM and
writes what it sees in plain prose; nothing tells it where to look, in what order, or what to mention.

  stdout, first line:   {"event": "ready", "backend": "torch", "model": "OpenGVLab/InternVL3-1B", "device": "cpu"}
  stdin, one per line:  {"id": 1, "image_b64": "<base64>", "mime": "image/jpeg", "history": ""}
  stdout, one per line: {"id": 1, "text": "<the model's description>", "ms": 1834, "model": "..."}
                 or     {"id": 1, "error": "..."}

Models (both on the ExecuTorch Qualcomm backend's supported list, see scripts/export_vlm.md):
  internvl3   OpenGVLab/InternVL3-1B              MIT         the better reader
  smolvlm     HuggingFaceTB/SmolVLM-500M-Instruct  Apache 2.0  smaller and faster

Backends: this worker runs the PyTorch weights through transformers (CPU, CUDA or Apple MPS), which works on
any laptop today. The ExecuTorch .pte path for the Hexagon NPU is the stretch goal in scripts/export_vlm.md;
when VLM_RUNNER is set the server calls that runner instead of this worker.

Research prototype. Not for clinical use. The radiologist decides.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

MODELS = {
    "internvl3": "OpenGVLab/InternVL3-1B",
    "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
}
PROMPT = (
    "You are a radiologist. This is a frontal chest radiograph{history}. "
    "Report what you see: findings, then an impression, in plain prose."
)


def log(*a):
    print("[vlm]", *a, file=sys.stderr, flush=True)


def pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class SmolVLM:
    def __init__(self, repo, device, dtype):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForVision2Seq.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        self.device = device

    def read(self, image, prompt, max_new_tokens):
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        full = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return full.split("Assistant:")[-1].strip()


class InternVL3:
    """InternVL3 ships its own `chat()` in remote code; images go through its dynamic tiling preprocessing."""

    def __init__(self, repo, device, dtype):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(repo, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True).to(device).eval()
        self.device, self.dtype = device, dtype

    def _pixels(self, image, size=448, max_tiles=6):
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        tf = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # Dynamic tiling as in the InternVL model card: pick the grid closest to the image's aspect ratio,
        # tile, and append a thumbnail when there is more than one tile.
        w, h = image.size
        ratio = w / h
        best, best_d = (1, 1), 9e9
        for n in range(1, max_tiles + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i * j <= max_tiles and abs(i / j - ratio) < best_d:
                        best, best_d = (i, j), abs(i / j - ratio)
        cols, rows = best
        resized = image.resize((size * cols, size * rows))
        tiles = [resized.crop((c * size, r * size, (c + 1) * size, (r + 1) * size)) for r in range(rows) for c in range(cols)]
        if len(tiles) > 1:
            tiles.append(image.resize((size, size)))
        return torch.stack([tf(t) for t in tiles]).to(self.dtype).to(self.device)

    def read(self, image, prompt, max_new_tokens):
        import torch

        pixels = self._pixels(image)
        with torch.inference_mode():
            return self.model.chat(self.tokenizer, pixels, "<image>\n" + prompt, dict(max_new_tokens=max_new_tokens, do_sample=False)).strip()


def load(name):
    repo = MODELS.get(name, name)
    device, dtype = pick_device()
    log(f"loading {repo} on {device} ({dtype}) ...")
    t = time.time()
    m = InternVL3(repo, device, dtype) if "internvl" in repo.lower() else SmolVLM(repo, device, dtype)
    log(f"loaded in {time.time() - t:.1f} s")
    return m, repo, device


def main():
    ap = argparse.ArgumentParser(description="On-device VLM chest X-ray reader (JSON lines).")
    ap.add_argument("--model", default=os.environ.get("LOCAL_READER", "internvl3"), help="internvl3 | smolvlm | a Hugging Face repo id")
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VLM_MAX_NEW_TOKENS", "400")))
    ap.add_argument("--once", metavar="IMAGE", help="read one file and print the text (debugging)")
    ap.add_argument("--history", default="")
    a = ap.parse_args()

    from PIL import Image

    try:
        model, repo, device = load(a.model)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"event": "error", "error": f"{type(e).__name__}: {e}"}), flush=True)
        sys.exit(1)

    def run(image, history):
        prompt = PROMPT.format(history=f" with the history: {history.strip()}" if history and history.strip() else "")
        return model.read(image, prompt, a.max_new_tokens)

    if a.once:
        t = time.time()
        print(run(Image.open(a.once), a.history))
        log(f"{time.time() - t:.1f} s")
        return

    print(json.dumps({"event": "ready", "backend": "torch", "model": repo, "device": device}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            t = time.time()
            image = Image.open(io.BytesIO(base64.b64decode(req["image_b64"])))
            text = run(image, req.get("history", ""))
            print(json.dumps({"id": req.get("id"), "text": text, "ms": int((time.time() - t) * 1000), "model": repo}), flush=True)
        except Exception as e:  # noqa: BLE001
            rid = None
            try:
                rid = json.loads(line).get("id")
            except Exception:  # noqa: BLE001
                pass
            print(json.dumps({"id": rid, "error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
