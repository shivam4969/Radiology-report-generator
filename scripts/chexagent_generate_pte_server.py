#!/usr/bin/env python3
"""Serve the generation-capable CheXagent ExecuTorch graphs.

  .venv/bin/python scripts/chexagent_generate_pte_server.py
  GET  /health
  POST /read   { imageBase64, mime, history? }

Loads three programs from chexagent-executorch/:

  vision_encoder.pte   film -> 1024 image embeddings
  text_prefill.pte     prompt embeddings -> logits and a KV cache
  text_decode.pte      one new embedding plus that cache -> the next logits

The embedding table stays outside the graphs. These programs run on the
ExecuTorch CPU runtime. Research weights only. Not for clinical use.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import traceback
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

PREFILL = 1536
MAX_SEQ = 2048
HIDDEN = 2560
HEADS = 32
HEAD_DIM = 80
LAYERS = 32


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_tokenizer(model_dir: Path):
    """Load the CheXagent tokenizer without its drawing dependencies."""
    import importlib.util

    import transformers.utils as tf_utils

    if not hasattr(tf_utils, "is_tf_available"):
        tf_utils.is_tf_available = lambda: False
    for name in (
        "matplotlib",
        "matplotlib.colors",
        "matplotlib.figure",
        "matplotlib.backends",
        "matplotlib.backends.backend_agg",
        "tensorflow",
        "requests",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["matplotlib.backends.backend_agg"].FigureCanvasAgg = object
    spec = importlib.util.spec_from_file_location(
        "tokenization_chexagent", model_dir / "tokenization_chexagent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CheXagentTokenizer.from_pretrained(str(model_dir))


def load_embeddings(model_dir: Path) -> torch.Tensor:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = model_dir / index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("model.embed_tokens.weight")
    return weight.to(torch.float16).contiguous()


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


def empty_cache(device: str = "cpu") -> torch.Tensor:
    return torch.zeros(LAYERS, 1, HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16, device=device)


def prepare_pt2(graph_module, device: str) -> None:
    """Drop the CPU device checks baked in by torch.export and move the graph to MPS."""
    target = torch.device(device)
    for module in graph_module.modules():
        graph = getattr(module, "graph", None)
        if graph is None:
            continue
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target is torch.ops.aten._assert_tensor_metadata.default:
                graph.erase_node(node)
                continue
            kwargs = dict(node.kwargs)
            if getattr(kwargs.get("device"), "type", None) == "cpu":
                kwargs["device"] = target
                node.kwargs = kwargs
            if "wrap_with_autocast" in str(node.target) and node.args and node.args[0] == "cpu":
                node.args = (device,) + tuple(node.args[1:])
        graph.lint()
        module.recompile()
    graph_module.to(device)


def pixels_from_b64(value: str) -> torch.Tensor:
    from PIL import Image

    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(text, validate=False)))
        image.load()
    except Exception as exc:
        raise ValueError(f"could not read the image ({exc})") from exc
    image = image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    pixels = torch.tensor(list(image.get_flattened_data()), dtype=torch.float32)
    pixels = pixels.view(1, 512, 512, 3).permute(0, 3, 1, 2).contiguous()
    return ((pixels / 255.0 - 0.5) / 0.5).to(torch.float16)


class App:
    def __init__(self) -> None:
        self.vision = None
        self.prefill = None
        self.decode = None
        self.tokenizer = None
        self.embeddings = None
        self.eos_id = 50256
        self.max_new_tokens = 160
        self.device = "cpu"
        self.runtime_name = "executorch"
        self.infer_lock = threading.Lock()

    def tensor_device(self) -> str:
        # MetalBackend copies CPU inputs onto the GPU. Keeping them on MPS
        # makes the runtime reject the call.
        return "cpu" if self.runtime_name == "metal" else self.device

    def load(self) -> None:
        forced = os.environ.get("CHEXAGENT_RUNTIME", "").strip().lower()
        if forced == "metal":
            self.load_metal()
            return
        if forced in {"cpu", "pte", "executorch", "local"}:
            self.load_pte()
            return
        if forced in {"pt2", "mps", "gpu"} or (not forced and torch.backends.mps.is_available()):
            self.load_gpu()
            return
        self.load_pte()

    def load_gpu(self) -> None:
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available")
        pte_dir = Path(os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch"))
        model_dir = Path(os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b"))
        names = ("vision_encoder", "text_prefill", "text_decode")
        paths = {name: pte_dir / f"{name}.pt2" for name in names}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise SystemExit("missing " + ", ".join(missing))
        loaded = {}
        for name, path in paths.items():
            print(f"loading {path} on mps (int4)", flush=True)
            module = torch.export.load(path).module()
            prepare_pt2(module, "mps")
            loaded[name] = module
        self.vision = loaded["vision_encoder"]
        self.prefill = loaded["text_prefill"]
        self.decode = loaded["text_decode"]
        self.device = "mps"
        self.runtime_name = "pt2"
        self._finish_load(model_dir)

    def load_metal(self) -> None:
        metal_dir = os.environ.get("CHEXAGENT_METAL_DIR", "chexagent-executorch/metal")
        os.environ["CHEXAGENT_PTE_DIR"] = metal_dir
        inserted = os.environ.get("DYLD_INSERT_LIBRARIES", "")
        if "libmetal_interpose.dylib" not in inserted:
            raise SystemExit(
                "Metal .pte files need the linked MPS runtime. Start with "
                f"DYLD_INSERT_LIBRARIES={Path(metal_dir) / 'libmetal_interpose.dylib'}"
            )
        self.runtime_name = "metal"
        self.device = "mps"
        self.load_pte(keep_runtime=True)

    def load_pte(self, keep_runtime: bool = False) -> None:
        from executorch.runtime import Runtime

        pte_dir = Path(os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch"))
        model_dir = Path(os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b"))
        paths = {
            "vision_encoder": pte_dir / "vision_encoder.pte",
            "text_prefill": pte_dir / "text_prefill.pte",
            "text_decode": pte_dir / "text_decode.pte",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise SystemExit("missing " + ", ".join(missing))

        runtime = Runtime.get()
        loaded = {}
        for name, path in paths.items():
            print(f"loading {path}", flush=True)
            loaded[name] = runtime.load_program(path).load_method("forward")
        self.vision = loaded["vision_encoder"]
        self.prefill = loaded["text_prefill"]
        self.decode = loaded["text_decode"]
        if not keep_runtime:
            self.device = "cpu"
            self.runtime_name = "executorch"
        self._finish_load(model_dir)

    def _finish_load(self, model_dir: Path) -> None:
        print("loading tokenizer and embeddings", flush=True)
        self.tokenizer = load_tokenizer(model_dir)
        parent_decode = self.tokenizer.__class__.__mro__[1]._decode

        def decode_tokens(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
            return parent_decode(
                self.tokenizer,
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )

        self.tokenizer._decode = decode_tokens
        self.embeddings = load_embeddings(model_dir).to(self.tensor_device())
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.max_new_tokens = int(os.environ.get("CHEXAGENT_MAX_NEW_TOKENS", "160"))
        print(
            f"ready  runtime={self.runtime_name} device={self.device} "
            f"prefill={PREFILL} max_new={self.max_new_tokens}",
            flush=True,
        )

    def run(self, fn, args):
        place = self.tensor_device()
        moved = [item.to(place) if torch.is_tensor(item) else item for item in args]
        if self.runtime_name == "pt2":
            with torch.inference_mode():
                out = fn(*moved)
            return (out,) if isinstance(out, torch.Tensor) else tuple(out)
        return tuple(fn.execute(moved))

    def read(self, body: dict) -> dict:
        image_b64 = body.get("imageBase64")
        if not image_b64 or not isinstance(image_b64, str):
            raise ValueError("imageBase64 is required")
        indication = (body.get("history") or body.get("indication") or "").strip() or "N/A"
        pixels = pixels_from_b64(image_b64)
        with self.infer_lock:
            findings = self.generate(pixels, indication)
        print(f"read  indication={indication!r}  chars={len(findings)}", flush=True)
        return {
            "findings": findings,
            "exam": f"CheXagent INT4 weights, group 32, on {self.device}.",
        }

    def generate(self, pixels: torch.Tensor, indication: str) -> str:
        prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
        query = self.tokenizer.from_list_format([{"image": "cxr.jpg"}, {"text": prompt}])
        encoded = self.tokenizer.apply_chat_template(
            [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") and "input_ids" in encoded else encoded
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.view(1, -1)
        real_len = int(input_ids.shape[1])
        if real_len > PREFILL:
            raise ValueError(f"prompt is {real_len} tokens and the prefill graph holds {PREFILL}")

        started = time.perf_counter()
        print("vision", flush=True)
        place = self.tensor_device()
        input_ids = input_ids.to(place)
        image = self.run(self.vision, [pixels.to(place)])[0]
        ids = input_ids[0].tolist()
        start = ids.index(self.tokenizer.img_start_id)
        end = ids.index(self.tokenizer.img_end_id)
        embeds = self.embeddings[input_ids].clone()
        embeds[0, start + 1 : end] = image[0].to(device=embeds.device, dtype=embeds.dtype)
        padded = torch.zeros(1, PREFILL, HIDDEN, dtype=torch.float16, device=place)
        padded[:, :real_len] = embeds

        print(f"prefill tokens={real_len}", flush=True)
        logits, cache_k, cache_v = self.run(
            self.prefill,
            [
                padded,
                torch.arange(PREFILL, device=place).view(1, -1),
                empty_cache(place),
                empty_cache(place),
                prefill_mask(real_len),
            ],
        )
        cache_k = cache_k.contiguous()
        cache_v = cache_v.contiguous()
        next_id = int(logits[0, real_len - 1].argmax())
        print(f"prefill {time.perf_counter() - started:.1f}s first={next_id}", flush=True)

        produced = []
        position = real_len
        for _ in range(self.max_new_tokens):
            if next_id == self.eos_id:
                break
            produced.append(next_id)
            step = self.embeddings[next_id].view(1, 1, HIDDEN).contiguous()
            logits, cache_k, cache_v = self.run(
                self.decode,
                [
                    step,
                    torch.tensor([[position]], dtype=torch.long, device=place),
                    cache_k,
                    cache_v,
                    decode_mask(position),
                ],
            )
            cache_k = cache_k.contiguous()
            cache_v = cache_v.contiguous()
            next_id = int(logits[0, -1].argmax())
            position += 1
            if len(produced) % 16 == 0:
                print(f"decode tokens={len(produced)} {time.perf_counter() - started:.1f}s", flush=True)

        text = self.tokenizer.decode(produced).replace("**", "").strip()
        print(f"done tokens={len(produced)} {time.perf_counter() - started:.1f}s", flush=True)
        return text or "(decoder returned no text)"


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(
                200,
                {
                    "ok": self.server is not None and app.prefill is not None,
                    "runtime": app.runtime_name,
                    "device": app.device,
                    "prefill": PREFILL,
                    "max_new_tokens": app.max_new_tokens,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/read":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0 or length > 40 * 1024 * 1024:
                self._send(400, {"error": "JSON body is required and must be under 40MB"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                payload = app.read(body)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:
                traceback.print_exc()
                self._send(500, {"error": str(exc)})
                return
            self._send(200, payload)

        def _send(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.close_connection = True
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def main() -> None:
    load_dotenv()
    host = os.environ.get("CHEXAGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("CHEXAGENT_PORT", "8010"))
    app = App()
    app.load()
    print(f"ready  http://{host}:{port}", flush=True)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
