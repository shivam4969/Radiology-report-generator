#!/usr/bin/env python3
"""Serve the CheXagent graphs for OpenRead.

  .venv/bin/python scripts/chexagent_pte_server.py
  CHEXAGENT_RUNTIME=pt2 .venv/bin/python scripts/chexagent_pte_server.py
  GET  /health
  POST /read   { imageBase64, mime, history? }

CHEXAGENT_RUNTIME=executorch (default) loads vision_encoder.pte and
text_decoder.pte and runs them on the CPU.

CHEXAGENT_RUNTIME=pt2 loads the float16 vision_encoder.pt2 and
text_decoder.pt2 graphs and runs them on MPS. Those graphs were exported
with CPU device checks baked in, so the loader strips those checks and
retargets the factory ops to MPS.

Both text graphs are one forward of 8 embeddings. There is no KV cache,
so the server decodes the argmax of the first decoder-length image
embeddings.

Research weights only. Not for clinical use.

Environment:
  CHEXAGENT_PTE_DIR   default chexagent-executorch
  CHEXAGENT_MODEL     tokenizer files, default models/CheXagent-2-3b
  CHEXAGENT_RUNTIME   executorch | pt2
  CHEXAGENT_DEVICE    pt2 only, default mps when available
  CHEXAGENT_HOST      default 127.0.0.1
  CHEXAGENT_PORT      default 8010
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _placeholder_shape(graph_module):
    for node in graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if hasattr(val, "shape"):
            return tuple(int(dim) for dim in val.shape)
    return None


def _prepare_pt2(graph_module, device: str) -> None:
    """Drop CPU metadata asserts and retarget factory ops onto MPS.

    torch.export recorded device(type='cpu') into the graph. A .to('mps')
    leaves those checks and the arange/zeros/autocast wrappers on CPU, so
    the forward then mixes devices.
    """
    import torch

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


class App:
    def __init__(self) -> None:
        self.runtime_name = "executorch"
        self.device = "cpu"
        self.runtime = None
        self.vision = None
        self.decoder = None
        self.tokenizer = None
        self.vision_tokens = 0
        self.decoder_tokens = 0
        self.infer_lock = threading.Lock()

    def load(self) -> None:
        runtime = os.environ.get("CHEXAGENT_RUNTIME", "executorch").strip().lower()
        if runtime in {"pt2", "mps", "pytorch"}:
            self.load_pt2()
            return
        from executorch.runtime import Runtime

        pte_dir = Path(os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch"))
        model_dir = Path(os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b"))
        vision_path = pte_dir / "vision_encoder.pte"
        decoder_path = pte_dir / "text_decoder.pte"
        for path in (vision_path, decoder_path):
            if not path.is_file():
                raise SystemExit(f"missing {path}")

        print(f"loading {vision_path}", flush=True)
        self.runtime = Runtime.get()
        vision_program = self.runtime.load_program(vision_path)
        decoder_program = self.runtime.load_program(decoder_path)
        self.vision = vision_program.load_method("forward")
        self.decoder = decoder_program.load_method("forward")
        self.vision_tokens = int(self.vision.metadata.output_tensor_meta(0).sizes()[1])
        self.decoder_tokens = int(self.decoder.metadata.input_tensor_meta(0).sizes()[1])
        self.tokenizer = CheXagentDecoder(model_dir)
        print(
            f"ready  runtime={self.runtime_name} device={self.device} "
            f"vision_tokens={self.vision_tokens}  decoder_tokens={self.decoder_tokens}",
            flush=True,
        )

    def load_pt2(self) -> None:
        import torch

        pte_dir = Path(os.environ.get("CHEXAGENT_PTE_DIR", "chexagent-executorch"))
        model_dir = Path(os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b"))
        vision_path = pte_dir / "vision_encoder.pt2"
        decoder_path = pte_dir / "text_decoder.pt2"
        for path in (vision_path, decoder_path):
            if not path.is_file():
                raise SystemExit(f"missing {path}")

        forced = os.environ.get("CHEXAGENT_DEVICE", "").strip().lower()
        if forced:
            device = forced
        else:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS is not available")

        print(f"loading {vision_path} on {device} (float16)", flush=True)
        vision = torch.export.load(vision_path).module()
        print(f"loading {decoder_path} on {device} (float16)", flush=True)
        decoder = torch.export.load(decoder_path).module()
        if device == "mps":
            _prepare_pt2(vision, device)
            _prepare_pt2(decoder, device)
        vision_shape = _placeholder_shape(vision)
        decoder_shape = _placeholder_shape(decoder)
        self.runtime_name = "pt2"
        self.device = device
        self.vision = vision
        self.decoder = decoder
        self.vision_tokens = 1024
        self.decoder_tokens = int(decoder_shape[1]) if decoder_shape and len(decoder_shape) > 1 else 8
        self.tokenizer = CheXagentDecoder(model_dir)
        print(
            f"ready  runtime={self.runtime_name} device={self.device} "
            f"vision_in={vision_shape} decoder_in={decoder_shape}",
            flush=True,
        )

    def read(self, body: dict) -> dict:
        image_b64 = body.get("imageBase64")
        if not image_b64 or not isinstance(image_b64, str):
            raise ValueError("imageBase64 is required")
        pixels = pixels_from_b64(image_b64)
        findings, exam = self.generate(pixels)
        print(f"read  chars={len(findings)}", flush=True)
        return {"findings": findings, "exam": exam}

    def generate(self, pixels) -> tuple[str, str]:
        import torch

        with self.infer_lock:
            if self.runtime_name == "pt2":
                pixels = pixels.to(self.device)
                with torch.inference_mode():
                    image_embeds = self.vision(pixels)
                    width = min(self.decoder_tokens, int(image_embeds.shape[1]))
                    logits = self.decoder(image_embeds[:, :width].contiguous())
            else:
                image_embeds = self.vision.execute([pixels])[0]
                width = min(self.decoder_tokens, int(image_embeds.shape[1]))
                logits = self.decoder.execute([image_embeds[:, :width].contiguous()])[0]
        ids = logits[0, :width].argmax(dim=-1).tolist()
        text = self.tokenizer.decode(ids).strip()
        if self.runtime_name == "pt2":
            exam = (
                f"CheXagent float16 on {self.device}. Vision encoder {self.vision_tokens} tokens, "
                f"text decoder {self.decoder_tokens} tokens."
            )
        else:
            exam = (
                f"ExecuTorch CheXagent. Vision encoder {self.vision_tokens} tokens, "
                f"text decoder {self.decoder_tokens} tokens."
            )
        if not text:
            text = "(decoder returned no text)"
        return text, exam


class CheXagentDecoder:
    """Turn token ids back into text using the CheXagent vocab.

    The base ids are GPT-2 byte-level tokens. Added ids are literal strings.
    """

    def __init__(self, model_dir: Path) -> None:
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        added = json.loads((model_dir / "added_tokens.json").read_text(encoding="utf-8"))
        self.added = {int(index): token for token, index in added.items()}
        self.base = {int(index): token for token, index in vocab.items()}
        self.byte_decoder = {value: key for key, value in _bytes_to_unicode().items()}

    def decode(self, ids: list[int]) -> str:
        pieces = []
        byte_chars = []

        def flush() -> None:
            if not byte_chars:
                return
            raw = "".join(byte_chars)
            pieces.append(bytearray(self.byte_decoder[char] for char in raw).decode("utf-8", errors="replace"))
            byte_chars.clear()

        for index in ids:
            if index in self.added:
                flush()
                pieces.append(self.added[index])
                continue
            token = self.base.get(index)
            if token is None:
                flush()
                pieces.append(f"<{index}>")
                continue
            byte_chars.append(token)
        flush()
        return "".join(pieces)


def _bytes_to_unicode() -> dict[int, str]:
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(values, (chr(code) for code in codes)))


def pixels_from_b64(value: str):
    import torch
    from PIL import Image

    text = value.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    try:
        raw = base64.b64decode(text, validate=False)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError(f"could not read the image ({exc})") from exc
    image = image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    pixels = torch.tensor(list(image.get_flattened_data()), dtype=torch.float32)
    pixels = pixels.view(1, 512, 512, 3).permute(0, 3, 1, 2).contiguous()
    return ((pixels / 255.0 - 0.5) / 0.5).to(torch.float16)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            # A client that connects and never finishes its headers must not
            # sit on this socket. The server is threaded, so the timeout only
            # ends this connection.
            self.request.settimeout(30)

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(
                200,
                {
                    "ok": app.vision is not None and app.decoder is not None,
                    "runtime": app.runtime_name,
                    "device": app.device,
                    "vision_tokens": app.vision_tokens,
                    "decoder_tokens": app.decoder_tokens,
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
