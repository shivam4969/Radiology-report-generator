#!/usr/bin/env python3
"""Serve StanfordAIMI/CheXagent-2-3b for OpenRead.

Loads the weights once, then answers POST /read with a findings section.
The image must be written to a file: the model reads a path out of its tokens.

  .venv-chexagent/bin/python scripts/chexagent_server.py
  GET  /health
  POST /read   { imageBase64, mime, history? }

Research weights only. Not for clinical use.

Environment:
  CHEXAGENT_MODEL   default models/CheXagent-2-3b
  CHEXAGENT_HOST    default 127.0.0.1
  CHEXAGENT_PORT    default 8009
  CHEXAGENT_DEVICE  default mps, else cpu
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer


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


class App:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def load(self) -> None:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                f"Missing Python package ({exc}). Install with:\n"
                "  python3.12 -m venv .venv-chexagent && source .venv-chexagent/bin/activate\n"
                "  pip install torch torchvision transformers==4.40.0 accelerate pillow einops "
                "albumentations opencv-python-headless pyarrow numpy matplotlib regex"
            ) from exc

        model_dir = os.environ.get("CHEXAGENT_MODEL", "models/CheXagent-2-3b")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        forced = os.environ.get("CHEXAGENT_DEVICE", "").strip().lower()
        if forced:
            self.device = forced
        dtype = torch.float16 if self.device == "mps" else torch.float32
        print(f"loading {model_dir} on {self.device} ({dtype})", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        host = os.environ.get("CHEXAGENT_HOST", "127.0.0.1")
        port = os.environ.get("CHEXAGENT_PORT", "8009")
        print(f"ready  http://{host}:{port}", flush=True)

    def read(self, body: dict) -> dict:
        image_b64 = body.get("imageBase64")
        if not image_b64 or not isinstance(image_b64, str):
            raise ValueError("imageBase64 is required")
        indication = (body.get("history") or body.get("indication") or "").strip() or "N/A"
        suffix = ".png" if "png" in str(body.get("mime") or "") else ".jpg"
        path = write_temp_image(image_b64, suffix)
        try:
            findings = self.generate(path, indication)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"read  indication={indication!r}  chars={len(findings)}", flush=True)
        return {"findings": findings, "indication": indication}

    def generate(self, image_path: str, indication: str) -> str:
        import torch

        prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
        query = self.tokenizer.from_list_format([{"image": image_path}, {"text": prompt}])
        conversation = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt"
        )
        with torch.inference_mode():
            output = self.model.generate(
                input_ids.to(self.device),
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=512,
            )[0]
        text = self.tokenizer.decode(output[input_ids.shape[1] : -1])
        return text.replace("**", "").strip()


def write_temp_image(value: str, suffix: str) -> str:
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
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        image.convert("RGB").save(handle, format="PNG" if suffix == ".png" else "JPEG")
    finally:
        handle.close()
    return handle.name


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"ok": app.model is not None, "device": app.device})

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
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def main() -> None:
    load_dotenv()
    host = os.environ.get("CHEXAGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("CHEXAGENT_PORT", "8009"))
    app = App()
    app.load()
    server = HTTPServer((host, port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
