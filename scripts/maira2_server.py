#!/usr/bin/env python3
"""Serve microsoft/maira-2 for OpenRead.

Loads the model once, then answers POST /read with grounded findings.
Boxes are [x1, y1, x2, y2] in fractions of the original frontal image.

  python3 scripts/maira2_server.py
  GET  /health
  POST /read   { imageBase64, mime, history, grounding?, lateralBase64?, technique?, comparison?, priorReport?, priorBase64? }

Research weights only. Not for clinical use.

Environment:
  MAIRA2_MODEL            default microsoft/maira-2
  MAIRA2_HOST             default 127.0.0.1
  MAIRA2_PORT             default 8008
  MAIRA2_DEVICE           cuda | mps | cpu   (default: first one that exists)
  MAIRA2_DTYPE            bfloat16 | float16 | float32
  MAIRA2_MAX_NEW_TOKENS   default 450 when grounding, else 300
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def adjust_box_for_original_image_size(norm_box, width: int, height: int):
    """Map a box on the centre crop back onto the original image.

    MAIRA-2 centre-crops to the short side before it predicts boxes, and the
    processor reports those boxes in that cropped frame. This is the same
    adjustment the model card describes.
    """
    crop_width = crop_height = min(width, height)
    x_offset = (width - crop_width) // 2
    y_offset = (height - crop_height) // 2
    norm_x_min, norm_y_min, norm_x_max, norm_y_max = norm_box
    abs_x_min = int(norm_x_min * crop_width + x_offset)
    abs_x_max = int(norm_x_max * crop_width + x_offset)
    abs_y_min = int(norm_y_min * crop_height + y_offset)
    abs_y_max = int(norm_y_max * crop_height + y_offset)
    return (
        abs_x_min / width,
        abs_y_min / height,
        abs_x_max / width,
        abs_y_max / height,
    )


def _box_count(prediction) -> int:
    if not isinstance(prediction, list):
        return 0
    total = 0
    for item in prediction:
        if isinstance(item, (list, tuple)) and len(item) > 1 and item[1]:
            total += len(list(item[1]))
    return total


def findings_from_prediction(prediction, width: int, height: int, adjust=adjust_box_for_original_image_size):
    """Normalise the processor output to [{text, boxes}] with original-image boxes."""
    if isinstance(prediction, str):
        parts = [part.strip() for part in _split_sentences(prediction) if part.strip()]
        return [{"text": part, "boxes": []} for part in parts]
    if not isinstance(prediction, list):
        return []

    findings = []
    for item in prediction:
        if isinstance(item, str):
            text = _plain_findings(item)
            boxes = []
        elif isinstance(item, dict):
            text = _plain_findings(str(item.get("text") or item.get("finding") or ""))
            boxes = item.get("boxes") or []
        elif isinstance(item, (list, tuple)) and item:
            text = _plain_findings(str(item[0]))
            boxes = item[1] if len(item) > 1 and item[1] else []
        else:
            continue
        if not text:
            continue
        norm_boxes = []
        for box in boxes:
            if hasattr(box, "tolist"):
                box = box.tolist()
            coords = [float(v) for v in list(box)[:4]]
            if len(coords) < 4:
                continue
            adjusted = adjust(tuple(coords), width, height)
            norm_boxes.append([round(float(v), 4) for v in adjusted])
        findings.append({"text": text, "boxes": norm_boxes})
    return findings


def _split_sentences(text: str):
    import re

    return re.split(r"(?<=[.!?])\s+", text.strip())


def _self_check() -> None:
    same = adjust_box_for_original_image_size((0.1, 0.2, 0.4, 0.5), 1000, 1000)
    assert same == (0.1, 0.2, 0.4, 0.5), same
    wide = adjust_box_for_original_image_size((0.0, 0.0, 1.0, 1.0), 200, 100)
    assert wide == (0.25, 0.0, 0.75, 1.0), wide
    grounded = findings_from_prediction(
        [("Right base opacity.", [(0.0, 0.0, 1.0, 1.0)]), ("Heart size is normal.", None)],
        200,
        100,
    )
    assert grounded[0]["boxes"] == [[0.25, 0.0, 0.75, 1.0]], grounded
    assert grounded[1]["boxes"] == []
    plain = findings_from_prediction("The lungs are clear. The heart is normal.", 10, 10)
    assert [item["text"] for item in plain] == ["The lungs are clear.", "The heart is normal."]
    print("maira2 self-check ok")


class App:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.lock = threading.Lock()
        self.jobs = queue.Queue()

    def call_on_main(self, fn):
        """Run a model call on the thread that loaded the weights."""
        done = queue.Queue(maxsize=1)
        self.jobs.put((fn, done))
        kind, value = done.get()
        if kind == "err":
            raise value
        return value

    def serve_jobs(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            fn, done = job
            try:
                done.put(("ok", fn()))
            except Exception as exc:
                done.put(("err", exc))

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise SystemExit(
                f"Missing Python package ({exc}). Install with:\n"
                "  python3 -m venv .venv && source .venv/bin/activate\n"
                "  pip install -r requirements-maira2.txt"
            ) from exc

        model_id = os.environ.get("MAIRA2_MODEL", "microsoft/maira-2")
        self.device = pick_device(torch)
        dtype = pick_dtype(torch, self.device)
        print(f"loading {model_id} on {self.device} ({dtype})", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"ready  http://{os.environ.get('MAIRA2_HOST', '127.0.0.1')}:{os.environ.get('MAIRA2_PORT', '8008')}", flush=True)

    def read(self, body: dict) -> dict:
        image_b64 = body.get("imageBase64")
        if not image_b64 or not isinstance(image_b64, str):
            raise ValueError("imageBase64 is required")
        frontal = decode_image(image_b64)
        lateral = decode_image(body["lateralBase64"]) if body.get("lateralBase64") else None
        prior = decode_image(body["priorBase64"]) if body.get("priorBase64") else None
        grounding = body.get("grounding", True)
        if isinstance(grounding, str):
            grounding = grounding not in ("0", "false", "False")
        indication = (body.get("history") or body.get("indication") or "").strip() or "N/A"
        technique = (body.get("technique") or "").strip() or "N/A"
        comparison = (body.get("comparison") or "").strip() or "N/A"
        prior_report = (body.get("priorReport") or "").strip() or "N/A"
        default_tokens = 450 if grounding else 300
        max_new_tokens = int(body.get("maxNewTokens") or os.environ.get("MAIRA2_MAX_NEW_TOKENS", default_tokens))

        import torch
        from transformers import LogitsProcessorList

        processed = self.processor.format_and_preprocess_reporting_input(
            current_frontal=frontal,
            current_lateral=lateral,
            prior_frontal=prior,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            return_tensors="pt",
            get_grounding=bool(grounding),
        )
        processed = processed.to(self.device)
        if "pixel_values" in processed and hasattr(processed["pixel_values"], "to"):
            processed["pixel_values"] = processed["pixel_values"].to(dtype=self.model.dtype)

        prompt_length = processed["input_ids"].shape[-1]
        # Greedy decoding on this checkpoint often emits the end-of-text token
        # after the first grounded sentence. Hold that token off until several
        # findings have been closed, which is what a chest report looks like.
        end_finding_id = self.processor.tokenizer.convert_tokens_to_ids("</obj>")
        eos_id = self.processor.tokenizer.eos_token_id
        processors = LogitsProcessorList([
            HoldEosUntilFindings(prompt_length, eos_id, end_finding_id, min_findings=5),
        ]) if grounding and isinstance(end_finding_id, int) and end_finding_id >= 0 else None
        with self.lock:
            with torch.inference_mode():
                output = self.model.generate(
                    **processed,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    logits_processor=processors,
                )
        new_count = output.shape[-1] - prompt_length
        decoded, prediction = decode_and_parse(self.processor, output[0][prompt_length:])
        print(f"generated {new_count} tokens  raw={decoded[:240]!r}", flush=True)
        width, height = frontal.size
        findings = findings_from_prediction(prediction, width, height, adjust=self._adjust)
        print(
            f"read  {width}x{height}  findings={len(findings)}  "
            f"boxes={sum(len(item['boxes']) for item in findings)}",
            flush=True,
        )
        return {"findings": findings, "raw": decoded, "lateral": lateral is not None, "width": width, "height": height}

    def _adjust(self, box, width, height):
        fn = getattr(self.processor, "adjust_box_for_original_image_size", None)
        if fn is not None:
            try:
                return tuple(fn(box, width, height))
            except TypeError:
                pass
        return adjust_box_for_original_image_size(box, width, height)


class HoldEosUntilFindings:
    """Block the end-of-text token until enough findings have been closed."""

    def __init__(self, prompt_length: int, eos_token_id: int, end_finding_id: int, min_findings: int) -> None:
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id
        self.end_finding_id = end_finding_id
        self.min_findings = min_findings

    def __call__(self, input_ids, scores):
        closed = (input_ids[:, self.prompt_length :] == self.end_finding_id).sum(dim=-1)
        block = closed < self.min_findings
        if bool(block.any()):
            scores = scores.clone()
            scores[block, self.eos_token_id] = float("-inf")
        return scores


def _plain_findings(text: str) -> str:
    """Drop an unfinished <obj>/<box> span so a cut-off generation still has text."""
    text = re.sub(r"</?obj>|</?box>|<x\d+>|<y\d+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def decode_and_parse(processor, token_ids):
    """Prefer the decode that keeps the most boxes.

    The model card skips special tokens. Some processor versions need the box
    tokens left in the string for convert_output_to_plaintext_or_grounded_sequence.
    A generation that stops inside an <obj> span fails that parser; keep the
    readable text instead of failing the request.
    """
    attempts = []
    plain = ""
    for skip in (True, False):
        decoded = processor.decode(token_ids, skip_special_tokens=skip).lstrip()
        if skip:
            plain = _plain_findings(decoded)
        try:
            prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
        except Exception:
            continue
        attempts.append((decoded, prediction, _box_count(prediction), 1 if skip else 0))
    if not attempts:
        return plain, plain
    attempts.sort(key=lambda item: (item[2], item[3]), reverse=True)
    return attempts[0][0], attempts[0][1]


def decode_image(value: str):
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
    return image.convert("RGB")


def pick_device(torch) -> str:
    forced = os.environ.get("MAIRA2_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(torch, device: str):
    name = os.environ.get("MAIRA2_DTYPE", "").strip().lower()
    if not name:
        name = "bfloat16" if device == "cuda" else "float16" if device == "mps" else "float32"
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise SystemExit(f"Unknown MAIRA2_DTYPE {name!r}. Use bfloat16, float16, or float32.")
    return table[name]


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # A client that connects and never sends a request would block the
        # single request thread forever. Drop it and keep serving.
        timeout = 10

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
                payload = app.call_on_main(lambda: app.read(body))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:  # model and processor errors reach the UI
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
    if "--self-check" in sys.argv:
        _self_check()
        return
    host = os.environ.get("MAIRA2_HOST", "127.0.0.1")
    port = int(os.environ.get("MAIRA2_PORT", "8008"))
    app = App()
    app.load()
    # Accept connections on worker threads. generate() stays on the thread that
    # loaded the weights; Metal and CUDA dislike a request thread running it.
    server = ThreadingHTTPServer((host, port), make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        app.serve_jobs()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        app.jobs.put(None)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
