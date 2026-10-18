# Radio Warriors (OpenRead)

> New: `LOCAL_READER=internvl3` or `smolvlm` puts a vision-language model on the film instead of the classifier (`scripts/export_vlm.md`).
>
> Looking for the minimal version? [cxr-read](https://github.com/dictora-ai/cxr-read): one film in, one report out, plus a scorer against NIH labels. Two files, no UI, no toggles.


**An open-source chest X-ray reader that shows its work.** A vision language model reads the film, reports the regions it looked at in order, and drafts a radiology report. The UI shows that reading path live as a red cloud moving over the image, then gives you an editable Exam / Findings / Impression, a one-click differential diagnosis and management panel, and a two-button rating that feeds a per-model benchmark.

Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, Ollama), the Anthropic API, or Amazon Bedrock. Ships with a mock engine so it runs with no API key, and a fully on-device `local` engine built on ExecuTorch (below).

![Reading](docs/screens/reading.png)

> **Research prototype. Not for clinical use. The radiologist decides.**
> OpenRead is for exploring how vision language models read radiographs. It is not a medical device, it has not been validated, and nothing it produces should be used to diagnose or treat a patient.

**Hackathon plan:** the five-task pipeline, the Snapdragon ladder, the Astra-reference benchmark and dataset notes are in [docs/PLAN.md](docs/PLAN.md).

## Run it fully on-device (ExecuTorch Hackathon)

`ENGINE=local` replaces the single vision language model with a pipeline of small specialist
models that run on the laptop with no network: a chest X-ray classifier exported to ExecuTorch,
a reading path built from its activations, and a small local LLM (or a template) that writes the
report. Built for the ExecuTorch Hackathon compute track (Dell Latitude 7455, Snapdragon X Elite,
Windows on ARM): QNN/HTP for the NPU, XNNPACK on the CPU as the fallback.

```
            image
              |
              v
   [1] findings model ......... py/cxr_worker.py  (Python, spawned once, JSON lines)
       TorchXRayVision DenseNet-121 "densenet121-res224-all" as an ExecuTorch .pte
       -> 18 pathology probabilities + a CAM box per pathology
       (no .pte? falls back to PyTorch on CPU and says so)
              |
              v
   [2] reading path ........... server/lib/findings.js  (JavaScript, deterministic)
       lungs overview -> each positive finding with its CAM box, zoom: true
       -> heart and mediastinum -> pleura and costophrenic angles -> bones and soft tissues
       streamed as `path` events with a small delay, so the cloud animates
              |
              v
   [3] report writer .......... LOCAL_LLM=
       executorch          ExecuTorch Llama runner (llama_main) with a 4-bit Llama 3.2 3B or Qwen3 4B
       openai-compatible   any local server (Ollama, LM Studio, vLLM) for testing on a Mac
       template            no LLM: a deterministic report straight from the probabilities
       -> `token` events while it streams, then `report`, then `done`
       The same step answers "Show me differential diagnosis" (rule-based with template).
```

Three model roles, three files you can swap independently:

| role | model | where | swap for |
|---|---|---|---|
| findings | TorchXRayVision DenseNet-121, 18 outputs, 224 px | `models/cxr-densenet121.pte` via `scripts/export_cxr.py` | any classifier that returns probs + feature maps (edit `py/cxr_model.py`) |
| reading path | none, rules over the findings | `server/lib/findings.js` | a segmentation model for true lung fields |
| report writer | Llama 3.2 3B Instruct or Qwen3 4B, 4-bit | `models/*.pte` via `scripts/export_llm.md` | any instruct model the runner loads |

### Quickstart on the Dell (details in `scripts/setup-snapdragon.md`)

```bash
# 0. the pipeline with no models at all (canned findings, templated report): proves the plumbing
ENGINE=local CXR_WORKER=py/cxr_worker_mock.py LOCAL_LLM=template npm run dev

# 1. real classifier on CPU through PyTorch (downloads 30 MB of weights once)
python -m venv .venv && . .venv/bin/activate         # .venv\Scripts\activate on Windows
pip install -r py/requirements.txt
ENGINE=local LOCAL_LLM=template npm run dev          # log says: falling back to PyTorch

# 2. export to ExecuTorch and run the .pte (XNNPACK; add --backend qnn for the NPU, see the docs)
python scripts/export_cxr.py                         # writes models/cxr-densenet121.pte + verification run
ENGINE=local LOCAL_LLM=template npm run dev          # log says: findings worker ready, backend=executorch

# 3. a local LLM writes the report
#    a) tonight on a Mac: ollama pull llama3.2:3b, then
ENGINE=local LOCAL_LLM=openai-compatible LOCAL_LLM_URL=http://localhost:11434/v1 LOCAL_LLM_MODEL=llama3.2:3b npm run dev
#    b) on the device with ExecuTorch: scripts/export_llm.md, then LOCAL_LLM=executorch LLM_RUNNER=... LLM_MODEL=... LLM_TOKENIZER=...

# 4. a labelled sample set and the agreement check
python scripts/fetch_samples.py --n 40               # NIH ChestX-ray14 sample via Kaggle (see samples/README.md)
ENGINE=local npm run bench                           # per-label agreement of the classifier's top finding
```

All variables are documented in `.env.example` under "local". `npm test` runs the local engine
through `py/cxr_worker_mock.py`, so the test suite needs Python 3 but no PyTorch.

### Honesty notes

- The classifier is a **research model** (TorchXRayVision, trained on public datasets with
  NLP-mined labels). Its **probabilities are not diagnoses**; they are the model's calibrated
  outputs on its own operating point, and they are quoted as percentages so nobody mistakes them
  for certainty.
- The reading path shows **where the classifier's activation was strongest** for each finding
  (class activation map, thresholded at 60% of its max, largest region). That is not a lesion
  outline and not eye tracking. The anatomic order around those boxes is fixed by us, not chosen
  by a model.
- The report writer has **not seen the image**. It rewrites the classifier's findings. Negatives
  are phrased "not flagged by the classifier", never "normal".
- Nothing here is a medical device. The radiologist decides.

## 5-minute quickstart

```bash
git clone <this repo> openread && cd openread
cp .env.example .env          # ENGINE=mock works with no key
npm install
npm run dev                   # API on :8787, web on :5173
```

Open http://localhost:5173, click **Try the sample image**, then **Analyze**. You will see the mock engine stream a six-region reading path and a report.

To use a real model, edit `.env`:

```ini
# OpenAI
ENGINE=openai-compatible
MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# Anthropic
ENGINE=anthropic
MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...

# Amazon Bedrock (standard AWS credential chain)
ENGINE=bedrock
MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0
AWS_REGION=us-east-1
```

Restart `npm run dev` after changing `.env`.

### Local models: vLLM or Ollama (MedGemma, Qwen3-VL)

Any server that speaks the OpenAI chat completions API with `image_url` content works.

```bash
# vLLM with MedGemma
vllm serve google/medgemma-4b-it --max-model-len 8192
# .env
ENGINE=openai-compatible
OPENAI_BASE_URL=http://localhost:8000/v1
MODEL=google/medgemma-4b-it

# Ollama with Qwen3-VL
ollama pull qwen3-vl
# .env
ENGINE=openai-compatible
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL=qwen3-vl
```

Leave `OPENAI_API_KEY` empty for local servers. If your server does not stream, set `OPENAI_NO_STREAM=1`; OpenRead will emit the whole reading path at the end instead of live. Small models sometimes wrap the JSON in code fences or add prose; the parser tolerates both. If a model keeps returning unparseable output, the UI shows the raw text so you can tune the prompt in `server/lib/prompt.js`.

### MAIRA-2

[MAIRA-2](https://huggingface.co/microsoft/maira-2) writes a chest X-ray Findings section and, in grounded mode, a box for each finding it can localise. OpenRead runs the weights in a Python sidecar and maps that output onto the usual reading path and report. It does not go through the OpenAI-compatible engine: the model has its own processor and its own box tokens.

Accept the model license on the Hugging Face page first. The weights are released for research.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-maira2.txt
npm run maira2          # loads microsoft/maira-2, listens on http://127.0.0.1:8008
```

In `.env`:

```ini
ENGINE=maira2
MODEL=microsoft/maira-2
MAIRA2_BASE_URL=http://127.0.0.1:8008
```

Restart `npm run dev`. The first `npm run maira2` downloads the checkpoint (on the order of 15 GB). The sidecar uses CUDA when it is present, otherwise Apple Metal, otherwise CPU. CPU will run and will be slow. Override with `MAIRA2_DEVICE` and `MAIRA2_DTYPE` if you need to.

What the sidecar returns becomes:

- one reading-path item per box, with `zoom` set, in the model’s finding order
- `findings`: every sentence, including ones with no box (normal anatomy, or a finding the model did not localise)
- `impression`: those same sentences, numbered. MAIRA-2 does not write an Impression section, so this field is the findings list and not a separate summary
- `exam`: frontal view, or frontal and lateral when a lateral image was sent to the sidecar

The history line is passed as the study indication. Lateral, prior film, and prior report are accepted by `POST /read` on the sidecar. The current intake screen sends the single frontal image.

Differential diagnosis is a text task this model was not trained for. Leave that button unused, or point it at another engine:

```ini
DDX_ENGINE=openai-compatible
DDX_MODEL=gpt-4o
```

### CheXagent

[CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) writes a findings section. It does not emit boxes, so the reading path stays empty and the text lands in the report. The weights run in a Python sidecar. The build that serves on an Apple GPU is three INT4 ExecuTorch Metal programs, not the PyTorch checkpoint.

```bash
DYLD_INSERT_LIBRARIES=$PWD/chexagent-executorch/metal/libmetal_interpose.dylib \
CHEXAGENT_RUNTIME=metal \
.venv/bin/python scripts/chexagent_generate_pte_server.py
```

In `.env`, point the app at that sidecar and restart `npm run dev`:

```ini
ENGINE=chexagent
MODEL=StanfordAIMI/CheXagent-2-3b
CHEXAGENT_BASE_URL=http://127.0.0.1:8010
```

`GET http://127.0.0.1:8010/health` should report `"runtime": "metal"` and `"device": "mps"`. Differential diagnosis is a text task this model was not trained for. Point that button at another engine with `DDX_ENGINE` and `DDX_MODEL`, the same way as MAIRA-2.

CheXagent's normal forward cannot be one `.pte`. A `.pte` is a frozen graph: shapes are fixed, and the graph cannot open files, call Python, or loop until it decides to stop. The stock model does all three. A real prompt, with the film spliced in, is about 1,085 tokens, and an earlier `text_decoder.pte` was one forward of 8 tokens with no cache, so it could not write a report. The export splits the model into three static programs and a Python runner stitches them back together.

| Program | File | Job |
| --- | --- | --- |
| Vision | `vision_encoder.pte` (about 206 MB) | 512×512 film → 1,024 embeddings of size 2,560 |
| Prefill | `text_prefill.pte` (about 1.4 GB) | The whole prompt, padded to 1,536 tokens → next-token logits and a KV cache |
| Decode | `text_decode.pte` (about 1.4 GB) | One new embedding plus that cache → the next logits |

The runner resizes the film and hands the vision program a float16 tensor. It looks up prompt token ids in the word-embedding table, which stays outside the graphs, and pastes the 1,024 image embeddings into the image span. Prefill then runs once. Decode repeats one token at a time until end-of-sequence (token 50256) or 160 tokens. Prefill and decode are the same Phi stack (32 layers, 32 heads, head size 80, hidden size 2,560) exported at two shapes, because a `.pte` cannot change its sequence length. The cache holds 2,048 positions, which is the model's maximum. Before the weights were packed, this stack matched the full model on the first two tokens (58, then 33).

The linear layers are stored as INT4, group size 32. Every 32 weights share one float16 scale. Each weight is rounded to an integer from −8 to 7, and two of those codes are packed into one byte. At runtime the bytes are unpacked and the multiply runs in float16. Activations, the embedding table, and the KV cache stay float16. Layer norms stay float32, because the float16 norm drifts. Both text programs carry a full copy of the language-model weights, which is why each one is about 1.4 GB instead of the earlier 5.3 GB float16 file.

Those three graphs are lowered twice. `chexagent-executorch/*.pte` is the portable CPU lowering. `chexagent-executorch/metal/*.pte` is the Apple GPU lowering, and that is what the sidecar above serves. The stock Metal runtime only accepts float32 and bfloat16, so the local backend accepts float16 and implements float16 matmul. Metal inputs stay on the CPU and the runtime copies them to the GPU. The same graphs also exist as PyTorch `.pt2` files. `torch.export` bakes in CPU device checks, so those files do not move to MPS until the checks are stripped and the factory ops are retargeted. That `.pt2` path is `npm run chexagent:pt2`. The CPU `.pte` files cannot be moved onto MPS at all: ExecuTorch lowered them to CPU kernels.

On the sample film the Metal path writes a findings paragraph, beginning `[Breathing: Lungs] There is a large area of consolidation in the right lower lobe.` The CPU `.pte` path produces the same kind of paragraph and is much slower: prefill of this prompt took about 71 minutes, then about 45 seconds per new token.

#### Snapdragon

The ExecuTorch runtime Derrick supplied is a CPU runtime. It was not built for arm64 with an NPU delegate or a GPU delegate, so the three CheXagent programs lowered against it contain CPU kernels only. Copying those `.pte` files onto a Snapdragon product does not move the work onto the Hexagon NPU or the Adreno GPU. The Mac Metal lowering is a different backend and does not run on that chip either. With that runtime, CheXagent could not be built onto the product NPU or GPU.

What runs on a GPU locally in this track is SmolVLM-500 in GGUF. A GGUF runtime executes that model on the GPU of the machine it is running on.

The next build on this compute track is an on-device ExecuTorch for the Snapdragon, aimed at arm64 and the NPU. The three CheXagent programs would be lowered again with that backend. The files in `chexagent-executorch/` stay the CPU build.

That build has four parts:

- Compile the ExecuTorch runtime with the Snapdragon toolchain for arm64 and link Qualcomm's NPU backend. The runtime Derrick supplied has no NPU or GPU delegate, so this is a new runtime, not a flag on the one we have.
- Lower `vision_encoder`, `text_prefill`, and `text_decode` again so the matmul and attention land on the Hexagon NPU. A CPU `.pte` copied onto the device still runs on the CPU.
- Quantize in the scheme that backend accepts. The INT4 in these files is packed bytes that get unpacked back to float16 before the multiply. The NPU path needs weights the NPU can multiply directly.
- Fit the model on the device. Each text program is about 1.4 GB because prefill and decode each carry a full copy of the weights. A phone cannot hold both. A Snapdragon laptop can load them, and sharing one weight blob between prefill and decode is what makes a smaller device realistic.

SmolVLM-500 in GGUF stays the local GPU model until that NPU build exists.

## What you see

1. **Intake.** Paste (Ctrl+V), drop, or choose a chest X-ray. Add a history line. Click the orange **Analyze**.
2. **Reading.** The image on the left, the reading path on the right. As each region arrives from the model, a red translucent cloud eases (600 ms) to the centre of that region's box. When the model flags `zoom: true`, the cloud tightens, a ring appears, and a magnified inset of that box shows top right. The list marks the current region. The caption says what this is: *the regions the model reports, in its order. Not eye tracking.* `prefers-reduced-motion` turns the easing off.
3. **Report.** Exam, Findings and Impression, all editable. **Copy report** copies plain text. **Replay read** runs the path again. **Show me differential diagnosis / management** sends only the report text (never the image) to the model and renders a tinted panel: for each finding, the clinical question, differentials with a reason, and a next step with the guideline it rests on.
4. **Rating.** Orange **Acceptable** or outlined red **Not acceptable** (with a reason box), plus chips for study type, ICU and modality. Stored in `data/openread.db` with the model name and a hash of the report.
5. **New** clears everything.

![Report](docs/screens/report.png)
![Differential](docs/screens/ddx.png)

## The JSON contract

Every engine uses the same system prompt (`server/lib/prompt.js`) and the model must answer with one JSON object:

```json
{
  "reading_path": [
    { "region": "Both lungs, overview", "box": [0.08, 0.12, 0.84, 0.72], "note": "Right base looks denser than the left.", "zoom": false },
    { "region": "Right lower lobe", "box": [0.12, 0.52, 0.32, 0.30], "note": "Airspace opacity with air bronchograms.", "zoom": true }
  ],
  "report": {
    "exam": "Chest radiograph, one view.",
    "findings": "Lungs: ...\nPleura: ...\nHeart and mediastinum: ...\nBones and soft tissues: ...\nLines and tubes: ...",
    "impression": "1. ...\n2. ..."
  }
}
```

- `box` is `[x, y, w, h]` as fractions of the image in 0..1 from the top-left. Values above 1 are treated as percentages and rescaled; everything is clamped.
- `zoom: true` marks a region worth a closer look. The UI tightens the cloud and shows the inset.
- The model is asked to emit `reading_path` before `report` so the path can be shown while the report is still being written.

The server parses tolerant of code fences, prose before or after the object, trailing commas and truncated output (`server/lib/json.js`). While the response streams, an incremental scanner (`PathScanner`) emits each `reading_path` item the moment its object closes, without waiting for the rest.

### Streaming API

`POST /api/analyze` with `{ imageBase64, mime, history, imageName }` answers with Server-Sent Events:

```
event: start   data: { "engine": "anthropic", "model": "claude-sonnet-4-5" }
event: path    data: { "item": { "index": 0, "region": "...", "box": [..], "note": "...", "zoom": false } }
event: path    data: ...
event: report  data: { "report": { "exam": "...", "findings": "...", "impression": "..." } }
event: done    data: { "readId": 12, "reportHash": "b4df0ecab34ee51d", "durationMs": 8120 }
event: error   data: { "message": "...", "raw": "first 2000 chars of model output" }
```

The client reads it with `fetch` and a `ReadableStream`, so it works over POST (see `web/src/api.js`).

Other routes: `GET /api/config`, `GET /api/samples`, `POST /api/ddx` (`{ report }` or `{ text }`), `POST /api/rate`, `GET /api/benchmark/summary`, `GET /api/benchmark.csv`, `GET /api/benchmark/next-unrated`.

## Benchmark workflow: comparing two models

Ratings are stored per model, so comparing models is a matter of generating reads with each and rating them blind.

```bash
# 1. Put your test images in samples/ (CC0 or your own de-identified set; nothing leaves your machine except to the model you configured).

# 2. Generate reads with model A, three per image
ENGINE=openai-compatible MODEL=google/medgemma-4b-it npm run bench -- --n 3

# 3. Generate reads with model B
ENGINE=anthropic MODEL=claude-sonnet-4-5 npm run bench -- --n 3

# 4. Rate them. In the UI, click "Rate bench reads" in the footer: it loads the oldest
#    unrated read (image, path and report) and you give the verdict. Repeat until empty.

# 5. Compare
curl -s localhost:8787/api/benchmark/summary | jq .
curl -s localhost:8787/api/benchmark.csv > results.csv
```

`summary` returns the acceptable rate per model per study type, plus how many reads each model has and how many are still unrated. The CSV is one row per read joined with its rating, including `duration_ms` and the `report_hash`, so you can also check how consistent a model is across its N runs of the same image (identical hashes mean identical reports).

Bench reads are rated blind: the UI does not show which model produced them. The verdict is joined to the model through the read id on the server.

## Project layout

```
server/            Express API (ESM)
  index.js         routes: /api/analyze (SSE), /api/ddx, /api/rate, /api/benchmark*
  engines/         one interface: mock, openai-compatible, anthropic, bedrock, local, maira2, chexagent
    base.js        makeEngine(): wraps a text generator into analyze()/complete()
    openai-compatible.js  anthropic.js  bedrock.js  mock.js  maira2.js  chexagent.js
    local.js       on-device pipeline: Python findings worker -> reading path -> local LLM/template
  lib/findings.js  reading path, templated report, rule-based ddx and the report-writer prompt
  lib/prompt.js    the system prompt and the JSON contract
  lib/json.js      tolerant parser + incremental PathScanner
  lib/maira2.js    maps grounded findings onto the reading path and report
  lib/db.js        better-sqlite3: reads, ratings, summary, CSV
web/               Vite + React, plain JSX
  src/App.jsx      intake, reading, replay
  src/Film.jsx     the image, the cloud, the ring, the zoom inset
  src/ReportView.jsx  report, ddx panel, rating
py/                cxr_model.py (shared wrapper), cxr_worker.py (ExecuTorch/PyTorch), cxr_worker_mock.py
docs/chexagent-executorch.md   how the INT4 Metal .pte programs were built
scripts/bench.mjs  run samples/ through the engine N times; label agreement with samples/labels.csv
scripts/maira2_server.py   microsoft/maira-2 sidecar used by ENGINE=maira2
scripts/chexagent_server.py              full PyTorch CheXagent sidecar
scripts/chexagent_generate_pte_server.py generation graphs: vision, prefill, decode
scripts/smoke.mjs  Playwright end-to-end run on the mock engine, writes docs/screens/
scripts/export_cxr.py       classifier -> ExecuTorch .pte (XNNPACK or QNN)
scripts/export_llm.md       Llama 3.2 3B / Qwen3 4B export and runner build
scripts/setup-snapdragon.md Windows on ARM setup for the Dell
scripts/fetch_samples.py    labelled public chest X-rays into samples/
models/            exported .pte files (gitignored)
test/              node:test unit tests
samples/           CC0 sample image; your own labelled set (gitignored), see samples/README.md
data/              SQLite database (gitignored)
```

## Tests

```bash
npm test          # node:test: JSON parser, PathScanner, engines, db, local pipeline via the mock worker (needs python3)
npm run smoke     # Playwright smoke on the mock engine; screenshots to docs/screens/
                  # set CHROME=/path/to/chrome if Chromium is not at the default path
```

## Hackathon ideas

- **Prompt tuning per model.** Small open models need firmer instructions on box coordinates and on emitting `reading_path` first. Try few-shot examples, or ask for one region per line and adapt `PathScanner`.
- **Lateral view support.** Accept two images, add a `view` field to each path item, and show two films side by side.
- **Prior comparison.** Drop a prior study, add a `change` field (`new`, `improved`, `worse`, `stable`) to the report, and let the cloud jump between current and prior.
- **CT series.** Replace the single image with a stack; `box` becomes `[x, y, w, h, slice]` and the film becomes a scrollable viewer.
- **Blind A/B rating mode.** Show two reports for the same image, hide the model, record which one the reader prefers.
- **Fine-tuned region heads.** Compare the model's reported boxes against a segmentation model's lung fields to measure how often the model looks where it says it looks.
- **Voice.** Dictate a correction, apply it to the draft, store the diff as a training signal.
- **Teaching mode.** Hide the model's path, ask a trainee to mark theirs, then overlay the two.

## Disclaimer

OpenRead is a research prototype. It is not a medical device, has not been cleared or approved by any regulator, and must not be used for diagnosis, treatment or any clinical decision. Output can be wrong, including confidently wrong. Never upload images that contain protected health information to a third-party model without the agreements your institution requires. The radiologist decides.

## Credits and license

MIT License, copyright 2026 DICTORA, Inc. and contributors. See `LICENSE`.

Sample image: lobar pneumonia chest radiograph by Mikael Häggström, CC0, via Wikimedia Commons. See `CREDITS.md`.
