# Exporting the report-writer LLM with ExecuTorch

The local engine's report writer is a small instruct model run by ExecuTorch's Llama runner.
Two candidates that fit a laptop NPU/CPU: **Llama 3.2 3B Instruct** (needs a Meta licence click)
and **Qwen3 4B** (Apache 2.0, no gate). Both export through the same `examples/models/llama`
path in the ExecuTorch repo, 4-bit weights, and both produce a `.pte` plus a tokenizer file that
`server/engines/local.js` hands to the runner.

This file is written against the ExecuTorch repo layout as of the v0.6 to v1.0 releases
(2025). Commands and flag names move between releases, so where this says **check**, open the
file named and compare before you paste. Primary references inside the repo:

- `examples/models/llama/README.md` (the export recipe and runner build)
- `examples/models/llama/config/` (YAML export configs used by the new config-driven exporter)
- `examples/models/qwen3/README.md`
- `examples/qualcomm/oss_scripts/llama/README.md` (the QNN/HTP path)
- `docs/source/llm/` and `docs/source/backends-qualcomm.md`

Plan for the hackathon: get the **XNNPACK** (CPU) build working first on whatever machine you
have, because every step below is known to work there. Then attempt **QNN** on the Snapdragon as
the stretch goal. Keep `LOCAL_LLM=template` or `LOCAL_LLM=openai-compatible` (Ollama) in `.env`
until the runner is ready, so the rest of the pipeline is never blocked.

---

## 0. One-time setup (Linux x86_64 or macOS arm64 host; WSL2 Ubuntu on the Dell works for CPU)

```bash
git clone https://github.com/pytorch/executorch.git
cd executorch
git checkout v1.0.0           # or the release the team agrees on; `git tag | grep '^v'`
git submodule sync && git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
./install_executorch.sh       # pip-installs executorch + the matching torch nightly/stable
# LLM extras (tokenizers, quantisation):
./examples/models/llama/install_requirements.sh    # check: may be named install_requirements.sh or requirements.txt
```

If `./install_executorch.sh` offers `--pybind xnnpack` or `--editable`, take the defaults.
Confirm: `python -c "import executorch; print(executorch.__version__)"`.

## 1. Get the weights

**Llama 3.2 3B Instruct** (gated):

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
  --include "original/*" --local-dir ~/models/llama3_2-3b-instruct
# original/consolidated.00.pth, original/params.json, original/tokenizer.model
```

**Qwen3 4B** (open):

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir ~/models/qwen3-4b
# ExecuTorch converts the HF safetensors itself for qwen3; see examples/models/qwen3/README.md
# The tokenizer is tokenizer.json (HF tokenizers format), which the runner reads directly.
```

## 2. Export for XNNPACK (CPU), 4-bit

ExecuTorch has two exporter front ends. The config-driven one (`extension.llm.export.export_llm`,
Hydra-style overrides) is the current one; the older flag-style `examples.models.llama.export_llama`
still exists in 0.6/1.0. Use whichever your checkout's README shows.

**Config-driven (check `examples/models/llama/config/*.yaml` for the exact file names):**

```bash
python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_xnnpack_q8da4w.yaml \
  +base.model_class="llama3_2" \
  +base.checkpoint=~/models/llama3_2-3b-instruct/original/consolidated.00.pth \
  +base.params=~/models/llama3_2-3b-instruct/original/params.json \
  +export.output_name="llama3_2-3b-instruct-xnnpack-4bit.pte"
```

**Flag-style (older checkouts):**

```bash
python -m examples.models.llama.export_llama \
  --model llama3_2 \
  --checkpoint ~/models/llama3_2-3b-instruct/original/consolidated.00.pth \
  --params ~/models/llama3_2-3b-instruct/original/params.json \
  -kv --use_sdpa_with_kv_cache -X \
  -qmode 8da4w --group_size 128 -d fp32 \
  --max_seq_length 2048 --max_context_length 2048 \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --output_name llama3_2-3b-instruct-xnnpack-4bit.pte
```

What the flags mean: `-kv` KV cache in the graph, `-X` lower to XNNPACK, `-qmode 8da4w` 8-bit
dynamic activations with 4-bit weights (the standard "4-bit" recipe), `--group_size 128`
quantisation group, `-d fp32` compute dtype (use `fp16`/`bf16` only if the README says the
backend supports it), `--max_context_length` must cover our prompt (~900 tokens) plus
`LLM_MAX_NEW_TOKENS` (700), so 2048 is the floor.

For **Qwen3 4B** swap `--model qwen3_4b` (check the exact `model_class` string in
`examples/models/llama/model_args.py` or `examples/models/qwen3/`), point `--checkpoint` at the
converted weights the Qwen3 README produces, and keep the same quantisation flags. Qwen3 uses the
ChatML template: set `LLM_CHAT_TEMPLATE=chatml` in `.env`.

Expected size: Llama 3.2 3B at 8da4w is roughly 2.0 GB; Qwen3 4B about 2.6 GB. Export takes
5 to 15 minutes and a lot of RAM (16 GB is tight; close other things).

Copy outputs into this repo's `models/` (gitignored):

```bash
cp llama3_2-3b-instruct-xnnpack-4bit.pte ~/openread/models/
cp ~/models/llama3_2-3b-instruct/original/tokenizer.model ~/openread/models/llama3_2-tokenizer.model
```

## 3. Build the runner (`llama_main`)

```bash
# from the executorch checkout root
cmake --preset llm -DCMAKE_INSTALL_PREFIX=cmake-out -DCMAKE_BUILD_TYPE=Release -Bcmake-out .
cmake --build cmake-out -j$(nproc) --target install

cmake -DCMAKE_PREFIX_PATH=cmake-out -DCMAKE_BUILD_TYPE=Release \
      -Bcmake-out/examples/models/llama examples/models/llama
cmake --build cmake-out/examples/models/llama -j$(nproc)
ls cmake-out/examples/models/llama/llama_main
```

If your checkout has no `llm` preset (older than 1.0), the README lists the equivalent `-D`
flags: `EXECUTORCH_BUILD_EXTENSION_MODULE`, `_DATA_LOADER`, `_TENSOR`, `_LLM`, `_LLM_RUNNER`,
`EXECUTORCH_BUILD_XNNPACK`, `EXECUTORCH_BUILD_KERNELS_QUANTIZED`, `_OPTIMIZED`, `_LLM` all `ON`.

Smoke test the runner directly:

```bash
cmake-out/examples/models/llama/llama_main \
  --model_path ~/openread/models/llama3_2-3b-instruct-xnnpack-4bit.pte \
  --tokenizer_path ~/openread/models/llama3_2-tokenizer.model \
  --prompt "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Say hi in JSON.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

" --max_new_tokens 40 --temperature 0
```

**Check** `llama_main --help`: older runners have `--seq_len` (total tokens, prompt included)
instead of `--max_new_tokens`. If so, set `LLM_MAX_TOKENS_FLAG=--seq_len` and raise
`LLM_MAX_NEW_TOKENS` to about 1600. The runner echoes the prompt, then the generated text, then a
`PyTorchObserver {...}` stats line; `local.js` strips the echo and the stats before parsing.

## 4. Wire it into OpenRead

```ini
ENGINE=local
LOCAL_LLM=executorch
LLM_RUNNER=/home/you/executorch/cmake-out/examples/models/llama/llama_main
LLM_MODEL=models/llama3_2-3b-instruct-xnnpack-4bit.pte
LLM_TOKENIZER=models/llama3_2-tokenizer.model
LLM_CHAT_TEMPLATE=llama3          # chatml for Qwen3
LLM_MAX_NEW_TOKENS=700
LLM_RUNNER_ARGS=--cpu_threads 8   # optional
```

`npm run dev`, Analyze the sample: the reading path arrives from the classifier first, then the
report streams token by token (`event: token`) and lands as `event: report`. If the runner
returns something unparseable, the engine logs it and falls back to the templated report, so the
demo never stalls.

## 5. QNN (Qualcomm HTP / NPU) variant

The HTP path has its own scripts because the LLM is split into prefill and decode graphs and
quantised with PT2E (`16a4w`: 16-bit activations, 4-bit weights). It requires:

- the Qualcomm AI Engine Direct SDK (QAIRT / QNN), `QNN_SDK_ROOT` exported; see
  `scripts/setup-snapdragon.md`
- an ExecuTorch build with `EXECUTORCH_BUILD_QNN=ON` (`backends/qualcomm/scripts/build.sh`)
- the export run on a **Linux x86_64 host** (the SDK's compiler is x86_64 only); the `.pte` then
  runs on the Snapdragon

**Check** `examples/qualcomm/oss_scripts/llama/README.md` for the current invocation; it is
roughly:

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
  --decoder_model llama3_2-3b_instruct \
  --checkpoint ~/models/llama3_2-3b-instruct/original/consolidated.00.pth \
  --params ~/models/llama3_2-3b-instruct/original/params.json \
  --tokenizer_model ~/models/llama3_2-3b-instruct/original/tokenizer.model \
  --model_mode hybrid --prefill_ar_len 128 --max_seq_len 2048 \
  --ptq 16a4w --compile_only \
  --model SC8380XP     # the Snapdragon X Elite SoC name; see QcomChipset for what your build knows
  --artifact ./qnn-llama
```

Open questions to settle on day one (five minutes with the repo beats guessing):

1. Is the X Elite (SC8380XP) in `QcomChipset` for your release, or do you target the nearest
   HTP architecture (v73)? Look in `backends/qualcomm/serialization/qc_schema.py`.
2. The QNN runner is `qnn_llama_runner`, built by the same script with `--build_folder`; its
   flags differ from `llama_main` (`--model_path`, `--tokenizer_path`, `--prompt`,
   `--seq_len`, `--eval_mode`). Use `LLM_RUNNER_ARGS` and `LLM_MAX_TOKENS_FLAG` to match.
3. Runner binary on **Windows on ARM**: the QNN runner is built for Android or Linux in the
   repo's scripts. A native Windows ARM64 build means CMake with the MSVC ARM64 toolchain and the
   Windows QNN libraries (`lib/aarch64-windows-msvc/` inside the SDK). Treat that as the stretch
   goal; CPU XNNPACK in WSL2 is the fallback that always works.

## 6. Prompt and size budget

`server/lib/findings.js` builds the prompt: a system message with the JSON schema, then the
history, the positive findings with percentages and locations, the below-threshold list and the
template. It is ~600 to 900 Llama tokens. A 3B/4B model at 4-bit on a laptop CPU generates
10 to 20 tokens/s, so a 300-token report takes 15 to 30 s; on the HTP expect several times faster.
Set `LLM_MAX_NEW_TOKENS` no lower than 500 or the JSON gets cut off (the parser repairs
truncation, but the impression suffers).
