# CheXagent on ExecuTorch

How OpenRead runs [CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) from ExecuTorch `.pte` files on an Apple GPU. Research prototype. Not for clinical use.

The app does not load the weights itself. The page on http://localhost:5173 talks to the Node API on port 8787, and that API posts the film to a Python sidecar. The sidecar that is serving now listens on http://127.0.0.1:8010 and runs three Metal `.pte` programs.

## What is serving

| Program | File | Job |
| --- | --- | --- |
| Vision | `chexagent-executorch/metal/vision_encoder.pte` | 512×512 film → 1024 image embeddings |
| Prefill | `chexagent-executorch/metal/text_prefill.pte` | The whole prompt → next-token logits and a KV cache |
| Decode | `chexagent-executorch/metal/text_decode.pte` | One new token plus that cache → the next logits |

Health from the sidecar:

```json
{"ok": true, "runtime": "metal", "device": "mps", "prefill": 1536, "max_new_tokens": 160}
```

`runtime=metal` means these are ExecuTorch Metal programs. `device=mps` means the GPU is Apple Metal. This is not the PyTorch `.pt2` path and not the CPU `.pte` files one directory up.

Start it from the repo root:

```bash
DYLD_INSERT_LIBRARIES=$PWD/chexagent-executorch/metal/libmetal_interpose.dylib \
CHEXAGENT_RUNTIME=metal \
.venv/bin/python scripts/chexagent_generate_pte_server.py
```

Point the app at it and restart `npm run dev` only if that process was not already started with these variables. `node --env-file` does not override variables the process already has.

```ini
ENGINE=chexagent
MODEL=StanfordAIMI/CheXagent-2-3b
CHEXAGENT_BASE_URL=http://127.0.0.1:8010
```

`DYLD_INSERT_LIBRARIES` has to be set before Python starts. It loads `libmetal_interpose.dylib`, which loads `libmetal_backend.dylib` and registers ExecuTorch's Metal backend. The model code looks up `aoti_torch_*` symbols at runtime. This process also has `libtorch_cpu.dylib`, which exports the same names, and macOS has no `RTLD_DEEPBIND`. Without the interpose, dyld binds those calls to libtorch and the process crashes. The interpose renames the ExecuTorch copies to `et_aoti_torch_*` and forwards the lookups there.

## What was built

CheXagent's normal `forward` cannot be a `.pte`. A `.pte` is a frozen graph: shapes are fixed, and the graph cannot open files, call Python, or loop until it decides to stop. The stock model does all three. The work was to split it into three static programs, quantize the matrices, lower them for Metal, and write a runner that stitches them back together.

```
film ──► vision_encoder.pte ──► 1024 embeddings
                                      │
prompt token ids ──► embedding table ─┤  (both outside the .pte)
                                      ▼
                              splice into the prompt
                                      │
                                      ▼
                            text_prefill.pte
                                      │
                            logits + KV cache
                                      │
                                      ▼
                         text_decode.pte  × one token
                                      │
                                      ▼
                              findings text
```

### 1. Pixels stay outside the graph

The stock forward hides an image path inside the token ids and opens it with PIL. Export moves that out. The runner resizes the film to 512×512 and hands the vision program a float16 tensor. The vision program returns 1024 vectors of width 2560, which is the same width as a word embedding.

### 2. INT4 weights

The size of the model is the linear layers: query, key, value, the attention output projection, the MLP, the vocabulary projection, and the same kind of matrices in the vision tower. `scripts/export_chexagent_pte.py` replaces each of those with `Int4Linear` before export.

- Group size 32, symmetric. Every 32 weights in a row share one float16 scale, `max(abs(group)) / 7`.
- Each weight is rounded to an integer from −8 to 7 and stored as 0–15. Two values are packed into one `uint8`: the low nibble, then the high nibble.
- Activations, the token embedding table, and the KV cache stay float16. Biases stay full precision. LayerNorm runs in float32 and casts back, because the portable float16 LayerNorm drifts.
- On Metal, a shader unpacks each byte, subtracts 8, multiplies by the group scale, and the matmul runs in float16.

That is why each text `.pte` is about 1.4 GB. Both text programs carry a full copy of the language-model weights, because each `.pte` is standalone. The vision program is about 205 MB.

### 3. Two text graphs, one language model

CheXagent's language model is a decoder-only Phi stack: 32 layers, 32 heads, head size 80, hidden size 2560. There is no text encoder. Prefill and decode are the same stack exported at two shapes, because a `.pte` cannot change its sequence length.

`scripts/export_chexagent_generate.py` checks that stack against the full model, then quantizes and exports:

- `text_prefill.pt2` — embeddings `[1, 1536, 2560]`, positions, an empty cache, and a mask. Returns logits `[1, 1536, 51200]` and a cache.
- `text_decode.pt2` — one new embedding `[1, 1, 2560]`, one position, the cache, and a mask. Returns the next logits and the updated cache.

The prompt is about 1080 tokens once the film is spliced in, so prefill is padded to 1536. The cache holds 2048 positions so decode can keep writing past the prompt. The mask fills unused positions with the smallest float32, which keeps attention off the padding.

An earlier `text_decoder.pte` was one forward of 8 tokens with no cache. It cannot hold a real prompt and cannot continue a report. Prefill plus decode replaces it.

### 4. Metal lowering

`scripts/lower_chexagent_pte.py` with `CHEXAGENT_BACKEND=metal` lowers the three `.pt2` graphs into `chexagent-executorch/metal/`. CPU `.pte` files in the parent directory are left alone.

The stock Metal runtime only accepts float32 and bfloat16. These graphs are float16, so the local backend (`libmetal_backend.dylib`, built from the ExecuTorch 1.5.1 sources) accepts dtype 5 and implements float16 matmul, batched matmul, and convolution. Attention in these graphs is explicit matmul, not scaled-dot-product attention. Float16 products accumulate in float32.

Metal inputs stay on the CPU. The runtime copies them to the GPU. Keeping them on MPS makes the runtime reject the call.

The three programs stay loaded for the life of the process. Dropping one while another is alive frees a shared weight map and the next call crashes with `Memory address ... is already being tracked`. Prefill and decode share one weights blob of about 1438 MB.

### 5. The runner

`scripts/chexagent_generate_pte_server.py` is the sidecar on port 8010. `POST /read` with `{ imageBase64, history }` does this:

1. The CheXagent tokenizer builds the prompt, with a hole between the image-start and image-end tokens. The history line is the indication.
2. The vision program encodes the film.
3. Token ids are looked up in `model.embed_tokens.weight`, which stays outside the graphs. The 1024 image vectors are pasted into the hole. The splice depends on where those tokens landed, so it stays in Python.
4. Prefill runs on the padded embeddings. The first generated token is the argmax at the last real prompt position.
5. Decode repeats with one new embedding at a time, updating the same cache, until end-of-sequence (token 50256) or 160 tokens.

The Node engine is `server/engines/chexagent.js`. It posts the film to `CHEXAGENT_BASE_URL` and puts the returned findings text on the report. CheXagent does not emit boxes, so the reading path stays empty.

## Why the key cache had to be fixed

A findings section is the decode loop. Decode is useless if prefill's key cache is wrong, because every later token attends to those keys.

Rotary position is applied to the first 32 of the 80 dimensions in each head. The exported graph writes those dimensions through a strided slice: the slice is 32 wide, and the next head dimension starts 80 elements later. ExecuTorch's tensor type cannot store a gapped stride like that. The first Metal runtime copied the slice into a fresh packed buffer so the tensor could be constructed. The shader still indexed with the original stride of 80, so the write landed in the copy and the cache the runner reads stayed zero. The value cache is densely packed, so it already matched PyTorch.

The runtime now keeps the parent allocation and only records contiguous strides on the tensor object, which is enough for ExecuTorch to accept it. The shader's own indexing writes the rotated keys into the real cache. On an all-zero prompt, layer-0 keys match the PyTorch `.pt2` graph to float16. On the sample film (`samples/cxr-pneumonia-cc0.jpg`, indication "cough and fever"), prefill's first token is `[` and the text begins `[Breathing: Lungs] There is a large area of consolidation in the right lower lobe.`

## Other ways to run the same model

| Command | What it loads | Where |
| --- | --- | --- |
| The launch command above | Metal `.pte` in `chexagent-executorch/metal/` | GPU, port 8010 |
| `npm run chexagent:generate` | CPU `.pte` (`text_prefill`, `text_decode`, `vision_encoder`) | CPU, port 8010 |
| `npm run chexagent:pte` | Older CPU pair, `vision_encoder.pte` and 8-token `text_decoder.pte` | CPU, port 8010 |
| `npm run chexagent:pt2` | Float16 `.pt2` graphs through PyTorch | GPU, port 8010 |
| `npm run chexagent` | Full PyTorch CheXagent, `scripts/chexagent_server.py` | GPU if present, port 8009 |

The `.pt2` files are the export products. They are PyTorch graphs. Lowering them is what produces a `.pte`. The app only calls whichever sidecar `CHEXAGENT_BASE_URL` names. Port 8009 and port 8010 are different servers.

## Snapdragon

The ExecuTorch runtime Derrick supplied is a CPU runtime. It was not built with an arm64 NPU delegate or a GPU delegate, so the three CheXagent programs lowered against it contain CPU kernels only. Copying those `.pte` files onto a Snapdragon product does not move the work onto the Hexagon NPU or the Adreno GPU. The Mac Metal lowering is a different backend and does not run on that chip either. CheXagent therefore could not be placed on the product NPU or GPU with the runtime we had.

The local GPU path used instead is SmolVLM-500 in GGUF. A GGUF runtime can execute that model on the GPU of the machine it is running on, which is how that model runs locally today.

The follow-up is an on-device ExecuTorch build. Build the runtime on the Snapdragon device toolchain for arm64, with the Qualcomm NPU backend linked in, then lower `vision_encoder`, `text_prefill`, and `text_decode` again with that backend. The resulting `.pte` files are the ones that can run on the NPU. The CPU files in `chexagent-executorch/` stay the portable CPU build.

## Files

```
scripts/export_chexagent_pte.py          vision export, INT4, the old 8-token decoder
scripts/export_chexagent_generate.py     prefill and decode export
scripts/lower_chexagent_pte.py           .pt2 → CPU or Metal .pte
scripts/chexagent_generate_pte_server.py sidecar for the three generation programs
scripts/chexagent_pte_server.py          sidecar for the old 8-token pair
scripts/chexagent_server.py              full PyTorch sidecar on port 8009
server/engines/chexagent.js              Node client
chexagent-executorch/                    exported graphs (gitignored)
chexagent-executorch/metal/              Metal .pte files and the linked runtime
```

The ExecuTorch checkout used to build `libmetal_backend.dylib` is not in this repo. The dylib in `chexagent-executorch/metal/` is the runtime that is actually loaded.
