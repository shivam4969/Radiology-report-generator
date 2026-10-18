# The vision-language readers: InternVL3 1B and SmolVLM 500M

Two small models that look at the film and write what they see. Both are on the ExecuTorch Qualcomm
backend's supported list, so they are the path to a reader that runs on the Snapdragon NPU.

| | InternVL3 1B | SmolVLM 500M |
|---|---|---|
| Hugging Face | `OpenGVLab/InternVL3-1B` | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Licence | MIT | Apache 2.0 |
| Size (bf16) | about 1.9 GB | about 1.0 GB |
| Use | the better reader | smaller, faster, fallback |

Neither was trained on radiology. Expect plain descriptions ("dense opacity in the right lower lung zone"),
not radiologist prose. That is the point of the benchmark: measure it, do not assume it.

## Run today (PyTorch, any laptop)

```
pip install -r py/requirements.txt transformers>=4.49 einops timm
python py/vlm_worker.py --model internvl3 --once samples/cxr-pneumonia-cc0.jpg
python py/vlm_worker.py --model smolvlm  --once samples/cxr-pneumonia-cc0.jpg
```

Then the server:

```
ENGINE=local LOCAL_READER=internvl3 LOCAL_LLM=template npm start        # the VLM's prose is the report
ENGINE=local LOCAL_READER=internvl3 LOCAL_LLM=openai-compatible LOCAL_LLM_MODEL=qwen3:1.7b npm start   # LLM rewrites it
```

With `LOCAL_READER` set to a VLM there is no reading path and no boxes: nothing tells the model where to
look. The prompt is three sentences (radiologist, frontal chest radiograph, findings then impression in plain
prose), the same one `cxr-read` uses, so results compare across repos. `LOCAL_READER=densenet` (default)
is the previous classifier pipeline, unchanged.

## Run on the Hexagon NPU (ExecuTorch, Qualcomm backend): the stretch goal

The export recipes live in the ExecuTorch repo, not here:
https://github.com/pytorch/executorch/tree/main/examples/qualcomm/oss_scripts
("Vision-Language Models: SmolVLM 500M, InternVL3 1B" in that README).

Steps, same ladder as `export_llm.md`:
1. On a Linux x86 host with the Qualcomm AI Engine Direct SDK (QNN) installed and `QNN_SDK_ROOT` set, follow
   that README's command for the model you want. Use `--compile_only` first, copy the `.pte` and the runner to
   the Dell, then run. The exact flags change between ExecuTorch releases; take them from that README, not
   from memory.
2. Keep the exported `.pte` out of git (`models/` is ignored) and never commit anything from the QNN SDK.
3. Point the server at the runner: `VLM_RUNNER`, `VLM_MODEL`, `VLM_TOKENIZER` (to be wired in
   `server/engines/local.js` once the runner's CLI is confirmed on the device; until then the PyTorch worker
   above is what runs).

If the QNN export fights you on the day, the PyTorch worker on the CPU is still fully offline and still a
valid demo; say so on the slide.

Research prototype. Not for clinical use. The radiologist decides.
