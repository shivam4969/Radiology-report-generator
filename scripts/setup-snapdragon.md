# Setting up the Dell Latitude 7455 (Snapdragon X Elite, Windows on ARM)

Goal: run OpenRead's `ENGINE=local` pipeline entirely on the laptop, with no network after
setup. Three layers, in the order you should get them working:

| layer | what | where it runs | fallback |
|---|---|---|---|
| 1 | Node 22 API + Vite UI | native Windows | none needed |
| 2 | findings worker (`py/cxr_worker.py`) with `cxr-densenet121.pte` | ExecuTorch runtime, XNNPACK (CPU) or QNN (NPU) | PyTorch + torchxrayvision on CPU, automatic |
| 3 | report writer LLM (`llama_main`) | ExecuTorch runtime, XNNPACK or QNN | `LOCAL_LLM=template` (no LLM) or an Ollama build for Windows ARM via `LOCAL_LLM=openai-compatible` |

Get 1 and `LOCAL_LLM=template` with the mock worker running in the first fifteen minutes, then
climb. Every rung of the ladder demos.

**Never commit anything from the Qualcomm SDK** (no `libQnn*.so`, `QnnHtp*.dll`, headers or
the SDK zip), no `.pte` files, no model weights. `models/` and `data/downloads/` are gitignored
for that reason. The SDK licence does not allow redistribution.

---

## 1. Base tools (native Windows)

1. **Git**: <https://git-scm.com/download/win> (ARM64 installer exists since 2.47).
2. **Node 22 ARM64**: <https://nodejs.org/en/download> choose Windows, ARM64, `.msi`.
   `node -v` should print `v22.x`.
3. **Python 3.12 ARM64**: <https://www.python.org/downloads/windows/> pick the "ARM64 installer".
   Tick "Add python.exe to PATH". (3.12, not 3.13: more wheels exist for 3.12 on ARM64.)
4. **Visual Studio 2022 Build Tools** with the "Desktop development with C++" workload and the
   ARM64 build tools component (needed by `better-sqlite3` if no prebuilt binary matches, and by
   any native ExecuTorch build): <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
5. **WSL2 with Ubuntu 24.04 (ARM64)**: `wsl --install -d Ubuntu-24.04` from an admin PowerShell,
   reboot. This is where ExecuTorch builds painlessly. The NPU is **not** visible inside WSL2, so
   WSL2 is the CPU path; QNN needs native Windows binaries (section 5).

```powershell
git clone https://github.com/dictora-ai/openread.git
cd openread
copy .env.example .env
npm install
npm run dev          # http://localhost:5173 with ENGINE=mock
```

If `npm install` fails on `better-sqlite3` it is compiling from source: make sure the VS Build
Tools are installed and rerun; or `npm install --ignore-scripts` and run the API inside WSL2.

## 2. The findings worker on CPU (no ExecuTorch yet)

Native Windows ARM64 PyTorch wheels exist since torch 2.7 for Python 3.12:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy pillow torchxrayvision
python py\cxr_worker.py --once samples\cxr-pneumonia-cc0.jpg
```

The first run downloads the DenseNet weights (~30 MB) to `%USERPROFILE%\.torchxrayvision`; after
that it is offline. You should see probabilities with Pneumonia / Consolidation / Lung Opacity
near the top and a box in the right lower zone. Then in `.env`:

```ini
ENGINE=local
CXR_PYTHON=.venv\Scripts\python.exe
LOCAL_LLM=template
```

`npm run dev`, Analyze the sample. The log says `no .pte at ... falling back to PyTorch`. That is
rung 2 working on the CPU. If torch wheels are not available for your Python, do this step in
WSL2 instead (`pip install torch` on aarch64 Linux works) and run the API from WSL2 too.

## 3. ExecuTorch runtime, XNNPACK (CPU)

In WSL2 Ubuntu (ARM64):

```bash
sudo apt update && sudo apt install -y build-essential cmake ninja-build python3.12-venv python3-dev git
git clone https://github.com/pytorch/executorch.git && cd executorch
git checkout v1.0.0 && git submodule update --init --recursive
python3 -m venv .venv && source .venv/bin/activate
./install_executorch.sh
```

`pip install executorch` may or may not have a linux/aarch64 wheel for your release; if the
install script falls back to building from source, let it (20 to 40 minutes on the X Elite).

Export the classifier (also downloads the weights the first time):

```bash
cd ~/openread
pip install -r py/requirements.txt
python scripts/export_cxr.py              # writes models/cxr-densenet121.pte and prints the verification
```

Expect the verification table to show eager PyTorch and the `.pte` agreeing to 3 decimals.
`CXR_PTE` defaults to `models/cxr-densenet121.pte`; `CXR_PYTHON` must point at the venv that has
`executorch` installed (`CXR_PYTHON=/home/you/executorch/.venv/bin/python` if you run the API
from WSL2, or run the whole thing from WSL2 with `npm run dev` there; the browser on Windows can
reach WSL2's localhost).

A `.pte` exported with XNNPACK on one machine runs on any other machine with an XNNPACK runtime
(the partitioned graph is architecture-neutral; XNNPACK picks kernels at runtime). So you can
export on a Mac and copy the file over.

## 4. The LLM on CPU

Follow `scripts/export_llm.md` sections 0 to 4 inside WSL2. `llama_main` is a Linux binary, so
the API must run in WSL2 for `LOCAL_LLM=executorch`, or you use the Windows-side alternative:
install Ollama for Windows ARM (<https://ollama.com/download/windows>, ARM64 build available
since mid-2025), `ollama pull llama3.2:3b`, and set

```ini
LOCAL_LLM=openai-compatible
LOCAL_LLM_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=llama3.2:3b
```

Ollama is also fully offline after the pull. It is not ExecuTorch, so for the judging story keep
it as the fallback and label it honestly.

## 5. QNN: the Hexagon NPU

This is the stretch goal. What you need:

1. **Qualcomm AI Engine Direct SDK (QAIRT)**: <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>
   (free Qualcomm developer account). Download the Linux x86_64 **and** Windows ARM64 packages
   of the same version (ExecuTorch 1.0 was tested with QNN 2.3x; check `backends/qualcomm/README.md`
   for the version it expects). Unzip outside the repo, e.g. `C:\qairt\2.3x.y` and `~/qairt/2.3x.y`.
2. **Export host**: a Linux **x86_64** machine (a teammate's laptop, or an x86 cloud VM; WSL2 on
   the Dell is ARM64 and the QNN compiler ships x86_64-only). There:

   ```bash
   export QNN_SDK_ROOT=~/qairt/2.3x.y
   cd ~/executorch && ./backends/qualcomm/scripts/build.sh --skip_aarch64   # builds the AOT library; check the flags
   cd ~/openread
   python scripts/export_cxr.py --backend qnn --soc SC8380XP   # or the nearest chipset in QcomChipset
   ```

   The classifier at fp16 on HTP is a few MB and exports in a minute. The LLM is
   `scripts/export_llm.md` section 5.
3. **Runtime on the Dell**: the QNN `.pte` needs an ExecuTorch runtime linked against the
   Qualcomm backend **and** the HTP libraries from the Windows ARM64 SDK package
   (`lib/aarch64-windows-msvc/QnnHtp.dll`, `QnnHtpV73Stub.dll`, `QnnSystem.dll`, plus the
   `lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so` DSP skel) on the `PATH`. Building that runtime
   natively on Windows ARM64 is the part with the least documentation: ExecuTorch's Windows
   support is young, and the Qualcomm scripts target Linux/Android. Budget half a day, start with
   the `python` pybindings (`EXECUTORCH_BUILD_PYBIND=ON -DEXECUTORCH_BUILD_QNN=ON`) so the
   existing `py/cxr_worker.py` can load the QNN `.pte` unchanged, and keep the XNNPACK `.pte`
   next to it so switching back is one `.env` line.
4. Verify the NPU is actually used: Task Manager shows an "NPU" graph on Snapdragon X; it should
   spike during Analyze. `py/cxr_worker.py` prints `ms` per image: XNNPACK CPU is ~50 to 100 ms
   for DenseNet-121 at 224, HTP should be well under 20 ms.

## 6. `.env` for the demo laptop

```ini
ENGINE=local
CXR_PYTHON=/home/you/executorch/.venv/bin/python      # the interpreter that has executorch (or your Windows venv)
CXR_WORKER=py/cxr_worker.py
CXR_PTE=models/cxr-densenet121.pte                    # XNNPACK; swap for the QNN export when it works
CXR_THRESHOLD=0.5
CXR_MAX_FINDINGS=4
LOCAL_PATH_DELAY_MS=600

LOCAL_LLM=executorch                                   # or template / openai-compatible
LLM_RUNNER=/home/you/executorch/cmake-out/examples/models/llama/llama_main
LLM_MODEL=models/llama3_2-3b-instruct-xnnpack-4bit.pte
LLM_TOKENIZER=models/llama3_2-tokenizer.model
LLM_CHAT_TEMPLATE=llama3
LLM_MAX_NEW_TOKENS=700
```

Where files live:

```
openread/
  models/             .pte files, tokenizers, .meta.json   (gitignored)
  samples/            demo images + labels.csv             (only the CC0 image is committed)
  data/openread.db    ratings                               (gitignored)
  data/downloads/     Kaggle zips                           (gitignored)
~/.torchxrayvision/   DenseNet weights cache (PyTorch fallback and export)
~/qairt/<ver>/        Qualcomm SDK, never inside the repo
```

## 7. Pre-demo checklist

- [ ] Wi-Fi off, `npm run dev` still starts and Analyze completes on the sample
- [ ] API log shows `findings worker ready, backend=executorch` (not `torch`, not `mock`)
- [ ] Report streams (`LOCAL_LLM=executorch`), or you have decided to show `template` and say so
- [ ] `ENGINE=local npm run bench` on `samples/` with `labels.csv` ran and you know the agreement number
- [ ] The honesty slide: research classifier, probabilities are not diagnoses, boxes show the
      classifier's strongest activation, the radiologist decides
