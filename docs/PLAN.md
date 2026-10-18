# OpenRead at the ExecuTorch Hackathon: the plan

Team: radio warriors. Track: Compute (Dell Latitude 7455, Snapdragon X Elite). Demo night: Sunday Oct 18, 2026.

Goal: a chest X-ray reader that runs fully on the laptop, no cloud call, built as a pipeline of small
open models, each doing one job, and benchmarked against a frontier reference reader.

## The five tasks, one model each

| # | Task | In | Out | Open models to try (under 25 GB) |
|---|------|----|-----|----------------------------------|
| 1 | **Triage**: readable PA film? anything urgent? | image | flags: projection, rotation, lines and tubes, pneumothorax, large effusion, free air | TorchXRayVision DenseNet-121 (30 MB), MedSigLIP + linear head, RAD-DINO + linear head |
| 2 | **Findings**: what is abnormal, where | image | checklist of findings with probability and box, or prose | DenseNet-121 (18 labels, Grad-CAM boxes, in `ENGINE=local` now); **InternVL3 1B and SmolVLM 500M** (vision-language, `LOCAL_READER=internvl3|smolvlm`, both on the Qualcomm backend list, see `scripts/export_vlm.md`); MedGemma 1.5 4B, CheXagent-2 3B |
| 3 | **Comparison**: what changed since the prior | findings + prior report or image | new / resolved / larger / smaller / stable | Qwen3 4B or Llama 3.2 3B for a prior report; the task 2 VLM for a prior image; MAIRA-2 does this natively |
| 4 | **Report**: write it the way a radiologist would | history + findings + change + template | Exam / Findings / Impression JSON | Llama 3.2 3B Instruct (best ExecuTorch path, NPU), Qwen3 4B, Phi-4 mini |
| 5 | **Reasoning**: differential, management, self-check | the report text only | differentials with reasons, next step with guideline, consistency critique | Qwen3 8B thinking, MedGemma 27B text (4-bit), DeepSeek-R1 distill 7B. Use a different family from task 4 so it is not grading its own prose |

Tasks 1, 2 and 4 are the demo. 3 and 5 are the stretch. Each task has a crisp input and output, so one
teammate owns one without touching the others, and the benchmark scores each stage separately.

## Running on the Snapdragon

See `scripts/setup-snapdragon.md`. The ladder, each rung a working demo:
1. Node + mock worker (`ENGINE=mock`): UI works, no models.
2. `ENGINE=local` with the PyTorch worker on CPU (`LOCAL_LLM=template`): real classifier, templated report.
3. Classifier exported to ExecuTorch, XNNPACK backend (`scripts/export_cxr.py`), still CPU.
4. LLM through the ExecuTorch Llama runner (`scripts/export_llm.md`). Text writers on the Qualcomm list, best first for the size: Qwen3 1.7B, Llama 3.2 3B, Phi4-mini, Gemma 3 1B. Stories 110M and SmolLM2 135M are smoke tests only.
4b. VLM reader (InternVL3 1B, then SmolVLM 500M) in PyTorch, then as a `.pte` on QNN (`scripts/export_vlm.md`).
5. QNN backend: classifier, then LLM, on the Hexagon NPU. The stretch goal, and what judges will ask about.

The NPU matters because a 3B model that runs at 3 to 8 tokens a second on the CPU runs at 15 to 30 on the
NPU, cool and on battery. QNN export happens on a Linux x86 host, not on the Dell.

## Benchmark: Astra as the reference reader, not the truth

1. **Reference set.** 100 films from NIH ChestX-ray14 (Kaggle `nih-chest-xrays/sample`), sampled so
   every label has at least 8 positives, plus 20 normals. Run each once through GPT-6 Astra with a
   structured prompt: the report, a present / absent / uncertain checklist of the 14 NIH findings, the
   single most important finding, laterality. Store as `reference` with prompt and model version. Run a
   subset twice: Astra's self-agreement is the ceiling.
2. **Score the local pipeline three ways.**
   - Finding level: present/absent per label versus Astra's checklist and versus the NIH label.
     Sensitivity, specificity, Cohen's kappa per finding.
   - Impression level: a judge model scores each local impression against the reference on four yes/no
     questions: critical finding missed, finding invented, wrong side, management wrong.
   - Radiologist tiebreak: every disagreement goes to the blind rating queue ("Rate bench reads" in
     the footer), both reports shown, model names hidden.
3. **Image-swap test** (from arXiv 2606.17710): swap the radiograph for one with the opposite label and
   ask the same question. A model that keeps its answer is reading the text, not the pixels. Cheap and
   decisive; run it on every VLM you try.

Headline number: critical-finding miss rate versus the reference, with the radiologist's agreement rate
alongside. Honest line for the slide: "NLP labels, frontier-model reference, radiologist adjudicated."

## Dataset notes

- NIH ChestX-ray14: 112,120 films, 1024 x 1024 8-bit PNG, 14 labels mined from reports by NLP (about
  90% accurate). Small nodules and subtle pneumothoraces lose detail at 1024; say so when presenting.
- RSNA Pneumonia is one finding only; good for training a detector, wrong for testing a report writer.
- CheXpert test set (500 films, radiologist panel labels) is the best public ground truth; needs a
  Stanford AIMI registration.

## Licences

MedGemma: Health AI Developer Foundations terms (accept on Hugging Face). MAIRA-2: Microsoft Research
licence, research only. CheXagent: Stanford research licence. Qualcomm QNN SDK binaries: never commit
them; link to Qualcomm's download. OpenRead itself is MIT.

Research prototype. Not for clinical use. The radiologist decides.
