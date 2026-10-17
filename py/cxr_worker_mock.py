#!/usr/bin/env python3
"""Mock findings worker: same JSON-lines protocol as cxr_worker.py, canned
answer for any image, standard library only. Used by `npm test` so the local
engine can be exercised without PyTorch, ExecuTorch or model weights.

  CXR_WORKER=py/cxr_worker_mock.py ENGINE=local LOCAL_LLM=template npm run dev
"""

import json
import sys

PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
    "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule",
    "Mass", "Hernia", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]

# A right lower lobe pneumonia, like the CC0 sample image.
PROBS = {p: 0.05 for p in PATHOLOGIES}
PROBS.update({"Pneumonia": 0.83, "Consolidation": 0.74, "Lung Opacity": 0.69, "Infiltration": 0.52, "Effusion": 0.21})

CAMS = {p: [0.2, 0.2, 0.6, 0.6] for p in PATHOLOGIES}
CAMS.update({
    "Pneumonia": [0.12, 0.52, 0.32, 0.3],
    "Consolidation": [0.14, 0.55, 0.28, 0.26],
    "Lung Opacity": [0.1, 0.5, 0.36, 0.34],
    "Infiltration": [0.15, 0.5, 0.3, 0.3],
})


def main():
    print(json.dumps({"event": "ready", "backend": "mock", "pathologies": PATHOLOGIES}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"id": None, "error": f"bad request: {e}"}), flush=True)
            continue
        if not req.get("image_b64"):
            print(json.dumps({"id": req.get("id"), "error": "image_b64 is required"}), flush=True)
            continue
        print(json.dumps({"id": req.get("id"), "probs": PROBS, "cams": CAMS, "backend": "mock", "ms": 1}), flush=True)


if __name__ == "__main__":
    main()
