"""Shared pieces for the on-device chest X-ray classifier.

Used by both `scripts/export_cxr.py` (export to ExecuTorch) and
`py/cxr_worker.py` (inference). Keeping preprocessing and the CAM head in one
place means the exported `.pte` and the PyTorch fallback compute exactly the
same thing.

Model: TorchXRayVision DenseNet-121, weights "densenet121-res224-all"
(18 pathology outputs, 224x224 single-channel input). It is a research model;
its outputs are calibrated probabilities on its training distribution, not
diagnoses.
"""

from __future__ import annotations

import numpy as np

# The 18 pathologies of densenet121-res224-all, in output order. Taken from
# torchxrayvision.datasets.default_pathologies. Verified against
# `xrv.models.DenseNet(weights="densenet121-res224-all").pathologies` at export
# time; the export script writes the live list into the .meta.json.
PATHOLOGIES = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

INPUT_SIZE = 224


def preprocess(pil_image, size: int = INPUT_SIZE):
    """PIL image -> (array[1,1,size,size] float32, crop) in TorchXRayVision units.

    TorchXRayVision expects a single channel in the range [-1024, 1024]
    (xrv.datasets.normalize) after a center crop to a square and a resize.
    `crop` is (x0, y0, side, W, H) in original pixels so the caller can map
    boxes found on the crop back onto the full image.
    """
    img = pil_image.convert("L")
    w, h = img.size
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    img = img.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), _BILINEAR())
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr / 255.0) * 2048.0 - 1024.0  # == xrv.datasets.normalize(arr, 255)
    return arr[None, None, :, :], (x0, y0, side, w, h)


def _BILINEAR():
    from PIL import Image

    return getattr(Image, "Resampling", Image).BILINEAR


def build_wrapper():
    """Return (wrapper_module, pathologies). Imports torch lazily.

    The wrapper outputs `(probs[1,18], cams[1,18,7,7])`:
      probs  sigmoid of the logits, then TorchXRayVision's operating-point
             normalisation (op_norm) so 0.5 is the model's decision threshold
      cams   class activation maps: classifier weights x the final feature map
             (DenseNet features after ReLU), one 7x7 map per pathology.
    Everything is written with torch.where so torch.export can trace it.
    """
    import torch
    import torch.nn.functional as F
    import torchxrayvision as xrv

    base = xrv.models.DenseNet(weights="densenet121-res224-all")
    base.eval()

    class CxrWithCam(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.features = m.features
            self.classifier = m.classifier
            thr = getattr(m, "op_threshs", None)
            if thr is None:
                thr = torch.full((len(m.pathologies),), float("nan"))
            self.register_buffer("op_threshs", torch.as_tensor(thr, dtype=torch.float32))

        def forward(self, x):
            feats = F.relu(self.features(x))  # [B, 1024, 7, 7]
            pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
            logits = self.classifier(pooled)  # [B, 18]
            p = torch.sigmoid(logits)
            # xrv.models.op_norm, rewritten without boolean indexing.
            t = self.op_threshs
            nan = torch.isnan(t)
            ts = torch.where(nan, torch.full_like(t, 0.5), t)
            low = p / (ts * 2.0)
            high = 1.0 - (1.0 - p) / ((1.0 - ts) * 2.0)
            probs = torch.where(p < ts, low, high)
            probs = torch.where(nan, p, probs)  # no operating point: raw sigmoid
            # CAM: weights [18, 1024] x features [B, 1024, 7, 7] -> [B, 18, 7, 7]
            w = self.classifier.weight
            cams = torch.einsum("oc,bchw->bohw", w, feats)
            return probs, cams

    return CxrWithCam(base), list(base.pathologies)
