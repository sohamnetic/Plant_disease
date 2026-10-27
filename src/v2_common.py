"""Constants and preprocessing shared by the v2 network, the PyTorch runtime and the
PyTorch-free ONNX runtime (src/onnx_runtime.py). Imports no torch."""

from __future__ import annotations

import numpy as np
from PIL import Image

CLASS_ORDER = ["bacterial_spot_rot", "black_rot", "downy_mildew", "no_disease"]
DISEASES = CLASS_ORDER[:3]
NO_DISEASE_IDX = 3
NUM_GRADES = 5
GRADE_LABELS = ["none", "early", "moderate", "severe", "critical"]
# Same scale as src/stages.json and the measured grades (Phase 5).
GRADE_MEANING = [
    "no visible symptoms",
    "<5% of the worst leaf/head affected, or a few small spots",
    "5-25% of the worst leaf/head affected",
    "25-50% of the worst leaf/head affected",
    ">50% affected or tissue collapse",
]
IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FILL = tuple(int(m * 255) for m in IMAGENET_MEAN)


def letterbox(im: Image.Image) -> Image.Image:
    """Pad to a square with ImageNet-mean colour; keeps the whole frame."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), FILL)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


def coral_to_probs_np(sev_logits: np.ndarray) -> np.ndarray:
    """numpy twin of v2_model.coral_to_probs (same arithmetic, float32)."""
    cum = (1.0 / (1.0 + np.exp(-sev_logits.astype(np.float32)))).astype(np.float32)
    ones = np.ones_like(cum[:, :1])
    zeros = np.zeros_like(cum[:, :1])
    return np.clip(np.concatenate([ones, cum], 1) - np.concatenate([cum, zeros], 1), 0, None)
