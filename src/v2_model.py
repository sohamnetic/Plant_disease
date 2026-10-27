"""v2 multi-task network and preprocessing, shared by training and the API.

One EfficientNet-B0 backbone, three heads on the pooled 1280-d feature:
  * diagnosis   - 4-way softmax (bacterial_spot_rot, black_rot, downy_mildew, no_disease)
  * presence    - 3 independent sigmoids, one per disease, so co-infection can be
                  expressed (trained with CutMix composites of two classes)
  * severity    - CORAL ordinal head over grades 0-4 (shared weight, 4 ordered
                  biases => rank-consistent cumulative probabilities)

Preprocessing letterboxes the full photo to a square (no centre crop, so edge
lesions survive) and runs at 384 px instead of 224 px so small spots survive.

Only torch / torchvision / PIL here: the API imports this without pandas.
"""

from __future__ import annotations

import io
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

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


class Letterbox:
    def __call__(self, im: Image.Image) -> Image.Image:
        return letterbox(im)


def eval_transforms(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            Letterbox(),
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class RandomLowRes:
    """Downscale then upscale: mimics small web / messaging-app photos."""

    def __init__(self, p: float = 0.25, sides: tuple[int, int] = (140, 320)):
        self.p, self.sides = p, sides

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return im
        side = random.randint(*self.sides)
        k = side / max(im.size)
        if k >= 1:
            return im
        small = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.Resampling.BILINEAR)
        return small.resize(im.size, Image.Resampling.BILINEAR)


class RandomJpeg:
    def __init__(self, p: float = 0.3, quality: tuple[int, int] = (25, 80)):
        self.p, self.quality = p, quality

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return im
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomWhiteBalance:
    """Per-channel gain: camera white balance / colour-cast differences."""

    def __init__(self, p: float = 0.3, gain: float = 0.12):
        self.p, self.gain = p, gain

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return im
        g = np.array([random.uniform(1 - self.gain, 1 + self.gain) for _ in range(3)], dtype=np.float32)
        a = np.asarray(im, dtype=np.float32) * g
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))


class RandomShadow:
    """Soft dark polygon: leaf/structure shadows in field light."""

    def __init__(self, p: float = 0.2, strength: tuple[float, float] = (0.35, 0.7)):
        self.p, self.strength = p, strength

    def __call__(self, im: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return im
        w, h = im.size
        mask = Image.new("L", im.size, 0)
        pts = [(random.randint(0, w), random.randint(0, h)) for _ in range(random.randint(3, 6))]
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) / 40))
        k = random.uniform(*self.strength)
        dark = Image.eval(im, lambda v: int(v * k))
        return Image.composite(dark, im, mask)


def train_transforms(size: int = IMAGE_SIZE, realworld: bool = False) -> transforms.Compose:
    photo = [RandomLowRes(), RandomJpeg(), RandomWhiteBalance(), RandomShadow()] if realworld else []
    return transforms.Compose(
        [
            *photo,
            transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR, fill=FILL),
            Letterbox(),
            transforms.RandomResizedCrop(
                size, scale=(0.45, 1.0), ratio=(0.8, 1.25),
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Mild colour jitter only: hue/yellowing is itself a symptom.
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), value=0),
        ]
    )


def load_rgb(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im).convert("RGB")


class CoralHead(nn.Module):
    """P(grade > k) = sigmoid(w.x + b_k), k = 0..K-2. Shared w keeps them ordered."""

    def __init__(self, in_features: int, num_grades: int = NUM_GRADES):
        super().__init__()
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.arange(num_grades - 1, 0, -1).float() / (num_grades - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias


class MultiTaskNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3, num_classes: int = len(CLASS_ORDER)):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        dim = base.classifier[1].in_features
        self.cls_head = nn.Linear(dim, num_classes)
        self.presence_head = nn.Linear(dim, len(DISEASES))
        self.sev_head = CoralHead(dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.embed(x)
        h = self.dropout(feat)
        return {
            "feat": feat,
            "cls": self.cls_head(h),
            "presence": self.presence_head(h),
            "sev": self.sev_head(h),
        }


def coral_to_probs(sev_logits: torch.Tensor) -> torch.Tensor:
    """Cumulative P(y>k) -> per-grade probabilities (monotone by construction)."""
    cum = torch.sigmoid(sev_logits)
    ones = torch.ones_like(cum[:, :1])
    zeros = torch.zeros_like(cum[:, :1])
    upper = torch.cat([ones, cum], dim=1)
    lower = torch.cat([cum, zeros], dim=1)
    return (upper - lower).clamp_min(0)


def coral_loss(sev_logits: torch.Tensor, grades: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE on the K-1 cumulative tasks. mask (N, K-1) marks which tasks are known.

    Fully graded images know every task. Augmented diseased images are only known
    to be grade >= 1, so only task 0 is supervised for them.
    """
    k = torch.arange(sev_logits.shape[1], device=sev_logits.device)
    target = (grades.unsqueeze(1) > k).float()
    loss = F.binary_cross_entropy_with_logits(sev_logits, target, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom
