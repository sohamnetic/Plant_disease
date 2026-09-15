"""Original-only leaf-disease dataset and transforms (experiment 001)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

import config as cfg

CLASS_TO_IDX = {c: i for i, c in enumerate(cfg.CLASS_ORDER)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

EXPECTED_ORIGINAL_COUNTS = {
    "train": {
        "bacterial_spot_rot": 121,
        "black_rot": 44,
        "downy_mildew": 123,
        "no_disease": 144,
    },
    "val": {
        "bacterial_spot_rot": 26,
        "black_rot": 9,
        "downy_mildew": 27,
        "no_disease": 31,
    },
}


def load_full_split(split_name: str) -> pd.DataFrame:
    """Load a frozen split with all source types. Never reads test.csv."""
    if split_name == "test":
        raise RuntimeError("test.csv must not be used for this experiment.")
    path = cfg.SPLITS_DIR / f"{split_name}.csv"
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Empty split: {path}")
    return df.reset_index(drop=True)


def load_split(split_name: str, source_type: str) -> pd.DataFrame:
    """Load a frozen split filtered by source_type. Never reads test.csv."""
    if split_name == "test":
        raise RuntimeError("test.csv must not be used for this experiment.")
    if source_type not in {"original", "augmented"}:
        raise ValueError(source_type)
    path = cfg.SPLITS_DIR / f"{split_name}.csv"
    df = pd.read_csv(path)
    out = df[df["source_type"] == source_type].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No {source_type} rows in {path}")
    return out


def load_original_split(split_name: str) -> pd.DataFrame:
    orig = load_split(split_name, "original")
    counts = (
        orig["class"].value_counts().reindex(cfg.CLASS_ORDER).fillna(0).astype(int).to_dict()
    )
    expected = EXPECTED_ORIGINAL_COUNTS[split_name]
    if counts != expected:
        raise RuntimeError(
            f"Original-only {split_name} counts {counts} != expected {expected}. "
            "Splits were not supposed to be regenerated."
        )
    return orig


def build_transforms(train: bool) -> transforms.Compose:
    """Train augs avoid black-triangle pads: rotate on the full image, then crop."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomRotation(
                    degrees=18,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=tuple(int(m * 255) for m in cfg.IMAGENET_MEAN),
                ),
                transforms.RandomResizedCrop(
                    cfg.IMAGE_SIZE,
                    scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.25, hue=0.0
                ),
                transforms.ToTensor(),
                transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(cfg.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
        ]
    )


class LeafImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_tfms: transforms.Compose):
        self.frame = frame.reset_index(drop=True)
        self.tfms = image_tfms
        self.root = cfg.PROJECT_ROOT

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = self.root / row["filepath"]
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            tensor = self.tfms(im)
        label = CLASS_TO_IDX[row["class"]]
        return tensor, label, row["filepath"]
