"""Collect symptom-free background patches for background-swap augmentation.

Black rot was learned partly as "leaf on cracked red soil": with the soil cut
away, black-rot accuracy fell from 100% to 43%. To break that link, training
pastes cut-out plants onto random backgrounds, so soil appears behind every
class. Backgrounds must never contain disease, so they come from:

  * bare soil: windows of train/val black-rot photos (clean blue-green leaf on
    soil) with no plant tissue within a safety margin. Downy-mildew photos are
    not used: their dead, bleached tissue is missed by the plant mask.
  * healthy foliage: healthy-leaf crops (src/healthy_leaf_crops.py, train/val)

Test photos are never used.

  python src/background_bank.py
Outputs: data_external/derived/backgrounds/*.jpg, outputs/review/background_bank.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from severity_prefill import estimate_leaf
from v2_cache import cache_path_for

OUT_DIR = cfg.PROJECT_ROOT / "data_external" / "derived" / "backgrounds"
LEAF_CROPS = cfg.PROJECT_ROOT / "data_external" / "derived" / "healthy_leaf_crops" / "manifest.csv"
SHEET = cfg.REVIEW_DIR / "background_bank.png"
MAX_PLANT_FRAC = 0.005


def soil_windows(rgb: np.ndarray) -> list[tuple[int, int, int]]:
    _, plant, lesion = estimate_leaf(rgb)
    occupied = cv2.dilate((plant | lesion).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    H, W = occupied.shape
    ii = np.pad(occupied.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    out = []
    for frac in (0.4, 0.3):
        s = int(min(H, W) * frac)
        step = max(s // 3, 8)
        for y in range(0, H - s + 1, step):
            for x in range(0, W - s + 1, step):
                occ = (ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]) / (s * s)
                if occ <= MAX_PLANT_FRAC and all(abs(x - a) + abs(y - b) > s for a, b, _ in out):
                    out.append((x, y, s))
        if out:
            break
    return out[:2]


def leaf_frac(im: Image.Image) -> float:
    """Share of leaf-coloured pixels (blue-green black-rot leaves are low-saturation)."""
    hsv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0
    return float(((h >= 70) & (h <= 230) & (s >= 0.08)).mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(cfg.SPLITS_DIR / f"{s}.csv") for s in ("train", "val")])
    df = df[(df["source_type"] == "original") & (df["class"] == "black_rot")]
    thumbs, n_soil = [], 0
    for fp in df["filepath"]:
        im = Image.open(cache_path_for(fp)).convert("RGB")
        small = im.copy()
        small.thumbnail((320, 320))
        k = im.width / small.width
        for i, (x, y, s) in enumerate(soil_windows(np.asarray(small))):
            box = tuple(int(v * k) for v in (x, y, x + s, y + s))
            crop = im.crop(box)
            if crop.width < 120 or leaf_frac(crop) > 0.03:
                continue
            crop.save(OUT_DIR / f"soil_{Path(fp).stem.replace(' ', '_')}_{i}.jpg", quality=92)
            thumbs.append(crop.resize((110, 110)))
            n_soil += 1
    n_leaf = 0
    if LEAF_CROPS.is_file():
        crops = pd.read_csv(LEAF_CROPS)
        for fp in crops.loc[crops["parent_split"] != "test", "filepath"]:
            im = Image.open(cfg.PROJECT_ROOT / fp).convert("RGB")
            im.save(OUT_DIR / f"foliage_{Path(fp).name}", quality=92)
            n_leaf += 1
    cols = 14
    sheet = Image.new("RGB", (cols * 110, max(1, (len(thumbs) + cols - 1) // cols) * 110), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 110, (i // cols) * 110))
    sheet.save(SHEET)
    print(f"{n_soil} soil + {n_leaf} foliage backgrounds -> {OUT_DIR}")


if __name__ == "__main__":
    main()
