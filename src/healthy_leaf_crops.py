"""Cut healthy-leaf patches out of our own no_disease photos.

Almost every no_disease original is a curd surrounded by wrapper leaves, so the
model never sees a healthy leaf on its own and calls leafy photos downy mildew.
This finds square windows that are mostly green leaf and contain no curd, and
saves them as extra no_disease images. Crops of train/val photos are training
data; crops of test photos (parent_split == "test") are only an evaluation set
for healthy-leaf false alarms and must never be trained on.

  python src/healthy_leaf_crops.py
Outputs: data_external/derived/healthy_leaf_crops/*.jpg + manifest.csv, and a
contact sheet for review in outputs/review/healthy_leaf_crops.png
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
from v2_cache import cache_path_for

OUT_DIR = cfg.PROJECT_ROOT / "data_external" / "derived" / "healthy_leaf_crops"
SHEET = cfg.REVIEW_DIR / "healthy_leaf_crops.png"
WINDOW_FRACS = (0.55, 0.45, 0.35)
MIN_GREEN = 0.75
MAX_CURD = 0.01
MAX_PER_IMAGE = 2


def masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    # Cream / yellow-green curd sits at hue 30-75, so "leaf green" starts at 75.
    green = (h >= 75) & (h <= 220) & (s >= 0.12) & (v >= 0.12)
    curd = (v >= 0.50) & ((s <= 0.35) | ((h >= 30) & (h < 75))) & ~green
    return green, curd


def best_windows(rgb: np.ndarray) -> list[tuple[int, int, int, float]]:
    green, curd = masks(rgb)
    H, W = green.shape
    gi = cv2.integral(green.astype(np.uint8))
    ci = cv2.integral(curd.astype(np.uint8))

    def frac(ii: np.ndarray, x: int, y: int, s: int) -> float:
        return (ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]) / (s * s)

    picked: list[tuple[int, int, int, float]] = []
    for wf in WINDOW_FRACS:
        s = int(min(H, W) * wf)
        step = max(s // 4, 8)
        cands = []
        for y in range(0, H - s + 1, step):
            for x in range(0, W - s + 1, step):
                g, c = frac(gi, x, y, s), frac(ci, x, y, s)
                if g >= MIN_GREEN and c <= MAX_CURD:
                    cands.append((g, x, y, s))
        for g, x, y, s in sorted(cands, reverse=True):
            if len(picked) >= MAX_PER_IMAGE:
                return picked
            # keep crops that overlap previous picks by < 30%
            ok = True
            for px, py, ps, _ in picked:
                ix = max(0, min(x + s, px + ps) - max(x, px))
                iy = max(0, min(y + s, py + ps) - max(y, py))
                if ix * iy > 0.3 * min(s * s, ps * ps):
                    ok = False
                    break
            if ok:
                picked.append((x, y, s, g))
    return picked


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(cfg.SPLITS_DIR / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")])
    df = df[(df["class"] == "no_disease") & (df["source_type"] == "original")]
    rows, thumbs = [], []
    for _, r in df.iterrows():
        im = Image.open(cache_path_for(r["filepath"])).convert("RGB")
        rgb = np.asarray(im)
        for k, (x, y, s, g) in enumerate(best_windows(rgb)):
            crop = im.crop((x, y, x + s, y + s))
            if s < 160:
                continue
            name = f"{Path(r['filepath']).name}_leaf{k}.jpg".replace(" ", "_")  # .name: (1).jpg and (1).jpeg are different photos
            crop.save(OUT_DIR / name, quality=95)
            rows.append({
                "filepath": str((OUT_DIR / name).relative_to(cfg.PROJECT_ROOT)).replace("\\", "/"),
                "class": "no_disease", "source": "healthy_leaf_crop", "parent_filepath": r["filepath"],
                "parent_group_id": r["group_id"], "parent_split": r["split"], "green_frac": round(g, 3), "side": s,
            })
            if r["split"] != "test":
                thumbs.append(crop.resize((140, 140)))
    pd.DataFrame(rows).to_csv(OUT_DIR / "manifest.csv", index=False)
    cols = 12
    sheet = Image.new("RGB", (cols * 140, ((len(thumbs) + cols - 1) // cols) * 140), "white")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * 140, (i // cols) * 140))
    sheet.save(SHEET)
    m = pd.DataFrame(rows)
    print(f"{len(m)} healthy leaf crops -> {OUT_DIR}")
    print(m["parent_split"].value_counts().to_string())


if __name__ == "__main__":
    main()
