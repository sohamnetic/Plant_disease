"""Balanced dataset built only from VegNet's own original photos.

The publisher's "Augmented Dataset" folder is NOT used. Instead every class in the
train split is brought to the same size (TARGET images = its originals + our own
augmented copies of those originals). Val and test keep their original photos only,
unaugmented, so scores on them stay honest. The split (splits/train|val|test.csv,
grouped so near-duplicates never cross splits) is reused as-is.

Augmentation per copy (the photo-level steps of src/v2_model.py train_transforms,
written to disk instead of applied on the fly):
  low-res p0.25 (140-320 px then back up), white balance p0.3 (+-12% per channel),
  shadow p0.2, rotation +-15 deg (then cropped to the largest rectangle with no empty
  corners), random crop 45-100% of the area at the photo's own
  aspect x0.8-1.25, horizontal / vertical flip p0.5 each, mild colour jitter
  (brightness / contrast 0.25, saturation 0.2, hue 0.02), JPEG q25-80 p0.3.
Not used here, because they are training-time only: random erasing, CutMix, scene
background swap. Copies are spread evenly over a class's originals.

Originals are copied byte-for-byte. Augmented copies are made from the original
downscaled to a 1024 px long side and saved as JPEG q95. Each copy has its own seed,
so the build is reproducible.

  python src/build_balanced_dataset.py
Output: dataset_balanced/{train,val,test}/{class}/   original photos only
        dataset_balanced/augmented/{class}/          all augmented copies (made from train originals)
        dataset_balanced/manifest.csv, counts.csv, samples.jpg
"""

from __future__ import annotations

import math
import random
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from v2_model import CLASS_ORDER, RandomJpeg, RandomLowRes, RandomShadow, RandomWhiteBalance

ROOT = cfg.PROJECT_ROOT
OUT = ROOT / "dataset_balanced"
TARGET = 700  # train images per class, originals included
LONG_SIDE = 1024
SEED = 42


def pre(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    k = LONG_SIDE / max(im.size)
    return im.resize((round(im.width * k), round(im.height * k)), Image.Resampling.LANCZOS) if k < 1 else im


def random_crop(im: Image.Image) -> Image.Image:
    """Crop 45-100% of the area, aspect within x0.8-1.25 of the photo's own."""
    w, h = im.size
    for _ in range(10):
        area = random.uniform(0.45, 1.0) * w * h
        ar = (w / h) * np.exp(random.uniform(np.log(0.8), np.log(1.25)))
        cw, ch = round((area * ar) ** 0.5), round((area / ar) ** 0.5)
        if 0 < cw <= w and 0 < ch <= h:
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            return im.crop((x, y, x + cw, y + ch))
    return im


def rotate(im: Image.Image) -> Image.Image:
    """Rotate +-15 deg and keep the largest centred rectangle that has no empty corners."""
    a = math.radians(random.uniform(-15, 15))
    w, h = im.size
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))
    long_, short = max(w, h), min(w, h)
    if short <= 2 * sin_a * cos_a * long_ or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short
        cw, ch = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        cw, ch = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    im = im.rotate(math.degrees(a), Image.Resampling.BILINEAR)
    cw, ch = min(w, int(cw)), min(h, int(ch))
    x, y = (w - cw) // 2, (h - ch) // 2
    return im.crop((x, y, x + cw, y + ch))


STEPS = [RandomLowRes(), RandomWhiteBalance(), RandomShadow()]
JITTER = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02)
JPEG = RandomJpeg()


def augment(im: Image.Image, seed: int) -> Image.Image:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    for t in STEPS:
        im = t(im)
    im = random_crop(rotate(im))
    if random.random() < 0.5:
        im = ImageOps.mirror(im)
    if random.random() < 0.5:
        im = ImageOps.flip(im)
    return JPEG(JITTER(im))


def work(task: tuple) -> list[dict]:
    src, cls, split, n_aug, seeds = task
    src = ROOT / src
    dst_dir = OUT / split / cls
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    rows = [{"filepath": dst.relative_to(ROOT).as_posix(), "split": split, "class": cls, "kind": "original",
             "parent": src.relative_to(ROOT).as_posix(), "seed": ""}]
    if n_aug:
        aug_dir = OUT / "augmented" / cls
        with Image.open(src) as im:
            base = pre(im)
        for k, s in enumerate(seeds):
            out = aug_dir / f"{src.stem}_{src.suffix[1:].lower()}__aug{k + 1:02d}.jpg"  # .jpg and .jpeg twins exist
            augment(base, s).save(out, "JPEG", quality=95)
            rows.append({"filepath": out.relative_to(ROOT).as_posix(), "split": split, "class": cls, "kind": "augmented",
                         "parent": src.relative_to(ROOT).as_posix(), "seed": s})
    return rows


def sample_sheet(man: pd.DataFrame) -> None:
    rng = random.Random(SEED)
    cell, cols = 220, 6
    tr = man[man["split"] == "train"]
    sheet = Image.new("RGB", (cell * cols, cell * len(CLASS_ORDER)), "white")
    for r, c in enumerate(CLASS_ORDER):
        parents = tr[(tr["class"] == c) & (tr["kind"] == "original")]["parent"].tolist()
        p = rng.choice(parents)
        files = [tr[(tr["parent"] == p) & (tr["kind"] == "original")]["filepath"].iloc[0]]
        files += tr[(tr["parent"] == p) & (tr["kind"] == "augmented")]["filepath"].head(cols - 1).tolist()
        for j, f in enumerate(files):
            with Image.open(ROOT / f) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((cell - 8, cell - 8))
            sheet.paste(im, (j * cell + (cell - im.width) // 2, r * cell + (cell - im.height) // 2))
    sheet.save(OUT / "samples.jpg", quality=90)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    tasks, rng = [], np.random.default_rng(SEED)
    for split in ("train", "val", "test"):
        df = pd.read_csv(cfg.SPLITS_DIR / f"{split}.csv")
        df = df[df["source_type"] == "original"].sort_values("filepath").reset_index(drop=True)
        for cls in CLASS_ORDER:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)
            if split == "train":
                (OUT / "augmented" / cls).mkdir(parents=True, exist_ok=True)
            files = df[df["class"] == cls]["filepath"].tolist()
            n_extra = TARGET - len(files) if split == "train" else 0
            per = [n_extra // len(files) + (1 if i < n_extra % len(files) else 0) for i in range(len(files))]
            for f, k in zip(files, per):
                tasks.append((f, cls, split, k, [int(s) for s in rng.integers(0, 2**31 - 1, k)]))
    with Pool(6) as pool:
        rows = [r for rs in pool.imap(work, tasks, chunksize=4) for r in rs]
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    counts = man.pivot_table(index=["split", "kind"], columns="class", values="filepath", aggfunc="count", fill_value=0)
    counts["total"] = counts.sum(1)
    counts.to_csv(OUT / "counts.csv")
    sample_sheet(man)
    print(counts.to_string())
    print("total images", len(man))


if __name__ == "__main__":
    main()
