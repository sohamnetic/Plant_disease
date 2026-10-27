"""Step I-2: YOLO-seg dataset = the Phase 3 set (leaf / curd / lesion) + reviewed
insect-hole outlines (class 3 "insect_hole").

  * Phase 3 images and labels are copied unchanged (same train / val split).
  * Insect photos reviewed "K" (outlines usable, outputs/insect_seg/review.csv):
      insect_train -> train, insect_test -> val (held out, never trained on)
    labels: the leaf outline (class 0, holes included: a leaf polygon has no holes,
    like every other leaf in the set) and one polygon per hole (class 3).
  * Every insect photo is a leaf cut out onto white, so its holes only ever show
    white / pale grey. For each training photo N_COMPOSITES extra copies put the
    leaf on a random symptom-free background (soil, foliage, colour) with the
    holes see-through, so holes in field photos (soil or leaves behind) are learnt.
  * Extra rounds (bootstrap) add photos listed in outputs/insect_seg/round<k>_picks.csv
    with their reviewed masks in cache/insect_prelabel/round<k>/.

  python src/insect_seg_dataset.py                 # -> data_external/seg_yolo_insect/
  python src/insect_seg_dataset.py --round 2 --out data_external/seg_yolo_insect_r2   # + round-2 reviewed photos
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from insect_prelabel import CACHE, cache_path, load_work
from scene_aug import SceneComposer
from seg_dataset import polygons

SRC = cfg.PROJECT_ROOT / "data_external" / "seg_yolo"
OUT = cfg.PROJECT_ROOT / "data_external" / "seg_yolo_insect"
REVIEW = cfg.OUTPUTS_DIR / "insect_seg" / "review.csv"
BG_DIR = cfg.PROJECT_ROOT / "data_external" / "derived" / "backgrounds"
CLASSES = ["leaf", "curd", "lesion", "insect_hole"]
MIN_AREA = {"leaf": 0.01, "insect_hole": 0.0002}
N_COMPOSITES = 2
SEED = 42


def label_lines(leaf: np.ndarray, hole: np.ndarray) -> list[str]:
    lines = []
    for k, (name, m) in ((0, ("leaf", leaf | hole)), (3, ("insect_hole", hole))):
        for poly in polygons(m, MIN_AREA[name]):
            lines.append(f"{k} " + " ".join(f"{v:.5f}" for v in poly.ravel()))
    return lines


def composite(rgb: np.ndarray, leaf: np.ndarray, scene: SceneComposer) -> np.ndarray:
    """Leaf tissue over a random background; backdrop and holes become background."""
    im = Image.fromarray(rgb)
    alpha = Image.fromarray(leaf.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.0))
    bg = scene.background(im.size)
    bg.paste(im, (0, 0), alpha)
    return np.asarray(bg)


def write(split: str, name: str, rgb: np.ndarray, lines: list[str]) -> None:
    Image.fromarray(rgb).save(OUT / "images" / split / f"{name}.jpg", quality=95)
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def insect_rows(round_k: int) -> pd.DataFrame:
    rev = pd.read_csv(REVIEW)
    rows = rev[rev["review"] == "K"].assign(mask_dir="")
    for k in range(2, round_k + 1):
        extra = pd.read_csv(cfg.OUTPUTS_DIR / "insect_seg" / f"round{k}_picks.csv")
        extra = extra[extra["review"] == "K"].assign(mask_dir=f"round{k}")
        rows = pd.concat([rows[~rows["filepath"].isin(extra["filepath"])], extra], ignore_index=True)
    return rows


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out).resolve()
    random.seed(SEED)
    np.random.seed(SEED)
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (OUT / sub / split).mkdir(parents=True)
            for f in (SRC / sub / split).iterdir():
                shutil.copy2(f, OUT / sub / split / f.name)

    scene = SceneComposer(BG_DIR)
    rows = insect_rows(args.round)
    counts = {"train": 0, "val": 0, "composites": 0, "holes": 0}
    for _, r in rows.iterrows():
        split = "val" if r["split"] == "insect_test" else "train"
        path = (CACHE / r["mask_dir"] / cache_path(r["filepath"]).name) if r["mask_dir"] else cache_path(r["filepath"])
        m = np.load(path)
        leaf, hole = m["leaf"], m["hole"]
        rgb = load_work(r["filepath"])
        lines = label_lines(leaf, hole)
        name = f"ins{int(r['idx']):04d}_cleaf_insect_hole"
        write(split, name, rgb, lines)
        counts[split] += 1
        counts["holes"] += sum(1 for line in lines if line.startswith("3 "))
        if split == "train":
            for c in range(N_COMPOSITES):
                write(split, f"{name}_bg{c}", composite(rgb, leaf, scene), lines)
                counts["composites"] += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"-> {OUT}: insect photos train {counts['train']} (+{counts['composites']} background composites), "
          f"val {counts['val']}; hole instances {counts['holes']}")
    for split in ("train", "val"):
        print(f"  {split}: {sum(1 for _ in (OUT / 'images' / split).glob('*.jpg'))} images")


if __name__ == "__main__":
    main()
