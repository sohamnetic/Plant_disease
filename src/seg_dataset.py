"""Phase 3, step 2: reviewed pre-labels -> YOLO-seg dataset.

Reads outputs/seg/label_set.csv + outputs/seg/review_picks.txt (A-D = chosen
candidate, X = rejected), loads the chosen candidate masks from
cache/seg_prelabel/, and writes polygons in YOLO format:

  data_external/seg_yolo/{images,labels}/{train,val}/  + data.yaml
  classes: 0 leaf, 1 curd, 2 lesion

Every connected component becomes one instance. Tiny specks are dropped.
seg_split "eval" -> val (held out, never trained on).

  python src/seg_dataset.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from seg_prelabel import LABEL_SET, SEG_DIR, load_candidate, load_work

PICKS = SEG_DIR / "review_picks.txt"
OUT = cfg.PROJECT_ROOT / "data_external" / "seg_yolo"
CLASSES = ["leaf", "curd", "lesion"]
MIN_AREA_FRAC = {"leaf": 0.01, "curd": 0.01, "lesion": 0.0002}


def load_picks() -> pd.DataFrame:
    rows = [line.split() for line in PICKS.read_text().splitlines() if line.strip() and not line.startswith("#")]
    p = pd.DataFrame(rows, columns=["idx", "pick"]).astype({"idx": int})
    ls = pd.read_csv(LABEL_SET).merge(p, on="idx", how="left")
    if ls["pick"].isna().any():
        raise RuntimeError(f"unreviewed images: {ls.loc[ls['pick'].isna(), 'idx'].tolist()[:10]}")
    return ls[ls["pick"] != "X"].reset_index(drop=True)


def polygons(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """One polygon per connected component, normalised to 0-1."""
    h, w = mask.shape
    out = []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area * h * w or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(np.float32)
        if len(c) < 3:
            continue
        c[:, 0] /= w
        c[:, 1] /= h
        out.append(c.clip(0, 1))
    return out


def main() -> None:
    kept = load_picks()
    if OUT.exists():
        shutil.rmtree(OUT)
    counts = {s: {c: 0 for c in CLASSES} for s in ("train", "val")}
    for _, r in kept.iterrows():
        split = "val" if r["seg_split"] == "eval" else "train"
        cand = load_candidate(r["filepath"], r["pick"])
        rgb = load_work(r["filepath"])
        name = f"{r['idx']:04d}_{r['source']}_{r['class']}"
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(OUT / "images" / split / f"{name}.jpg", quality=95)
        lines = []
        for k, cls in enumerate(CLASSES):
            m = cand[cls] if cls != "lesion" else cand["lesion"] & (cand["leaf"] | cand["curd"])
            for poly in polygons(m, MIN_AREA_FRAC[cls]):
                lines.append(f"{k} " + " ".join(f"{v:.5f}" for v in poly.ravel()))
                counts[split][cls] += 1
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    kept.to_csv(SEG_DIR / "labelled_images.csv", index=False)
    print(f"{len(kept)} images -> {OUT}")
    for s, c in counts.items():
        print(f"  {s}: {sum(1 for _ in (OUT / 'images' / s).glob('*.jpg'))} images, instances {c}")


if __name__ == "__main__":
    main()
