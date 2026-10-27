"""Ingest external image sources and find copies of our own photos.

Several public "cauliflower" datasets are re-packaged VegNet photos (background
cut out, resized, flipped, rotated). pHash misses those, and whole-image
embedding similarity cannot separate "same photo, background removed" from
"another photo of a similar plant". So matching is two-step:

  1. embedding shortlist: 20 nearest of our 4281 images (ImageNet EfficientNet-B0)
  2. geometric check: SIFT keypoints + RANSAC homography against each shortlisted
     photo (robust to crop, resize, rotation, flips and background removal).
     A real copy shares >= DUP_INLIERS inlier points (observed 48-750); a
     different photo of a similar plant usually shares < 10. Vein-rich single
     leaves can reach 25-48 by chance (22 of 2661 in "cleaf"); the threshold
     deliberately errs that way: a false copy is only dropped, a missed copy
     would leak a test photo into training.

Each external image gets nn_filepath / nn_split / nn_group_id / match_inliers and
`is_copy`. Copies inherit the split and group of the photo they copy, so a copy
of a test photo is never trained on.

  python src/external_data.py --source c2h
  python src/external_data.py --source all
Outputs: data_external/manifests/<source>.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from v2_cache import cache_path_for
from v2_model import eval_transforms

RAW = cfg.PROJECT_ROOT / "data_external" / "raw"
MANIFEST_DIR = cfg.PROJECT_ROOT / "data_external" / "manifests"
EMB_CACHE = cfg.PROJECT_ROOT / "cache" / "ours_emb.npz"
DUP_INLIERS = 25
SHORTLIST = 20

# Cauliflower sources only. source -> (root, {folder name: class}); unmapped folders are skipped. Classes
# outside our 4 are kept in the manifest for the record; build_v3_data drops them.
SOURCES: dict[str, tuple[Path, dict[str, str]]] = {
    "c2h": (
        RAW / "c2hrwntzgh" / "Original Cauliflower Dataset" / "Original Cauliflower Dataset",
        {"Bacterial Spot Rot": "bacterial_spot_rot", "Black Rot": "black_rot",
         "Downy Mildew": "downy_mildew", "Healthy Fruit": "no_disease"},
    ),
    "cleaf": (
        RAW / "x995snz7p3" / "Processed Image",
        {"Black Rot": "black_rot", "Healthy": "no_disease", "Insect Hole": "insect_hole"},
    ),
    "vegleaf": (
        RAW / "rh8c5tgg5k" / "Cauliflower Healthy leaf",
        {"Cauliflower Healthy leaf": "no_disease"},
    ),
    "sppu": (
        RAW / "x26px3xnmy" / "Cauliflower Dataset",
        {"Bacterial Spot": "bacterial_spot_rot", "Black Spot": "black_spot_curd", "Healthy": "no_disease",
         "Downy Mildew": "downy_mildew", "Alternaria Brassicae": "alternaria",
         "Bacterial Soft Rot": "bacterial_soft_rot", "Purple Tinges": "purple_tinge",
         "Alternaria Leaf Spot": "alternaria"},
    ),
}


def _backbone(device: torch.device):
    return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features.to(device).eval()




@torch.inference_mode()
def _embed(model, ims: list[Image.Image], device) -> np.ndarray:
    tfm = eval_transforms(224)
    out = []
    for i in range(0, len(ims), 64):
        x = torch.stack([tfm(im) for im in ims[i : i + 64]]).to(device)
        out.append(torch.nn.functional.normalize(model(x).mean((2, 3)), dim=1).cpu().numpy())
    return np.concatenate(out)


def ours(device) -> tuple[pd.DataFrame, np.ndarray]:
    inv = pd.concat([pd.read_csv(cfg.SPLITS_DIR / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")])
    inv = inv.reset_index(drop=True)
    if EMB_CACHE.is_file():
        z = np.load(EMB_CACHE, allow_pickle=True)
        if list(z["filepath"]) == inv["filepath"].tolist():
            return inv, z["emb"]
    ims = [Image.open(cache_path_for(f)).convert("RGB") for f in inv["filepath"]]
    emb = _embed(_backbone(device), ims, device)
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_CACHE, filepath=inv["filepath"].to_numpy(), emb=emb)
    return inv, emb




_SIFT = cv2.SIFT_create(nfeatures=1500)
_BF = cv2.BFMatcher()


def sift_feats(im: Image.Image):
    g = np.asarray(im.convert("L"))
    scale = 400 / max(g.shape)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return _SIFT.detectAndCompute(g, None)


def inliers(a, b) -> int:
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in (p for p in _BF.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    for flip in (False, True):  # homography cannot express a mirror; try the mirrored query too
        pts = src.copy()
        if flip:
            pts[:, 0] = -pts[:, 0]
        _, mask = cv2.findHomography(pts, dst, cv2.RANSAC, 8.0)
        if mask is not None and mask.sum() >= DUP_INLIERS:
            return int(mask.sum())
    return int(mask.sum()) if mask is not None else 0


def ingest(name: str, device) -> pd.DataFrame:
    root, cmap = SOURCES[name]
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in cfg.IMAGE_EXTENSIONS]
    files = [p for p in files if p.parent.name.strip() in cmap]
    inv, emb = ours(device)
    ims, keep, widths, heights = [], [], {}, {}
    for p in files:
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                widths[p], heights[p] = im.size
                im.thumbnail((640, 640))  # matching needs <= 400 px; 12 MP originals exhaust RAM
                ims.append(im)
            keep.append(p)
        except Exception:
            print("unreadable:", p)
    e = _embed(_backbone(device), ims, device)
    sims = e @ emb.T
    rows = []
    our_feats: dict[int, tuple] = {}
    for i, (p, im) in enumerate(zip(keep, ims)):
        fa = sift_feats(im)
        top = np.argsort(-sims[i])[:SHORTLIST]
        scores = []
        for j in top:
            if j not in our_feats:
                our_feats[j] = sift_feats(Image.open(cache_path_for(inv.at[j, "filepath"])))
            scores.append(inliers(fa, our_feats[j]))
        b = int(np.argmax(scores))
        j = top[b]
        rows.append({
            "filepath": str(p.relative_to(cfg.PROJECT_ROOT)).replace("\\", "/"),
            "source": name,
            "source_folder": p.parent.name.strip(),
            "class": cmap[p.parent.name.strip()],
            "width": widths[p],
            "height": heights[p],
            "emb_sim": round(float(sims[i, j]), 4),
            "match_inliers": scores[b],
            "nn_filepath": inv.at[j, "filepath"],
            "nn_class": inv.at[j, "class"],
            "nn_split": inv.at[j, "split"],
            "nn_group_id": inv.at[j, "group_id"],
            "nn_source_type": inv.at[j, "source_type"],
        })
    df = pd.DataFrame(rows)
    df["is_copy"] = df["match_inliers"] >= DUP_INLIERS
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_DIR / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} images, copies of our photos: {int(df.is_copy.sum())}")
    print(df.groupby("class")["is_copy"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "copies"}))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all", choices=["all", *SOURCES])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in (SOURCES if args.source == "all" else [args.source]):
        if SOURCES[name][0].exists():
            ingest(name, device)
        else:
            print(f"{name}: not downloaded, skipped")


if __name__ == "__main__":
    main()
