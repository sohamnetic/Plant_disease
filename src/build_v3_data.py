"""Assemble the v3 training pool and the frozen EXTERNAL test set.

Inputs
  splits/{train,val,test}.csv                    frozen VegNet split (unchanged)
  data_external/manifests/<source>.csv           from src/external_data.py
  data_external/derived/healthy_leaf_crops/      from src/healthy_leaf_crops.py

Routing rules (no image or near-copy of a test photo is ever trained on):
  * our train/val rows                       -> train pool
  * healthy-leaf crops of train/val photos   -> train pool (parent's group)
  * healthy-leaf crops of test photos        -> external test, subset "healthy_leaf_crops"
  * c2h copies of our photos (background removed):
        copy of a train/val photo            -> train pool (same group as the copied photo)
        copy of a test photo                 -> external test, subset "bg_removed_copies"
  * copies of our photos from other sources  -> dropped (no new information)
  * genuinely new external images: grouped by SIFT near-duplicates inside each
    source, then EXTERNAL_TEST_FRAC of the groups (per source x class) go to
    the external test, the rest to the train pool
  * EXCLUDED folders (labels that contradict ours) and images in
    outputs/review/external_review_excluded.csv   -> dropped

  python src/build_v3_data.py
Outputs: splits_v3/train_pool.csv, splits_v3/external_test.csv, splits_v3/SUMMARY.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from external_data import DUP_INLIERS, MANIFEST_DIR, SHORTLIST, SOURCES, _backbone, _embed, inliers, sift_feats
from v2_model import CLASS_ORDER

OUT = cfg.PROJECT_ROOT / "splits_v3"
LEAF_CROPS = cfg.PROJECT_ROOT / "data_external" / "derived" / "healthy_leaf_crops" / "manifest.csv"
EXTERNAL_TEST_FRAC = 0.35
SEED = 42
REVIEW_EXCLUDED = cfg.REVIEW_DIR / "external_review_excluded.csv"
# Source folders whose labels contradict ours or cannot be verified. Evidence from
# copy detection (copies of our photos, and what WE label them):
EXCLUDED = {
    ("sppu", "Alternaria Leaf Spot"): "182/182 checkable images are our downy_mildew photos relabelled",
    ("sppu", "Bacterial Soft Rot"): "its checkable images are our bacterial_spot_rot photos relabelled",
    ("sppu", "Black Spot"): "black spots on the curd: not the leaf black rot we label",
    ("sppu", "Purple Tinges"): "includes plain white heads; VegNet labels purple curds bacterial_spot_rot",
    ("sppu", "Alternaria Brassicae"): "45 augmented copies of one single photo",
    ("cleaf", "Insect Hole"): "pest damage: not one of our 4 classes",
}
COLS = ["filepath", "class", "source", "source_type", "group_id", "subset"]


def group_new_images(df: pd.DataFrame, device) -> pd.Series:
    """Union-find over SIFT near-duplicates within each source x class.

    All-pairs SIFT is quadratic (a 1,000-image class = 500k matches), so each
    image is only checked against its SHORTLIST nearest neighbours by embedding.
    """
    parent = {i: i for i in df.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    model = _backbone(device)
    for (_, _), g in df.groupby(["source", "class"]):
        idx = list(g.index)
        ims = []
        for i in idx:
            with Image.open(cfg.PROJECT_ROOT / df.at[i, "filepath"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((640, 640))
                ims.append(im)
        feats = [sift_feats(im) for im in ims]
        emb = _embed(model, ims, device)
        sims = emb @ emb.T
        for a in range(len(idx)):
            for b in np.argsort(-sims[a])[1 : SHORTLIST + 1]:
                ia, ib = idx[a], idx[int(b)]
                if find(ia) != find(ib) and inliers(feats[a], feats[int(b)]) >= DUP_INLIERS:
                    parent[find(ia)] = find(ib)
    return pd.Series({i: f"ext_{df.at[find(i), 'source']}_{find(i)}" for i in df.index})


def main() -> None:
    rng = np.random.default_rng(SEED)
    parts_train, parts_test = [], []
    # Images a visual review found not clearly matching their label (see REVIEW_EXCLUDED).
    review_excluded = set(pd.read_csv(REVIEW_EXCLUDED)["filepath"]) if REVIEW_EXCLUDED.is_file() else set()

    ours = pd.concat([pd.read_csv(cfg.SPLITS_DIR / f"{s}.csv").assign(split=s) for s in ("train", "val")])
    ours = ours.assign(source="vegnet", subset="")
    parts_train.append(ours[COLS])

    crops = pd.read_csv(LEAF_CROPS)
    crops = crops.assign(source="healthy_leaf_crop", source_type="derived", group_id=crops["parent_group_id"])
    parts_train.append(crops[crops["parent_split"] != "test"].assign(subset="")[COLS])
    parts_test.append(crops[crops["parent_split"] == "test"].assign(subset="healthy_leaf_crops")[COLS])

    new_frames = []
    for name in SOURCES:
        path = MANIFEST_DIR / f"{name}.csv"
        if not path.is_file():
            print(f"{name}: no manifest, skipped")
            continue
        m = pd.read_csv(path)
        m = m[~m.apply(lambda r: (r["source"], r["source_folder"]) in EXCLUDED, axis=1)]
        m = m[~m["filepath"].isin(review_excluded)]
        copies, new = m[m["is_copy"]], m[~m["is_copy"]]
        if name == "c2h":
            c = copies.assign(source_type="bg_removed_copy", group_id=copies["nn_group_id"])
            parts_train.append(c[c["nn_split"] != "test"].assign(subset="")[COLS])
            parts_test.append(c[c["nn_split"] == "test"].assign(subset="bg_removed_copies")[COLS])
        new_frames.append(new.assign(source_type="external"))

    new = pd.concat(new_frames, ignore_index=True)
    assert new["class"].isin(CLASS_ORDER).all(), f"unexpected classes: {sorted(set(new['class']) - set(CLASS_ORDER))}"
    new["group_id"] = group_new_images(new, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    new["subset"] = "new_" + new["source"]
    for (_, _), g in new.groupby(["source", "class"]):
        groups = g["group_id"].unique()
        rng.shuffle(groups)
        n_test = int(round(len(groups) * EXTERNAL_TEST_FRAC))
        test_groups = set(groups[:n_test])
        is_test = g["group_id"].isin(test_groups)
        parts_test.append(g[is_test][COLS])
        parts_train.append(g[~is_test].assign(subset="")[COLS])

    train = pd.concat(parts_train, ignore_index=True)
    test = pd.concat(parts_test, ignore_index=True)
    leak = set(train["group_id"]) & set(test["group_id"])
    assert not leak, f"group leakage between train pool and external test: {sorted(leak)[:5]}"
    assert not set(train["filepath"]) & set(test["filepath"])

    OUT.mkdir(exist_ok=True)
    train.to_csv(OUT / "train_pool.csv", index=False)
    test.to_csv(OUT / "external_test.csv", index=False)
    summary = [
        "TRAIN POOL (class x source)",
        pd.crosstab(train["source"] + "/" + train["source_type"], train["class"], margins=True).to_string(),
        "",
        "EXTERNAL TEST (subset x class) - frozen, never trained on",
        pd.crosstab(test["subset"], test["class"], margins=True).to_string(),
    ]
    (OUT / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
