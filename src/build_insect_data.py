"""Prepare the cleaf "Insect Hole" photos (Mendeley x995snz7p3) for a later model
version. Builds data only: nothing here trains, and splits_v3/ (including the
frozen external test) is read, never written.

Steps
  1. exact duplicates (file MD5) inside the folder
  2. hole measurement on every cleaf photo (all 3 folders, for comparison):
     the photos are leaves cut out onto white, so white pixels enclosed by the
     leaf are holes. Edge bites cannot be told apart from the leaf's own wavy
     outline, so they are reported (convexity deficit) but not trusted.
  3. cross-folder near-duplicates: each insect photo vs every cleaf Black Rot /
     Healthy photo (embedding shortlist + SIFT/RANSAC, same rule as
     external_data.py). A match means the same leaf is in two folders.
  4. near-duplicate groups inside the insect folder (same union-find as v3)
  5. split by group: groups linked to a frozen external-test photo go to the
     insect test; of the rest, INSECT_TEST_FRAC of the groups go to the insect
     test (stratified by hole tier), the others to the train candidates.

  python src/build_insect_data.py
Outputs: splits_insect/insect_pool.csv, splits_insect/SUMMARY.txt,
         outputs/insect_probe/cleaf_hole_metrics.csv, outputs/insect_probe/sheet_*.jpg
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from build_v3_data import group_new_images
from external_data import DUP_INLIERS, MANIFEST_DIR, _backbone, _embed, inliers, sift_feats

OUT = cfg.PROJECT_ROOT / "splits_insect"
PROBE = cfg.OUTPUTS_DIR / "insect_probe"
V3 = cfg.PROJECT_ROOT / "splits_v3"
INSECT_TEST_FRAC = 0.35
SEED = 42
SHORTLIST = 10
WHITE = 225          # all three channels above this = cut-out background
# Through a hole the photos show a pale grey surface (~180,182,196), not white:
# grey = low saturation and fairly bright. Leaf tissue, veins included, is saturated.
GREY_S_MAX, GREY_V_MIN = 40, 140
MIN_HOLE_FRAC = 5e-4  # a hole must cover >= 0.05% of the leaf (drops specks of glare)
CLEAR_HOLE_PCT = 0.5  # >= 0.5% of the leaf missing through enclosed holes = "clear" tier


def hole_metrics(im: Image.Image) -> dict:
    a = np.asarray(im.convert("RGB"))
    hsv = np.asarray(im.convert("HSV"))
    bg = (a > WHITE).all(axis=2) | ((hsv[..., 1] < GREY_S_MAX) & (hsv[..., 2] > GREY_V_MIN))
    leaf = ~bg
    leaf = ndimage.binary_opening(leaf, iterations=1)
    lab, n = ndimage.label(leaf)
    if n == 0:
        return {"leaf_px": 0, "n_holes": 0, "hole_pct": 0.0, "convexity_deficit": 0.0}
    sizes = ndimage.sum(leaf, lab, range(1, n + 1))
    leaf = lab == (int(np.argmax(sizes)) + 1)
    filled = ndimage.binary_fill_holes(leaf)
    holes = filled & ~leaf
    hl, hn = ndimage.label(holes)
    area = filled.sum()
    hs = ndimage.sum(holes, hl, range(1, hn + 1)) if hn else np.array([])
    keep = hs >= max(MIN_HOLE_FRAC * area, 6)
    hull = cv2.convexHull(np.argwhere(filled)[:, ::-1].astype(np.int32))
    hull_area = cv2.contourArea(hull)
    return {
        "leaf_px": int(area),
        "n_holes": int(keep.sum()),
        "hole_pct": round(100 * float(hs[keep].sum()) / area, 3) if area else 0.0,
        "convexity_deficit": round(1 - area / hull_area, 4) if hull_area else 0.0,
    }


def load_thumbs(paths: list[str]) -> list[Image.Image]:
    ims = []
    for f in paths:
        with Image.open(cfg.PROJECT_ROOT / f) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((640, 640))
            ims.append(im)
    return ims


def contact_sheet(df: pd.DataFrame, ims: dict[str, Image.Image], path: Path, n: int = 30) -> None:
    pick = df.sample(min(n, len(df)), random_state=SEED)
    w = 200
    sheet = Image.new("RGB", (w * 6, (w + 16) * ((len(pick) + 5) // 6)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (_, r) in enumerate(pick.iterrows()):
        t = ims[r["filepath"]].copy()
        t.thumbnail((w, w))
        x, y = (k % 6) * w, (k // 6) * (w + 16)
        sheet.paste(t, (x, y))
        d.text((x + 2, y + w), f"{Path(r['filepath']).name}  holes {r['n_holes']} / {r['hole_pct']}%", fill="black")
    sheet.save(path, quality=88)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = pd.read_csv(MANIFEST_DIR / "cleaf.csv")
    v3 = pd.concat([
        pd.read_csv(V3 / "train_pool.csv").assign(v3_split="train_pool"),
        pd.read_csv(V3 / "external_test.csv").assign(v3_split="external_test"),
    ])
    v3 = v3[v3["source"] == "cleaf"].set_index("filepath")

    print(f"loading {len(m)} cleaf photos ...")
    ims = dict(zip(m["filepath"], load_thumbs(m["filepath"].tolist())))

    # 1. exact duplicates
    m["md5"] = [hashlib.md5((cfg.PROJECT_ROOT / f).read_bytes()).hexdigest() for f in m["filepath"]]
    # 2. holes, for every folder
    met = pd.DataFrame([hole_metrics(ims[f]) for f in m["filepath"]], index=m.index)
    m = pd.concat([m, met], axis=1)
    m["v3_split"] = m["filepath"].map(v3["v3_split"]).fillna("not_used")
    m["v3_group_id"] = m["filepath"].map(v3["group_id"])
    PROBE.mkdir(parents=True, exist_ok=True)
    m[["filepath", "source_folder", "class", "v3_split", "md5", "leaf_px", "n_holes", "hole_pct",
       "convexity_deficit"]].to_csv(PROBE / "cleaf_hole_metrics.csv", index=False)

    ins = m[m["source_folder"] == "Insect Hole"].copy()
    other = m[m["source_folder"] != "Insect Hole"].reset_index(drop=True)

    # 3. same leaf in another folder?
    model = _backbone(device)
    e_ins = _embed(model, [ims[f] for f in ins["filepath"]], device)
    e_oth = _embed(model, [ims[f] for f in other["filepath"]], device)
    sims = e_ins @ e_oth.T
    oth_feats: dict[int, tuple] = {}
    links = []
    for i, f in enumerate(ins["filepath"]):
        fa = sift_feats(ims[f])
        best, bj = 0, -1
        for j in np.argsort(-sims[i])[:SHORTLIST]:
            if j not in oth_feats:
                oth_feats[j] = sift_feats(ims[other.at[j, "filepath"]])
            s = inliers(fa, oth_feats[j])
            if s > best:
                best, bj = s, int(j)
        o = other.loc[bj] if bj >= 0 else None
        links.append({
            "xfolder_inliers": best,
            "xfolder_match": o["filepath"] if o is not None and best >= DUP_INLIERS else "",
            "xfolder_class": o["class"] if o is not None and best >= DUP_INLIERS else "",
            "xfolder_v3_split": o["v3_split"] if o is not None and best >= DUP_INLIERS else "",
            "xfolder_v3_group": o["v3_group_id"] if o is not None and best >= DUP_INLIERS else "",
        })
    ins = pd.concat([ins.reset_index(drop=True), pd.DataFrame(links)], axis=1)

    # 4. groups inside the insect folder; a leaf shared with a v3 photo joins that photo's group
    # v3 ids are "ext_cleaf_<row>" too; prefix so an insect group can never collide with an unrelated v3 group
    ins["group_id"] = "ins_" + group_new_images(ins, device).astype(str).to_numpy()
    shared = ins["xfolder_v3_group"].fillna("").astype(str) != ""
    for g, gg in ins[shared].groupby("group_id"):
        ins.loc[ins["group_id"] == g, "group_id"] = gg["xfolder_v3_group"].iloc[0]

    # exact duplicates and same-leaf-labelled-healthy are not trustworthy labels
    ins["exclude_reason"] = ""
    # same rule as build_v3_data: a flagged copy of one of our photos is dropped (usually a chance match on veins)
    ins.loc[ins["is_copy"], "exclude_reason"] = "flagged as a copy of one of our photos"
    ins.loc[ins.duplicated("md5", keep="first") & (ins["exclude_reason"] == ""), "exclude_reason"] = "exact duplicate of another insect photo"
    conflict = ins["xfolder_class"] == "no_disease"
    ins.loc[conflict & (ins["exclude_reason"] == ""), "exclude_reason"] = "same leaf is also in the Healthy folder"
    ins["tier"] = np.where(ins["hole_pct"] >= CLEAR_HOLE_PCT, "clear_holes", "edge_or_subtle")

    # 5. split by group
    rng = np.random.default_rng(SEED)
    frozen = set(ins.loc[ins["xfolder_v3_split"] == "external_test", "group_id"])
    ins["split"] = ""
    ins.loc[ins["group_id"].isin(frozen), "split"] = "insect_test"
    free = ins[ins["split"] == ""]
    gtier = free.groupby("group_id")["tier"].agg(lambda s: s.mode().iloc[0])
    target = int(round(INSECT_TEST_FRAC * ins["group_id"].nunique())) - len(frozen)
    for t, gs in gtier.groupby(gtier):
        groups = gs.index.to_numpy()
        rng.shuffle(groups)
        k = max(int(round(len(groups) / len(gtier) * max(target, 0))), 0)
        ins.loc[ins["group_id"].isin(groups[:k]), "split"] = "insect_test"
    ins.loc[ins["split"] == "", "split"] = "insect_train"
    ins.loc[ins["exclude_reason"] != "", "split"] = "excluded"
    leak = set(ins.loc[ins.split == "insect_train", "group_id"]) & set(ins.loc[ins.split == "insect_test", "group_id"])
    assert not leak, f"group leakage: {sorted(leak)[:5]}"
    v3_test_groups = set(pd.read_csv(V3 / "external_test.csv")["group_id"])
    assert not set(ins.loc[ins.split == "insect_train", "group_id"]) & v3_test_groups, "train row shares a frozen-test group"

    ins = ins.assign(source="cleaf", source_type="external")
    OUT.mkdir(exist_ok=True)
    cols = ["filepath", "class", "source", "source_type", "group_id", "split", "tier", "exclude_reason",
            "n_holes", "hole_pct", "convexity_deficit", "width", "height", "md5",
            "xfolder_inliers", "xfolder_match", "xfolder_class", "xfolder_v3_split"]
    ins[cols].to_csv(OUT / "insect_pool.csv", index=False)

    for t in ("clear_holes", "edge_or_subtle"):
        contact_sheet(ins[ins["tier"] == t], ims, PROBE / f"sheet_{t}.jpg")
    if (ins["exclude_reason"] != "").any():
        contact_sheet(ins[ins["exclude_reason"] != ""], ims, PROBE / "sheet_excluded.jpg")

    by_folder = m.groupby("source_folder").agg(
        photos=("filepath", "size"),
        with_enclosed_hole=("n_holes", lambda s: int((s > 0).sum())),
        clear_tier=("hole_pct", lambda s: int((s >= CLEAR_HOLE_PCT).sum())),
        median_hole_pct=("hole_pct", "median"),
        median_convexity_deficit=("convexity_deficit", "median"),
    )
    summary = [
        "INSECT HOLE PHOTOS (cleaf / Mendeley x995snz7p3) - data only, not trained on",
        f"photos: {len(ins)}  groups: {ins['group_id'].nunique()}  exact-duplicate files: {int(ins.duplicated('md5').sum())}",
        f"same leaf found in another cleaf folder: {int((ins['xfolder_match'] != '').sum())} "
        f"({ins.loc[ins['xfolder_match'] != '', 'xfolder_class'].value_counts().to_dict()}); "
        f"of these in the frozen external test: {int((ins['xfolder_v3_split'] == 'external_test').sum())}",
        "",
        "SPLIT x TIER",
        pd.crosstab(ins["split"], ins["tier"], margins=True).to_string(),
        "",
        "EXCLUDED",
        ins.loc[ins["exclude_reason"] != "", "exclude_reason"].value_counts().to_string() or "none",
        "",
        f"HOLES BY FOLDER (enclosed holes >= {MIN_HOLE_FRAC:.2%} of leaf; clear tier >= {CLEAR_HOLE_PCT}% of leaf missing)",
        by_folder.to_string(),
    ]
    (OUT / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
