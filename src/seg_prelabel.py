"""Phase 3, step 1: machine pre-labels for leaf / curd / lesion segmentation.

For every photo in the labelling set this builds up to 4 CANDIDATE tissue masks
with SAM2 (segment-anything), guided in different ways, and marks lesions inside
each candidate with the colour rules from severity_prefill.py:

  A  SAM2 points inside the colour-rule LEAF region
  B  SAM2 points inside the colour-rule CURD region
  C  SAM2 points around the image centre (photographers centre the subject)
  D  SAM2 box over the central 90% (dominant object)
  white-backdrop photos (c2h, cleaf) get one exact candidate: every non-white pixel

A reviewer then picks the right candidate per photo (or rejects it) on the
contact sheets; src/seg_dataset.py turns the picks into YOLO-seg labels.

Rules that do not need review:
  * candidate type: mostly curd-coloured -> curd (+ surrounding green = leaf),
    otherwise leaf
  * no_disease photos have no lesions (that is what the label says)
  * holes in a leaf (shot-holes, torn tissue) count as lesion: tissue was lost

  python src/seg_prelabel.py select      # build outputs/seg/label_set.csv
  python src/seg_prelabel.py candidates  # SAM2 + colour rules -> cache/seg_prelabel/*.npz + sheets
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from severity_prefill import _fill_holes, _largest_components, estimate_curd, estimate_leaf
from v2_cache import cache_path_for

SEG_DIR = cfg.OUTPUTS_DIR / "seg"
SHEET_DIR = SEG_DIR / "review_sheets"
CAND_DIR = cfg.PROJECT_ROOT / "cache" / "seg_prelabel"
LABEL_SET = SEG_DIR / "label_set.csv"
SAM_WEIGHTS = cfg.PROJECT_ROOT / "cache" / "weights" / "sam2.1_b.pt"
WORK_SIDE = 640
CUTOUT_SOURCES = {"c2h", "cleaf"}
CANDIDATES = ["A", "B", "C", "D"]
LEAF_DISEASES = {"black_rot", "downy_mildew"}
SEED = 42


# ---------------------------------------------------------------- selection
def select() -> pd.DataFrame:
    """Labelling set from the training pool (split=train) and a held-out set (split=eval)."""
    rng = np.random.default_rng(SEED)
    grades = pd.read_csv(cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv")
    pool = pd.read_csv(cfg.PROJECT_ROOT / "splits_v3" / "train_pool.csv")
    ext = pd.read_csv(cfg.PROJECT_ROOT / "splits_v3" / "external_test.csv")

    def take(df, n):
        return df.sample(min(n, len(df)), random_state=int(rng.integers(1 << 31)))

    rows = []
    # VegNet originals with a reviewed severity grade.
    g_tr, g_te = grades[grades["split"] != "test"], grades[grades["split"] == "test"]
    tr = pd.concat([
        g_tr[g_tr["grade"] == 1],  # every early-stage photo
        take(g_tr[(g_tr["class"] == "bacterial_spot_rot") & (g_tr["grade"] > 1)], 45),
        g_tr[(g_tr["class"] == "black_rot") & (g_tr["grade"] > 1)],
        take(g_tr[(g_tr["class"] == "downy_mildew") & (g_tr["grade"] > 1)], 50),
        take(g_tr[g_tr["class"] == "no_disease"], 35),
    ])
    te = pd.concat([
        g_te[g_te["grade"] == 1],
        take(g_te[(g_te["class"] == "bacterial_spot_rot") & (g_te["grade"] > 1)], 10),
        g_te[(g_te["class"] == "black_rot") & (g_te["grade"] > 1)],
        take(g_te[(g_te["class"] == "downy_mildew") & (g_te["grade"] > 1)], 10),
        take(g_te[g_te["class"] == "no_disease"], 10),
    ])
    for df, split in ((tr, "train"), (te, "eval")):
        rows.append(df.assign(source="vegnet", seg_split=split)[["filepath", "class", "source", "grade", "seg_split"]])

    def ext_rows(df, source, cls, n, split):
        sub = take(df[(df["source"] == source) & (df["class"] == cls)], n)
        return sub.assign(grade=np.nan, seg_split=split)[["filepath", "class", "source", "grade", "seg_split"]]

    ext_new = ext[ext["subset"].str.startswith("new_")]
    for source, cls, n_tr, n_te in [
        ("cleaf", "black_rot", 80, 15), ("cleaf", "no_disease", 40, 10),
        ("sppu", "bacterial_spot_rot", 25, 8), ("sppu", "no_disease", 30, 8), ("sppu", "downy_mildew", 20, 5),
        ("c2h", "black_rot", 10, 4), ("c2h", "downy_mildew", 10, 4),
        ("vegleaf", "no_disease", 0, 6),
    ]:
        if n_tr:
            rows.append(ext_rows(pool, source, cls, n_tr, "train"))
        rows.append(ext_rows(ext_new, source, cls, n_te, "eval"))
    crops = pool[pool["source"] == "healthy_leaf_crop"]
    rows.append(take(crops, 15).assign(grade=0, seg_split="train")[["filepath", "class", "source", "grade", "seg_split"]])

    out = pd.concat(rows, ignore_index=True).drop_duplicates("filepath").reset_index(drop=True)
    out["idx"] = np.arange(len(out))
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LABEL_SET, index=False)
    print(pd.crosstab([out["seg_split"], out["source"]], out["class"], margins=True))
    return out


# ---------------------------------------------------------------- colour rules on a given tissue mask
def _hsv(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    return hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def curd_like(rgb: np.ndarray) -> np.ndarray:
    h, s, v = _hsv(rgb)
    green = (h >= 75) & (h <= 220) & (s >= 0.15)
    return ~green & (v >= 0.45) & ((s <= 0.35) | ((h >= 20) & (h < 75)))


def leaf_lesions(rgb: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    h, s, v = _hsv(rgb)
    healthy = (h >= 70) & (h <= 220) & (s >= 0.12) & (v >= 0.18)
    chlorotic = (h >= 40) & (h < 70) & (s >= 0.30) & (v >= 0.30)
    necrotic = ((h < 40) | (h >= 330)) & (s >= 0.15) & (v >= 0.12) & (v <= 0.85)
    dark = v < 0.14
    # Pale midribs and veins are thin; dead bleached tissue comes in wide patches.
    bleached = (s < 0.12) & (v >= 0.65)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bleached = cv2.morphologyEx(bleached.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    les = tissue & (chlorotic | necrotic | dark | bleached) & ~healthy
    return _clean(les, tissue)


def curd_lesions(rgb: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    if tissue.sum() < 50:
        return np.zeros_like(tissue)
    h, s, v = _hsv(rgb)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A = lab[..., 0] * 100 / 255.0, lab[..., 1] - 128
    medL, medA = float(np.median(L[tissue])), float(np.median(A[tissue]))
    B = lab[..., 2] - 128
    medB = float(np.median(B[tissue]))
    # Darker alone is a shadow between florets; rot is darker AND browner/redder.
    browned = (L < medL - 15) & ((A > medA + 3) | (B > medB + 6))
    les = tissue & (browned | (A > medA + 10) | ((h < 40) & (s >= 0.40) & (v < 0.75)))
    return _clean(les, tissue)


def _clean(les: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    les = cv2.morphologyEx(les.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(les.astype(np.uint8), 8)
    min_area = max(12, int(0.0002 * les.size))
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab] & tissue


def _tidy(mask: np.ndarray, keep_frac: float = 0.10) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    mask = _largest_components(mask, keep_frac)
    return _fill_holes(mask)


# ---------------------------------------------------------------- candidates
def build_candidate(rgb: np.ndarray, tissue: np.ndarray, cls: str) -> dict:
    """tissue -> {curd, leaf, lesion} masks. Type decided by colour."""
    tissue = _tidy(tissue)
    frac_curd = curd_like(rgb)[tissue].mean() if tissue.any() else 0.0
    # Black rot and downy mildew are leaf diseases: a yellow dying leaf is not a curd.
    if frac_curd > 0.5 and cls not in LEAF_DISEASES:
        curd = tissue
        h, s, v = _hsv(rgb)
        green = ((h >= 60) & (h <= 220) & (s >= 0.12) & (v >= 0.15))
        ring = cv2.dilate(curd.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        leaf = _tidy(cv2.morphologyEx((green & ~ring).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0, 0.15)
        leaf &= ~ring
        lesion = curd_lesions(rgb, curd) | leaf_lesions(rgb, leaf)
    else:
        curd = np.zeros_like(tissue)
        leaf = tissue
        lesion = leaf_lesions(rgb, leaf)
    if cls == "no_disease":
        lesion = np.zeros_like(tissue)
    return {"curd": curd, "leaf": leaf, "lesion": lesion}


def _points_in(mask: np.ndarray, n: int, rng) -> np.ndarray | None:
    if mask.sum() < 30:
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    cand = np.argwhere(dist > dist.max() * 0.5)
    pick = cand[rng.choice(len(cand), min(n, len(cand)), replace=False)]
    return pick[:, ::-1].astype(np.float32)  # (x, y)


def candidates_for(sam, rgb: np.ndarray, cls: str, source: str) -> dict[str, dict]:
    if source in CUTOUT_SOURCES:
        tissue = rgb.min(2) <= 235
        return {"A": build_candidate(rgb, tissue, cls)}
    rng = np.random.default_rng(0)
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (320, int(320 * H / W)), interpolation=cv2.INTER_AREA)
    k = W / 320

    def up(m):
        return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    prompts: dict[str, dict] = {}
    _, leaf_roi, _ = estimate_leaf(small)
    _, curd_roi, _ = estimate_curd(small)
    for key, roi in (("A", leaf_roi), ("B", curd_roi)):
        pts = _points_in(up(roi), 5, rng)
        if pts is not None:
            prompts[key] = {"points": [pts.tolist()], "labels": [[1] * len(pts)]}
    cx, cy, dx, dy = W / 2, H / 2, W * 0.12, H * 0.12
    c_pts = [[cx, cy], [cx - dx, cy], [cx + dx, cy], [cx, cy - dy], [cx, cy + dy]]
    prompts["C"] = {"points": [c_pts], "labels": [[1] * 5]}
    prompts["D"] = {"bboxes": [[W * 0.05, H * 0.05, W * 0.95, H * 0.95]]}
    out = {}
    for key, pr in prompts.items():
        res = sam(rgb, verbose=False, **pr)[0]
        if res.masks is None or len(res.masks.data) == 0:
            continue
        m = res.masks.data[0].cpu().numpy().astype(bool)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if 0.02 < m.mean() < 0.995:
            out[key] = build_candidate(rgb, m, cls)
    return out


def load_work(filepath: str) -> np.ndarray:
    path = cache_path_for(filepath)
    im = Image.open(path if path.is_file() else cfg.PROJECT_ROOT / filepath).convert("RGB")
    im.thumbnail((WORK_SIDE, WORK_SIDE))
    return np.asarray(im)


def cand_path(filepath: str) -> Path:
    return CAND_DIR / f"{hashlib.md5(filepath.encode()).hexdigest()}.npz"


def overlay(rgb: np.ndarray, c: dict) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.45
    tissue = c["leaf"] | c["curd"]
    out[tissue] = rgb[tissue] * 0.95
    out[c["lesion"]] = out[c["lesion"]] * 0.3 + np.array([255, 0, 200]) * 0.7
    for m, col in ((c["leaf"], (0, 255, 255)), (c["curd"], (255, 220, 0))):
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        out[edge] = col
    return out.clip(0, 255).astype(np.uint8)


def pct(c: dict) -> float:
    t = (c["leaf"] | c["curd"]).sum()
    return 100.0 * (c["lesion"] & (c["leaf"] | c["curd"])).sum() / max(t, 1)


def run_candidates() -> None:
    from ultralytics import SAM

    sam = SAM(str(SAM_WEIGHTS))
    ls = pd.read_csv(LABEL_SET)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for old in SHEET_DIR.glob("*.png"):
        old.unlink()
    font = ImageFont.load_default()
    tile, per_sheet = 190, 8
    sheet, draw, slot, n_sheet = None, None, 0, 0

    def flush():
        nonlocal sheet, n_sheet
        if sheet is not None:
            sheet.save(SHEET_DIR / f"sheet_{n_sheet:03d}.png")
            n_sheet += 1
        sheet = None

    for _, r in ls.iterrows():
        rgb = load_work(r["filepath"])
        cands = candidates_for(sam, rgb, r["class"], r["source"])
        np.savez_compressed(
            cand_path(r["filepath"]), rgb_shape=np.array(rgb.shape),
            **{f"{k}_{part}": np.packbits(v[part]) for k, v in cands.items() for part in ("leaf", "curd", "lesion")},
        )
        if sheet is None:
            sheet = Image.new("RGB", (5 * tile, per_sheet * (tile + 14)), (245, 245, 245))
            draw, slot = ImageDraw.Draw(sheet), 0
        y = slot * (tile + 14)
        base = Image.fromarray(rgb)
        base.thumbnail((tile, tile))
        sheet.paste(base, (0, y))
        draw.text((2, y + tile), f"#{r['idx']} {r['class'][:12]} {r['source']} g{'' if pd.isna(r['grade']) else int(r['grade'])}",
                  fill=(0, 0, 0), font=font)
        for j, key in enumerate(CANDIDATES):
            if key not in cands:
                continue
            t = Image.fromarray(overlay(rgb, cands[key]))
            t.thumbnail((tile, tile))
            sheet.paste(t, ((j + 1) * tile, y))
            draw.text(((j + 1) * tile + 2, y + tile), f"{key}  {pct(cands[key]):.0f}%", fill=(0, 0, 0), font=font)
        slot += 1
        if slot == per_sheet:
            flush()
    flush()
    print(f"{len(ls)} images, {n_sheet} review sheets -> {SHEET_DIR}")


def load_candidate(filepath: str, key: str) -> dict:
    z = np.load(cand_path(filepath))
    shape = tuple(z["rgb_shape"][:2])
    n = shape[0] * shape[1]
    return {part: np.unpackbits(z[f"{key}_{part}"])[:n].reshape(shape).astype(bool) for part in ("leaf", "curd", "lesion")}


if __name__ == "__main__":
    {"select": select, "candidates": run_candidates}[sys.argv[1]]()
