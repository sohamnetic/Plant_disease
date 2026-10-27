"""Step I-1: machine-drafted insect-hole outlines for the cleaf "Insect Hole" photos.

The photos are single leaves cut out onto white. Where an insect ate through
the leaf the photo shows what was under it: either the white backdrop or a pale
grey surface. A hole pixel is therefore
  * white, or
  * pale, neutral grey (no green/yellow tint) AND much brighter than this
    leaf's own tissue (a pale leaf's waxy patches, glare and midribs are
    tinted and only a little brighter; the colour-only rule failed on those)
and it must lie inside the leaf outline. Holes that open onto the leaf edge are
caught by closing the outline with a small disc first (a narrow-mouthed bite is
treated as a hole); wide bites from the edge are not drafted at all, because
the original edge is gone and a natural lobe looks the same.

Every draft is reviewed on contact sheets before it becomes a label
(outputs/insect_seg/review_picks.txt: K keep, X reject).

  python src/insect_prelabel.py drafts     # masks -> cache/insect_prelabel/*.npz, sheets
  python src/insect_prelabel.py controls   # same rule on cleaf Healthy / Black Rot (should be ~0)
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

POOL = cfg.PROJECT_ROOT / "splits_insect" / "insect_pool.csv"
METRICS = cfg.OUTPUTS_DIR / "insect_probe" / "cleaf_hole_metrics.csv"
OUT_DIR = cfg.OUTPUTS_DIR / "insect_seg"
SHEET_DIR = OUT_DIR / "review_sheets"
CACHE = cfg.PROJECT_ROOT / "cache" / "insect_prelabel"
WORK_SIDE = 640
WHITE = 232            # min channel above this = backdrop
HOLE_S_MAX = 50        # 0-255 saturation
HOLE_V_MIN = 150
HOLE_V_ABOVE_LEAF = 45  # hole must be this much brighter than the leaf's median tissue
HOLE_A_ABS_MAX, HOLE_B_MAX = 5, 6  # OpenCV LAB, offset removed
MOUTH_FRAC = 0.035     # closing disc radius as a fraction of sqrt(leaf area)
MIN_HOLE_FRAC = 4e-4   # drop specks below 0.04% of the leaf
MIN_RING_TISSUE = 0.75  # a bridged-mouth hole must be >= 75% surrounded by leaf tissue
PER_SHEET = 24


def load_work(filepath: str) -> np.ndarray:
    with Image.open(cfg.PROJECT_ROOT / filepath) as im:
        im.draft("RGB", (WORK_SIDE * 2, WORK_SIDE * 2))  # fast JPEG downscale while decoding
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE))
        return np.asarray(im)


def _largest(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def draft(rgb: np.ndarray) -> dict:
    """-> {"leaf": remaining tissue, "hole": eaten area inside the leaf outline}."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    white = rgb.min(2) > WHITE
    silhouette = _largest(ndimage.binary_opening(~white, iterations=1))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(int)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    # The surface under a hole is neutral grey (a ~ 0, b <= 3 on reviewed holes); pale
    # midribs, waxy patches and glare stay green-yellow (a <= -7, b >= 10).
    neutral = (np.abs(a) <= HOLE_A_ABS_MAX) & (b <= HOLE_B_MAX)
    pale = (s <= HOLE_S_MAX) & (v >= HOLE_V_MIN) & neutral
    leaf_v = float(np.median(v[silhouette & ~pale])) if (silhouette & ~pale).any() else 0.0
    see_through = white | (pale & (v >= leaf_v + HOLE_V_ABOVE_LEAF))
    # a see-through blob must be a blob, not a 1-2 px seam along a vein
    see_through = ndimage.binary_opening(see_through, structure=np.ones((3, 3)))
    tissue = silhouette & ~see_through
    tissue = _largest(ndimage.binary_opening(tissue, iterations=1))
    outline = ndimage.binary_fill_holes(tissue)
    r = max(int(MOUTH_FRAC * np.sqrt(outline.sum())), 2)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pad = r + 2
    closed = cv2.morphologyEx(np.pad(outline, pad).astype(np.uint8), cv2.MORPH_CLOSE, disc)[pad:-pad, pad:-pad] > 0
    enclosed = outline
    outline = ndimage.binary_fill_holes(closed | outline)
    hole = outline & ~tissue
    comps, n = ndimage.label(hole)
    if n:
        sizes = ndimage.sum(hole, comps, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= max(MIN_HOLE_FRAC * outline.sum(), 8)
        # A hole that only exists because the closing bridged a mouth must be mostly ringed by
        # tissue; the notch where the blade meets the stalk, or a wavy margin, is not.
        edge_ids = np.unique(np.concatenate([comps[0], comps[-1], comps[:, 0], comps[:, -1]]))
        keep[edge_ids] = False  # cut off by the photo frame: cannot tell hole from backdrop
        for i in np.flatnonzero(keep):
            comp = comps == i
            if (comp & ~enclosed).any():
                border = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
                keep[i] = tissue[border].mean() >= MIN_RING_TISSUE
        hole = keep[comps]
    return {"leaf": enclosed & ~hole, "hole": hole}


def eaten_pct(m: dict) -> float:
    return round(100.0 * m["hole"].sum() / max(m["leaf"].sum() + m["hole"].sum(), 1), 3)


def cache_path(filepath: str) -> Path:
    return CACHE / f"{hashlib.md5(filepath.encode()).hexdigest()}.npz"


def overlay(rgb: np.ndarray, m: dict) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[m["hole"]] = out[m["hole"]] * 0.25 + np.array([255, 0, 40]) * 0.75
    outline = m["leaf"] | m["hole"]
    edge = cv2.morphologyEx(outline.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    out[edge] = (0, 200, 255)
    return out.clip(0, 255).astype(np.uint8)


def sheets(rows: pd.DataFrame, prefix: str, tile: int = 300, mask_dir: Path | None = None) -> None:
    """Each tile: original (left half) | draft overlay (right half). mask_dir: other drafts (bootstrap rounds)."""
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for p in SHEET_DIR.glob(f"{prefix}_*.jpg"):
        p.unlink()
    cols = 4
    for s0 in range(0, len(rows), PER_SHEET):
        chunk = rows.iloc[s0 : s0 + PER_SHEET]
        nrow = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile * 2, nrow * (tile + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (_, r) in enumerate(chunk.iterrows()):
            rgb = load_work(r["filepath"])
            path = cache_path(r["filepath"]) if mask_dir is None else mask_dir / cache_path(r["filepath"]).name
            m = {key: v for key, v in np.load(path).items()}
            x, y = (k % cols) * tile * 2, (k // cols) * (tile + 18)
            for j, img in enumerate((rgb, overlay(rgb, m))):
                t = Image.fromarray(img)
                t.thumbnail((tile, tile))
                sheet.paste(t, (x + j * tile, y))
            d.text((x + 3, y + tile + 2), f"#{r['idx']}  {Path(r['filepath']).name}  eaten {r['eaten_pct']:.1f}%", fill="black")
        sheet.save(SHEET_DIR / f"{prefix}_{s0 // PER_SHEET:02d}.jpg", quality=85)


def run(rows: pd.DataFrame) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for _, r in rows.iterrows():
        m = draft(load_work(r["filepath"]))
        np.savez_compressed(cache_path(r["filepath"]), **m)
        out.append(eaten_pct(m))
    return rows.assign(eaten_pct=out)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "drafts"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if cmd == "drafts":
        pool = pd.read_csv(POOL)
        pool = pool[pool["split"] != "excluded"].reset_index(drop=True)
        pool.insert(0, "idx", range(len(pool)))
        res = run(pool)
        res.to_csv(OUT_DIR / "drafts.csv", index=False)
        sheets(res, "insect")
        print(res.groupby("split")["eaten_pct"].describe(percentiles=[0.1, 0.5, 0.9]).round(2))
        print("no hole drafted:", int((res["eaten_pct"] == 0).sum()), "of", len(res))
    elif cmd == "controls":
        m = pd.read_csv(METRICS)
        ctl = pd.concat([m[m["source_folder"] == f].sample(120, random_state=0) for f in ("Healthy", "Black Rot")])
        ctl = ctl.reset_index(drop=True)
        ctl.insert(0, "idx", range(len(ctl)))
        res = run(ctl)
        res.to_csv(OUT_DIR / "controls.csv", index=False)
        sheets(res.sort_values("eaten_pct", ascending=False).head(48), "control_worst")
        print(res.groupby("source_folder")["eaten_pct"].describe(percentiles=[0.5, 0.9, 0.97]).round(2))
    else:
        raise SystemExit("usage: insect_prelabel.py drafts|controls")


if __name__ == "__main__":
    main()
