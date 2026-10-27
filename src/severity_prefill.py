"""Approach A: colour-rule lesion estimate used ONLY to pre-fill severity grades.

For each ORIGINAL diseased image this estimates the fraction of visible plant
tissue that looks symptomatic, maps it to a 0-4 grade, and renders contact
sheets (photo | lesion overlay) so a reviewer can confirm or correct the grade.

It is a labelling aid, not the severity model. Known weak spots:
  * curd colour varies naturally (white / cream / yellow / pink),
  * fully collapsed leaves have no healthy tissue to anchor the plant mask,
  * soil is brown, so lesions are only counted inside the plant hull.

  python src/severity_prefill.py
Outputs: outputs/severity/prefill_grades.csv, outputs/severity/sheets/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from v2_cache import cache_path_for

OUT_DIR = cfg.OUTPUTS_DIR / "severity"
SHEET_DIR = OUT_DIR / "sheets"
WORK_SIDE = 320

# Grade scale shared by prefill, grading tool, model and API.
GRADE_BINS = [(0, "none"), (1, "early"), (2, "moderate"), (3, "severe"), (4, "critical")]
GRADE_EDGES_PCT = [10.0, 25.0, 50.0]  # grade 1: <10%, 2: 10-25%, 3: 25-50%, 4: >50%
LEAF_CLASSES = {"black_rot", "downy_mildew"}
CURD_CLASSES = {"bacterial_spot_rot"}


def pct_to_grade(pct: float) -> int:
    grade = 1
    for edge in GRADE_EDGES_PCT:
        if pct >= edge:
            grade += 1
    return grade


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    # Pad so the flood reaches background touching any border, not just (0,0).
    m = np.pad(mask.astype(np.uint8), 1)
    h, w = m.shape
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    return (m | (1 - flood)).astype(bool)[1:-1, 1:-1]


def _largest_components(mask: np.ndarray, keep_frac: float = 0.15) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.where(areas >= keep_frac * areas.max())[0] + 1
    return np.isin(lab, keep)


def estimate_leaf(rgb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    healthy = (h >= 70) & (h <= 220) & (s >= 0.12) & (v >= 0.18)
    chlorotic = (h >= 40) & (h < 70) & (s >= 0.30) & (v >= 0.30)
    necrotic = ((h < 40) | (h >= 330)) & (s >= 0.15) & (v >= 0.12) & (v <= 0.85)
    dark = v < 0.14
    bleached = (s < 0.15) & (v >= 0.60)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    anchor = cv2.morphologyEx(healthy.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    anchor = _largest_components(anchor > 0, 0.10)
    # Chlorotic tissue is plant, not soil: let it extend the hull.
    plant = cv2.morphologyEx((anchor | chlorotic).astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    plant = _fill_holes(plant)
    lesion = plant & (chlorotic | necrotic | dark | bleached) & ~healthy
    # Large lesion blobs touching the border inside a small plant hull are soil.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lesion.astype(np.uint8), 8)
    H, W = lesion.shape
    for i in range(1, n):
        x, y, w, h_, area = stats[i]
        touches = x == 0 or y == 0 or x + w >= W or y + h_ >= H
        if touches and area > 0.03 * lesion.size and plant.mean() < 0.9:
            lesion[lab == i] = False
    tissue_px = plant.sum()
    if tissue_px < 0.05 * plant.size:
        # Almost no healthy anchor: leaf probably collapsed. Treat brown/yellow
        # non-soil-dark foreground as symptomatic tissue over the whole frame.
        plant = np.ones_like(plant)
        lesion = chlorotic | (necrotic & (s >= 0.20)) | bleached
        tissue_px = plant.sum()
    pct = 100.0 * lesion.sum() / max(tissue_px, 1)
    return pct, plant, lesion


def estimate_curd(rgb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A = lab[..., 0] * 100 / 255.0, lab[..., 1] - 128
    green = (h >= 70) & (h <= 220) & (s >= 0.18)
    curd_like = ~green & (v >= 0.35) & ((s <= 0.55) | ((h < 70) | (h >= 290)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    curd = cv2.morphologyEx(curd_like.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    curd = _largest_components(curd, 0.25)
    curd = _fill_holes(cv2.morphologyEx(curd.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0)
    curd &= ~green
    if curd.sum() < 0.03 * curd.size:
        return 0.0, curd, np.zeros_like(curd)
    medL = float(np.median(L[curd]))
    medA = float(np.median(A[curd]))
    browned = curd & (L < medL - 18)
    reddish = curd & (A > medA + 10)
    brown_hue = curd & (h < 40) & (s >= 0.40) & (v < 0.75)
    lesion = browned | reddish | brown_hue
    lesion = cv2.morphologyEx(lesion.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    pct = 100.0 * lesion.sum() / max(curd.sum(), 1)
    return pct, curd, lesion


def load_small(filepath: str) -> np.ndarray:
    im = Image.open(cache_path_for(filepath)).convert("RGB")
    im.thumbnail((WORK_SIDE, WORK_SIDE))
    return np.asarray(im)


def estimate(filepath: str, cls: str) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    rgb = load_small(filepath)
    if cls in CURD_CLASSES:
        pct, roi, lesion = estimate_curd(rgb)
    else:
        pct, roi, lesion = estimate_leaf(rgb)
    return pct, rgb, roi, lesion


def overlay(rgb: np.ndarray, roi: np.ndarray, lesion: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.55
    out[roi] = rgb[roi] * 0.9
    out[lesion] = out[lesion] * 0.3 + np.array([255, 0, 200]) * 0.7
    edges = cv2.Canny(roi.astype(np.uint8) * 255, 50, 150) > 0
    out[edges] = [0, 255, 255]
    return out.clip(0, 255).astype(np.uint8)


def render_sheets(df: pd.DataFrame, per_sheet: int = 12, tile: int = 220) -> None:
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for old in SHEET_DIR.glob("*.png"):
        old.unlink()
    font = ImageFont.load_default()
    cols, label_h = 3, 26
    for sheet_i, start in enumerate(range(0, len(df), per_sheet)):
        chunk = df.iloc[start : start + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        W = cols * (2 * tile + 12)
        H = rows * (tile + label_h)
        sheet = Image.new("RGB", (W, H), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for j, (_, r) in enumerate(chunk.iterrows()):
            pct, rgb, roi, lesion = estimate(r["filepath"], r["class"])
            a = Image.fromarray(rgb)
            b = Image.fromarray(overlay(rgb, roi, lesion))
            a.thumbnail((tile, tile))
            b.thumbnail((tile, tile))
            x0 = (j % cols) * (2 * tile + 12)
            y0 = (j // cols) * (tile + label_h)
            sheet.paste(a, (x0, y0))
            sheet.paste(b, (x0 + tile + 2, y0))
            name = Path(r["filepath"]).name
            draw.text(
                (x0 + 2, y0 + tile + 2),
                f"#{r['idx']} {name[:26]}  A={r['prefill_pct']:.0f}% g{r['prefill_grade']}",
                fill=(0, 0, 0),
                font=font,
            )
        sheet.save(SHEET_DIR / f"sheet_{sheet_i:03d}.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(cfg.SPLITS_DIR / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["source_type"] == "original"].copy()
    df = df.sort_values(["class", "filepath"]).reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        if r["class"] == "no_disease":
            rows.append((0.0, 0))
            continue
        pct, *_ = estimate(r["filepath"], r["class"])
        rows.append((round(pct, 1), pct_to_grade(pct)))
    df["prefill_pct"] = [p for p, _ in rows]
    df["prefill_grade"] = [g for _, g in rows]
    df["idx"] = np.arange(len(df))
    df.to_csv(OUT_DIR / "prefill_grades.csv", index=False)
    diseased = df[df["class"] != "no_disease"]
    print(diseased.groupby("class")["prefill_grade"].value_counts().unstack(fill_value=0))
    render_sheets(diseased)
    print("sheets ->", SHEET_DIR)


if __name__ == "__main__":
    main()
