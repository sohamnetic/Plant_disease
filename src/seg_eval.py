"""Phase 3 evaluation of the YOLO11-seg model (never-trained-on images only).

  1. held-out reviewed labels (data_external/seg_yolo val, 73 images):
       affected-% error (worst region and overall), lesion Dice, tissue IoU,
       early-stage recall, healthy false alarms
  2. PlantSeg cauliflower images with EXPERT lesion masks (90, other diseases):
       lesion Dice / precision / recall against human annotation
  3. VegNet test-split originals (93): measured % vs the reviewed severity grade
  4. CPU speed, the user's field photo overlay, and a contact sheet

Plan gates: affected-% MAE <= 5 points, lesion Dice >= 0.6, early recall >= 0.8,
healthy false-lesion area < 1%.

  python src/seg_eval.py
  python src/seg_eval.py --weights experiments/seg_002_insect/weights/best.pt --out-dir outputs/insect_seg/seg_002_phase3_eval
Outputs: outputs/seg/seg_eval.json, outputs/seg/val_predictions.png, outputs/seg/user_photo_1_overlay.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from seg_runtime import CLASSES, WORK_SIDE, SegEngine, measure, overlay, stage_for
from v2_cache import cache_path_for

YOLO_DIR = cfg.PROJECT_ROOT / "data_external" / "seg_yolo"
PLANTSEG = cfg.PROJECT_ROOT / "data_external" / "raw" / "plantseg" / "cauliflower" / "plantsegv3"
SEG_DIR = cfg.OUTPUTS_DIR / "seg"
USER_PHOTO = cfg.OUTPUTS_DIR / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"
GATES = {"affected_pct_mae_max": 5.0, "lesion_dice_min": 0.6, "early_recall_min": 0.8, "healthy_false_area_max_pct": 1.0}


def gt_masks(label_path: Path, h: int, w: int):
    regions, lesion = [], np.zeros((h, w), bool)
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        v = line.split()
        cls = CLASSES[int(v[0])]
        pts = (np.array(v[1:], dtype=np.float32).reshape(-1, 2) * [w, h]).round().astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        if cls == "lesion":
            lesion |= m.astype(bool)
        else:
            regions.append((cls, m.astype(bool)))
    return regions, lesion


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def load_rgb(path: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((WORK_SIDE, WORK_SIDE))
    return np.asarray(im)


def eval_heldout(eng: SegEngine) -> tuple[dict, pd.DataFrame, list]:
    labelled = pd.read_csv(SEG_DIR / "labelled_images.csv")
    rows, sheet_items = [], []
    for img in sorted((YOLO_DIR / "images" / "val").glob("*.jpg")):
        idx = int(img.stem.split("_")[0])
        meta = labelled[labelled["idx"] == idx].iloc[0]
        rgb = np.asarray(Image.open(img).convert("RGB"))
        h, w = rgb.shape[:2]
        g_regions, g_les = gt_masks(YOLO_DIR / "labels" / "val" / f"{img.stem}.txt", h, w)
        p_regions, p_les = eng.raw_masks(rgb)
        g, p = measure(g_regions, g_les), measure(p_regions, p_les)
        rows.append({
            "idx": idx, "source": meta["source"], "class": meta["class"], "grade": meta["grade"],
            "gt_worst_pct": g["worst_region_pct"], "pred_worst_pct": p["worst_region_pct"],
            "gt_overall_pct": g["overall_pct"], "pred_overall_pct": p["overall_pct"],
            "gt_lesions": g["lesion_count"], "pred_lesions": p["lesion_count"],
            "gt_stage": g["stage"], "pred_stage": p["stage"],
            "lesion_dice": dice(p_les, g_les) if g_les.any() else np.nan,
            "tissue_iou": iou(p["_tissue"], g["_tissue"]),
            "pred_false_lesion_pct": 100.0 * (p_les & p["_tissue"]).sum() / max(p["_tissue"].sum(), 1),
        })
        sheet_items.append((rgb, overlay(rgb, g, g_les), overlay(rgb, p, p_les), rows[-1]))
    df = pd.DataFrame(rows)
    dis = df["class"] != "no_disease"
    healthy = ~dis
    early_labelled = df[(df["source"] == "vegnet") & (df["grade"] == 1)]
    early_measured = df[dis & (df["gt_worst_pct"] >= 0.5) & (df["gt_worst_pct"] < 5)]
    res = {
        "n_images": int(len(df)),
        "affected_pct_mae_worst_region": float((df.pred_worst_pct - df.gt_worst_pct).abs().mean()),
        "affected_pct_mae_overall": float((df.pred_overall_pct - df.gt_overall_pct).abs().mean()),
        "affected_pct_mae_by_source": df.assign(e=(df.pred_worst_pct - df.gt_worst_pct).abs()).groupby("source")["e"].mean().round(2).to_dict(),
        "affected_pct_spearman": float(spearmanr(df.gt_worst_pct, df.pred_worst_pct).statistic),
        "lesion_dice_mean_diseased": float(df.loc[dis, "lesion_dice"].mean()),
        "lesion_dice_by_source": df[dis].groupby("source")["lesion_dice"].mean().round(3).to_dict(),
        "tissue_iou_mean": float(df["tissue_iou"].mean()),
        "stage_exact_vs_labels": float((df.gt_stage == df.pred_stage).mean()),
        "early_recall_grade1_photos": {"n": int(len(early_labelled)),
                                       "detected": float((early_labelled.pred_stage != "none").mean()) if len(early_labelled) else None},
        "early_recall_measured_0p5_to_5pct": {"n": int(len(early_measured)),
                                              "detected": float((early_measured.pred_stage != "none").mean()) if len(early_measured) else None},
        "healthy": {"n": int(healthy.sum()),
                    "mean_false_lesion_pct": float(df.loc[healthy, "pred_false_lesion_pct"].mean()),
                    "flagged_diseased_rate": float((df.loc[healthy, "pred_stage"] != "none").mean())},
        "stage_confusion": pd.crosstab(df.gt_stage, df.pred_stage).to_dict(),
    }
    return res, df, sheet_items


def eval_plantseg(eng: SegEngine) -> dict:
    rows = []
    tp = fp = fn = 0
    for img in sorted((PLANTSEG / "images").rglob("*.jpg")):
        ann = PLANTSEG / "annotations" / img.parent.name / f"{img.stem}.png"
        if not ann.is_file():
            continue
        rgb = load_rgb(img)
        h, w = rgb.shape[:2]
        gt = np.asarray(Image.open(ann).resize((w, h), Image.Resampling.NEAREST)) > 0
        _, les = eng.raw_masks(rgb)
        rows.append({"image": img.name, "dice": dice(les, gt), "gt_pct_of_image": 100 * gt.mean()})
        tp += int((les & gt).sum()); fp += int((les & ~gt).sum()); fn += int((~les & gt).sum())
    df = pd.DataFrame(rows)
    df["disease"] = df["image"].str.extract(r"cauliflower_(alternaria_leaf_spot|bacterial_soft_rot)")[0]
    return {
        "n_images": int(len(df)),
        "lesion_dice_mean": float(df["dice"].mean()),
        "lesion_dice_by_disease": df.groupby("disease")["dice"].mean().round(3).to_dict(),
        "pixel_precision": tp / max(tp + fp, 1),
        "pixel_recall": tp / max(tp + fn, 1),
        "note": "Expert masks; these diseases (Alternaria, soft rot) are NOT in our training labels.",
    }


def eval_grades(eng: SegEngine) -> dict:
    g = pd.read_csv(cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv")
    g = g[g["split"] == "test"].reset_index(drop=True)
    pcts, stages = [], []
    for fp in g["filepath"]:
        p = eng.predict_pil(Image.open(cache_path_for(fp)), with_overlay=False)
        pcts.append(p["worst_region_pct"])
        stages.append(p["stage"])
    g["measured_pct"], g["measured_stage"] = pcts, stages
    dis = g["class"] != "no_disease"
    return {
        "n": int(len(g)),
        "spearman_measured_pct_vs_grade_all": float(spearmanr(g["measured_pct"], g["grade"]).statistic),
        "spearman_diseased_only": float(spearmanr(g.loc[dis, "measured_pct"], g.loc[dis, "grade"]).statistic),
        "median_measured_pct_by_grade": g.groupby("grade")["measured_pct"].median().round(1).to_dict(),
        "stage_by_grade": pd.crosstab(g["grade"], g["measured_stage"]).to_dict(),
        "healthy_flagged_diseased": float((g.loc[~dis, "measured_stage"] != "none").mean()),
    }


def contact_sheet(items: list, path: Path, tile: int = 200) -> None:
    items = items[:: max(1, len(items) // 24)][:24]
    cols = 2
    W = cols * (3 * tile + 10)
    H = ((len(items) + cols - 1) // cols) * (tile + 16)
    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, (rgb, g, p, r) in enumerate(items):
        x, y = (i % cols) * (3 * tile + 10), (i // cols) * (tile + 16)
        for j, a in enumerate((rgb, g, p)):
            t = Image.fromarray(a)
            t.thumbnail((tile, tile))
            sheet.paste(t, (x + j * tile, y))
        d.text((x + 2, y + tile + 2),
               f"#{r['idx']} {r['class'][:12]}  label {r['gt_worst_pct']:.0f}% {r['gt_stage']} | model {r['pred_worst_pct']:.0f}% {r['pred_stage']}",
               fill=(0, 0, 0))
    sheet.save(path)


def main() -> None:
    import argparse

    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="outline weights (default: the deployed ones)")
    ap.add_argument("--out-dir", default=str(SEG_DIR), help="where to write the report (Phase 3 inputs stay in outputs/seg)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {"weights": Path(args.weights)} if args.weights else {}
    device = "0" if torch.cuda.is_available() else "cpu"
    eng = SegEngine(device=device, **weights)
    held, df, items = eval_heldout(eng)
    df.to_csv(out_dir / "val_predictions.csv", index=False)
    contact_sheet(items, out_dir / "val_predictions.png")
    report = {"gates": GATES, "heldout": held, "plantseg_expert": eval_plantseg(eng), "vs_severity_grades": eval_grades(eng)}

    user = eng.predict_pil(Image.open(USER_PHOTO))
    Image.fromarray(user.pop("_overlay")).save(out_dir / "user_photo_1_overlay.png")
    report["user_photo_1"] = user

    cpu = SegEngine(device="cpu", **weights)
    im = Image.open(USER_PHOTO)
    cpu.predict_pil(im)
    t0 = time.perf_counter()
    for _ in range(5):
        cpu.predict_pil(im)
    report["cpu_seconds_per_image"] = round((time.perf_counter() - t0) / 5, 3)

    h = report["heldout"]
    report["gate_results"] = {
        "affected_pct_mae": h["affected_pct_mae_worst_region"] <= GATES["affected_pct_mae_max"],
        "lesion_dice": h["lesion_dice_mean_diseased"] >= GATES["lesion_dice_min"],
        "early_recall": (h["early_recall_measured_0p5_to_5pct"]["detected"] or 0) >= GATES["early_recall_min"],
        "healthy_false_area": h["healthy"]["mean_false_lesion_pct"] < GATES["healthy_false_area_max_pct"],
    }
    (out_dir / "seg_eval.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "heldout"} | {"heldout": {k: v for k, v in h.items() if k != "stage_confusion"}},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
