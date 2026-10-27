"""Phase 5: replace by-eye severity grades with MEASURED grades.

  python src/measure_severity.py measure   # measure all 618 originals, write overlays + review sheets
  python src/measure_severity.py apply     # merge the reviewed disagreements into severity_grades.csv

measure
  For every VegNet original in outputs/severity/severity_grades.csv:
    * if the photo has a reviewed segmentation label (Phase 3), measure from that label
    * otherwise run the YOLO11-seg model (seg_runtime.SegEngine)
  % affected = worst leaf / head, grade = stage index from src/stages.json
  (none 0, early 1, moderate 2, severe 3, critical 4) - the same scale the app shows.
  Diseased photos where nothing measurable is found are "unmeasurable" and keep
  their by-eye grade. Healthy photos stay 0.
  Photos whose measured and by-eye grades differ by 2+ go to review:
  outputs/severity/measured_review_sheets/*.png (photo | overlay).

apply
  Reads outputs/severity/measured_review.txt ("idx M" keep measured, "idx V" keep
  by-eye, "idx <0-4>" set grade) and writes the final grade column:
    grade          final grade used for training
    grade_visual   the earlier by-eye grade
    measured_pct / measured_grade / measurement_source
    grade_source   measured | measured_reviewed | visual_reviewed | reviewed_set |
                   visual_unmeasurable | visual_measurement_untrusted (bacterial spot rot) | rule_healthy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from seg_eval import gt_masks
from seg_runtime import STAGES, WORK_SIDE, SegEngine, measure, overlay
from v2_cache import cache_path_for

GRADES = cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv"
MEASURED = cfg.OUTPUTS_DIR / "severity" / "measured_grades.csv"
REVIEW = cfg.OUTPUTS_DIR / "severity" / "measured_review.txt"
OVERLAY_DIR = cfg.OUTPUTS_DIR / "severity" / "overlays"
SHEET_DIR = cfg.OUTPUTS_DIR / "severity" / "measured_review_sheets"
YOLO_DIR = cfg.PROJECT_ROOT / "data_external" / "seg_yolo"
STAGE_INDEX = {s: i for i, s in enumerate(STAGES["stages"])}
# Diseases whose measurement is trusted. Bacterial spot rot is excluded: the
# segmentation model treats dark purple/brown rotten curd as background, so its
# measured % stays ~5-8% from grade 2 to 4 (see outputs/final/PHASE5_REPORT.md).
MEASURED_CLASSES = {"black_rot", "downy_mildew"}


def overlay_path(filepath: str) -> Path:
    return OVERLAY_DIR / (cache_path_for(filepath).stem + ".jpg")


def label_file_for() -> dict[str, Path]:
    """filepath -> reviewed YOLO label file (Phase 3), for the photos that have one."""
    lab = pd.read_csv(cfg.OUTPUTS_DIR / "seg" / "labelled_images.csv")
    out = {}
    for _, r in lab.iterrows():
        name = f"{r['idx']:04d}_{r['source']}_{r['class']}"
        for split in ("train", "val"):
            f = YOLO_DIR / "labels" / split / f"{name}.txt"
            if f.is_file():
                out[r["filepath"]] = f
    return out


def run_measure() -> None:
    import torch

    g = pd.read_csv(GRADES)
    eng = SegEngine(device="0" if torch.cuda.is_available() else "cpu")
    labels = label_file_for()
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in g.iterrows():
        im = Image.open(cache_path_for(r["filepath"])).convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE))
        rgb = np.asarray(im)
        if r["filepath"] in labels:
            regions, lesion = gt_masks(labels[r["filepath"]], *rgb.shape[:2])
            src = "reviewed_label"
        else:
            regions, lesion = eng.raw_masks(rgb)
            src = "seg_model"
        m = measure(regions, lesion)
        Image.fromarray(overlay(rgb, m, lesion)).save(overlay_path(r["filepath"]), quality=88)
        stage = m["stage"]
        if r["class"] == "no_disease":
            mg, note = 0, "healthy label"
        elif stage == "none":
            mg, note = None, "unmeasurable: no lesions found"
        else:
            mg, note = STAGE_INDEX[stage], ""
        rows.append({"idx": i, "measured_pct": m["worst_region_pct"], "measured_lesions": m["lesion_count"],
                     "measured_stage": stage, "measured_grade": mg, "measurement_source": src, "note": note})
    out = pd.concat([g.reset_index(drop=True), pd.DataFrame(rows).drop(columns="idx")], axis=1)
    out.insert(0, "idx", np.arange(len(out)))
    out.to_csv(MEASURED, index=False)

    dis = out["class"] != "no_disease"
    meas = dis & out["measured_grade"].notna()
    diff = (out["measured_grade"] - out["grade"]).abs()
    review = out[meas & (diff >= 2)]
    print(f"diseased {int(dis.sum())}: measured {int(meas.sum())}, unmeasurable {int((dis & ~meas).sum())}")
    print("measured vs by-eye (diseased, measured):")
    print(pd.crosstab(out.loc[meas, "grade"], out.loc[meas, "measured_grade"].astype(int),
                      rownames=["by-eye"], colnames=["measured"]))
    print(f"exact {float((diff[meas] == 0).mean()):.3f}  within1 {float((diff[meas] <= 1).mean()):.3f}  "
          f"to review (2+ apart): {len(review)}")
    render_sheets(review)


def render_sheets(df: pd.DataFrame, per_sheet: int = 8, tile: int = 260) -> None:
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    for old in SHEET_DIR.glob("*.png"):
        old.unlink()
    for s_i, start in enumerate(range(0, len(df), per_sheet)):
        chunk = df.iloc[start : start + per_sheet]
        sheet = Image.new("RGB", (2 * (2 * tile + 12), ((len(chunk) + 1) // 2) * (tile + 16)), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for j, (_, r) in enumerate(chunk.iterrows()):
            a = Image.open(cache_path_for(r["filepath"])).convert("RGB")
            b = Image.open(overlay_path(r["filepath"])).convert("RGB")
            a.thumbnail((tile, tile))
            b.thumbnail((tile, tile))
            x, y = (j % 2) * (2 * tile + 12), (j // 2) * (tile + 16)
            sheet.paste(a, (x, y))
            sheet.paste(b, (x + tile, y))
            d.text((x + 2, y + tile + 2),
                   f"#{r['idx']} {r['class'][:14]}  by-eye g{int(r['grade'])} | measured g{int(r['measured_grade'])} "
                   f"({r['measured_pct']:.0f}%, {r['measured_lesions']} spots, {r['measurement_source'][:4]})",
                   fill=(0, 0, 0))
        sheet.save(SHEET_DIR / f"sheet_{s_i:03d}.png")
    print(f"{len(df)} photos on review sheets -> {SHEET_DIR}")


def run_apply() -> None:
    m = pd.read_csv(MEASURED)
    decisions = {}
    if REVIEW.is_file():
        for line in REVIEW.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                idx, dec, *_ = line.split()
                decisions[int(idx)] = dec
    diff = (m["measured_grade"] - m["grade"]).abs()
    need = m[(m["class"] != "no_disease") & m["measured_grade"].notna() & (diff >= 2)]["idx"]
    missing = sorted(set(need) - set(decisions))
    if missing:
        raise SystemExit(f"{len(missing)} disputed photos not reviewed yet, e.g. {missing[:10]}")

    final, source = [], []
    for _, r in m.iterrows():
        if r["class"] == "no_disease":
            final.append(0); source.append("rule_healthy")
        elif pd.isna(r["measured_grade"]):
            final.append(int(r["grade"])); source.append("visual_unmeasurable")
        elif r["class"] not in MEASURED_CLASSES and r["idx"] not in decisions:
            final.append(int(r["grade"])); source.append("visual_measurement_untrusted")
        elif r["idx"] in decisions:
            dec = decisions[r["idx"]]
            if dec == "M":
                final.append(int(r["measured_grade"])); source.append("measured_reviewed")
            elif dec == "V":
                final.append(int(r["grade"])); source.append("visual_reviewed")
            else:
                final.append(int(dec)); source.append("reviewed_set")
        else:
            final.append(int(r["measured_grade"])); source.append("measured")
    out = m[["filepath", "class", "split", "group_id", "prefill_pct", "prefill_grade"]].copy()
    out["grade_visual"] = m["grade"]
    out["measured_pct"] = m["measured_pct"]
    out["measured_lesions"] = m["measured_lesions"]
    out["measured_grade"] = m["measured_grade"]
    out["measurement_source"] = m["measurement_source"]
    out["grade"] = final
    out["grade_source"] = source
    out.to_csv(GRADES, index=False)
    print(out["grade_source"].value_counts().to_string())
    print(pd.crosstab(out["class"], out["grade"]))


if __name__ == "__main__":
    {"measure": run_measure, "apply": run_apply}[sys.argv[1]]()
