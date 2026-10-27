"""Step I-2: how well the outline model finds insect holes and measures % eaten.

Never-trained-on photos only:
  * insect test, reviewed K (outlines checked): measured % eaten vs the reviewed outline %
    (also with the leaf pasted on a random field background, holes see-through)
  * insect test, reviewed D (damage visible, no checked outline): share where holes are found
  * reviewed N (no visible damage, never used for training): share where holes are found
  * healthy controls: cleaf Healthy photos of the frozen external test (white backdrop; some
    really have holes) and field healthy photos (SPPU, vegleaf) of the frozen external test

"found" = the plant's insect stage is not "none" (stages.json `insect.min_pct`, >= 1 hole).

  python src/insect_seg_eval.py --weights experiments/seg_002_insect/weights/best.pt
Outputs: <out-dir>/insect_seg_eval.json, insect_seg_predictions.csv, sheet_k_test.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from insect_prelabel import cache_path, load_work
from insect_seg_dataset import BG_DIR, composite
from scene_aug import SceneComposer
from seg_runtime import WORK_SIDE, SegEngine, overlay

REVIEW = cfg.OUTPUTS_DIR / "insect_seg" / "review.csv"
FROZEN = cfg.PROJECT_ROOT / "splits_v3" / "external_test.csv"


def run(eng: SegEngine, rgb: np.ndarray) -> tuple[dict, np.ndarray]:
    return eng.measure_all(rgb)


def load_any(filepath: str) -> np.ndarray:
    im = Image.open(cfg.PROJECT_ROOT / filepath).convert("RGB")
    im.thumbnail((WORK_SIDE, WORK_SIDE))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-dir", default=str(cfg.OUTPUTS_DIR / "insect_seg" / "eval"))
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eng = SegEngine(Path(args.weights), device=args.device)
    random.seed(0)
    np.random.seed(0)
    scene = SceneComposer(BG_DIR)

    rev = pd.read_csv(REVIEW)
    frozen = pd.read_csv(FROZEN)
    sets = {
        "insect_test_K": rev[(rev["split"] == "insect_test") & (rev["review"] == "K")],
        "insect_test_D": rev[(rev["split"] == "insect_test") & (rev["review"] == "D")],
        "no_visible_damage_N": rev[rev["review"] == "N"],
        "healthy_cleaf_white": frozen[(frozen["source"] == "cleaf") & (frozen["class"] == "no_disease")],
        "healthy_field_sppu_vegleaf": frozen[frozen["source"].isin(["sppu", "vegleaf"]) & (frozen["class"] == "no_disease")],
    }
    rows, sheet = [], []
    for name, df in sets.items():
        for _, r in df.iterrows():
            rgb = load_work(r["filepath"]) if name.startswith(("insect", "no_visible")) else load_any(r["filepath"])
            m, les = run(eng, rgb)
            row = {"set": name, "filepath": r["filepath"], "pred_eaten_pct": m["worst_region_eaten_pct"],
                   "pred_holes": m["hole_count"], "found": m["insect_stage"] != "none", "pred_stage": m["insect_stage"]}
            if name == "insect_test_K":
                row["ref_eaten_pct"] = r["eaten_pct"]
                masks = np.load(cache_path(r["filepath"]))
                field = composite(rgb, masks["leaf"], scene)
                mf, _ = run(eng, field)
                row["field_pred_eaten_pct"], row["field_found"] = mf["worst_region_eaten_pct"], mf["insect_stage"] != "none"
                if len(sheet) < 24:
                    sheet.append((rgb, overlay(rgb, m, les), field, overlay(field, mf, np.zeros_like(les)), row))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "insect_seg_predictions.csv", index=False)

    k = df[df["set"] == "insect_test_K"]
    rep = {"weights": str(args.weights),
           "found_rate": df.groupby("set")["found"].mean().round(4).to_dict(),
           "n": df.groupby("set").size().to_dict(),
           "mean_pred_eaten_pct": df.groupby("set")["pred_eaten_pct"].mean().round(3).to_dict(),
           "K_eaten_mae_points": float((k["pred_eaten_pct"] - k["ref_eaten_pct"]).abs().mean()),
           "K_eaten_median_abs_err": float((k["pred_eaten_pct"] - k["ref_eaten_pct"]).abs().median()),
           "K_eaten_spearman": float(spearmanr(k["ref_eaten_pct"], k["pred_eaten_pct"]).statistic),
           "K_ref_mean_eaten_pct": float(k["ref_eaten_pct"].mean()),
           "K_field_background_found_rate": float(k["field_found"].mean()),
           "K_field_background_eaten_mae_points": float((k["field_pred_eaten_pct"] - k["ref_eaten_pct"]).abs().mean()),
           "K_field_background_spearman": float(spearmanr(k["ref_eaten_pct"], k["field_pred_eaten_pct"]).statistic)}
    (out / "insect_seg_eval.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))

    t = 220
    img = Image.new("RGB", (4 * t * 2, ((len(sheet) + 1) // 2) * (t + 16)), "white")
    d = ImageDraw.Draw(img)
    for i, (a, b, c, e, row) in enumerate(sheet):
        x, y = (i % 2) * 4 * t, (i // 2) * (t + 16)
        for j, arr in enumerate((a, b, c, e)):
            tile = Image.fromarray(arr)
            tile.thumbnail((t, t))
            img.paste(tile, (x + j * t, y))
        d.text((x + 3, y + t + 1), f"{Path(row['filepath']).name}  reviewed {row['ref_eaten_pct']:.1f}%  "
               f"model {row['pred_eaten_pct']:.1f}%  on field bg {row['field_pred_eaten_pct']:.1f}%", fill="black")
    img.save(out / "sheet_k_test.jpg", quality=85)


if __name__ == "__main__":
    main()
