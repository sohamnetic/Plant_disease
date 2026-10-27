"""Phase 4 evaluation: whole-photo classifier (v3) vs region-based diagnosis.

Same frozen test sets as before (external 1,142 + in-dataset 93 originals), same
scoring (src/eval_external.score). Also: healthy false alarms, stage vs grade,
status changes, the user's field photo, and CPU time.

Uses the policy in src/phase4_policy.json (chosen on validation data by
src/select_phase4_policy.py). Gates (plan): user photo -> head bacterial_spot_rot
with leaves not flagged; external not worse than v3 (correct not lower by > 1
point, wrong answers not higher).

  python src/eval_phase4.py
  python src/eval_phase4.py --out-dir outputs/insect_seg/phase4_app_v4   # re-check after the insect switch
Outputs: outputs/final/phase4_eval.json, outputs/final/phase4_predictions.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from eval_external import EXTERNAL_TEST, USER_PHOTO, healthy_false_alarm, score
from model_runtime import RELIABLE_STATUSES
from plant_runtime import PlantEngine, plant_public_result

OUT = cfg.OUTPUTS_DIR / "final"
STAGE_INDEX = {"none": 0, "early": 1, "moderate": 2, "severe": 3, "critical": 4}


def open_im(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1600, 1600))
        return im


def main() -> None:
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    OUT = Path(ap.parse_args().out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng = PlantEngine(device=dev, seg_device="0" if dev.type == "cuda" else "cpu")
    ext = pd.read_csv(EXTERNAL_TEST).assign(set="external")
    test = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    test = test[test["source_type"] == "original"].assign(set="in_dataset", subset="vegnet_test")
    grades = pd.read_csv(cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv")[["filepath", "grade"]]
    df = pd.concat([ext, test], ignore_index=True).merge(grades, on="filepath", how="left")

    rows, whole_res, plant_res = [], [], []
    for fp in df["filepath"]:
        r = eng.predict_pil(open_im(cfg.PROJECT_ROOT / fp), with_overlay=False)
        w = r.get("plant") or {}
        plant_res.append(r)
        rows.append({
            "status": r["status"], "prediction": r["prediction"], "confidence": r["confidence"],
            "stage": w.get("stage"), "affected_pct": w.get("affected_pct"),
            "whole_prediction": w.get("whole_photo_prediction", r["prediction"]),
            "n_regions": len(w.get("regions", [])),
            "unidentified_spots": any(g["verdict"].startswith("small unidentified") for g in w.get("regions", [])),
            "reason": r.get("status_reason"),
        })
    # whole-photo baseline: the same engine's classifier on the full photo
    for fp in df["filepath"]:
        whole_res.append(eng.cls.predict_pil(open_im(cfg.PROJECT_ROOT / fp)))
    pred = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    pred["v3_status"] = [r["status"] for r in whole_res]
    pred["v3_prediction"] = [r["prediction"] for r in whole_res]
    pred["v3_grade"] = [r["severity"]["grade"] for r in whole_res]
    pred.to_csv(OUT / "phase4_predictions.csv", index=False)

    report: dict = {}
    for name, sel in (("external", pred["set"] == "external"), ("in_dataset", pred["set"] == "in_dataset")):
        idx = np.nonzero(sel.to_numpy())[0]
        truth = pred.loc[sel, "class"].tolist()
        v3 = [whole_res[i] for i in idx]
        p4 = [plant_res[i] for i in idx]
        report[name] = {
            "v3_whole_photo": score(v3, truth) | {"healthy_false_alarm": healthy_false_alarm(v3, truth)},
            "phase4_region_based": score(p4, truth) | {"healthy_false_alarm": healthy_false_alarm(p4, truth)},
        }
        if name == "external":
            by = {}
            for subset, g in pred[sel].groupby("subset"):
                ii = g.index.to_numpy()
                by[subset] = {"v3": score([whole_res[i] for i in ii], g["class"].tolist()),
                              "phase4": score([plant_res[i] for i in ii], g["class"].tolist())}
            report[name]["by_subset"] = by

    rel = pred["status"].isin(RELIABLE_STATUSES)
    healthy = pred["class"] == "no_disease"
    report["healthy_photos"] = {
        "n": int(healthy.sum()),
        "phase4_stage_not_none_rate": float(((pred["stage"].fillna("none") != "none") & rel & healthy).sum() / healthy.sum()),
        "phase4_unidentified_spot_note_rate": float((pred["unidentified_spots"] & healthy).sum() / healthy.sum()),
    }
    changed = pred[(pred["status"] != pred["v3_status"]) | (pred["prediction"] != pred["v3_prediction"])]
    report["changes_vs_v3"] = {
        "n_changed": int(len(changed)),
        "status_transitions": changed.groupby(["v3_status", "status"]).size().rename("n").reset_index().to_dict("records"),
        "fixed": int(((changed["prediction"] == changed["class"]) & changed["status"].isin(RELIABLE_STATUSES)
                      & ~((changed["v3_prediction"] == changed["class"]) & changed["v3_status"].isin(RELIABLE_STATUSES))).sum()),
        "broken": int(((changed["v3_prediction"] == changed["class"]) & changed["v3_status"].isin(RELIABLE_STATUSES)
                       & ~((changed["prediction"] == changed["class"]) & changed["status"].isin(RELIABLE_STATUSES))).sum()),
        "top_uncertain_reasons": pred.loc[pred["status"] == "uncertain", "reason"].str.split(";").str[0].value_counts().head(6).to_dict(),
    }
    g = pred[(pred["set"] == "in_dataset") & rel & (pred["class"] != "no_disease") & (pred["prediction"] == pred["class"])]
    si = g["stage"].map(STAGE_INDEX)
    report["stage_vs_grade_in_dataset"] = {
        "n": int(len(g)),
        "within_one": float((si - g["grade"]).abs().le(1).mean()),
        "exact": float((si == g["grade"]).mean()),
        "early_grade1_detected_as_diseased": float(g.loc[g["grade"] == 1, "stage"].ne("none").mean()) if (g["grade"] == 1).any() else None,
        "crosstab": pd.crosstab(g["grade"], g["stage"]).to_dict(),
    }

    u = plant_public_result(eng.predict_pil(Image.open(USER_PHOTO)))
    report["user_photo_1"] = {"status": u["status"], "prediction": u["prediction"], "message": u["message"],
                              "regions": [{k: r[k] for k in ("type", "disease", "affected_pct", "verdict")} for r in u["plant"]["regions"]]}
    cpu = PlantEngine()
    im = Image.open(USER_PHOTO)
    cpu.predict_pil(im)
    t0 = time.perf_counter()
    for _ in range(3):
        cpu.predict_pil(im)
    report["cpu_seconds_user_photo"] = round((time.perf_counter() - t0) / 3, 2)

    e4, e3 = report["external"]["phase4_region_based"], report["external"]["v3_whole_photo"]
    heads = [r for r in report["user_photo_1"]["regions"] if r["type"] == "curd"]
    leaves = [r for r in report["user_photo_1"]["regions"] if r["type"] == "leaf"]
    report["gates"] = {
        "user_photo_head_bsr_leaves_not_flagged": bool(heads and heads[0]["disease"] == "bacterial_spot_rot"
                                                        and all(r["disease"] == "no_disease" for r in leaves)),
        "external_correct_not_lower_by_more_than_1pt": e4["correct"] >= e3["correct"] - 0.01,
        "external_wrong_answers_not_higher": e4["harmful"] <= e3["harmful"] + 1e-9,
    }
    (OUT / "phase4_eval.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
