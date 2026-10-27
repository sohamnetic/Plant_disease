"""Head-to-head: the CNN alone vs the CNN + YOLO11-seg pipeline the app runs.

Both systems are scored on the SAME photos from ONE pass each:
  * CNN only     = the EfficientNet-B0 ensemble's whole-photo answer (diagnosis,
                   status, CORAL severity grade -> stage). No % affected exists.
  * CNN + YOLO   = plant_runtime.PlantEngine (the app): CNN decides the disease,
                   YOLO11-seg outlines leaves/heads/lesions and measures % affected;
                   stage from the measurement (bacterial spot rot: from the grade).

Sets
  diagnosis : frozen external test (1,142), in-dataset VegNet test originals (93),
              web photos (new_pics, copies of known photos excluded)
  severity  : the 73 segmentation held-out photos with reviewed lesion outlines
              (never trained on by either model): reference % and stage come
              from those outlines
  speed     : 30 photos, wall-clock per photo on this machine

  python src/eval_cnn_vs_cnn_yolo.py
Output: outputs/final/cnn_vs_cnn_yolo.json (+ printed tables)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from model_runtime import RELIABLE_STATUSES
from plant_runtime import PlantEngine
from v2_model import GRADE_LABELS

ROOT = cfg.PROJECT_ROOT
OUT = cfg.OUTPUTS_DIR / "final" / "cnn_vs_cnn_yolo.json"
STAGE_ORDER = ["none", "early", "moderate", "severe", "critical"]


def run_one(eng: PlantEngine, path: Path) -> dict:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    a = eng.analyse(im)
    whole = a["whole"]
    cnn_status = "retake_photo" if a["issues"] else whole["status"]
    cnn_grade = whole["severity"]["grade"]
    cnn_stage = "none" if whole["prediction"] == "no_disease" else GRADE_LABELS[max(cnn_grade, 1)]
    d = eng.decide(a)
    plant = d.get("plant") or {}
    return {
        "cnn_status": cnn_status, "cnn_pred": whole["prediction"], "cnn_conf": whole["confidence"],
        "cnn_stage": cnn_stage,
        "cy_status": d["status"], "cy_pred": d["prediction"], "cy_conf": d.get("confidence"),
        "cy_stage": plant.get("stage"), "cy_pct": plant.get("affected_pct"),
        "cy_regions": len(plant.get("regions", [])),
    }


def diag_scores(df: pd.DataFrame, p: str) -> dict:
    st, pred, y = df[f"{p}_status"], df[f"{p}_pred"], df["y"]
    rel = st.isin(RELIABLE_STATUSES)
    ok = pred == y
    healthy = y == "no_disease"
    return {
        "n": int(len(df)),
        "correct": float((rel & ok).mean()),
        "wrong_answer": float((rel & ~ok).mean()),
        "confident_wrong": float(((st == "confident") & ~ok).mean()),
        "abstained": float((~rel).mean()),
        "accuracy_when_answering": float(ok[rel].mean()) if rel.any() else None,
        "healthy_false_alarm": float((rel & healthy & (pred != "no_disease")).sum() / max(healthy.sum(), 1)),
    }


def main() -> None:
    eng = PlantEngine()
    sets: dict[str, pd.DataFrame] = {}
    ext = pd.read_csv(ROOT / "splits_v3" / "external_test.csv")
    sets["external_test"] = pd.DataFrame({"file": ext["filepath"], "y": ext["class"]})
    t = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    t = t[t["source_type"] == "original"]
    sets["in_dataset_test"] = pd.DataFrame({"file": t["filepath"], "y": t["class"]})
    w = pd.read_csv(ROOT / "new_pics" / "manifest.csv", dtype={"is_copy": str})
    w = w[w["is_copy"].isin(["False", "0.0"])]
    sets["web_photos"] = pd.DataFrame({"file": w["file"], "y": w["cls"]})
    v = pd.read_csv(cfg.OUTPUTS_DIR / "seg" / "val_predictions.csv")
    li = pd.read_csv(cfg.OUTPUTS_DIR / "seg" / "labelled_images.csv")[["idx", "filepath"]]
    v = v.merge(li, on="idx")
    sets["seg_heldout"] = pd.DataFrame({"file": v["filepath"], "y": v["class"], "ref_pct": v["gt_worst_pct"],
                                        "ref_stage": v["gt_stage"]})

    report: dict = {"diagnosis": {}, "severity_seg_heldout": {}, "speed": {}}
    frames = {}
    for name, df in sets.items():
        rows = [run_one(eng, ROOT / f) for f in df["file"]]
        frames[name] = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        print(f"{name}: {len(df)} photos done", flush=True)
    for name in ("external_test", "in_dataset_test", "web_photos"):
        f = frames[name]
        report["diagnosis"][name] = {"cnn_only": diag_scores(f, "cnn"), "cnn_yolo": diag_scores(f, "cy"),
                                     "same_answer_rate": float(((f.cnn_pred == f.cy_pred) & (f.cnn_status == f.cy_status)).mean())}

    s = frames["seg_heldout"]
    idx = {k: i for i, k in enumerate(STAGE_ORDER)}
    ref = s["ref_stage"].map(idx)
    out = {}
    for p, col in (("cnn_only", "cnn_stage"), ("cnn_yolo", "cy_stage")):
        pred = s[col].fillna("none").map(idx)
        dis = s["y"] != "no_disease"
        early = s["ref_stage"] == "early"
        out[p] = {
            "stage_exact": float((pred == ref).mean()),
            "stage_within_1": float(((pred - ref).abs() <= 1).mean()),
            "stage_exact_diseased": float((pred == ref)[dis].mean()),
            "early_cases_flagged": float((pred[early] >= 1).mean()),
            "early_cases_exact_early": float((pred[early] == 1).mean()),
            "healthy_given_a_stage": float((pred[~dis] >= 1).mean()),
        }
    m = s["cy_pct"].notna() & (s["y"] != "no_disease")
    out["cnn_yolo"]["affected_pct_mae_points"] = float((s.loc[m, "cy_pct"] - s.loc[m, "ref_pct"]).abs().mean())
    out["cnn_yolo"]["affected_pct_measured_share"] = float(m.sum() / (s["y"] != "no_disease").sum())
    out["cnn_only"]["affected_pct_mae_points"] = None
    out["n"] = int(len(s))
    out["n_diseased"] = int((s["y"] != "no_disease").sum())
    out["n_early"] = int((s["ref_stage"] == "early").sum())
    report["severity_seg_heldout"] = out

    # speed: same 30 photos, classifier only vs full pipeline
    files = [ROOT / f for f in sets["in_dataset_test"]["file"].head(30)]
    ims = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in files]
    eng.cls.interpret(eng.cls.forward_batch(eng.cls.tfm(ims[0])[None]), 0, [])  # warm-up
    eng.predict_pil(ims[0], with_overlay=False)
    t0 = time.perf_counter()
    for im in ims:
        eng.cls.interpret(eng.cls.forward_batch(eng.cls.tfm(im)[None]), 0, [])
    t1 = time.perf_counter()
    for im in ims:
        eng.predict_pil(im, with_overlay=True)
    t2 = time.perf_counter()
    report["speed"] = {"cnn_only_s_per_photo": (t1 - t0) / len(ims), "cnn_yolo_s_per_photo": (t2 - t1) / len(ims),
                       "device_classifier": str(eng.cls.device), "device_segmentation": "cpu"}

    OUT.write_text(json.dumps(report, indent=2))
    for name, r in report["diagnosis"].items():
        print(f"\n{name}  (same answer in both: {r['same_answer_rate']:.1%})")
        print(pd.DataFrame({"CNN only": r["cnn_only"], "CNN + YOLO": r["cnn_yolo"]}).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nseverity on 73 segmentation held-out photos")
    print(pd.DataFrame({k: v for k, v in out.items() if isinstance(v, dict)}).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nspeed", report["speed"])


if __name__ == "__main__":
    main()
