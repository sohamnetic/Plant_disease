"""One-time held-out evaluation of the frozen v2 ensemble (Experiment 4).

Runs the exact API runtime (model_runtime.V2Engine) on:
  1. splits/test.csv ORIGINAL photos (n=93) - diagnosis, calibration, conformal
     coverage, status/abstention, severity vs reviewed grades, vs Experiment 3
  2. splits/test.csv augmented photos - informational only
  3. synthetic out-of-scope / bad-photo probes built from test photos + pic.jpg
  4. synthetic co-infection composites (half disease A | half disease B)
Nothing here feeds back into training or calibration.

  python src/eval_v2_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from calibrate_v2 import brier, ece, nll, softmax
from model_runtime import V2Engine, quality_issues, quality_metrics
from v2_model import CLASS_ORDER, DISEASES

EXP_DIR = cfg.PROJECT_ROOT / "experiments" / "experiment_004_v2_multitask_cv"
EXP3_TEST = cfg.PROJECT_ROOT / "experiments" / "experiment_003_effnetb0_combined" / "test_predictions.csv"
GRADES_CSV = cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv"
RNG = np.random.default_rng(42)


def run(engine: V2Engine, images: list[Image.Image], batch: int = 16) -> list[dict]:
    out = []
    for s in range(0, len(images), batch):
        chunk = [ImageOps.exif_transpose(im).convert("RGB") for im in images[s : s + batch]]
        x = torch.stack([engine.tfm(im) for im in chunk])
        raw = engine.forward_batch(x)
        for i, im in enumerate(chunk):
            q = quality_metrics(im)
            issues = quality_issues(q, engine.cal["quality"])
            r = engine.interpret(raw, i, issues)
            r["_raw_cls"] = raw["cls"][i].cpu().numpy()
            out.append(r)
    return out


def open_rgb(fp: str) -> Image.Image:
    with Image.open(cfg.PROJECT_ROOT / fp) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1600, 1600))
        return im


def status_counts(res: list[dict]) -> dict:
    s = pd.Series([r["status"] for r in res]).value_counts()
    return {k: int(v) for k, v in s.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = V2Engine(EXP_DIR, device=device)
    test = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    grades = pd.read_csv(GRADES_CSV)[["filepath", "grade"]]
    orig = test[test.source_type == "original"].merge(grades, on="filepath", how="left").reset_index(drop=True)
    y = orig["class"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()

    # ---- 1. originals ------------------------------------------------------
    res = run(engine, [open_rgb(fp) for fp in orig.filepath])
    pred = np.array([CLASS_ORDER.index(r["prediction"]) for r in res])
    p_cal = np.array([[r["class_probabilities"][c] for c in CLASS_ORDER] for r in res])
    p_raw = softmax(np.stack([r["_raw_cls"] for r in res]))
    in_set = np.array([orig["class"][i] in r["prediction_set_95"] for i, r in enumerate(res)])
    set_size = np.array([len(r["prediction_set_95"]) for r in res])
    status = np.array([r["status"] for r in res])
    confident = status == "confident"
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=range(4), zero_division=0)

    dis = orig["class"].to_numpy() != "no_disease"
    diag_ok = pred == y
    sev_pred = np.array([r["severity"]["grade"] for r in res])
    sev_lo = np.array([r["severity"]["grade_range_90"][0] for r in res])
    sev_hi = np.array([r["severity"]["grade_range_90"][1] for r in res])
    g = orig["grade"].to_numpy().astype(int)
    m = dis & diag_ok  # severity judged where the diagnosis is right and a disease is present
    pres = np.array([[r["disease_presence"][d] for d in DISEASES] for r in res])
    false_other = np.array([len(r["other_findings"]) > 0 for r in res])

    # Experiment 3 baseline on the same 93 photos (stored predictions, no rerun)
    e3 = pd.read_csv(EXP3_TEST)
    e3_conf, e3_ok = e3["confidence"].to_numpy(), (e3["true_class"] == e3["predicted_class"]).to_numpy()
    e3_bins = np.linspace(0, 1, 11)
    e3_ece = sum(
        ((e3_conf > lo) & (e3_conf <= hi)).mean() * abs(e3_ok[(e3_conf > lo) & (e3_conf <= hi)].mean() - e3_conf[(e3_conf > lo) & (e3_conf <= hi)].mean())
        for lo, hi in zip(e3_bins[:-1], e3_bins[1:]) if ((e3_conf > lo) & (e3_conf <= hi)).any()
    )

    report: dict = {
        "test_originals": {
            "n": int(len(y)),
            "accuracy": float(diag_ok.mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, c in enumerate(CLASS_ORDER)},
            "confusion_matrix": confusion_matrix(y, pred, labels=range(4)).tolist(),
            "calibration_raw": {"nll": nll(p_raw, y), "ece": ece(p_raw, y), "brier": brier(p_raw, y)},
            "calibration_temperature_scaled": {"nll": nll(p_cal, y), "ece": ece(p_cal, y), "brier": brier(p_cal, y)},
            "mean_confidence": float(p_cal.max(1).mean()),
            "conformal_95_coverage": float(in_set.mean()),
            "conformal_mean_set_size": float(set_size.mean()),
            "status_counts": status_counts(res),
            "accuracy_when_confident": float(diag_ok[confident].mean()) if confident.any() else None,
            "accuracy_when_not_confident": float(diag_ok[~confident].mean()) if (~confident).any() else None,
            "false_other_findings_rate": float(false_other.mean()),
            "true_disease_presence_detected": float((pres[dis, y[dis]] >= 0.5).mean()),
        },
        "severity_test_originals": {
            "n_diseased_correct_diagnosis": int(m.sum()),
            "exact": float((sev_pred[m] == g[m]).mean()),
            "within1": float((np.abs(sev_pred[m] - g[m]) <= 1).mean()),
            "mae": float(np.abs(sev_pred[m] - g[m]).mean()),
            "quadratic_weighted_kappa": float(cohen_kappa_score(g[m], sev_pred[m], weights="quadratic")),
            "range90_coverage": float(((g[m] >= sev_lo[m]) & (g[m] <= sev_hi[m])).mean()),
            "mean_range_width": float((sev_hi[m] - sev_lo[m] + 1).mean()),
            "healthy_graded_0": float((sev_pred[~dis & diag_ok] == 0).mean()),
            "grade_confusion_1to4": confusion_matrix(g[m], sev_pred[m], labels=[1, 2, 3, 4]).tolist(),
        },
        "experiment_003_same_photos": {
            "accuracy": float(e3_ok.mean()),
            "mean_confidence": float(e3_conf.mean()),
            "ece": float(e3_ece),
        },
    }
    rows = []
    for i, r in enumerate(res):
        rows.append({
            "filepath": orig.filepath[i], "true_class": orig["class"][i], "true_grade": g[i],
            "prediction": r["prediction"], "confidence": r["confidence"], "status": r["status"],
            "prediction_set_95": "|".join(r["prediction_set_95"]), "grade": r["severity"]["grade"],
            "grade_range_90": f"{r['severity']['grade_range_90'][0]}-{r['severity']['grade_range_90'][1]}",
            "grade_confidence": r["severity"]["grade_confidence"], "ood_score": r["ood_score"],
            "other_findings": "|".join(o["disease"] for o in r["other_findings"]),
        })
    pd.DataFrame(rows).to_csv(EXP_DIR / "test_predictions.csv", index=False)
    hard = [r for r, fp in zip(rows, orig.filepath) if "Bacterial Spot. (166)" in fp]
    report["known_hard_case_bacterial_spot_166"] = hard[0] if hard else None

    # ---- 2. augmented (informational) --------------------------------------
    aug = test[test.source_type == "augmented"].reset_index(drop=True)
    res_aug = run(engine, [open_rgb(fp) for fp in aug.filepath])
    ya = aug["class"].to_numpy()
    report["test_augmented_informational"] = {
        "n": int(len(aug)),
        "accuracy": float(np.mean([r["prediction"] == t for r, t in zip(res_aug, ya)])),
        "status_counts": status_counts(res_aug),
    }

    # ---- 3. out-of-scope and bad-photo probes -------------------------------
    base = [open_rgb(fp) for fp in orig.sample(30, random_state=1).filepath]
    probes: dict[str, list[Image.Image]] = {
        "gaussian_noise": [Image.fromarray(RNG.integers(0, 256, (480, 480, 3), dtype=np.uint8)) for _ in range(20)],
        "flat_colour": [Image.new("RGB", (480, 480), tuple(int(v) for v in RNG.integers(0, 256, 3))) for _ in range(20)],
        "tile_scrambled_test_photos": [],
        "colour_inverted_test_photos": [ImageOps.invert(im) for im in base],
        "heavily_blurred_test_photos": [im.filter(ImageFilter.GaussianBlur(radius=max(im.size) / 60)) for im in base],
        "very_dark_test_photos": [Image.eval(im, lambda v: v // 12) for im in base],
        "non_brassica_leaf_pic_jpg": [open_rgb("pic.jpg")] if (cfg.PROJECT_ROOT / "pic.jpg").is_file() else [],
    }
    for im in base:
        a = np.asarray(im.resize((480, 480)))
        tiles = [a[r : r + 60, c : c + 60] for r in range(0, 480, 60) for c in range(0, 480, 60)]
        order = RNG.permutation(len(tiles))
        rows_ = [np.concatenate([tiles[order[r * 8 + c]] for c in range(8)], 1) for r in range(8)]
        probes["tile_scrambled_test_photos"].append(Image.fromarray(np.concatenate(rows_, 0)))
    report["probes"] = {}
    for name, ims in probes.items():
        if not ims:
            continue
        rr = run(engine, ims)
        report["probes"][name] = {
            "n": len(ims),
            "status_counts": status_counts(rr),
            "flagged_rate": float(np.mean([r["status"] != "confident" for r in rr])),
            "mean_confidence": float(np.mean([r["confidence"] for r in rr])),
            "example": {k: rr[0][k] for k in ("status", "status_reason", "prediction", "confidence", "ood_score")},
        }

    # ---- 4. co-infection composites -----------------------------------------
    by_cls = {c: orig[orig["class"] == c].filepath.tolist() for c in DISEASES}
    comps, pairs = [], []
    for a_i, a in enumerate(DISEASES):
        for b in DISEASES[a_i + 1 :]:
            for _ in range(10):
                fa, fb = RNG.choice(by_cls[a]), RNG.choice(by_cls[b])
                ia, ib = open_rgb(fa).resize((384, 384)), open_rgb(fb).resize((384, 384))
                canvas = Image.new("RGB", (768, 384))
                canvas.paste(ia, (0, 0))
                canvas.paste(ib, (384, 0))
                comps.append(canvas)
                pairs.append((a, b))
    rc = run(engine, comps)
    both = [r["disease_presence"][a] >= 0.5 and r["disease_presence"][b] >= 0.5 for r, (a, b) in zip(rc, pairs)]
    reported = [
        {a, b} <= ({r["prediction"]} | {o["disease"] for o in r["other_findings"]} | set(r["prediction_set_95"]))
        for r, (a, b) in zip(rc, pairs)
    ]
    report["coinfection_composites"] = {
        "n": len(comps),
        "both_presence_ge_0.5": float(np.mean(both)),
        "both_reported_in_prediction_set_or_other_findings": float(np.mean(reported)),
        "flagged_not_confident": float(np.mean([r["status"] != "confident" for r in rc])),
        "status_counts": status_counts(rc),
        "prediction_is_one_of_pair": float(np.mean([r["prediction"] in p for r, p in zip(rc, pairs)])),
    }

    (EXP_DIR / "test_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
