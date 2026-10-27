"""Fit the trust layer of the v2 ensemble on OUT-OF-FOLD predictions only.

Inputs : experiments/experiment_004_v2_multitask_cv/oof_fold*.npz (from train_v2.py)
Outputs: calibration.json (read by model_runtime.V2Engine), calibration_report.json,
         reliability.png

  * diagnosis temperature (NLL-optimal)          -> probabilities mean what they say
  * split-conformal LAC threshold, alpha = 0.05  -> 95% prediction sets
  * severity temperature + conformal (alpha=0.10) over grades 1-4 for diseased photos
  * presence-head temperature
  * OOD threshold = 99th percentile of OOF Mahalanobis scores (1% false rejects)
  * image-quality thresholds from the real (original) photos

  python src/calibrate_v2.py                                        # v2
  python src/calibrate_v2.py --exp-dir experiments/experiment_005_v3_multisource
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from model_runtime import quality_metrics
from train_v2 import EXP_DIR, HP, REAL_SOURCE_TYPES, SEV_FULL
from v2_cache import cache_path_for
from v2_model import CLASS_ORDER, DISEASES, coral_to_probs

ALPHA_CLS = 0.05
ALPHA_SEV = 0.10
OOD_PERCENTILE = 99.0
CONFIDENT_MIN_PROB = 0.80
PRESENCE_THRESHOLD = 0.5


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(1).mean())


def fit_temperature(fn) -> float:
    return float(minimize_scalar(fn, bounds=(0.05, 10.0), method="bounded").x)


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def sev_probs(sev_logits: np.ndarray, t: float, diseased_only: bool) -> np.ndarray:
    p = coral_to_probs(torch.as_tensor(sev_logits / t)).numpy()
    if diseased_only:
        p[:, 0] = 0.0
    return p / np.clip(p.sum(1, keepdims=True), 1e-9, None)


def load_oof(exp_dir: Path, n_folds: int) -> dict[str, np.ndarray]:
    parts = [np.load(p, allow_pickle=True) for p in sorted(exp_dir.glob("oof_fold*.npz"))]
    if len(parts) != n_folds:
        raise RuntimeError(f"expected {n_folds} OOF files, found {len(parts)}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def reliability_plot(p_before: np.ndarray, p_after: np.ndarray, y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    edges = np.linspace(0, 1, 11)
    for p, label in ((p_before, "raw"), (p_after, "temperature-scaled")):
        conf, ok = p.max(1), p.argmax(1) == y
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean())
                ys.append(ok[m].mean())
        ax.plot(xs, ys, marker="o", label=f"{label} (ECE {ece(p, y):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Diagnosis reliability (out-of-fold real photos)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def quality_thresholds(filepaths: list[str]) -> tuple[dict, dict]:
    rows = []
    for fp in filepaths:
        with Image.open(cache_path_for(fp)) as im:
            rows.append(quality_metrics(im.convert("RGB")))
    q = pd.DataFrame(rows)
    th = {
        "min_side": 128.0,
        # Half of the blurriest real photo we trained on.
        "min_sharpness": float(q["sharpness"].min() * 0.5),
        "min_brightness": float(min(25.0, q["brightness"].min() * 0.8)),
        "max_brightness": float(max(235.0, q["subject_brightness"].max())),
        "max_clipped": 0.6,
    }
    summary = {c: {"min": float(q[c].min()), "p1": float(q[c].quantile(0.01)), "median": float(q[c].median()),
                   "max": float(q[c].max())} for c in q.columns}
    return th, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=str, default=str(EXP_DIR))
    ap.add_argument("--allow-partial", action="store_true", help="ablations: calibrate on the folds that exist")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)
    exp_dir = exp_dir if exp_dir.is_absolute() else cfg.PROJECT_ROOT / exp_dir
    conf_json = json.loads((exp_dir / "config.json").read_text())
    hp = conf_json.get("hp", HP)
    classes = conf_json.get("classes", CLASS_ORDER)

    n_found = len(list(exp_dir.glob("oof_fold*.npz")))
    o = load_oof(exp_dir, n_found if args.allow_partial else hp["n_folds"])
    y, g = o["label"].astype(int), o["grade"].astype(int)
    graded = o["sev_kind"].astype(int) == SEV_FULL if "sev_kind" in o else np.ones(len(y), bool)
    logits = o["cls"]

    # Diagnosis temperature
    p_raw = softmax(logits)
    t_cls = fit_temperature(lambda t: nll(softmax(logits / t), y))
    p_cal = softmax(logits / t_cls)
    qhat_cls = conformal_qhat(1 - p_cal[np.arange(len(y)), y], ALPHA_CLS)
    sets = p_cal >= 1 - qhat_cls
    sets[np.arange(len(y)), p_cal.argmax(1)] = True

    # Severity (graded diseased photos, grades 1-4, as used at inference)
    dis = y < len(DISEASES)
    sev_d, g_d = o["sev"][dis & graded], g[dis & graded]
    t_sev = fit_temperature(lambda t: nll(sev_probs(sev_d, t, True), g_d))
    sp_raw = sev_probs(sev_d, 1.0, True)
    sp_cal = sev_probs(sev_d, t_sev, True)
    qhat_sev = conformal_qhat(1 - sp_cal[np.arange(len(g_d)), g_d], ALPHA_SEV)
    members = sp_cal >= 1 - qhat_sev
    members[:, 0] = False
    arg = sp_cal.argmax(1)
    lo = np.array([min([k for k in range(5) if members[i, k]] + [arg[i]]) for i in range(len(g_d))])
    hi = np.array([max([k for k in range(5) if members[i, k]] + [arg[i]]) for i in range(len(g_d))])

    # Presence head: in OOF data each photo has exactly its labelled disease (or none).
    pres_t = np.zeros_like(o["presence"])
    pres_t[dis, y[dis]] = 1.0

    def pres_nll(t: float) -> float:
        p = expit(o["presence"] / t)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(pres_t * np.log(p) + (1 - pres_t) * np.log(1 - p)).mean())

    t_pres = fit_temperature(pres_nll)
    p_pres = expit(o["presence"] / t_pres)

    ood_th = float(np.percentile(o["maha"], OOD_PERCENTILE))
    # Evidence floor: real photos below this calibrated top probability are rare
    # (100 - OOD_PERCENTILE %) and near coin flips; flat / junk images sit below it.
    min_evidence = float(np.percentile(p_cal.max(1), 100 - OOD_PERCENTILE))
    folds = pd.read_csv(exp_dir / "folds.csv")
    q_th, q_summary = quality_thresholds(folds.loc[folds["source_type"].isin(REAL_SOURCE_TYPES), "filepath"].tolist())

    conf = p_cal.max(1)
    confident = (sets.sum(1) == 1) & (conf >= CONFIDENT_MIN_PROB) & (o["maha"] <= ood_th) & (conf >= min_evidence)
    pred = p_cal.argmax(1)

    cal = {
        "model": exp_dir.name,
        "classes": classes,
        "image_size": hp["image_size"],
        "n_models": len(list(exp_dir.glob("fold*.pt"))),
        "tta": hp["tta"],
        "cls_temperature": t_cls,
        "cls_conformal_alpha": ALPHA_CLS,
        "cls_conformal_qhat": qhat_cls,
        "sev_temperature": t_sev,
        "sev_conformal_alpha": ALPHA_SEV,
        "sev_conformal_qhat": qhat_sev,
        "presence_temperature": t_pres,
        "presence_threshold": PRESENCE_THRESHOLD,
        "ood_threshold": ood_th,
        "ood_percentile": OOD_PERCENTILE,
        "confident_min_prob": CONFIDENT_MIN_PROB,
        "min_evidence_prob": min_evidence,
        "quality": q_th,
        "fitted_on": f"out-of-fold predictions on {len(y)} real training-pool photos; test sets untouched",
    }
    (exp_dir / "calibration.json").write_text(json.dumps(cal, indent=2))

    report = {
        "n_oof": int(len(y)),
        "diagnosis": {
            "accuracy": float((pred == y).mean()),
            "raw": {"nll": nll(p_raw, y), "ece": ece(p_raw, y), "brier": brier(p_raw, y)},
            "calibrated": {"nll": nll(p_cal, y), "ece": ece(p_cal, y), "brier": brier(p_cal, y)},
            "temperature": t_cls,
            "conformal_coverage": float(sets[np.arange(len(y)), y].mean()),
            "mean_set_size": float(sets.sum(1).mean()),
            "singleton_rate": float((sets.sum(1) == 1).mean()),
        },
        "severity_diseased": {
            "n": int(len(g_d)),
            "exact": float((arg == g_d).mean()),
            "within1": float((np.abs(arg - g_d) <= 1).mean()),
            "mae": float(np.abs(arg - g_d).mean()),
            "raw_nll": nll(sp_raw, g_d),
            "calibrated_nll": nll(sp_cal, g_d),
            "temperature": t_sev,
            "range_coverage": float(((g_d >= lo) & (g_d <= hi)).mean()),
            "mean_range_width": float((hi - lo + 1).mean()),
        },
        "presence": {
            "temperature": t_pres,
            "false_other_findings_rate": float(((p_pres >= PRESENCE_THRESHOLD) & (pres_t == 0)).any(1).mean()),
            "true_disease_detected": float((p_pres[dis, y[dis]] >= PRESENCE_THRESHOLD).mean()),
        },
        "ood": {"threshold": ood_th, "oof_reject_rate": float((o["maha"] > ood_th).mean())},
        "evidence_floor": {"min_evidence_prob": min_evidence,
                           "oof_reject_rate": float((p_cal.max(1) < min_evidence).mean())},
        "status_on_oof": {
            "confident_rate": float(confident.mean()),
            "accuracy_when_confident": float((pred[confident] == y[confident]).mean()) if confident.any() else None,
            "accuracy_when_not_confident": float((pred[~confident] == y[~confident]).mean()) if (~confident).any() else None,
        },
        "quality_metric_summary_originals": q_summary,
    }
    if "source" in o:
        report["diagnosis"]["accuracy_by_source"] = {
            str(src): float((pred[o["source"] == src] == y[o["source"] == src]).mean()) for src in np.unique(o["source"])
        }
    (exp_dir / "calibration_report.json").write_text(json.dumps(report, indent=2))
    reliability_plot(p_raw, p_cal, y, exp_dir / "reliability.png")
    print(json.dumps({k: v for k, v in report.items() if k != "quality_metric_summary_originals"}, indent=2))


if __name__ == "__main__":
    main()
