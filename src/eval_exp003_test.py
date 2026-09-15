"""One-time held-out evaluation of frozen Experiment 3. No training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import IDX_TO_CLASS, LeafImageDataset, build_transforms
from train_baseline import build_model, evaluate, save_confusion_heatmap
from viz import save_image_grid

EXP_DIR = cfg.PROJECT_ROOT / "experiments" / "experiment_003_effnetb0_combined"
CKPT = EXP_DIR / "best_model.pt"
VAL_PRED = EXP_DIR / "validation_predictions.csv"
BATCH_SIZE = 32


def load_test(source_type: str) -> pd.DataFrame:
    df = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    out = df[df["source_type"] == source_type].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"No {source_type} rows in test.csv")
    return out


def low_confidence_val_review() -> pd.DataFrame:
    val = pd.read_csv(VAL_PRED)
    low = val[val["confidence"] < 0.60].sort_values("confidence")
    low.to_csv(EXP_DIR / "val_low_confidence.csv", index=False)
    if not low.empty:
        paths = [str(cfg.PROJECT_ROOT / p) for p in low["filepath"]]
        labels = [
            f"{Path(p).name}\n{t}->{pr} {c:.2f}"
            for p, t, pr, c in zip(
                low["filepath"], low["true_class"], low["predicted_class"], low["confidence"]
            )
        ]
        save_image_grid(
            paths,
            labels,
            cfg.REVIEW_DIR / "exp003_low_confidence.png",
            title=f"Exp3 val predictions with confidence < 0.60 (n={len(low)}; all still correct)",
            cols=min(5, len(low)),
        )
    print("Low-confidence val (<0.60):", len(low))
    print(low[["filepath", "true_class", "predicted_class", "confidence"]].to_string(index=False))
    return low


def run_split(model, device, criterion, frame: pd.DataFrame, tag: str) -> dict:
    loader = DataLoader(
        LeafImageDataset(frame, build_transforms(False)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val = evaluate(model, loader, criterion, device)
    y_true, y_pred = val["y_true"], val["y_pred"]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0
    )
    pred_df = pd.DataFrame(
        {
            "filepath": val["paths"],
            "true_class": [IDX_TO_CLASS[i] for i in y_true],
            "predicted_class": [IDX_TO_CLASS[i] for i in y_pred],
            "confidence": val["conf"],
        }
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))
    wrong = pred_df[pred_df["true_class"] != pred_df["predicted_class"]]
    metrics = {
        "subset": tag,
        "n": int(len(pred_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            cfg.CLASS_ORDER[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(4)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=cfg.CLASS_ORDER, zero_division=0
        ),
        "n_wrong": int(len(wrong)),
        "confidence": {
            "min": float(pred_df["confidence"].min()),
            "max": float(pred_df["confidence"].max()),
            "mean": float(pred_df["confidence"].mean()),
            "median": float(pred_df["confidence"].median()),
            "n_below_0_60": int((pred_df["confidence"] < 0.60).sum()),
        },
    }
    return metrics, pred_df, wrong, cm


def main() -> None:
    cfg.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print("=== TASK 1: val low-confidence review ===")
    low_confidence_val_review()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device", device)
    model = build_model()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("=== TASK 2: original-only TEST ===")
    orig_test = load_test("original")
    print("original test counts", orig_test["class"].value_counts().reindex(cfg.CLASS_ORDER).to_dict())
    metrics, pred_df, wrong, cm = run_split(model, device, criterion, orig_test, "test_original")
    pred_df.to_csv(EXP_DIR / "test_predictions.csv", index=False)
    save_confusion_heatmap(cm, EXP_DIR / "test_confusion_matrix.png")
    if not wrong.empty:
        wrong.sort_values("confidence", ascending=False).to_csv(
            EXP_DIR / "test_highest_confidence_wrong.csv", index=False
        )
        paths = [str(cfg.PROJECT_ROOT / p) for p in wrong["filepath"]]
        labels = [
            f"{t[:8]}>{p[:8]} {c:.2f}"
            for t, p, c in zip(wrong["true_class"], wrong["predicted_class"], wrong["confidence"])
        ]
        save_image_grid(
            paths,
            labels,
            cfg.REVIEW_DIR / "exp003_test_misclassified.png",
            title=f"Exp3 ORIGINAL test misclassifications (n={len(wrong)})",
            cols=min(5, max(1, len(wrong))),
        )
    pred_df.sort_values("confidence").head(10).to_csv(EXP_DIR / "test_lowest_confidence.csv", index=False)
    (EXP_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics["classification_report"])
    print("orig test acc", metrics["accuracy"], "macroF1", metrics["macro_f1"], "wrong", metrics["n_wrong"])
    print("conf", metrics["confidence"])

    print("=== TASK 3: augmented-only TEST (informational) ===")
    aug_test = load_test("augmented")
    aug_metrics, aug_pred, aug_wrong, aug_cm = run_split(
        model, device, criterion, aug_test, "test_augmented_informational"
    )
    aug_pred.to_csv(EXP_DIR / "test_predictions_augmented_informational.csv", index=False)
    (EXP_DIR / "test_metrics_augmented_informational.json").write_text(
        json.dumps(aug_metrics, indent=2), encoding="utf-8"
    )
    print(aug_metrics["classification_report"])
    print("aug test acc", aug_metrics["accuracy"], "macroF1", aug_metrics["macro_f1"], "wrong", aug_metrics["n_wrong"])

    print("Frozen checkpoint used:", CKPT)
    print("No retraining.")


if __name__ == "__main__":
    main()
