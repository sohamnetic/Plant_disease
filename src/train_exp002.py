"""Experiment 002: EfficientNet-B0, augmented images only, validation-only eval.

Same methodology as Experiment 001. Does not read splits/test.csv.
Does not modify experiment_001_effnetb0_original.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import IDX_TO_CLASS, LeafImageDataset, build_transforms, load_split
from train_baseline import (
    DROPOUT,
    EARLY_STOP_PATIENCE,
    HEAD_LR_S1,
    HEAD_LR_S2,
    BACKBONE_LR_S2,
    STAGE1_EPOCHS,
    STAGE2_MAX_EPOCHS,
    WEIGHT_DECAY,
    build_model,
    evaluate,
    freeze_backbone,
    plot_history,
    report_device,
    run_epoch,
    save_confusion_heatmap,
    set_seed,
    trainable_param_groups,
    unfreeze_final_mbconv,
)
from viz import save_image_grid

EXPERIMENT_DIR = cfg.PROJECT_ROOT / "experiments" / "experiment_002_effnetb0_augmented"
BEST_CKPT = EXPERIMENT_DIR / "best_model.pt"
HISTORY_CSV = EXPERIMENT_DIR / "training_history.csv"
CURVES_PNG = EXPERIMENT_DIR / "training_curves.png"
VAL_PRED_CSV = EXPERIMENT_DIR / "validation_predictions.csv"
VAL_METRICS_JSON = EXPERIMENT_DIR / "validation_metrics.json"
CONFIG_JSON = EXPERIMENT_DIR / "config.json"

BATCH_SIZE = 32
NUM_WORKERS = 0
SHEET_PREFIX = "exp002"


def write_error_analysis(pred_df: pd.DataFrame, cm: np.ndarray) -> dict:
    wrong = pred_df[pred_df["true_class"] != pred_df["predicted_class"]].copy()
    high = wrong.sort_values("confidence", ascending=False).head(10)
    low_all = pred_df.sort_values("confidence", ascending=True).head(10)

    pairs = []
    for i, t in enumerate(cfg.CLASS_ORDER):
        for j, p in enumerate(cfg.CLASS_ORDER):
            if i == j:
                continue
            pairs.append({"true": t, "pred": p, "count": int(cm[i, j])})
    pairs = sorted(pairs, key=lambda x: -x["count"])

    br = pred_df[pred_df["true_class"] == "black_rot"]
    br_wrong = br[br["true_class"] != br["predicted_class"]]
    br_fp = pred_df[
        (pred_df["predicted_class"] == "black_rot") & (pred_df["true_class"] != "black_rot")
    ]

    for item in pairs:
        if item["count"] == 0:
            continue
        sub = wrong[
            (wrong["true_class"] == item["true"]) & (wrong["predicted_class"] == item["pred"])
        ]
        paths = [str(cfg.PROJECT_ROOT / p) for p in sub["filepath"].tolist()]
        labels = [
            f"{Path(p).name} c={c:.2f}"
            for p, c in zip(sub["filepath"], sub["confidence"])
        ]
        save_image_grid(
            paths,
            labels,
            cfg.REVIEW_DIR / f"{SHEET_PREFIX}_miscls_{item['true']}_as_{item['pred']}.png",
            title=f"Exp002 val miscls: {item['true']} → {item['pred']} (n={item['count']})",
            cols=min(5, max(1, min(item["count"], 8))),
        )

    if not wrong.empty:
        show = wrong.head(40)
        paths = [str(cfg.PROJECT_ROOT / p) for p in show["filepath"].tolist()]
        labels = [
            f"{t[:8]}>{p[:8]} {c:.2f}"
            for t, p, c in zip(show["true_class"], show["predicted_class"], show["confidence"])
        ]
        save_image_grid(
            paths,
            labels,
            cfg.REVIEW_DIR / f"{SHEET_PREFIX}_all_val_misclassified.png",
            title=f"Exp002 validation misclassifications (showing {len(show)} of {len(wrong)})",
            cols=5,
        )

    high.to_csv(EXPERIMENT_DIR / "highest_confidence_wrong.csv", index=False)
    low_all.to_csv(EXPERIMENT_DIR / "lowest_confidence.csv", index=False)
    pd.DataFrame(pairs).to_csv(EXPERIMENT_DIR / "confusion_pairs.csv", index=False)

    return {
        "n_val": int(len(pred_df)),
        "n_wrong": int(len(wrong)),
        "black_rot_support": int(len(br)),
        "black_rot_errors": int(len(br_wrong)),
        "black_rot_false_positives": int(len(br_fp)),
        "black_rot_true_positives": int(len(br) - len(br_wrong)),
    }


def main() -> None:
    set_seed(cfg.RANDOM_SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    device = report_device()
    if device.type != "cuda":
        print("WARNING: CUDA not available; proceeding on CPU with batch 32 is slow.")

    train_df = load_split("train", "augmented")
    val_df = load_split("val", "augmented")
    print("Augmented-only frozen-split counts:")
    print("train", train_df["class"].value_counts().reindex(cfg.CLASS_ORDER).to_dict(), "n", len(train_df))
    print("val  ", val_df["class"].value_counts().reindex(cfg.CLASS_ORDER).to_dict(), "n", len(val_df))

    pin = device.type == "cuda"
    train_loader = DataLoader(
        LeafImageDataset(train_df, build_transforms(True)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        LeafImageDataset(val_df, build_transforms(False)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_cfg = {
        "experiment": "experiment_002_effnetb0_augmented",
        "model": "efficientnet_b0",
        "weights": "IMAGENET1K_V1",
        "classes": cfg.CLASS_ORDER,
        "source_filter": "augmented",
        "seed": cfg.RANDOM_SEED,
        "cuda": torch.cuda.is_available(),
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "label_smoothing": 0.1,
        "class_weights": None,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_max_epochs": STAGE2_MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "head_lr_stage1": HEAD_LR_S1,
        "backbone_lr_stage2": BACKBONE_LR_S2,
        "head_lr_stage2": HEAD_LR_S2,
        "weight_decay": WEIGHT_DECAY,
        "checkpoint_metric": "val_macro_f1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Same methodology as exp001. Do not evaluate test.csv. Splits not regenerated.",
        "comparison_baseline": "experiment_001_effnetb0_original val macro-F1 0.973",
    }
    CONFIG_JSON.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    history_rows = []
    best_f1 = -1.0
    global_epoch = 0

    freeze_backbone(model)
    opt = AdamW(trainable_param_groups(model, 0.0, HEAD_LR_S1), weight_decay=WEIGHT_DECAY)
    print(f"=== Stage 1: frozen backbone, {STAGE1_EPOCHS} epochs, head lr={HEAD_LR_S1}, bs={BATCH_SIZE} ===")
    for e in range(1, STAGE1_EPOCHS + 1):
        global_epoch += 1
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        val = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": global_epoch,
            "stage": 1,
            "stage_epoch": e,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr_head": HEAD_LR_S1,
            "lr_backbone": 0.0,
        }
        history_rows.append(row)
        print(
            f"S1 {e}/{STAGE1_EPOCHS}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={val['loss']:.4f} acc={val['acc']:.3f} macroF1={val['macro_f1']:.3f}"
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state": model.state_dict(),
                    "val_macro_f1": best_f1,
                    "val_acc": val["acc"],
                },
                BEST_CKPT,
            )
            print(f"  saved best (macro-F1={best_f1:.4f})")

    unfreeze_final_mbconv(model)
    opt = AdamW(
        trainable_param_groups(model, BACKBONE_LR_S2, HEAD_LR_S2),
        weight_decay=WEIGHT_DECAY,
    )
    print(
        f"=== Stage 2: last MBConv + head, max {STAGE2_MAX_EPOCHS} epochs, "
        f"bb={BACKBONE_LR_S2} head={HEAD_LR_S2} ==="
    )
    stale = 0
    for e in range(1, STAGE2_MAX_EPOCHS + 1):
        global_epoch += 1
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        val = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": global_epoch,
            "stage": 2,
            "stage_epoch": e,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr_head": HEAD_LR_S2,
            "lr_backbone": BACKBONE_LR_S2,
        }
        history_rows.append(row)
        print(
            f"S2 {e}/{STAGE2_MAX_EPOCHS}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={val['loss']:.4f} acc={val['acc']:.3f} macroF1={val['macro_f1']:.3f}"
        )
        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1 = val["macro_f1"]
            stale = 0
            torch.save(
                {
                    "epoch": global_epoch,
                    "model_state": model.state_dict(),
                    "val_macro_f1": best_f1,
                    "val_acc": val["acc"],
                },
                BEST_CKPT,
            )
            print(f"  saved best (macro-F1={best_f1:.4f})")
        else:
            stale += 1
            print(f"  no improvement ({stale}/{EARLY_STOP_PATIENCE})")
            if stale >= EARLY_STOP_PATIENCE:
                print("Early stopping on val macro-F1.")
                break

    hist = pd.DataFrame(history_rows)
    hist.to_csv(HISTORY_CSV, index=False)
    plot_history(hist, CURVES_PNG)

    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val = evaluate(model, val_loader, criterion, device)
    y_true, y_pred = val["y_true"], val["y_pred"]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(4)), zero_division=0
    )
    metrics = {
        "best_epoch": int(ckpt["epoch"]),
        "best_val_macro_f1_during_train": float(ckpt["val_macro_f1"]),
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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=cfg.CLASS_ORDER, zero_division=0
        ),
        "experiment_001_macro_f1": 0.973,
        "macro_f1_delta_vs_exp001": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0) - 0.973
        ),
    }
    pred_df = pd.DataFrame(
        {
            "filepath": val["paths"],
            "true_class": [IDX_TO_CLASS[i] for i in y_true],
            "predicted_class": [IDX_TO_CLASS[i] for i in y_pred],
            "confidence": val["conf"],
        }
    )
    pred_df.to_csv(VAL_PRED_CSV, index=False)
    cm = np.array(metrics["confusion_matrix"])
    save_confusion_heatmap(cm, EXPERIMENT_DIR / "confusion_matrix.png")
    err = write_error_analysis(pred_df, cm)
    metrics["error_analysis"] = err
    VAL_METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics["classification_report"])
    print("Best epoch", metrics["best_epoch"], "val acc", metrics["accuracy"], "macroF1", metrics["macro_f1"])
    print("Wrote", EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
