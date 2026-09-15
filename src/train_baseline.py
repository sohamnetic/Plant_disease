"""Experiment 001: EfficientNet-B0, original images only, validation-only eval.

Does not read or evaluate splits/test.csv.
Does not regenerate splits.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    LeafImageDataset,
    build_transforms,
    load_original_split,
)
from viz import save_image_grid

EXPERIMENT_DIR = cfg.PROJECT_ROOT / "experiments" / "experiment_001_effnetb0_original"
BEST_CKPT = EXPERIMENT_DIR / "best_model.pt"
HISTORY_CSV = EXPERIMENT_DIR / "training_history.csv"
CURVES_PNG = EXPERIMENT_DIR / "training_curves.png"
VAL_PRED_CSV = EXPERIMENT_DIR / "validation_predictions.csv"
VAL_METRICS_JSON = EXPERIMENT_DIR / "validation_metrics.json"
CONFIG_JSON = EXPERIMENT_DIR / "config.json"

STAGE1_EPOCHS = 8
STAGE2_MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 8
DROPOUT = 0.4
HEAD_LR_S1 = 1e-3
BACKBONE_LR_S2 = 1e-5
HEAD_LR_S2 = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8  # CPU
NUM_WORKERS = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def report_device() -> torch.device:
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("Running on CPU (torch CPU build).")
    return torch.device("cpu")


def build_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT, inplace=True),
        nn.Linear(in_features, len(cfg.CLASS_ORDER)),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_final_mbconv(model: nn.Module) -> None:
    """Unfreeze EfficientNet features[7] (final MBConv stage) + classifier."""
    for p in model.features.parameters():
        p.requires_grad = False
    last_mbconv = model.features[7]
    for p in last_mbconv.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    n_bb = sum(p.numel() for p in last_mbconv.parameters())
    n_head = sum(p.numel() for p in model.classifier.parameters())
    print(f"Unfroze final MBConv features[7] ({n_bb} params) + head ({n_head} params)")


def trainable_param_groups(model: nn.Module, backbone_lr: float, head_lr: float):
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head.append(p)
        else:
            backbone.append(p)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    return groups


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_y, all_pred, all_conf, all_paths = [], [], [], []
    for xb, yb, paths in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        bs = yb.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_y.extend(yb.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())
        all_paths.extend(list(paths))
    y_true = np.array(all_y)
    y_pred = np.array(all_pred)
    acc = float((y_true == y_pred).mean()) if n else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
        "conf": np.array(all_conf),
        "paths": all_paths,
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
        bs = yb.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        n += bs
    return total_loss / max(n, 1), correct / max(n, 1)


def plot_history(history: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0, 0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(history["epoch"], history["train_acc"], label="train")
    axes[0, 1].plot(history["epoch"], history["val_acc"], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()
    axes[1, 0].plot(history["epoch"], history["val_macro_f1"], color="#6A1B9A")
    axes[1, 0].set_title("Val macro-F1 (checkpoint metric)")
    axes[1, 1].plot(history["epoch"], history["lr_head"], label="head LR")
    axes[1, 1].plot(history["epoch"], history["lr_backbone"], label="backbone LR")
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.axvline(
            history.loc[history["stage"] == 2, "epoch"].min()
            if (history["stage"] == 2).any()
            else 0,
            color="gray",
            ls="--",
            alpha=0.5,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_heatmap(cm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(cfg.CLASS_ORDER, rotation=25, ha="right")
    ax.set_yticklabels(cfg.CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation confusion matrix (original-only)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
        (pred_df["predicted_class"] == "black_rot")
        & (pred_df["true_class"] != "black_rot")
    ]

    # confusion-type contact sheets
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
            cfg.REVIEW_DIR / f"exp001_miscls_{item['true']}_as_{item['pred']}.png",
            title=f"Val miscls: {item['true']} → {item['pred']} (n={item['count']})",
            cols=min(5, max(1, item["count"])),
        )

    if not wrong.empty:
        paths = [str(cfg.PROJECT_ROOT / p) for p in wrong["filepath"].tolist()]
        labels = [
            f"{t[:8]}>{p[:8]} {c:.2f}"
            for t, p, c in zip(wrong["true_class"], wrong["predicted_class"], wrong["confidence"])
        ]
        save_image_grid(
            paths,
            labels,
            cfg.REVIEW_DIR / "exp001_all_val_misclassified.png",
            title=f"All validation misclassifications (n={len(wrong)})",
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
        "confusion_pairs_desc": pairs,
        "highest_confidence_wrong": high.to_dict(orient="records"),
        "lowest_confidence": low_all.to_dict(orient="records"),
    }


def main() -> None:
    set_seed(cfg.RANDOM_SEED)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    device = report_device()

    train_df = load_original_split("train")
    val_df = load_original_split("val")
    print("Original-only counts MATCH expected frozen split.")
    print("train", train_df["class"].value_counts().reindex(cfg.CLASS_ORDER).to_dict())
    print("val  ", val_df["class"].value_counts().reindex(cfg.CLASS_ORDER).to_dict())

    train_loader = DataLoader(
        LeafImageDataset(train_df, build_transforms(True)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        LeafImageDataset(val_df, build_transforms(False)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model().to(device)
    print("features blocks:", list(model.features.named_children())[:9])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_cfg = {
        "experiment": "experiment_001_effnetb0_original",
        "model": "efficientnet_b0",
        "weights": "IMAGENET1K_V1",
        "classes": cfg.CLASS_ORDER,
        "source_filter": "original",
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
        "notes": "Do not evaluate test.csv. Splits not regenerated.",
    }
    CONFIG_JSON.write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    history_rows = []
    best_f1 = -1.0
    best_epoch = 0
    global_epoch = 0

    freeze_backbone(model)
    opt = AdamW(trainable_param_groups(model, 0.0, HEAD_LR_S1), weight_decay=WEIGHT_DECAY)
    print(f"=== Stage 1: frozen backbone, {STAGE1_EPOCHS} epochs, head lr={HEAD_LR_S1} ===")
    for e in range(1, STAGE1_EPOCHS + 1):
        global_epoch += 1
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        val = evaluate(model, val_loader, criterion, device)
        lrs = [g["lr"] for g in opt.param_groups]
        row = {
            "epoch": global_epoch,
            "stage": 1,
            "stage_epoch": e,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr_head": lrs[-1],
            "lr_backbone": 0.0,
        }
        history_rows.append(row)
        print(
            f"S1 {e}/{STAGE1_EPOCHS}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={val['loss']:.4f} acc={val['acc']:.3f} macroF1={val['macro_f1']:.3f}"
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = global_epoch
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
        f"bb={BACKBONE_LR_S2} head={HEAD_LR_S2}, early stop patience={EARLY_STOP_PATIENCE} ==="
    )
    stale = 0
    for e in range(1, STAGE2_MAX_EPOCHS + 1):
        global_epoch += 1
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, opt, device, True)
        val = evaluate(model, val_loader, criterion, device)
        lrs = {k: g["lr"] for k, g in zip(["backbone", "head"][: len(opt.param_groups)], opt.param_groups)}
        row = {
            "epoch": global_epoch,
            "stage": 2,
            "stage_epoch": e,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr_head": lrs.get("head", HEAD_LR_S2),
            "lr_backbone": lrs.get("backbone", BACKBONE_LR_S2),
        }
        if len(opt.param_groups) == 2:
            row["lr_backbone"] = opt.param_groups[0]["lr"]
            row["lr_head"] = opt.param_groups[1]["lr"]
        history_rows.append(row)
        print(
            f"S2 {e}/{STAGE2_MAX_EPOCHS}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={val['loss']:.4f} acc={val['acc']:.3f} macroF1={val['macro_f1']:.3f}"
        )
        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1 = val["macro_f1"]
            best_epoch = global_epoch
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
            y_true,
            y_pred,
            target_names=cfg.CLASS_ORDER,
            zero_division=0,
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
    metrics["error_analysis"] = {
        k: v
        for k, v in err.items()
        if k not in ("highest_confidence_wrong", "lowest_confidence", "confusion_pairs_desc")
    }
    VAL_METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(metrics["classification_report"])
    print("Best epoch", metrics["best_epoch"], "val acc", metrics["accuracy"], "macroF1", metrics["macro_f1"])
    print("Wrote", EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
