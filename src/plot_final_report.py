"""Write consolidated plots under outputs/final/. Read-only on experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

OUT = cfg.OUTPUTS_DIR / "final"
LABELS = ["bacterial_spot_rot", "black_rot", "downy_mildew", "no_disease"]
DISPLAY = [
    "bacterial\nspot rot",
    "black rot",
    "downy\nmildew",
    "no disease",
]


def plot_macro_f1() -> None:
    names = [
        "Exp 1\noriginal train\noriginal val (n=93)",
        "Exp 2\naugmented train\naugmented val (n=549)\nNOT comparable\n(same-domain / inflated)",
        "Exp 3 (selected)\ncombined train\noriginal val (n=93)",
    ]
    values = [0.973, 0.998, 1.000]
    colors = ["#4C78A8", "#E45756", "#54A24B"]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.bar(range(3), values, color=colors, width=0.62, edgecolor="black", linewidth=0.6)
    bars[1].set_hatch("//")
    ax.set_ylim(0.90, 1.02)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Validation macro-F1 by experiment\n(Exp 1 and Exp 3 share the original-only val set; Exp 2 does not)")
    ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "exp_macro_f1_comparison.png", dpi=160)
    plt.close(fig)


def plot_test_cm() -> None:
    metrics = json.loads(
        (cfg.PROJECT_ROOT / "experiments/experiment_003_effnetb0_combined/test_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(max(r) for r in cm))
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(DISPLAY, fontsize=9)
    ax.set_yticklabels(DISPLAY, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Held-out test confusion matrix\nExp 3 EfficientNet-B0 · original-only test (n=93)")
    for i in range(4):
        for j in range(4):
            val = int(cm[i][j])
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > 16 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "exp003_test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_exp3_curves() -> None:
    hist = pd.read_csv(cfg.PROJECT_ROOT / "experiments/experiment_003_effnetb0_combined/training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train loss", color="#4C78A8")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val loss (original)", color="#F58518")
    axes[0].axvline(8.5, color="0.55", linestyle="--", linewidth=0.9, label="stage 2 start")
    axes[0].axvline(13, color="#54A24B", linestyle=":", linewidth=1.0, label="best ckpt (epoch 13)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (CE, label smoothing 0.1)")
    axes[0].set_title("Experiment 3 — loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(hist["epoch"], hist["train_acc"], label="train acc", color="#4C78A8")
    axes[1].plot(hist["epoch"], hist["val_acc"], label="val acc (original)", color="#F58518")
    axes[1].axvline(8.5, color="0.55", linestyle="--", linewidth=0.9, label="stage 2 start")
    axes[1].axvline(13, color="#54A24B", linestyle=":", linewidth=1.0, label="best ckpt (epoch 13)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Experiment 3 — accuracy")
    axes[1].set_ylim(0.85, 1.01)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Selected model: EfficientNet-B0, combined train, original-only validation", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "exp003_training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_macro_f1()
    plot_test_cm()
    plot_exp3_curves()
    print("Wrote", list(OUT.glob("*.png")))


if __name__ == "__main__":
    main()
