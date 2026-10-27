"""Plot the per-epoch curves from train_curves_v3.py (classifier) and the YOLO11-seg run.

    python src/plot_curves_v3.py
Output: outputs/final/report_figs/curves_v3.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

HIST = cfg.PROJECT_ROOT / "experiments" / "curves_v3_fold0" / "history_fold0.csv"
YOLO = cfg.PROJECT_ROOT / "experiments" / "seg_001_yolo11s" / "results.csv"
OUT = cfg.OUTPUTS_DIR / "final" / "report_figs" / "curves_v3.png"
TRAIN, VAL, INK, GRID = "#40916C", "#1E2B23", "#1E2B23", "#E3E8E5"


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, color=INK, loc="left", fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def main() -> None:
    h = pd.read_csv(HIST)
    y = pd.read_csv(YOLO)
    y.columns = [c.strip() for c in y.columns]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    ax.plot(h.epoch, h.train_acc * 100, color=TRAIN, lw=2, marker="o", ms=3, label="Train")
    ax.plot(h.epoch, h.val_acc * 100, color=VAL, lw=2, ls="--", marker="o", ms=3, label="Validation")
    last = h.iloc[-1]
    ax.annotate(f"{last.val_acc:.1%}", (last.epoch, last.val_acc * 100), textcoords="offset points", xytext=(-8, -16),
                fontsize=9, color=VAL, ha="right")
    style(ax, "Accuracy (EfficientNet-B0)", "Accuracy (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    ax = axes[1]
    ax.plot(h.epoch, h.train_loss, color=TRAIN, lw=2, marker="o", ms=3, label="Train")
    ax.plot(h.epoch, h.val_loss, color=VAL, lw=2, ls="--", marker="o", ms=3, label="Validation")
    style(ax, "Loss (EfficientNet-B0)", "Cross-entropy loss")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[2]
    ax.plot(y["epoch"], y["train/seg_loss"], color=TRAIN, lw=2, label="Train")
    ax.plot(y["epoch"], y["val/seg_loss"], color=VAL, lw=2, ls="--", label="Validation")
    style(ax, "Loss (YOLO11-seg, lesion outlines)", "Mask loss")
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
