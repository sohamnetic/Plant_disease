"""Run our algorithm (EfficientNet-B0 ensemble + YOLO11s-seg, the app's PlantEngine) on the
test photos and show its confusion matrices in a window.

  Disease  outside photos (1,142, splits_v3/external_test.csv) and the VegNet test
           originals (93): the disease the algorithm names (its top answer). YOLO never
           changes the disease, so the classifier part answers this fast.
  Stage    the 73 YOLO held-out photos with reviewed lesion outlines: the full pipeline,
           where YOLO measures the affected area and sets the stage.
None of these photos was used for training. Results are also printed in this console
(scikit-learn classification report) and saved to outputs/final/our_algorithm_confusion.png.

  python src/show_confusion.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from eval_cnn_vs_cnn_yolo import run_one
from eval_three_way import STAGE_ORDER, load, sets
from plant_runtime import PlantEngine
from v2_model import CLASS_ORDER

ROOT = cfg.PROJECT_ROOT
NAMES = ["Bacterial\nspot rot", "Black rot", "Downy\nmildew", "Healthy"]
STAGES = ["None", "Early", "Moderate", "Severe", "Critical"]


def progress(name: str, i: int, n: int, t0: float) -> None:
    print(f"\r  {name}: {i}/{n} photos  ({time.time() - t0:.0f} s)", end="", flush=True)


def main() -> None:
    print("Loading our algorithm (5 EfficientNet-B0 models + YOLO11s-seg) ...", flush=True)
    eng = PlantEngine()
    s = sets()
    results = {}
    for key, title in (("external_test", "Outside photos"), ("in_dataset_test", "VegNet test")):
        df, pred, t0 = s[key], [], time.time()
        for i, f in enumerate(df["file"], 1):
            x = eng.cls.tfm(load(ROOT / f))[None]
            pred.append(eng.cls.interpret(eng.cls.forward_batch(x), 0, [])["prediction"])
            if i % 20 == 0 or i == len(df):
                progress(title, i, len(df), t0)
        print()
        results[key] = (title, df["y"].to_numpy(), np.array(pred))

    df, ref, got, t0 = s["seg_heldout"], [], [], time.time()
    for i, (f, r) in enumerate(zip(df["file"], df["ref_stage"]), 1):
        o = run_one(eng, ROOT / f)
        ref.append(r)
        got.append(o["cy_stage"] or "none")
        progress("Stage (full pipeline)", i, len(df), t0)
    print("\n")

    labels = [n.replace("\n", " ") for n in NAMES]
    for key, (title, y, p) in results.items():
        print("=" * 70)
        print(f"{title} ({len(y)} photos): accuracy {100 * (y == p).mean():.1f}%")
        print(pd.DataFrame(confusion_matrix(y, p, labels=CLASS_ORDER),
                           index=[f"true {n}" for n in labels], columns=labels).to_string())
        print()
        print(classification_report(y, p, labels=CLASS_ORDER, target_names=labels, digits=3, zero_division=0))
    ref, got = np.array(ref), np.array(got)
    print("=" * 70)
    print(f"Stage on {len(ref)} held-out photos: exact {100 * (ref == got).mean():.1f}%")
    print(pd.DataFrame(confusion_matrix(ref, got, labels=STAGE_ORDER),
                       index=[f"true {n}" for n in STAGES], columns=STAGES).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (key, (title, y, p)) in zip(axes, results.items()):
        ConfusionMatrixDisplay(confusion_matrix(y, p, labels=CLASS_ORDER), display_labels=NAMES).plot(
            ax=ax, cmap="Greens", colorbar=False, values_format="d")
        ax.set_title(f"Disease: {title} ({len(y)})\n{100 * (y == p).mean():.1f}% correct", fontweight="bold")
    ConfusionMatrixDisplay(confusion_matrix(ref, got, labels=STAGE_ORDER), display_labels=STAGES).plot(
        ax=axes[2], cmap="Greens", colorbar=False, values_format="d")
    axes[2].set_title(f"Stage: YOLO held-out ({len(ref)})\n{100 * (ref == got).mean():.1f}% exact", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Our algorithm: EfficientNet-B0 + YOLO11s-seg", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = cfg.OUTPUTS_DIR / "final" / "our_algorithm_confusion.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}\nClose the window to finish.")
    plt.show()


if __name__ == "__main__":
    main()
