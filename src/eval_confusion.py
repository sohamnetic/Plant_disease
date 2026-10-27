"""Confusion matrices for the report: normal CNN vs EfficientNet-B0 vs EfficientNet-B0 + YOLO.

Diagnosis (every photo answered with the top class, no 'not sure'), on the frozen
external test (1,142), the VegNet test originals (93) and the web photos (81):
  normal CNN       experiments/baseline_simple_cnn
  EfficientNet-B0  the deployed ensemble's whole-photo answer
  + YOLO           names the same disease as EfficientNet-B0 on every photo
                   (cnn_vs_cnn_yolo.json, same_answer_rate 1.0), so its matrix equals column 2's
Stage, on the 73 segmentation held-out photos with reviewed lesion outlines:
  all three columns; column 3 runs the app's PlantEngine (YOLO measures the stage).

The accuracies are checked against outputs/final/three_way_comparison.json.
Nothing here changes the deployed models.

  python src/eval_confusion.py               (add --plot-only to redraw from the saved CSV)
Output: outputs/final/confusion_predictions.csv, outputs/final/confusion_matrices.json,
        outputs/final/report_figs/confusion_*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from eval_cnn_vs_cnn_yolo import run_one
from eval_three_way import STAGE_ORDER, load, sets, simple_predict
from plant_runtime import PlantEngine
from train_simple_cnn import OUT as SIMPLE_DIR, SimpleCNN
from v2_model import CLASS_ORDER, eval_transforms

ROOT = cfg.PROJECT_ROOT
FINAL = cfg.OUTPUTS_DIR / "final"
FIG = FINAL / "report_figs"
CLASS_SHORT = {"bacterial_spot_rot": "Bacterial\nspot rot", "black_rot": "Black rot",
               "downy_mildew": "Downy\nmildew", "no_disease": "Healthy"}
SET_TITLE = {"external_test": "Outside photos (1,142)", "in_dataset_test": "VegNet test (93)",
             "web_photos": "Web photos (81)", "seg_heldout": "Segmentation held-out (73)"}


def predict_all() -> pd.DataFrame:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(SIMPLE_DIR / "model.pt", map_location="cpu", weights_only=False)
    simple = SimpleCNN()
    simple.load_state_dict(ck["model_state"])
    simple.to(dev).eval()
    eng = PlantEngine()
    tfm = eval_transforms(384)
    rows = []
    for name, df in sets().items():
        for _, r in df.iterrows():
            im = load(ROOT / r["file"])
            p, g = simple_predict(simple, tfm(im)[None].to(dev))
            s_pred = CLASS_ORDER[int(p[0].argmax())]
            row = {"set": name, "file": r["file"], "y": r["y"], "normal_cnn": s_pred}
            if name == "seg_heldout":
                o = run_one(eng, ROOT / r["file"])
                row.update({"efficientnet_b0": o["cnn_pred"], "ref_stage": r["ref_stage"],
                            "stage_normal_cnn": "none" if s_pred == "no_disease" else STAGE_ORDER[max(int(g[0]), 1)],
                            "stage_efficientnet_b0": o["cnn_stage"],
                            "stage_efficientnet_b0_yolo": o["cy_stage"] or "none"})
            else:
                x = eng.cls.tfm(im)[None]
                row["efficientnet_b0"] = eng.cls.interpret(eng.cls.forward_batch(x), 0, [])["prediction"]
            rows.append(row)
        print(name, "done", flush=True)
    return pd.DataFrame(rows)


def heatmap(ax, cm: np.ndarray, labels: list[str], title: str, ylabel: str = "True") -> None:
    """Counts in each cell, shaded by the share of that row (true class) so classes of any size compare."""
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(share, cmap="Greens", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm.sum(1)[i] == 0:
                continue
            dark = share[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]}\n{100 * share[i, j]:.0f}%", ha="center", va="center", fontsize=7.6,
                    color="white" if dark else "#1d2b24", fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(labels)), labels, fontsize=7.6)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.6)
    ax.set_xlabel("Predicted", fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    for k in np.arange(0.5, len(labels) - 1):
        ax.axhline(k, color="white", linewidth=2)
        ax.axvline(k, color="white", linewidth=2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    plt.rcParams.update({"font.family": "Arial"})
    FIG.mkdir(parents=True, exist_ok=True)
    csv = FINAL / "confusion_predictions.csv"
    if "--plot-only" in sys.argv and csv.exists():
        df = pd.read_csv(csv)
    else:
        df = predict_all()
        df.to_csv(csv, index=False)
    tw = json.loads((FINAL / "three_way_comparison.json").read_text())["raw_diagnosis"]
    out: dict = {"class_order": CLASS_ORDER, "stage_order": STAGE_ORDER, "diagnosis": {}, "stage_seg_heldout": {}}
    labels = [CLASS_SHORT[c] for c in CLASS_ORDER]

    for name in ("external_test", "in_dataset_test", "web_photos"):
        d = df[df["set"] == name]
        out["diagnosis"][name] = {}
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
        for ax, m, t in zip(axes, ("normal_cnn", "efficientnet_b0"),
                            ("Normal CNN", "EfficientNet-B0  (= EfficientNet-B0 + YOLO)")):
            cm = confusion_matrix(d["y"], d[m], labels=CLASS_ORDER)
            acc = np.trace(cm) / cm.sum()
            ref = tw[name][m]["accuracy"]
            assert abs(acc - ref) < 0.006, f"{name} {m}: {acc:.4f} vs stored {ref:.4f}"
            out["diagnosis"][name][m] = cm.tolist()
            heatmap(ax, cm, labels, f"{t}\n{100 * acc:.1f}% correct")
        fig.suptitle(SET_TITLE[name], fontsize=10.5, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(FIG / f"confusion_{name}.png", dpi=180)
        plt.close()

    s = df[df["set"] == "seg_heldout"]
    st_labels = [x.capitalize() if x != "none" else "None\n(healthy)" for x in STAGE_ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1))
    for ax, m, t in zip(axes, ("normal_cnn", "efficientnet_b0", "efficientnet_b0_yolo"),
                        ("Normal CNN", "EfficientNet-B0", "EfficientNet-B0 + YOLO")):
        cm = confusion_matrix(s["ref_stage"], s[f"stage_{m}"], labels=STAGE_ORDER)
        out["stage_seg_heldout"][m] = cm.tolist()
        heatmap(ax, cm, st_labels, f"{t}\n{100 * np.trace(cm) / cm.sum():.1f}% stage exact",
                "Reference stage" if m == "normal_cnn" else "")
    plt.tight_layout()
    plt.savefig(FIG / "confusion_stage.png", dpi=180)
    plt.close()

    (FINAL / "confusion_matrices.json").write_text(json.dumps(out, indent=2))
    for name, r in out["diagnosis"].items():
        print(f"\n{name}  rows = true {CLASS_ORDER}")
        for m, cm in r.items():
            print(m, cm)
    print("\nstage rows = reference", STAGE_ORDER)
    for m, cm in out["stage_seg_heldout"].items():
        print(m, cm)


if __name__ == "__main__":
    main()
