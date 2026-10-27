"""Grad-CAM heat maps for the deployed v3 classifier (5-fold ensemble), for slides / reports.

Each map is the average of the 5 fold models' Grad-CAM on the last EfficientNet-B0
block, for the class the ensemble predicts. Only photos no fold trained on are used:
the frozen external test set, or (--set vegnet) the 93 VegNet original test photos.

    python src/gradcam_v3.py --candidates [--set vegnet]   # contact sheet per class
    python src/gradcam_v3.py --files a.jpg b.jpg   # final figure from chosen photos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from model_runtime import DEFAULT_EXP
from v2_model import CLASS_ORDER, MultiTaskNet, eval_transforms, letterbox, load_rgb

OUT = cfg.OUTPUTS_DIR / "final" / "report_figs"
PRED_CSV = cfg.OUTPUTS_DIR / "final" / "confusion_predictions.csv"
EXT_CSV = cfg.PROJECT_ROOT / "splits_v3" / "external_test.csv"
NICE = {"bacterial_spot_rot": "Bacterial spot rot", "black_rot": "Black rot",
        "downy_mildew": "Downy mildew", "no_disease": "Healthy"}
# Field photos where a source has them; black rot only exists on leaves on white / 2025 set.
PREFERRED = {"bacterial_spot_rot": ["sppu"], "black_rot": ["c2h", "cleaf"],
             "downy_mildew": ["sppu"], "no_disease": ["sppu"]}


def load_models(exp_dir: Path) -> list[MultiTaskNet]:
    models = []
    for p in sorted(exp_dir.glob("fold*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = MultiTaskNet(pretrained=False, num_classes=len(CLASS_ORDER))
        m.load_state_dict({k: v.float() for k, v in ck["model_state"].items()})
        models.append(m.eval())
    return models


def gradcam(models: list[MultiTaskNet], x: torch.Tensor) -> tuple[np.ndarray, int, float]:
    with torch.no_grad():
        probs = torch.stack([F.softmax(m(x)["cls"], 1) for m in models]).mean(0)[0]
    c = int(probs.argmax())
    cams = []
    for m in models:
        store = {}
        h1 = m.features[-1].register_forward_hook(lambda _m, _i, o: store.__setitem__("a", o))
        h2 = m.features[-1].register_full_backward_hook(lambda _m, _gi, go: store.__setitem__("g", go[0]))
        m.zero_grad(set_to_none=True)
        m(x)["cls"][0, c].backward()
        h1.remove()
        h2.remove()
        w = store["g"][0].mean((1, 2))
        cam = F.relu((w[:, None, None] * store["a"][0]).sum(0)).detach()
        cams.append(cam / cam.max() if cam.max() > 0 else cam)
    cam = torch.stack(cams).mean(0)[None, None]
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
    return cam / max(cam.max(), 1e-8), c, float(probs[c])


def prepare(path: Path, size: int) -> tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as im:
        rgb = load_rgb(im)
    x = eval_transforms(size)(rgb)[None]
    shown = np.asarray(letterbox(rgb).resize((size, size), Image.Resampling.BILINEAR))
    return x, shown


def figure(items: list[tuple[np.ndarray, np.ndarray, str]], out: Path, ncols: int) -> None:
    nrows = 2 * ((len(items) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.1 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (img, cam, title) in enumerate(items):
        r, col = 2 * (i // ncols), i % ncols
        axes[r, col].imshow(img)
        axes[r, col].set_title(title, fontsize=11)
        axes[r + 1, col].imshow(img)
        axes[r + 1, col].imshow(cam, cmap="jet", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--set", choices=["external", "vegnet"], default="external")
    ap.add_argument("--out", default="gradcam_v3.png")
    args = ap.parse_args()

    models = load_models(DEFAULT_EXP)
    size = 384
    if args.candidates:
        pred = pd.read_csv(PRED_CSV)
        if args.set == "vegnet":
            pred = pred[(pred["set"] == "in_dataset_test") & (pred["efficientnet_b0"] == pred["y"])]
            pred = pred.assign(source="vegnet")
        else:
            pred = pred[(pred["set"] == "external_test") & (pred["efficientnet_b0"] == pred["y"])]
            src = pd.read_csv(EXT_CSV)[["filepath", "source"]].rename(columns={"filepath": "file"})
            pred = pred.merge(src, on="file")
        for cls in CLASS_ORDER:
            pool = pred[(pred["y"] == cls) & pred["source"].isin(PREFERRED[cls] + ["vegnet"])]
            pick = pool.sample(min(args.per_class, len(pool)), random_state=0)
            items = []
            for f in pick["file"]:
                x, shown = prepare(cfg.PROJECT_ROOT / f, size)
                cam, c, p = gradcam(models, x)
                items.append((shown, cam, f"{Path(f).name[:28]}\n{NICE[CLASS_ORDER[c]]} {p:.2f}"))
            name = f"gradcam_v3_candidates_{args.set}_{cls}.png"
            figure(items, OUT / name, ncols=len(items))
            print("wrote", OUT / name)
    if args.files:
        items = []
        for f in args.files:
            x, shown = prepare(cfg.PROJECT_ROOT / f, size)
            cam, c, p = gradcam(models, x)
            items.append((shown, cam, f"{NICE[CLASS_ORDER[c]]} ({p:.0%})"))
        figure(items, OUT / args.out, ncols=len(items))
        print("wrote", OUT / args.out)


if __name__ == "__main__":
    main()
