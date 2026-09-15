"""Grad-CAM diagnostic for Experiment 001 (no training, validation only).

Uses the saved EfficientNet-B0 checkpoint. Does not read splits/test.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.models import efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import CLASS_TO_IDX, IDX_TO_CLASS, build_transforms

EXPERIMENT_DIR = cfg.PROJECT_ROOT / "experiments" / "experiment_001_effnetb0_original"
CKPT = EXPERIMENT_DIR / "best_model.pt"
PRED_CSV = EXPERIMENT_DIR / "validation_predictions.csv"
OUT_DIR = cfg.REVIEW_DIR / "gradcam"
N_CORRECT_PER_CLASS = 7
SEED = cfg.RANDOM_SEED
DROPOUT = 0.4


class GradCAM:
    """Vanilla Grad-CAM on EfficientNet-B0's last conv (features[-1])."""

    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _m, _inp, out):
        self.activations = out

    def _backward_hook(self, _m, _gin, gout):
        self.gradients = gout[0]

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()
        grads = self.gradients[0]  # C,H,W
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logits.detach()


def denorm(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(cfg.IMAGENET_MEAN, device=t.device)[:, None, None]
    std = torch.tensor(cfg.IMAGENET_STD, device=t.device)[:, None, None]
    img = (t * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return arr


def colorize_cam(cam: np.ndarray, size: int) -> Image.Image:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    cam_img = cam_img.resize((size, size), Image.BILINEAR)
    heat = np.array(cam_img).astype(np.float32) / 255.0
    # matplotlib-like jet without importing pyplot in the hot path
    r = np.clip(1.5 - np.abs(4 * heat - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heat - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[heat < 0.05] = 0
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = Image.fromarray(rgb).convert("RGB")
    heat = colorize_cam(cam, base.size[0])
    return Image.blend(base, heat, alpha)


def panel(left: Image.Image, right: Image.Image, caption: str) -> Image.Image:
    pad, cap_h = 8, 48
    w, h = left.size
    canvas = Image.new("RGB", (2 * w + 3 * pad, h + 2 * pad + cap_h), (245, 245, 245))
    canvas.paste(left, (pad, pad + cap_h))
    canvas.paste(right, (2 * pad + w, pad + cap_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 8), caption, fill=(20, 20, 20), font=font)
    draw.text((pad, pad + cap_h - 2), "model crop", fill=(80, 80, 80), font=font)
    draw.text((2 * pad + w, pad + cap_h - 2), "Grad-CAM overlay", fill=(80, 80, 80), font=font)
    return canvas


def load_model(device: torch.device, ckpt_path: Path) -> torch.nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=DROPOUT, inplace=True),
        torch.nn.Linear(in_features, len(cfg.CLASS_ORDER)),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True  # grads needed for CAM; no optimizer step
    return model


def preprocess(path: Path, device: torch.device) -> torch.Tensor:
    tfm = build_transforms(train=False)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        t = tfm(im)
    return t.unsqueeze(0).to(device)


def slug(name: str) -> str:
    return Path(name).stem.replace(" ", "_").replace("(", "").replace(")", "")


def select_samples(preds: pd.DataFrame, include_wrong: bool = True) -> pd.DataFrame:
    rows = []
    correct = preds[preds["true_class"] == preds["predicted_class"]]
    rng = np.random.RandomState(SEED)
    for cls in cfg.CLASS_ORDER:
        sub = correct[correct["true_class"] == cls]
        n = min(N_CORRECT_PER_CLASS, len(sub))
        # mix: take some high-confidence and some lower-confidence
        sub = sub.sort_values("confidence", ascending=False)
        hi = sub.head(max(n // 2, 1))
        lo = sub.tail(max(n - len(hi), 0))
        chosen = pd.concat([hi, lo]).drop_duplicates("filepath")
        if len(chosen) < n:
            rest = sub[~sub["filepath"].isin(chosen["filepath"])]
            extra_n = n - len(chosen)
            extra = rest.sample(n=extra_n, random_state=SEED) if len(rest) >= extra_n else rest
            chosen = pd.concat([chosen, extra])
        chosen = chosen.head(n).copy()
        chosen["subset"] = "correct"
        rows.append(chosen)

    if include_wrong:
        wrong = preds[preds["true_class"] != preds["predicted_class"]].copy()
        wrong["subset"] = "misclassified"
        rows.append(wrong)
    out = pd.concat(rows, ignore_index=True)
    _ = rng  # seed reserved for any future sampling
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment-dir",
        default=str(cfg.PROJECT_ROOT / "experiments" / "experiment_001_effnetb0_original"),
    )
    p.add_argument("--out-dir", default="")
    p.add_argument("--pred-csv", default="")
    p.add_argument("--correct-only", action="store_true")
    p.add_argument("--n-correct", type=int, default=N_CORRECT_PER_CLASS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    ckpt_path = experiment_dir / "best_model.pt"
    pred_csv = Path(args.pred_csv) if args.pred_csv else (experiment_dir / "validation_predictions.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (cfg.REVIEW_DIR / "gradcam")
    n_correct = args.n_correct

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was expected for this Grad-CAM run.")
    device = torch.device("cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("checkpoint", ckpt_path)

    preds = pd.read_csv(pred_csv)
    global N_CORRECT_PER_CLASS
    N_CORRECT_PER_CLASS = n_correct
    samples = select_samples(preds, include_wrong=not args.correct_only)
    print("samples:", samples.groupby(["subset", "true_class"]).size().to_dict())

    model = load_model(device, ckpt_path)
    cammer = GradCAM(model, model.features[-1])

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []

    for _, row in samples.iterrows():
        rel = row["filepath"]
        path = cfg.PROJECT_ROOT / rel
        x = preprocess(path, device)
        true_idx = CLASS_TO_IDX[row["true_class"]]
        pred_idx = CLASS_TO_IDX[row["predicted_class"]]

        crop = denorm(x[0])
        crop_im = Image.fromarray(crop)

        cam_true, logits = cammer(x, true_idx)
        overlay_true = overlay_cam(crop, cam_true)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        folder = out_dir / ("misclassified" if row["subset"] == "misclassified" else row["true_class"])
        folder.mkdir(parents=True, exist_ok=True)
        fname = slug(Path(rel).name)
        cap = (
            f"{Path(rel).name} | true={row['true_class']} pred={row['predicted_class']} "
            f"conf={row['confidence']:.3f} | CAM=true class"
        )
        panel(crop_im, overlay_true, cap).save(folder / f"{fname}_trueclass.png")

        out_pred = ""
        if row["subset"] == "misclassified":
            cam_pred, _ = cammer(x, pred_idx)
            overlay_pred = overlay_cam(crop, cam_pred)
            cap_p = (
                f"{Path(rel).name} | true={row['true_class']} pred={row['predicted_class']} "
                f"| CAM=predicted class"
            )
            panel(crop_im, overlay_pred, cap_p).save(folder / f"{fname}_predclass.png")
            out_pred = str((folder / f"{fname}_predclass.png").relative_to(cfg.PROJECT_ROOT).as_posix())

        meta_rows.append(
            {
                "filepath": rel,
                "filename": Path(rel).name,
                "true_class": row["true_class"],
                "predicted_class": row["predicted_class"],
                "confidence": float(row["confidence"]),
                "subset": row["subset"],
                "prob_true_class": float(probs[true_idx]),
                "prob_pred_class": float(probs[pred_idx]),
                "cam_true_path": str((folder / f"{fname}_trueclass.png").relative_to(cfg.PROJECT_ROOT).as_posix()),
                "cam_pred_path": out_pred,
                "attention_focus": "",
                "notes": "",
            }
        )
        print("saved", fname, row["subset"], row["true_class"])

    pd.DataFrame(meta_rows).to_csv(out_dir / "_cam_index.csv", index=False)
    print("Wrote overlays. Fill notes after visual inspection.")


if __name__ == "__main__":
    main()
