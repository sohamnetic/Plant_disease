"""Single-image / folder inference for the selected EfficientNet-B0 model.

Uses the same eval preprocessing as training (Resize 256, CenterCrop 224,
ImageNet mean/std). No augmentation. Does not train or touch experiment files
beyond loading a checkpoint.

Examples:
  python src/predict.py --image "dataset/Original Dataset/Original Dataset/No disease/No disease. (99).jpg"
  python src/predict.py --folder "path/to/images"
  python src/predict.py --folder "path/to/images" --checkpoint experiments/experiment_001_effnetb0_original/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import efficientnet_b0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import IDX_TO_CLASS, build_transforms

DEFAULT_CKPT = (
    cfg.PROJECT_ROOT
    / "experiments"
    / "experiment_003_effnetb0_combined"
    / "best_model.pt"
)
PRED_DIR = cfg.OUTPUTS_DIR / "predictions"
DROPOUT = 0.4


def build_inference_model() -> torch.nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=DROPOUT, inplace=True),
        torch.nn.Linear(in_features, len(cfg.CLASS_ORDER)),
    )
    return model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Device: cpu")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = build_inference_model()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def collect_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in cfg.IMAGE_EXTENSIONS:
            files.append(p)
    return files


def predict_pil(model: torch.nn.Module, im: Image.Image, tfm, device: torch.device) -> dict:
    im = ImageOps.exif_transpose(im).convert("RGB")
    tensor = tfm(im).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    row = {
        "predicted_class": IDX_TO_CLASS[pred_idx],
        "confidence": float(probs[pred_idx]),
    }
    for i, name in enumerate(cfg.CLASS_ORDER):
        row[f"prob_{name}"] = float(probs[i])
    return row


def predict_path(model: torch.nn.Module, image_path: Path, tfm, device: torch.device) -> dict:
    with Image.open(image_path) as im:
        row = predict_pil(model, im, tfm, device)
    row["filepath"] = str(image_path)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leaf-disease inference (frozen checkpoint).")
    p.add_argument("--image", type=str, default="", help="Path to a single image.")
    p.add_argument("--folder", type=str, default="", help="Path to a folder of images (recursive).")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CKPT),
        help="Path to a best_model.pt checkpoint.",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional CSV path for folder runs (default: outputs/predictions/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image_arg = args.image.strip()
    folder_arg = args.folder.strip()
    if bool(image_arg) == bool(folder_arg):
        raise SystemExit("Provide exactly one of --image or --folder.")

    device = pick_device()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = cfg.PROJECT_ROOT / ckpt_path
    print("Checkpoint:", ckpt_path)
    model = load_checkpoint(ckpt_path, device)
    tfm = build_transforms(train=False)

    if image_arg:
        path = Path(image_arg)
        if not path.is_absolute():
            path = (cfg.PROJECT_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        row = predict_path(model, path, tfm, device)
        print(f"{path.name}: {row['predicted_class']}  conf={row['confidence']:.4f}")
        for name in cfg.CLASS_ORDER:
            print(f"  {name}: {row[f'prob_{name}']:.4f}")
        return

    folder = Path(folder_arg)
    if not folder.is_absolute():
        folder = (cfg.PROJECT_ROOT / folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    images = collect_images(folder)
    if not images:
        raise SystemExit(f"No images found under {folder}")

    rows = []
    for path in images:
        row = predict_path(model, path, tfm, device)
        rows.append(row)
        print(f"{path.name}: {row['predicted_class']}  conf={row['confidence']:.4f}")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if args.out_csv.strip():
        out_csv = Path(args.out_csv)
        if not out_csv.is_absolute():
            out_csv = cfg.PROJECT_ROOT / out_csv
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_csv = PRED_DIR / f"{folder.name}_{stamp}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} predictions -> {out_csv}")


if __name__ == "__main__":
    main()
