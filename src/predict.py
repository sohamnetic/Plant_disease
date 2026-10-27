"""Single-image / folder inference: same runtime as the API (plant_runtime.PlantEngine).

Classifier ensemble + segmentation: diagnosis per leaf / head, calibrated
confidence, measured affected area, spot count and stage.

Examples:
  python src/predict.py --image "dataset/Original Dataset/Original Dataset/No disease/No disease. (99).jpg"
  python src/predict.py --folder "path/to/images"
  python src/predict.py --image leaf.jpg --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from model_runtime import DEFAULT_EXP, RELIABLE_STATUSES
from plant_runtime import PlantEngine, plant_public_result

PRED_DIR = cfg.OUTPUTS_DIR / "predictions"


def collect_images(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in cfg.IMAGE_EXTENSIONS]


def flat_row(path: Path, r: dict) -> dict:
    sev = r["severity"]
    plant = r.get("plant") or {}
    return {
        "filepath": str(path),
        "status": r["status"],
        "reliable": r["status"] in RELIABLE_STATUSES,  # other columns are only a guess when False
        "prediction": r["prediction"],
        "confidence": r["confidence"],
        "prediction_set_95": "|".join(r["prediction_set_95"]),
        "severity_grade": sev["grade"],
        "severity_stage": sev["stage"],
        "severity_range_90": "-".join(map(str, sev["grade_range_90"])),
        "severity_score_0_100": sev["severity_score_0_100"],
        "measured_stage": plant.get("stage"),
        "measured_affected_pct": plant.get("affected_pct"),
        "lesion_count": plant.get("lesion_count"),
        "parts": "|".join(f"{g['type']}:{g['disease']}:{g['measured_affected_pct']}" for g in plant.get("regions", [])),
        "other_findings": "|".join(o["disease"] for o in r["other_findings"]),
        "ood_score": r["ood_score"],
        "quality_issues": "|".join(r["image_quality"]["issues"]),
        **{f"prob_{k}": v for k, v in r["class_probabilities"].items()},
    }


def summary(path: Path, r: dict) -> str:
    return f"{path.name}: [{r['status']}] {plant_public_result(r)['message']}"


def main() -> None:
    p = argparse.ArgumentParser(description="Cauliflower disease inference (classifier + segmentation).")
    p.add_argument("--image", type=str, default="")
    p.add_argument("--folder", type=str, default="")
    p.add_argument("--exp-dir", type=str, default=str(DEFAULT_EXP))
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--json", action="store_true", help="print the full API-style JSON for --image")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    if bool(args.image.strip()) == bool(args.folder.strip()):
        raise SystemExit("Provide exactly one of --image or --folder.")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    engine = PlantEngine(Path(args.exp_dir), device=device, seg_device="0" if device.type == "cuda" else "cpu")

    def resolve(s: str) -> Path:
        path = Path(s)
        return path if path.is_absolute() else (cfg.PROJECT_ROOT / path).resolve()

    if args.image.strip():
        path = resolve(args.image.strip())
        with Image.open(path) as im:
            r = engine.predict_pil(im, with_overlay=False)
        print(json.dumps(plant_public_result(r), indent=2) if args.json else summary(path, r))
        return

    folder = resolve(args.folder.strip())
    images = collect_images(folder)
    if not images:
        raise SystemExit(f"No images found under {folder}")
    rows = []
    for path in images:
        with Image.open(path) as im:
            r = engine.predict_pil(im, with_overlay=False)
        rows.append(flat_row(path, r))
        print(summary(path, r))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = resolve(args.out_csv) if args.out_csv.strip() else PRED_DIR / f"{folder.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} predictions -> {out_csv}")


if __name__ == "__main__":
    main()
