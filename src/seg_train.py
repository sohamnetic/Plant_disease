"""Phase 3, step 3: train YOLO11-seg (leaf / curd / lesion).

  python src/seg_train.py            # -> experiments/seg_001_yolo11s/weights/best.pt
  python src/seg_train.py --data data_external/seg_yolo_insect/data.yaml --name seg_002_insect
                                     # step I-2: + insect_hole
  python src/seg_train.py --data data_external/seg_yolo_insect_r2/data.yaml --name seg_003_insect_r2
         --model experiments/seg_002_insect/weights/best.pt --epochs 60 --patience 20
                                     # bootstrap round 2: fine-tune round 1 on the extra reviewed outlines
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

DATA = cfg.PROJECT_ROOT / "data_external" / "seg_yolo" / "data.yaml"
EXP_ROOT = cfg.PROJECT_ROOT / "experiments"
RUN_NAME = "seg_001_yolo11s"
WEIGHTS_DIR = cfg.PROJECT_ROOT / "cache" / "weights"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--name", default=RUN_NAME)
    ap.add_argument("--model", default="yolo11s-seg.pt", help="start weights (default: COCO-pretrained YOLO11s-seg)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()
    start = args.model if args.model == "yolo11s-seg.pt" else str(Path(args.model).resolve())
    data = Path(args.data).resolve()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_DIR)  # ultralytics downloads base weights into the cwd
    from ultralytics import YOLO

    model = YOLO(start)
    model.train(
        data=str(data),
        imgsz=640,
        epochs=args.epochs,
        patience=args.patience,
        batch=16,
        project=str(EXP_ROOT),
        name=args.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        # Colour is the symptom: keep hue shifts tiny, allow brightness/saturation.
        hsv_h=0.005,
        hsv_s=0.4,
        hsv_v=0.35,
        degrees=15,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=15,
        workers=4,
        plots=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
