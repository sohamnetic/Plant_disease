"""Interim fix check: plant stage for bacterial spot rot, measured % vs the model's grade.

Chosen on non-test photos (117 BSR originals measured by the segmentation model;
measured stage 30% exact / 80% within 1 vs model grade 78% / 100%). This script
confirms once on the VegNet test split through the real runtime (PlantEngine).

  python src/eval_bsr_stage.py
Output: outputs/final/bsr_stage_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import plant_runtime
from plant_runtime import PlantEngine
from v2_cache import cache_path_for

STAGE_INDEX = {"none": 0, "early": 1, "moderate": 2, "severe": 3, "critical": 4}


def main() -> None:
    g = pd.read_csv(cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv")
    g = g[(g["split"] == "test") & (g["class"] == "bacterial_spot_rot")].reset_index(drop=True)
    eng = PlantEngine()
    out = {}
    for name, stage_from_grade in (("measured_pct", set()), ("model_grade", {"bacterial_spot_rot"})):
        plant_runtime.STAGE_FROM_GRADE = stage_from_grade
        pred, keep = [], []
        for fp in g["filepath"]:
            r = eng.predict_pil(Image.open(cache_path_for(fp)), with_overlay=False)
            ok = r.get("plant") is not None and r["prediction"] == "bacterial_spot_rot"
            keep.append(ok)
            pred.append(STAGE_INDEX[r["plant"]["stage"]] if ok else -1)
        pred, keep = np.array(pred), np.array(keep)
        e = np.abs(pred[keep] - g["grade"].to_numpy()[keep])
        out[name] = {"n": int(keep.sum()), "exact": float((e == 0).mean()), "within1": float((e <= 1).mean()),
                     "mae": float(e.mean())}
    plant_runtime.STAGE_FROM_GRADE = {"bacterial_spot_rot"}
    path = cfg.OUTPUTS_DIR / "final" / "bsr_stage_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
