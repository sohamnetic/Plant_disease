"""Phase 5 evaluation: severity head refit on measured grades.

Compares the old head (experiment 005) and the refit head (006) against the NEW
grades, on (a) out-of-fold training photos and (b) the VegNet test split
(93 originals, never trained on). Diagnosis must be identical (same backbone).

Gate (plan): refit grade within +-1 of the (measured) grade >= 95%.

  python src/eval_phase5.py
Output: outputs/final/phase5_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from calibrate_v2 import load_oof
from model_runtime import V2Engine
from train_v2 import SEV_FULL
from v2_cache import cache_path_for

E5 = cfg.PROJECT_ROOT / "experiments" / "experiment_005_v3_multisource"
E6 = cfg.PROJECT_ROOT / "experiments" / "experiment_006_v3_measured_severity"
OUT = cfg.OUTPUTS_DIR / "final" / "phase5_eval.json"


def stats(pred: np.ndarray, g: np.ndarray) -> dict:
    return {"n": int(len(g)), "exact": float((pred == g).mean()), "within1": float((np.abs(pred - g) <= 1).mean()),
            "mae": float(np.abs(pred - g).mean())}


def oof_block() -> dict:
    new = pd.read_csv(cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv").set_index("filepath")
    out = {}
    for name, exp in (("old_head_005", E5), ("refit_head_006", E6)):
        o = load_oof(exp, 5)
        cal = json.loads((exp / "calibration.json").read_text())
        keep = (o["sev_kind"] == SEV_FULL) & (o["label"] < 3) & np.isin(o["filepath"], new.index)
        g = new.loc[o["filepath"][keep], "grade"].to_numpy()
        src = new.loc[o["filepath"][keep], "grade_source"].to_numpy()
        logits = torch.as_tensor(o["sev"][keep]) / cal["sev_temperature"]
        from v2_model import coral_to_probs
        p = coral_to_probs(logits).numpy()
        p[:, 0] = 0
        pred = p.argmax(1)
        meas = np.isin(src, ["measured", "measured_reviewed", "reviewed_set"])
        out[name] = {"all_diseased": stats(pred, g), "measured_labels_only": stats(pred[meas], g[meas]),
                     "by_eye_labels_only": stats(pred[~meas], g[~meas])}
    return out


def test_block() -> dict:
    g = pd.read_csv(cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv")
    g = g[(g["split"] == "test") & (g["class"] != "no_disease")].reset_index(drop=True)
    out, diag = {}, {}
    for name, exp in (("old_head_005", E5), ("refit_head_006", E6)):
        eng = V2Engine(exp, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        res = [eng.predict_pil(Image.open(cache_path_for(fp))) for fp in g["filepath"]]
        ok = np.array([r["prediction"] == c for r, c in zip(res, g["class"])])
        pred = np.array([r["severity"]["grade"] for r in res])
        meas = g["grade_source"].isin(["measured", "measured_reviewed", "reviewed_set"]).to_numpy()
        gt = g["grade"].to_numpy()
        out[name] = {"correctly_diagnosed": stats(pred[ok], gt[ok]),
                     "measured_labels_only": stats(pred[ok & meas], gt[ok & meas]),
                     "by_class": {c: stats(pred[ok & (g["class"] == c).to_numpy()], gt[ok & (g["class"] == c).to_numpy()])
                                  for c in sorted(g["class"].unique())},
                     "crosstab_true_vs_pred": pd.crosstab(gt[ok], pred[ok]).to_dict()}
        diag[name] = [r["prediction"] for r in res]
    out["diagnosis_identical"] = diag["old_head_005"] == diag["refit_head_006"]
    return out


def main() -> None:
    report = {"out_of_fold": oof_block(), "test_split": test_block()}
    t = report["test_split"]["refit_head_006"]["correctly_diagnosed"]["within1"]
    o = report["out_of_fold"]["refit_head_006"]["all_diseased"]["within1"]
    report["gate_within1_ge_0.95"] = {"test_split": t >= 0.95, "out_of_fold": o >= 0.95}
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
