"""Step I-4: classifiers on insect-damaged leaves (never trained on).

  * insect test (splits_v4/insect_test.csv, reviewed K + D): how often "insect damage"
    is the answer, and how often a disease is confidently named instead
  * reviewed N (labelled "Insect Hole" by the dataset, no visible damage; never trained on)
  * frozen external test: how often a non-insect photo is called insect damage

Runs the classifier exactly as eval_external.py does (model_runtime.V2Engine).

  python src/eval_insect.py experiments/experiment_006_v3_measured_severity experiments/experiment_007_v4_insect
Outputs: outputs/insect_seg/eval_insect.json (+ printed table)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from eval_external import run
from model_runtime import RELIABLE_STATUSES, V2Engine

INSECT_TEST = cfg.PROJECT_ROOT / "splits_v4" / "insect_test.csv"
REVIEW = cfg.OUTPUTS_DIR / "insect_seg" / "review.csv"
FROZEN = cfg.PROJECT_ROOT / "splits_v3" / "external_test.csv"
OUT = cfg.OUTPUTS_DIR / "insect_seg" / "eval_insect.json"


def summarise(res: list[dict]) -> dict:
    st = pd.Series([r["status"] for r in res])
    pred = pd.Series([r["prediction"] for r in res])
    rel = st.isin(list(RELIABLE_STATUSES))
    conf = st == "confident"
    return {
        "n": len(res),
        "answer_insect": float((rel & (pred == "insect_damage")).mean()),
        "confident_insect": float((conf & (pred == "insect_damage")).mean()),
        "confident_disease": float((conf & ~pred.isin(["insect_damage", "no_disease"])).mean()),
        "confident_black_rot": float((conf & (pred == "black_rot")).mean()),
        "answer_no_disease": float((rel & (pred == "no_disease")).mean()),
        "abstained": float((~rel).mean()),
        "status": st.value_counts().to_dict(),
        "reliable_predictions": pred[rel].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv(INSECT_TEST)
    rev = pd.read_csv(REVIEW)
    n_set = rev[rev["review"] == "N"]
    frozen = pd.read_csv(FROZEN)
    reports = []
    for d in args.exp_dirs:
        p = Path(d)
        eng = V2Engine(p if p.is_absolute() else cfg.PROJECT_ROOT / p, device=device)
        rep = {"model": p.name, "classes": eng.classes}
        res = run(eng, [cfg.PROJECT_ROOT / f for f in test["filepath"]])
        rep["insect_test"] = summarise(res)
        for k in ("K", "D"):
            rep[f"insect_test_{k}"] = summarise([r for r, v in zip(res, test["review"]) if v == k])
        rep["no_visible_damage_N"] = summarise(run(eng, [cfg.PROJECT_ROOT / f for f in n_set["filepath"]]))
        fr = run(eng, [cfg.PROJECT_ROOT / f for f in frozen["filepath"]])
        ins = [r["status"] in RELIABLE_STATUSES and r["prediction"] == "insect_damage" for r in fr]
        rep["frozen_external_called_insect"] = {
            "all": float(pd.Series(ins).mean()),
            "by_subset_class": pd.Series(ins).groupby(frozen["subset"] + "/" + frozen["class"]).mean().round(4).to_dict(),
            "files": [f for f, i in zip(frozen["filepath"], ins) if i],
        }
        reports.append(rep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    cols = ["n", "answer_insect", "confident_insect", "confident_disease", "confident_black_rot", "answer_no_disease", "abstained"]
    for rep in reports:
        print(f"\n=== {rep['model']}")
        tab = pd.DataFrame({k: {c: rep[k][c] for c in cols} for k in ("insect_test", "insect_test_K", "insect_test_D", "no_visible_damage_N")}).T
        print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
        print("frozen external test called insect damage:", f"{rep['frozen_external_called_insect']['all']:.3%}",
              {k: v for k, v in rep["frozen_external_called_insect"]["by_subset_class"].items() if v > 0})


if __name__ == "__main__":
    main()
