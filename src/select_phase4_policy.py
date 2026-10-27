"""Choose the Phase 4 decision policy on VALIDATION data (not the test sets).

Validation photos: real training-pool photos (VegNet originals + external
sources) that the segmentation model never saw (not in outputs/seg/label_set.csv).
Each photo is classified by the ONE fold model that held it out during training,
so the classifier has not seen it either.

Candidates: seg_veto x grade_check (4 region-decides policies), "classifier only"
(v3), and "classifier decides, parts measure" (added after the first run).
Selection rule, fixed before looking at the numbers: lowest wrong-answer rate;
ties (within 0.5 point) broken by the highest correct rate.

  python src/select_phase4_policy.py
Outputs: outputs/final/phase4_policy_selection.json, src/phase4_policy.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from eval_external import healthy_false_alarm, score
from plant_runtime import POLICY_FILE, PlantEngine
from train_v2 import REAL_SOURCE_TYPES

EXP = cfg.PROJECT_ROOT / "experiments" / "experiment_005_v3_multisource"
OUT = cfg.OUTPUTS_DIR / "final" / "phase4_policy_selection.json"
POLICIES = {f"veto={v},grade_check={g}": {"seg_veto": v, "grade_check": g}
            for v, g in itertools.product([True, False], [True, False])}
# Added after the first run showed every region-decides policy losing to the
# classifier alone: the classifier decides, the parts only measure.
POLICIES["classifier_decides,parts_measure"] = {"diagnosis": "classifier", "seg_veto": False, "grade_check": False}


def main() -> None:
    folds = pd.read_csv(EXP / "folds.csv")
    seen_by_seg = set(pd.read_csv(cfg.OUTPUTS_DIR / "seg" / "label_set.csv")["filepath"])
    val = folds[folds["source_type"].isin(REAL_SOURCE_TYPES) & ~folds["filepath"].isin(seen_by_seg)].reset_index(drop=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eng = PlantEngine(device=dev, seg_device="0" if dev.type == "cuda" else "cpu")
    all_models, all_maha = eng.cls.models, eng.cls.maha
    res = {name: [] for name in ["classifier_only", *POLICIES]}
    order = []
    for k in sorted(val["fold"].unique()):
        eng.cls.models, eng.cls.maha = [all_models[k]], [all_maha[k]]
        for i in np.nonzero(val["fold"].to_numpy() == k)[0]:
            with Image.open(cfg.PROJECT_ROOT / val.at[i, "filepath"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1600, 1600))
            a = eng.analyse(im)
            w = dict(a["whole"])
            if a["issues"]:
                w["status"] = "retake_photo"
            res["classifier_only"].append(w)
            for name, pol in POLICIES.items():
                res[name].append(eng.decide(a, pol))
            order.append(i)
    val = val.loc[order].reset_index(drop=True)
    truth = val["class"].tolist()

    table = {}
    for name, rr in res.items():
        s = score(rr, truth)
        s["healthy_false_alarm"] = healthy_false_alarm(rr, truth)
        s["uncertain_rate"] = float(np.mean([r["status"] == "uncertain" for r in rr]))
        s["by_source"] = {src: score([rr[i] for i in g.index], g["class"].tolist())["harmful"]
                          for src, g in val.groupby("source")}
        table[name] = s

    cands = {n: s for n, s in table.items() if n in POLICIES}
    best_wrong = min(s["harmful"] for s in cands.values())
    tied = {n: s for n, s in cands.items() if s["harmful"] <= best_wrong + 0.005}
    chosen = max(tied, key=lambda n: tied[n]["correct"])
    report = {"n_validation_photos": int(len(val)),
              "by_class": val["class"].value_counts().to_dict(),
              "by_source": val["source"].value_counts().to_dict(),
              "selection_rule": "lowest wrong-answer rate; ties within 0.5 pt -> highest correct",
              "results": table, "chosen": chosen, "chosen_policy": POLICIES[chosen]}
    OUT.write_text(json.dumps(report, indent=2, default=float))
    POLICY_FILE.write_text(json.dumps({"policy": POLICIES[chosen], "chosen_on": "validation set, see "
                                       "outputs/final/phase4_policy_selection.json"}, indent=2))
    cols = ["correct", "abstained", "harmful", "conf_wrong", "uncertain_rate", "healthy_false_alarm"]
    print(f"validation photos: {len(val)}")
    print(pd.DataFrame(table).T[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
