"""Step I-4b: per-photo answers of the 4-class (v3) and 5-class (v4) classifiers plus
the outline model's eaten %, on the frozen external test, the insect test and the
reviewed "no visible damage" (N) photos; then scores decision rules that combine them.

The v4 classifier alone failed the external-test gate (black rot leaves were called
insect damage). The pre-registered fallback rule ("v3 diagnoses, v4 adds insect"):
  * v3 decides every disease answer, exactly as today;
  * only when v3's answer is "no disease" (or v3 abstains) and v4 answers
    "insect damage" does the result become insect damage.
So a photo v3 calls a disease is never changed, and the external-test disease
numbers cannot get worse; what can change is healthy -> insect damage.
Other rules are printed for information only.

Result: the pre-registered rule changed 13 frozen-test answers (correct 92.7 -> 92.0%),
every one of them from an "uncertain" v4 answer. The ADOPTED rule therefore requires a
confident v4 answer (and lets a very sure one replace an unsure v3 disease answer). That
choice was made after looking at the frozen test, so its frozen-test score (identical
to v3) is optimistic; the v4 held-out folds and the web photos are the independent
checks (INSECT_MODEL_V4_REPORT.md).
  python src/insect_hybrid_eval.py --rescore   # re-score rules from per_photo.csv

  python src/insect_hybrid_eval.py --seg-weights experiments/seg_002_insect/weights/best.pt
Outputs: outputs/insect_seg/hybrid/per_photo.csv, hybrid_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from eval_external import run
from model_runtime import RELIABLE_STATUSES, V2Engine
from seg_runtime import WORK_SIDE, SegEngine

V3 = cfg.PROJECT_ROOT / "experiments" / "experiment_006_v3_measured_severity"
V4 = cfg.PROJECT_ROOT / "experiments" / "experiment_007_v4_insect"
INSECT_TEST = cfg.PROJECT_ROOT / "splits_v4" / "insect_test.csv"
REVIEW = cfg.OUTPUTS_DIR / "insect_seg" / "review.csv"
FROZEN = cfg.PROJECT_ROOT / "splits_v3" / "external_test.csv"
OUT = cfg.OUTPUTS_DIR / "insect_seg" / "hybrid"
INSECT, HEALTHY = "insect_damage", "no_disease"


def photos() -> pd.DataFrame:
    fr = pd.read_csv(FROZEN)
    fr = pd.DataFrame({"filepath": fr["filepath"], "set": "frozen/" + fr["subset"], "truth": fr["class"]})
    it = pd.read_csv(INSECT_TEST)
    it = pd.DataFrame({"filepath": it["filepath"], "set": "insect_test_" + it["review"], "truth": INSECT})
    rev = pd.read_csv(REVIEW)
    n = rev[rev["review"] == "N"]
    n = pd.DataFrame({"filepath": n["filepath"], "set": "no_visible_damage_N", "truth": "none_visible"})
    return pd.concat([fr, it, n], ignore_index=True)


def answers(exp: Path, paths: list[Path], device) -> pd.DataFrame:
    res = run(V2Engine(exp, device=device), paths)
    return pd.DataFrame({
        "status": [r["status"] for r in res],
        "pred": [r["prediction"] for r in res],
        "conf": [r["confidence"] for r in res],
        "p_insect": [r["class_probabilities"].get(INSECT, 0.0) for r in res],
    })


def eaten(weights: Path, paths: list[Path], device: str) -> pd.DataFrame:
    eng = SegEngine(weights, device=device)
    out = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((WORK_SIDE, WORK_SIDE))
            m, _ = eng.measure_all(np.asarray(im))
        out.append((m.get("worst_region_eaten_pct", 0.0) or 0.0, m.get("hole_count", 0) or 0, m.get("insect_stage")))
    return pd.DataFrame(out, columns=["eaten_pct", "hole_count", "insect_stage"])


def rule_v4(d):  # 5-class classifier alone
    return d["v4_pred"].where(d["v4_rel"], "abstain")


def rule_hybrid(d):  # pre-registered: v3 diagnoses; v4 only turns "no disease"/abstain into insect
    base = d["v3_pred"].where(d["v3_rel"], "abstain")
    promote = base.isin([HEALTHY, "abstain"]) & d["v4_rel"] & (d["v4_pred"] == INSECT)
    return base.mask(promote, INSECT)


def rule_hybrid_holes(d, min_pct: float = 0.3):  # information only: + measured holes promote "no disease"
    out = rule_hybrid(d)
    return out.mask((out == HEALTHY) & (d["eaten_pct"] >= min_pct), INSECT)


def rule_adopted(d, p: float = 0.8):  # the app's rule (plant_runtime._insect_second_opinion)
    base = d["v3_pred"].where(d["v3_rel"], "abstain")
    sure = (d["v4_status"] == "confident") & (d["v4_pred"] == INSECT)
    promote = sure & (base.isin([HEALTHY, "abstain"]) | ((d["v3_status"] != "confident") & (d["v4_p_insect"] >= p)))
    return base.mask(promote, INSECT)


def rule_hybrid_override(d, p: float = 0.8):  # information only: very sure v4 insect also overrides a v3 disease
    out = rule_hybrid(d)
    return out.mask(d["v4_rel"] & (d["v4_pred"] == INSECT) & (d["v4_p_insect"] >= p) & (d["v3_status"] != "confident"), INSECT)


def score(ans: pd.Series, d: pd.DataFrame) -> dict:
    rep = {}
    fr = d["set"].str.startswith("frozen/")
    a, t = ans[fr], d.loc[fr, "truth"]
    answered = a != "abstain"
    rep["frozen_correct"] = float((a == t).mean())
    rep["frozen_harmful"] = float((answered & (a != t)).mean())
    rep["frozen_abstained"] = float((~answered).mean())
    h = t == HEALTHY
    rep["frozen_healthy_false_alarm"] = float((answered[h] & (a[h] != HEALTHY)).mean())  # any non-healthy answer
    rep["frozen_healthy_called_insect"] = float((a[h] == INSECT).mean())
    rep["frozen_disease_called_insect"] = float((a[~h] == INSECT).mean())
    rep["frozen_correct_by_set_class"] = (a == t).groupby(d.loc[fr, "set"] + "/" + t).mean().round(4).to_dict()
    for s in ("insect_test_K", "insect_test_D", "no_visible_damage_N"):
        m = d["set"] == s
        rep[s] = {
            "n": int(m.sum()),
            "insect": float((ans[m] == INSECT).mean()),
            "no_disease": float((ans[m] == HEALTHY).mean()),
            "disease": float((~ans[m].isin([INSECT, HEALTHY, "abstain"])).mean()),
            "abstain": float((ans[m] == "abstain").mean()),
        }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-weights", default="experiments/seg_002_insect/weights/best.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--rescore", action="store_true", help="reuse per_photo.csv")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.rescore:
        d = pd.read_csv(OUT / "per_photo.csv")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = photos()
        paths = [cfg.PROJECT_ROOT / f for f in d["filepath"]]
        for tag, exp in (("v3", V3), ("v4", V4)):
            r = answers(exp, paths, device)
            for c in r:
                d[f"{tag}_{c}"] = r[c].values
            d[f"{tag}_rel"] = d[f"{tag}_status"].isin(list(RELIABLE_STATUSES))
        d = pd.concat([d, eaten(Path(a.seg_weights), paths, a.device)], axis=1)
        d.to_csv(OUT / "per_photo.csv", index=False)
    rules = {
        "v3_alone": d["v3_pred"].where(d["v3_rel"], "abstain"),
        "v4_alone": rule_v4(d),
        "hybrid (pre-registered)": rule_hybrid(d),
        "hybrid + holes>=0.3% (info)": rule_hybrid_holes(d),
        "hybrid + v4 override p>=0.8 (info)": rule_hybrid_override(d),
        "ADOPTED: v4 confident only, + override of unsure v3 at p>=0.8": rule_adopted(d),
    }
    rep = {"seg_weights": a.seg_weights, "rules": {k: score(v, d) for k, v in rules.items()}}
    (OUT / "hybrid_eval.json").write_text(json.dumps(rep, indent=2))
    keys = ["frozen_correct", "frozen_harmful", "frozen_healthy_false_alarm", "frozen_healthy_called_insect", "frozen_disease_called_insect"]
    tab = pd.DataFrame({k: {**{c: v[c] for c in keys},
                            **{f"{s}_insect": v[s]["insect"] for s in ("insect_test_K", "insect_test_D", "no_visible_damage_N")},
                            **{f"{s}_disease": v[s]["disease"] for s in ("insect_test_K", "insect_test_D")}}
                        for k, v in rep["rules"].items()}).T
    print(tab.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
