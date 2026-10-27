"""Compare model folders on the frozen EXTERNAL test set (and the in-dataset test).

Every model is run through the real API runtime (model_runtime.V2Engine), so
abstentions (out_of_scope / retake_photo) count exactly as a user would see them.

Per subset it reports:
  correct      reliable answer (confident/uncertain) with the right class
  abstained    out_of_scope or retake_photo (safe: no diagnosis given)
  harmful      reliable answer with the WRONG class (the number that matters most)
  conf_wrong   status "confident" and wrong
  acc_answered accuracy among reliable answers

  python src/eval_external.py experiments/experiment_004_v2_multitask_cv experiments/experiment_005_v3_multisource
Outputs: outputs/final/external_eval.json (+ printed table)
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
from model_runtime import RELIABLE_STATUSES, V2Engine, quality_issues, quality_metrics

EXTERNAL_TEST = cfg.PROJECT_ROOT / "splits_v3" / "external_test.csv"
OUT_JSON = cfg.OUTPUTS_DIR / "final" / "external_eval.json"
USER_PHOTO = cfg.PROJECT_ROOT / "outputs" / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"


def run(engine: V2Engine, paths: list[Path], batch: int = 24) -> list[dict]:
    out = []
    for s in range(0, len(paths), batch):
        ims = []
        for p in paths[s : s + batch]:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1600, 1600))
                ims.append(im)
        raw = engine.forward_batch(torch.stack([engine.tfm(im) for im in ims]))
        for i, im in enumerate(ims):
            out.append(engine.interpret(raw, i, quality_issues(quality_metrics(im), engine.cal["quality"])))
    return out


def score(res: list[dict], truth: list[str]) -> dict:
    y = np.array(truth)
    pred = np.array([r["prediction"] for r in res])
    st = np.array([r["status"] for r in res])
    rel = np.isin(st, list(RELIABLE_STATUSES))
    ok = pred == y
    return {
        "n": int(len(y)),
        "correct": float((rel & ok).mean()),
        "abstained": float((~rel).mean()),
        "harmful": float((rel & ~ok).mean()),
        "conf_wrong": float(((st == "confident") & ~ok).mean()),
        "acc_answered": float(ok[rel].mean()) if rel.any() else None,
        "raw_top1": float(ok.mean()),
    }


def healthy_false_alarm(res: list[dict], truth: list[str]) -> float | None:
    """Healthy photos given a reliable disease answer."""
    idx = [i for i, t in enumerate(truth) if t == "no_disease"]
    if not idx:
        return None
    return float(np.mean([res[i]["status"] in RELIABLE_STATUSES and res[i]["prediction"] != "no_disease" for i in idx]))


def evaluate(exp_dir: Path, device) -> dict:
    engine = V2Engine(exp_dir, device=device)
    ext = pd.read_csv(EXTERNAL_TEST)
    res = run(engine, [cfg.PROJECT_ROOT / f for f in ext["filepath"]])
    report: dict = {"model": exp_dir.name, "n_models": len(engine.models), "classes": engine.classes, "external": {}}
    for subset, g in ext.groupby("subset"):
        rr = [res[i] for i in g.index]
        report["external"][subset] = score(rr, g["class"].tolist()) | {"per_class": {
            c: score([res[i] for i in gg.index], gg["class"].tolist()) for c, gg in g.groupby("class")}}
    report["external"]["ALL"] = score(res, ext["class"].tolist())
    report["external"]["ALL"]["healthy_false_alarm"] = healthy_false_alarm(res, ext["class"].tolist())

    test = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    test = test[test["source_type"] == "original"].reset_index(drop=True)
    report["in_dataset_test"] = score(run(engine, [cfg.PROJECT_ROOT / f for f in test["filepath"]]), test["class"].tolist())

    if USER_PHOTO.is_file():
        r = run(engine, [USER_PHOTO])[0]
        report["user_photo_1_bsr"] = {k: r[k] for k in ("status", "prediction", "confidence", "class_probabilities")}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--out", default=str(OUT_JSON), help="where to write the JSON report")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports = []
    for d in args.exp_dirs:
        p = Path(d)
        reports.append(evaluate(p if p.is_absolute() else cfg.PROJECT_ROOT / p, device))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))

    cols = ["correct", "abstained", "harmful", "conf_wrong", "acc_answered"]
    for rep in reports:
        print(f"\n=== {rep['model']} ({rep['n_models']} models, {len(rep['classes'])} classes)")
        rows = {k: v for k, v in rep["external"].items()}
        rows["IN-DATASET TEST"] = rep["in_dataset_test"]
        tab = pd.DataFrame({k: {c: v.get(c) for c in ["n", *cols]} for k, v in rows.items()}).T
        print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
        print("healthy false alarm (external):", rep["external"]["ALL"]["healthy_false_alarm"])
        if "user_photo_1_bsr" in rep:
            u = rep["user_photo_1_bsr"]
            print("user photo 1 (true: bacterial_spot_rot):", u["status"], u["prediction"], u["confidence"])


if __name__ == "__main__":
    main()
