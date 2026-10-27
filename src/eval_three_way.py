"""Three-way comparison for the report: normal CNN vs EfficientNet-B0 vs EfficientNet-B0 + YOLO.

  1. normal CNN        = experiments/baseline_simple_cnn (src/train_simple_cnn.py): one plain
                         5-block CNN from scratch, same data / preprocessing / labels
  2. EfficientNet-B0   = the deployed 5-model ensemble with its trust layer (no YOLO)
  3. EfficientNet-B0 + YOLO = the app (src/plant_runtime.py)

Raw model quality (every photo answered with the top class) is computed here for
1 and 2 on the same photos with the same 384 px letterbox input and flip TTA. Column 3
names the same disease as column 2 on every photo (outputs/final/cnn_vs_cnn_yolo.json),
so its diagnosis numbers equal column 2's; its stage / % numbers and the app-level
statuses of 2 and 3 come from that file. Nothing here changes the deployed models.

  python src/eval_three_way.py
Output: outputs/final/three_way_comparison.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from model_runtime import V2Engine
from train_simple_cnn import OUT as SIMPLE_DIR, SimpleCNN
from v2_model import CLASS_ORDER, GRADE_LABELS, MultiTaskNet, coral_to_probs, eval_transforms

ROOT = cfg.PROJECT_ROOT
OUT = cfg.OUTPUTS_DIR / "final" / "three_way_comparison.json"
STAGE_ORDER = ["none", "early", "moderate", "severe", "critical"]
CONF = 0.8  # "confidently wrong" for the normal CNN: wrong with top probability >= 0.8 (it has no trust layer)


def load(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def sets() -> dict[str, pd.DataFrame]:
    s = {}
    ext = pd.read_csv(ROOT / "splits_v3" / "external_test.csv")
    s["external_test"] = pd.DataFrame({"file": ext["filepath"], "y": ext["class"]})
    t = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    t = t[t["source_type"] == "original"]
    s["in_dataset_test"] = pd.DataFrame({"file": t["filepath"], "y": t["class"]})
    w = pd.read_csv(ROOT / "new_pics" / "manifest.csv", dtype={"is_copy": str})
    w = w[w["is_copy"].isin(["False", "0.0"])]
    s["web_photos"] = pd.DataFrame({"file": w["file"], "y": w["cls"]})
    v = pd.read_csv(cfg.OUTPUTS_DIR / "seg" / "val_predictions.csv").merge(
        pd.read_csv(cfg.OUTPUTS_DIR / "seg" / "labelled_images.csv")[["idx", "filepath"]], on="idx")
    s["seg_heldout"] = pd.DataFrame({"file": v["filepath"], "y": v["class"], "ref_stage": v["gt_stage"]})
    return s


@torch.inference_mode()
def simple_predict(model: SimpleCNN, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    o1, o2 = model(x), model(torch.flip(x, dims=[3]))
    p = (o1["cls"].float().softmax(1) + o2["cls"].float().softmax(1)) / 2
    sev = coral_to_probs((o1["sev"].float() + o2["sev"].float()) / 2)
    return p.cpu().numpy(), sev.argmax(1).cpu().numpy()


def scores(y: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> dict:
    ok = pred == y
    healthy = y == "no_disease"
    return {
        "n": int(len(y)),
        "accuracy": float(ok.mean()),
        "macro_f1": float(f1_score(y, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "recall_by_class": {c: (float(ok[y == c].mean()) if (y == c).any() else None) for c in CLASS_ORDER},
        "confidently_wrong": float((~ok & (conf >= CONF)).mean()),
        "healthy_called_diseased": float((pred[healthy] != "no_disease").mean()) if healthy.any() else None,
    }


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(SIMPLE_DIR / "model.pt", map_location="cpu", weights_only=False)
    simple = SimpleCNN()
    simple.load_state_dict(ck["model_state"])
    simple.to(dev).eval()
    eff = V2Engine(device=dev)
    tfm = eval_transforms(384)

    report: dict = {"raw_diagnosis": {}, "stage_seg_heldout": {}, "size": {}, "speed_cpu_s_per_photo": {}}
    for name, df in sets().items():
        sp, sg, ep, ec = [], [], [], []
        for f in df["file"]:
            x = tfm(load(ROOT / f))[None].to(dev)
            p, g = simple_predict(simple, x)
            sp.append(p[0])
            sg.append(int(g[0]))
            r = eff.interpret(eff.forward_batch(x.cpu()), 0, [])
            ep.append(r["prediction"])
            ec.append(r["confidence"])
        sp = np.stack(sp)
        y = df["y"].to_numpy()
        s_pred = np.array(CLASS_ORDER)[sp.argmax(1)]
        if name == "seg_heldout":
            ref = df["ref_stage"].map({k: i for i, k in enumerate(STAGE_ORDER)}).to_numpy()
            st = np.array([0 if p == "no_disease" else max(g, 1) for p, g in zip(s_pred, sg)])
            dis, early = y != "no_disease", df["ref_stage"].to_numpy() == "early"
            report["stage_seg_heldout"]["normal_cnn"] = {
                "stage_exact": float((st == ref).mean()), "stage_within_1": float((np.abs(st - ref) <= 1).mean()),
                "stage_exact_diseased": float((st == ref)[dis].mean()),
                "early_cases_flagged": float((st[early] >= 1).mean()),
                "early_cases_exact_early": float((st[early] == 1).mean()),
                "healthy_given_a_stage": float((st[~dis] >= 1).mean()),
            }
        else:
            report["raw_diagnosis"][name] = {"normal_cnn": scores(y, s_pred, sp.max(1)),
                                             "efficientnet_b0": scores(y, np.array(ep), np.array(ec))}
        print(name, "done", flush=True)

    report["size"]["normal_cnn_parameters"] = sum(p.numel() for p in simple.parameters())
    report["size"]["efficientnet_b0_parameters_per_model"] = sum(p.numel() for p in MultiTaskNet(pretrained=False).parameters())
    from ultralytics import YOLO
    report["size"]["yolo11s_seg_parameters"] = sum(p.numel() for p in YOLO(str(ROOT / "experiments/seg_001_yolo11s/weights/best.pt")).model.parameters())

    files = [ROOT / f for f in sets()["in_dataset_test"]["file"].head(30)]
    xs = [tfm(load(f))[None] for f in files]
    simple.cpu()
    simple_predict(simple, xs[0])
    t0 = time.perf_counter()
    for x in xs:
        simple_predict(simple, x)
    report["speed_cpu_s_per_photo"]["normal_cnn"] = (time.perf_counter() - t0) / len(xs)
    prev = json.loads((cfg.OUTPUTS_DIR / "final" / "cnn_vs_cnn_yolo.json").read_text())
    report["speed_cpu_s_per_photo"]["efficientnet_b0"] = prev["speed"]["cnn_only_s_per_photo"]
    report["speed_cpu_s_per_photo"]["efficientnet_b0_yolo"] = prev["speed"]["cnn_yolo_s_per_photo"]
    report["stage_seg_heldout"]["efficientnet_b0"] = prev["severity_seg_heldout"]["cnn_only"]
    report["stage_seg_heldout"]["efficientnet_b0_yolo"] = prev["severity_seg_heldout"]["cnn_yolo"]
    report["app_level_status"] = prev["diagnosis"]
    report["normal_cnn_training"] = json.loads((SIMPLE_DIR / "config.json").read_text())
    OUT.write_text(json.dumps(report, indent=2))
    for name, r in report["raw_diagnosis"].items():
        print(f"\n{name}")
        print(pd.DataFrame({k: {m: v for m, v in d.items() if m != "recall_by_class"} for k, d in r.items()}).to_string())
    print("\nstage", json.dumps(report["stage_seg_heldout"]["normal_cnn"]))
    print("size", report["size"], "\nspeed", report["speed_cpu_s_per_photo"])


if __name__ == "__main__":
    main()
