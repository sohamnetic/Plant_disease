"""Phase 5: refit ONLY the severity (CORAL) head on the new grades.

The backbone, diagnosis head and presence head of every fold model stay
bit-identical, so diagnoses, confidences and OOD checks cannot change. For each
fold k:
  * features (pooled 1280-d, identity + hflip averaged) from fold k's own backbone
  * fit the CORAL head (1 shared weight vector + 4 ordered thresholds) on the
    graded rows of the OTHER folds (VegNet originals + every healthy photo = grade 0)
  * write held-out predictions of fold k into oof_fold{k}.npz (for calibration)

  python src/refit_severity.py
Output: experiments/experiment_006_v3_measured_severity/ (copy of 005 with new heads)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from train_v2 import GRADES_CSV, SEV_FULL, load_frame
from v2_cache import cache_path_for
from v2_model import NUM_GRADES, MultiTaskNet, coral_loss, eval_transforms

SRC = cfg.PROJECT_ROOT / "experiments" / "experiment_005_v3_multisource"
DST = cfg.PROJECT_ROOT / "experiments" / "experiment_006_v3_measured_severity"
EPOCHS = 400
LR = 0.05
WEIGHT_DECAY = 1e-3


@torch.inference_mode()
def features(model: MultiTaskNet, paths: list[str], device) -> torch.Tensor:
    tfm = eval_transforms(384)
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack([tfm(Image.open(cache_path_for(p)).convert("RGB")) for p in paths[i : i + 64]]).to(device)
        out.append(((model.embed(x) + model.embed(torch.flip(x, dims=[3]))) / 2).float().cpu())
    return torch.cat(out)


def fit_head(feat: torch.Tensor, grades: torch.Tensor, init: dict) -> dict:
    mu, sd = feat.mean(0), feat.std(0) + 1e-6
    w = torch.zeros(feat.shape[1], requires_grad=True)
    b = init["bias"].clone().float().requires_grad_(True)
    opt = torch.optim.Adam([w, b], lr=LR)
    z = (feat - mu) / sd
    mask = torch.ones(len(grades), NUM_GRADES - 1)
    for _ in range(EPOCHS):
        logits = (z @ w)[:, None] + b
        loss = coral_loss(logits, grades, mask) + WEIGHT_DECAY * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # fold the standardisation into the linear layer: w.(x-mu)/sd = (w/sd).x - w.(mu/sd)
    w_eff = (w / sd).detach()
    return {"fc.weight": w_eff[None, :], "bias": (b - (w * mu / sd).sum()).detach()}


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    conf = json.loads((SRC / "config.json").read_text())
    df = load_frame("v3", conf["classes"])
    folds = pd.read_csv(SRC / "folds.csv")[["filepath", "fold"]]
    df = df.merge(folds, on="filepath")
    graded = df[df["sev_kind"] == SEV_FULL].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = []
    for ckpt_path in sorted(DST.glob("fold*.pt")):
        k = int(ckpt_path.stem[4:])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = MultiTaskNet(pretrained=False, num_classes=len(conf["classes"]))
        model.load_state_dict({n: t.float() for n, t in ckpt["model_state"].items()})
        model.to(device).eval()
        feat = features(model, graded["filepath"].tolist(), device)
        tr = (graded["fold"] != k).to_numpy()
        g = torch.as_tensor(graded["grade"].to_numpy(), dtype=torch.long)
        head = fit_head(feat[tr], g[tr], {"bias": model.sev_head.bias.detach().cpu()})
        ckpt["model_state"]["sev_head.fc.weight"] = head["fc.weight"].half()
        ckpt["model_state"]["sev_head.bias"] = head["bias"].half()
        ckpt["severity_labels"] = "measured (Phase 5), see outputs/severity/severity_grades.csv"
        torch.save(ckpt, ckpt_path)

        # held-out severity logits for calibration (same rows the OOF file already holds)
        oof = dict(np.load(DST / f"oof_fold{k}.npz", allow_pickle=True))
        ho_paths = list(oof["filepath"])
        ho_feat = features(model, ho_paths, device)
        w16, b16 = head["fc.weight"].half().float(), head["bias"].half().float()
        oof["sev"] = (ho_feat @ w16.T + b16).numpy()
        new = graded.set_index("filepath")["grade"]
        oof["grade"] = np.array([int(new.get(p, g_old)) for p, g_old in zip(ho_paths, oof["grade"])])
        np.savez_compressed(DST / f"oof_fold{k}.npz", **oof)

        pred = np.array([(torch.sigmoid(torch.tensor(r)) > 0.5).sum().item() for r in oof["sev"]])
        m = oof["sev_kind"] == SEV_FULL
        log.append(f"fold {k}: held-out graded n={int(m.sum())} exact={np.mean(pred[m] == oof['grade'][m]):.3f} "
                   f"within1={np.mean(np.abs(pred[m] - oof['grade'][m]) <= 1):.3f}")
        print(log[-1])
    conf["experiment"] = DST.name
    conf["severity_labels"] = str(GRADES_CSV.relative_to(cfg.PROJECT_ROOT)) + " (measured, Phase 5)"
    conf["severity_head"] = f"refit on frozen fold features: CORAL, Adam lr {LR}, {EPOCHS} epochs, L2 {WEIGHT_DECAY}"
    (DST / "config.json").write_text(json.dumps(conf, indent=2))
    (DST / "refit_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
