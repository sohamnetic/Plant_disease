"""Baseline for the report: a plain CNN trained from scratch on the same data.

Comparison column 1 ("normal CNN") against the deployed EfficientNet-B0 ensemble
and EfficientNet-B0 + YOLO11-seg. Everything except the network is the same as
src/train_v2.py --recipe v3: the v3 training pool, the 384 px letterbox input,
real-world + scene augmentation, CutMix, class x source balanced sampling, the
diagnosis labels and the (measured) severity grades.

Network: 5 blocks of [3x3 conv - BN - ReLU] x 2 + 2x2 max-pool (32-64-128-256-256
channels), global average pool, dropout 0.3, a 4-way diagnosis head and the same
CORAL severity head. No ImageNet pretraining. One model, no ensemble, no
calibration / out-of-scope layer.

Training uses folds 1-4 of experiment_005's grouped folds and picks the epoch with
the best macro-F1 on fold 0's real photos (the same 80/20 grouping each
EfficientNet fold model saw). Nothing here touches the deployed models.

  python src/train_simple_cnn.py
Output: experiments/baseline_simple_cnn/ (model.pt, train_log.txt, config.json)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from scene_aug import SceneComposer
from train_v2 import BG_DIR, HP, RECIPES, SEV_GE1, V2Dataset, cutmix, load_frame, make_sampler
from v2_model import CLASS_ORDER, NUM_GRADES, CoralHead, coral_loss, eval_transforms, train_transforms

OUT = cfg.PROJECT_ROOT / "experiments" / "baseline_simple_cnn"
FOLDS = cfg.PROJECT_ROOT / "experiments" / "experiment_005_v3_multisource" / "folds.csv"
EPOCHS = 40
LR = 2e-3
WEIGHT_DECAY = 1e-4
VAL_EVERY = 2


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_ORDER), widths=(32, 64, 128, 256, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in widths:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(c_in, num_classes)
        self.sev_head = CoralHead(c_in)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.dropout(torch.flatten(self.pool(self.features(x)), 1))
        return {"cls": self.cls_head(h), "sev": self.sev_head(h)}


@torch.inference_mode()
def predict_frame(model: SimpleCNN, frame: pd.DataFrame, device) -> np.ndarray:
    model.eval()
    dl = DataLoader(V2Dataset(frame, eval_transforms(HP["image_size"])), batch_size=64, num_workers=6)
    out = []
    for x, *_ in dl:
        x = x.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = (model(x)["cls"].float().softmax(1) + model(torch.flip(x, dims=[3]))["cls"].float().softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out).numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_f = open(OUT / "train_log.txt", "w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(HP["seed"])
    np.random.seed(HP["seed"])
    recipe = RECIPES["v3"]
    df = load_frame("v3", CLASS_ORDER).merge(pd.read_csv(FOLDS)[["filepath", "fold"]], on="filepath")
    tr = df[df["fold"] != 0].reset_index(drop=True)
    va = df[(df["fold"] == 0) & df["source_type"].isin(["original", "external"])].reset_index(drop=True)
    log(f"train {len(tr)} rows, validation {len(va)} real photos, device {device}")

    model = SimpleCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"SimpleCNN parameters: {n_params:,}")
    ds = V2Dataset(tr, train_transforms(HP["image_size"], realworld=True), SceneComposer(BG_DIR))
    dl = DataLoader(ds, batch_size=HP["batch_size"], sampler=make_sampler(tr, recipe["bucket_shares"], HP["seed"]),
                    num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    steps_total, warm, step = EPOCHS * len(dl), len(dl), 0
    best, best_ep = -1.0, -1
    for ep in range(EPOCHS):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, g, kind, _ in dl:
            x, y, g, kind = x.to(device, non_blocking=True), y.to(device), g.to(device), kind.to(device)
            scale = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps_total - warm)))
            for pg in opt.param_groups:
                pg["lr"] = LR * scale
            sev_mask = torch.ones(len(y), NUM_GRADES - 1, device=device)
            sev_mask[kind == SEV_GE1, 1:] = 0.0
            use_mix = np.random.rand() < HP["cutmix_p"]
            if use_mix:
                x, y_b, lam = cutmix(x, y)
                sev_mask.zero_()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            logits = out["cls"].float()
            l_cls = (lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y_b)) if use_mix \
                else F.cross_entropy(logits, y)
            loss = l_cls + HP["w_severity"] * coral_loss(out["sev"].float(), g, sev_mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot, nb, step = tot + float(loss), nb + 1, step + 1
        msg = f"epoch {ep + 1}/{EPOCHS} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % VAL_EVERY == 0 or ep + 1 == EPOCHS:
            p = predict_frame(model, va, device)
            pred, y_va = p.argmax(1), va["label"].to_numpy()
            mf1 = f1_score(y_va, pred, average="macro")
            msg += f" | val acc {(pred == y_va).mean():.4f} macro-F1 {mf1:.4f}"
            if mf1 > best:
                best, best_ep = mf1, ep + 1
                torch.save({"model_state": model.state_dict(), "classes": CLASS_ORDER, "epoch": ep + 1}, OUT / "model.pt")
                msg += "  (saved)"
        log(msg)
    (OUT / "config.json").write_text(json.dumps({
        "model": "SimpleCNN (5 conv blocks, from scratch)", "parameters": n_params, "epochs": EPOCHS, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "best_epoch": best_ep, "best_val_macro_f1": best,
        "train": "splits_v3/train_pool.csv folds 1-4 (experiment_005 grouping)", "val": "fold 0 real photos",
        "image_size": HP["image_size"], "augmentation": "same as train_v2 recipe v3 (realworld + scene + CutMix)",
    }, indent=2))
    log(f"done: best epoch {best_ep}, val macro-F1 {best:.4f}")


if __name__ == "__main__":
    main()
