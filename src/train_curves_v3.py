"""Accuracy / loss curves for the v3 recipe (slides and reports only).

The v3 training log only records the training loss per epoch. This re-trains ONE
fold with the exact v3 recipe (same folds, sampler, augmentation, schedule) and,
after every epoch, measures on that fold's held-out real photos:
  * classification loss (cross-entropy) and accuracy, train vs validation.
Train accuracy / loss are taken from the batches that were not CutMix-ed.

Output: experiments/curves_v3_fold0/history_fold{k}.csv.
Does NOT touch the deployed experiments; the weights trained here are not saved.

    python src/train_curves_v3.py --fold 0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg
from train_v2 import (
    BG_DIR, HP, RECIPES, REAL_SOURCE_TYPES, SEV_GE1, NUM_GRADES, MultiTaskNet, SceneComposer, V2Dataset,
    assign_folds, coral_loss, cutmix, eval_transforms, load_frame, make_sampler, param_groups,
    presence_targets, train_transforms,
)

OUT_DIR = cfg.PROJECT_ROOT / "experiments" / "curves_v3_fold0"


@torch.inference_mode()
def evaluate(model: MultiTaskNet, frame: pd.DataFrame, device: torch.device) -> tuple[float, float]:
    model.eval()
    dl = DataLoader(V2Dataset(frame, eval_transforms(HP["image_size"])), batch_size=64, num_workers=6)
    loss, correct, n = 0.0, 0, 0
    for x, y, *_ in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)["cls"].float()
        loss += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        n += len(y)
    return loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    recipe = RECIPES["v3"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_frame(recipe["data"], recipe["classes"])
    df["fold"] = assign_folds(df)
    # Same folds as the deployed v3 run.
    ref = pd.read_csv(recipe["exp_dir"] / "folds.csv")
    assert (ref["filepath"].to_numpy() == df["filepath"].to_numpy()).all()
    assert (ref["fold"].to_numpy() == df["fold"].to_numpy()).all()

    k = args.fold
    tr = df[df["fold"] != k].reset_index(drop=True)
    ho = df[(df["fold"] == k) & (df["source_type"].isin(REAL_SOURCE_TYPES))].reset_index(drop=True)
    print(f"fold {k}: train {len(tr)} rows, validation {len(ho)} real photos, device {device}", flush=True)

    torch.manual_seed(HP["seed"] + k)
    np.random.seed(HP["seed"] + k)
    model = MultiTaskNet(pretrained=True, dropout=HP["dropout"], num_classes=len(recipe["classes"])).to(device)
    ds = V2Dataset(tr, train_transforms(HP["image_size"], realworld=recipe["realworld"]), SceneComposer(BG_DIR))
    dl = DataLoader(
        ds, batch_size=HP["batch_size"], sampler=make_sampler(tr, recipe["bucket_shares"], HP["seed"] + k),
        num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    opt = torch.optim.AdamW(param_groups(model), weight_decay=HP["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs = HP["epochs"]
    steps_total, warm, step = epochs * len(dl), HP["warmup_epochs"] * len(dl), 0
    rows = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, nb, cls_loss, cls_correct, cls_n = 0.0, 0, 0.0, 0, 0
        for x, y, g, kind, _ in dl:
            x, y, g, kind = x.to(device, non_blocking=True), y.to(device), g.to(device), kind.to(device)
            scale = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps_total - warm, 1)))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * scale
            use_mix = np.random.rand() < HP["cutmix_p"]
            pres_t = presence_targets(y)
            sev_mask = torch.ones(len(y), NUM_GRADES - 1, device=device)
            sev_mask[kind == SEV_GE1, 1:] = 0.0
            if use_mix:
                x, y_b, lam = cutmix(x, y)
                full = HP["presence_full_area"]
                wa, wb = min(lam / full, 1.0), min((1 - lam) / full, 1.0)
                pres_t = torch.maximum(pres_t * wa, presence_targets(y_b) * wb)
                sev_mask.zero_()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            logits = out["cls"].float()
            if use_mix:
                l_cls = lam * F.cross_entropy(logits, y) + (1 - lam) * F.cross_entropy(logits, y_b)
            else:
                l_cls = F.cross_entropy(logits, y)
                cls_loss += float(l_cls) * len(y)
                cls_correct += int((logits.argmax(1) == y).sum())
                cls_n += len(y)
            loss = l_cls + HP["w_presence"] * F.binary_cross_entropy_with_logits(out["presence"].float(), pres_t) \
                + HP["w_severity"] * coral_loss(out["sev"].float(), g, sev_mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
            step += 1
        val_loss, val_acc = evaluate(model, ho, device)
        rows.append({"epoch": ep + 1, "train_total_loss": tot / nb, "train_loss": cls_loss / cls_n,
                     "train_acc": cls_correct / cls_n, "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {ep + 1}/{epochs} train loss {cls_loss / cls_n:.4f} acc {cls_correct / cls_n:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"history_fold{k}.csv", index=False)
    print("saved", OUT_DIR / f"history_fold{k}.csv")


if __name__ == "__main__":
    main()
