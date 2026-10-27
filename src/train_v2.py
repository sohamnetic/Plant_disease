"""Experiment 4 (v2): 5-fold grouped CV ensemble, multi-task, calibrated.

Data: splits/train.csv + splits/val.csv (originals + augmented). splits/test.csv is
never read here. Severity grades come from outputs/severity/severity_grades.csv.

For each fold k (StratifiedGroupKFold on group_id, stratified by class x source):
  * train MultiTaskNet on the other 4 folds for a FIXED number of epochs (no early
    stopping, so the held-out predictions stay unbiased for calibration)
  * predict the held-out ORIGINAL images with TTA -> out-of-fold (OOF) outputs
  * fit feature-space OOD statistics (PCA + tied-covariance Mahalanobis) on the
    training images of that fold
The 5 fold models together are the deployed ensemble; src/calibrate_v2.py fits
temperature / conformal / OOD thresholds on the OOF outputs.

Fixes over Experiment 3 (see README "v2"):
  full backbone fine-tuning with layer-wise LR decay; 384 px letterboxed input;
  class+source balanced sampling; no label smoothing (calibrate afterwards);
  CutMix composites teach the presence head that two diseases can co-occur.

v4 recipe (--recipe v4, Experiment 7, step I-3): the v3 recipe on splits_v4/train_pool.csv
  with a 5th class "insect_damage" (reviewed cleaf insect-hole photos). Insect photos have
  no disease grade: their severity loss is masked (SEV_NONE); the % eaten comes from
  the outline model instead.

v3 recipe (--recipe v3, Experiment 5) additionally:
  trains on splits_v3/train_pool.csv (VegNet + healthy-leaf crops + background-removed
  copies + new external photos; the external test set is never read), adds
  real-world photo degradations (low-res, JPEG, white balance, shadows) and scene
  augmentation (background swap, zoom-out), and balances sampling over
  class x source bucket.

  python src/train_v2.py                              # v2 full run
  python src/train_v2.py --recipe v3                  # v3 full run
  python src/train_v2.py --recipe v4                  # v4 full run (+ insect_damage)
  python src/train_v2.py --recipe v3 --data v3-noext --folds 0 --exp-dir experiments/ablation_x
  python src/train_v2.py --recipe v3 --no-multitask --folds 0 --exp-dir experiments/ablation_y
  python src/train_v2.py --epochs 1 --folds 0 --limit 256   # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from scene_aug import SceneComposer
from v2_cache import cache_path_for
from v2_model import (
    CLASS_ORDER,
    DISEASES,
    NUM_GRADES,
    MultiTaskNet,
    coral_loss,
    coral_to_probs,
    eval_transforms,
    train_transforms,
)

EXP_DIR = cfg.PROJECT_ROOT / "experiments" / "experiment_004_v2_multitask_cv"
GRADES_CSV = cfg.OUTPUTS_DIR / "severity" / "severity_grades.csv"
V3_POOL = cfg.PROJECT_ROOT / "splits_v3" / "train_pool.csv"
V4_POOL = cfg.PROJECT_ROOT / "splits_v4" / "train_pool.csv"
V4_CLASSES = CLASS_ORDER + ["insect_damage"]  # appended: indices of the v3 classes stay the same
# Sources photographed on a plain white background: background-swap candidates.
CUTOUT_SOURCES = {"c2h", "cleaf"}
BG_DIR = cfg.PROJECT_ROOT / "data_external" / "derived" / "backgrounds"

RECIPES = {
    "v2": {"exp_dir": EXP_DIR, "data": "v2", "classes": CLASS_ORDER, "realworld": False, "scene": False,
           "bucket_shares": {"real": 0.4, "aug": 0.6}},
    "v3": {"exp_dir": cfg.PROJECT_ROOT / "experiments" / "experiment_005_v3_multisource", "data": "v3",
           "classes": CLASS_ORDER, "realworld": True, "scene": True,
           "bucket_shares": {"real": 0.45, "aug": 0.2, "cutout": 0.15, "external": 0.2}},
    "v4": {"exp_dir": cfg.PROJECT_ROOT / "experiments" / "experiment_007_v4_insect", "data": "v4",
           "classes": V4_CLASSES, "realworld": True, "scene": True,
           "bucket_shares": {"real": 0.45, "aug": 0.2, "cutout": 0.15, "external": 0.2}},
}
# Rows whose held-out predictions are used for calibration: real photos only.
REAL_SOURCE_TYPES = {"original", "external"}

HP = {
    "n_folds": 5,
    "epochs": 18,
    "batch_size": 32,
    "image_size": 384,
    "head_lr": 1e-3,
    "coral_bias_lr": 5e-2,
    "backbone_top_lr": 3e-4,
    "layer_decay": 0.8,
    "weight_decay": 0.01,
    "warmup_epochs": 1,
    "dropout": 0.3,
    "cutmix_p": 0.35,
    "presence_full_area": 0.3,
    "w_presence": 0.5,
    "w_severity": 0.5,
    "pca_dims": 128,
    "tta": ["identity", "hflip"],
    "seed": 42,
}

# Severity supervision kinds (SEV_NONE: not a disease, e.g. insect damage -> no severity loss)
SEV_FULL, SEV_GE1, SEV_NONE = 0, 1, 2


def load_frame(data: str, classes: list[str]) -> pd.DataFrame:
    if data == "v2":
        df = pd.concat(
            [pd.read_csv(cfg.SPLITS_DIR / "train.csv"), pd.read_csv(cfg.SPLITS_DIR / "val.csv")],
            ignore_index=True,
        ).assign(source="vegnet")
    elif data == "v4":
        df = pd.read_csv(V4_POOL)
    else:
        df = pd.read_csv(V3_POOL)
        if data == "v3-noext":
            df = df[df["source"].isin(["vegnet", "healthy_leaf_crop"])]
    grades = pd.read_csv(GRADES_CSV)[["filepath", "grade"]]
    df = df.merge(grades, on="filepath", how="left")
    df["label"] = df["class"].map({c: i for i, c in enumerate(classes)})
    if df["label"].isna().any():
        raise RuntimeError(f"classes not in recipe: {sorted(df.loc[df['label'].isna(), 'class'].unique())}")
    df["label"] = df["label"].astype(int)
    is_nd = df["class"] == "no_disease"
    df.loc[is_nd, "grade"] = 0
    known = df["grade"].notna()
    df["sev_kind"] = np.where(known, SEV_FULL, SEV_GE1)
    df["grade"] = df["grade"].fillna(1).astype(int)  # GE1 rows: only "grade>0" is supervised
    not_disease = df["class"] == "insect_damage"
    df.loc[not_disease, "sev_kind"] = SEV_NONE
    df.loc[not_disease, "grade"] = 0
    missing = df[(df.source_type == "original") & ~known]
    if len(missing):
        raise RuntimeError(f"{len(missing)} originals have no severity grade")
    return df.reset_index(drop=True)


def bucket(row: pd.Series) -> str:
    if row["source"] in ("vegnet", "healthy_leaf_crop"):
        return "aug" if row["source_type"] == "augmented" else "real"
    return "cutout" if row["source"] in CUTOUT_SOURCES else "external"


class V2Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tfm, scene: SceneComposer | None = None):
        self.f = frame.reset_index(drop=True)
        self.tfm = tfm
        self.scene = scene

    def __len__(self) -> int:
        return len(self.f)

    def __getitem__(self, i: int):
        r = self.f.iloc[i]
        with Image.open(cache_path_for(r["filepath"])) as im:
            im = im.convert("RGB")
            if self.scene is not None:
                im = self.scene(im, is_cutout=r.get("source") in CUTOUT_SOURCES)
            x = self.tfm(im)
        return x, int(r["label"]), int(r["grade"]), int(r["sev_kind"]), i


def make_sampler(frame: pd.DataFrame, shares: dict[str, float], seed: int) -> WeightedRandomSampler:
    """Every class gets an equal share; within a class, source buckets get `shares`
    (renormalised over the buckets that class actually has)."""
    w = np.zeros(len(frame))
    buckets = frame.apply(bucket, axis=1).to_numpy()
    cls = frame["class"].to_numpy()
    classes = sorted(set(cls))
    for c in classes:
        present = {b: sh for b, sh in shares.items() if ((cls == c) & (buckets == b)).any()}
        total = sum(present.values())
        for b, share in present.items():
            m = (cls == c) & (buckets == b)
            w[m] = share / total / len(classes) / m.sum()
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(frame), replacement=True, generator=g)


def param_groups(model: MultiTaskNet) -> list[dict]:
    groups = []
    n = len(model.features)
    for i, block in enumerate(model.features):
        lr = HP["backbone_top_lr"] * HP["layer_decay"] ** (n - 1 - i)
        groups.append({"params": list(block.parameters()), "lr": lr})
    heads = list(model.cls_head.parameters()) + list(model.presence_head.parameters()) + list(model.sev_head.fc.parameters())
    groups.append({"params": heads, "lr": HP["head_lr"]})
    # CORAL thresholds must travel several logits; at the head LR Adam moves them
    # ~1e-3/step and they end up almost at their init. Own LR, no weight decay.
    groups.append({"params": [model.sev_head.bias], "lr": HP["coral_bias_lr"], "weight_decay": 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


def presence_targets(labels: torch.Tensor) -> torch.Tensor:
    t = torch.zeros(len(labels), len(DISEASES), device=labels.device)
    dis = labels < len(DISEASES)
    t[dis, labels[dis]] = 1.0
    return t


def cutmix(x: torch.Tensor, labels: torch.Tensor):
    """Paste a box from a shuffled partner. Returns mixed x, partner labels, kept-area lam."""
    n, _, h, w = x.shape
    perm = torch.randperm(n, device=x.device)
    lam = float(np.random.beta(1.0, 1.0))
    cut = math.sqrt(1 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y0, y1 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x0, x1 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x = x.clone()
    x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return x, labels[perm], lam


def train_one_fold(fold: int, tr: pd.DataFrame, device: torch.device, epochs: int, recipe: dict, log) -> MultiTaskNet:
    torch.manual_seed(HP["seed"] + fold)
    np.random.seed(HP["seed"] + fold)
    model = MultiTaskNet(pretrained=True, dropout=HP["dropout"], num_classes=len(recipe["classes"])).to(device)
    scene = SceneComposer(BG_DIR) if recipe["scene"] else None
    ds = V2Dataset(tr, train_transforms(HP["image_size"], realworld=recipe["realworld"]), scene)
    dl = DataLoader(
        ds, batch_size=HP["batch_size"], sampler=make_sampler(tr, recipe["bucket_shares"], HP["seed"] + fold),
        num_workers=6, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    opt = torch.optim.AdamW(param_groups(model), weight_decay=HP["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    steps_total = epochs * len(dl)
    warm = HP["warmup_epochs"] * len(dl)
    step = 0
    for ep in range(epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, g, kind, _ in dl:
            x, y, g, kind = x.to(device, non_blocking=True), y.to(device), g.to(device), kind.to(device)
            scale = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(steps_total - warm, 1)))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * scale
            use_mix = np.random.rand() < HP["cutmix_p"]
            pres_t = presence_targets(y)
            sev_mask = torch.ones(len(y), NUM_GRADES - 1, device=device)
            sev_mask[kind == SEV_GE1, 1:] = 0.0
            sev_mask[kind == SEV_NONE] = 0.0
            if use_mix:
                x, y_b, lam = cutmix(x, y)
                full = HP["presence_full_area"]
                wa, wb = min(lam / full, 1.0), min((1 - lam) / full, 1.0)
                pres_t = torch.maximum(pres_t * wa, presence_targets(y_b) * wb)
                sev_mask.zero_()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            cls_logits = out["cls"].float()
            if use_mix:
                l_cls = lam * F.cross_entropy(cls_logits, y) + (1 - lam) * F.cross_entropy(cls_logits, y_b)
            else:
                l_cls = F.cross_entropy(cls_logits, y)
            l_pres = F.binary_cross_entropy_with_logits(out["presence"].float(), pres_t)
            l_sev = coral_loss(out["sev"].float(), g, sev_mask)
            loss = l_cls + HP["w_presence"] * l_pres + HP["w_severity"] * l_sev
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss)
            nb += 1
            step += 1
        log(f"fold {fold} epoch {ep + 1}/{epochs} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)")
    return model


@torch.inference_mode()
def run_outputs(model: MultiTaskNet, frame: pd.DataFrame, device: torch.device, tta: bool) -> dict[str, np.ndarray]:
    model.eval()
    dl = DataLoader(V2Dataset(frame, eval_transforms(HP["image_size"])), batch_size=64, num_workers=6)
    acc: dict[str, list] = {"cls": [], "presence": [], "sev": [], "feat": []}
    for x, *_ in dl:
        x = x.to(device)
        views = [x, torch.flip(x, dims=[3])] if tta else [x]
        outs = [model(v) for v in views]
        for k in acc:
            acc[k].append(torch.stack([o[k].float() for o in outs]).mean(0).cpu())
    return {k: torch.cat(v).numpy() for k, v in acc.items()}


def fit_maha(feat: np.ndarray, labels: np.ndarray, dims: int) -> dict[str, np.ndarray]:
    """PCA -> class means + tied covariance. Score = min over classes of Mahalanobis^2."""
    mu = feat.mean(0)
    xc = feat - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:dims]
    z = xc @ comps.T
    # Classes absent from this fold's training rows get no centroid.
    present = np.unique(labels)
    means = np.stack([z[labels == c].mean(0) for c in present])
    resid = z - means[np.searchsorted(present, labels)]
    cov = resid.T @ resid / len(z) + 1e-3 * np.eye(dims)
    prec = np.linalg.inv(cov)
    return {"mu": mu.astype(np.float32), "comps": comps.astype(np.float32),
            "means": means.astype(np.float32), "prec": prec.astype(np.float32)}


def maha_score(feat: np.ndarray, st: dict[str, np.ndarray]) -> np.ndarray:
    z = (feat - st["mu"]) @ st["comps"].T
    d = z[:, None, :] - st["means"][None]
    return np.einsum("nci,ij,ncj->nc", d, st["prec"], d).min(1)


def quick_metrics(frame: pd.DataFrame, out: dict[str, np.ndarray]) -> dict:
    pred = out["cls"].argmax(1)
    sev_p = coral_to_probs(torch.as_tensor(out["sev"])).numpy()
    sev_pred = sev_p.argmax(1)
    y, g = frame["label"].to_numpy(), frame["grade"].to_numpy()
    graded = frame["sev_kind"].to_numpy() == SEV_FULL
    m = {"acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro"))}
    for src in sorted(frame["source"].unique()):
        sel = frame["source"].to_numpy() == src
        m[f"acc_{src}"] = float((pred[sel] == y[sel]).mean())
    m |= {
        "sev_exact": float((sev_pred[graded] == g[graded]).mean()),
        "sev_within1": float((np.abs(sev_pred[graded] - g[graded]) <= 1).mean()),
        "sev_mae": float(np.abs(sev_pred[graded] - g[graded]).mean()),
    }
    return m


def assign_folds(df: pd.DataFrame) -> np.ndarray:
    """Two-stage grouped folds so the scarce, heavily clustered black_rot originals
    spread evenly: first split groups that contain originals (stratified on the
    originals only), then split augmented-only groups. Groups never straddle folds."""
    df = df.assign(group_id=df["group_id"].astype(str))
    orig_groups = set(df.loc[df["source_type"] == "original", "group_id"])
    stage1 = df[(df["source_type"] == "original")]
    sgkf = StratifiedGroupKFold(n_splits=HP["n_folds"], shuffle=True, random_state=HP["seed"])
    g2f = {}
    for k, (_, te) in enumerate(sgkf.split(stage1, stage1["class"], groups=stage1["group_id"])):
        for g in stage1.iloc[te]["group_id"]:
            g2f[g] = k
    rest = df[~df["group_id"].isin(orig_groups)]
    for k, (_, te) in enumerate(sgkf.split(rest, rest["class"], groups=rest["group_id"])):
        for g in rest.iloc[te]["group_id"]:
            g2f[g] = k
    fold = df["group_id"].map(g2f)
    assert fold.notna().all()
    return fold.astype(int).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    ap.add_argument("--folds", type=str, default="all")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: subsample training rows")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="v2")
    ap.add_argument("--data", choices=["v2", "v3", "v3-noext", "v4"], default=None, help="override the recipe's data")
    ap.add_argument("--exp-dir", type=str, default="", help="override the recipe's output folder")
    ap.add_argument("--no-multitask", action="store_true", help="ablation: diagnosis loss only (presence/severity weights 0)")
    args = ap.parse_args()
    if args.no_multitask:
        HP["w_presence"] = HP["w_severity"] = 0.0

    recipe = dict(RECIPES[args.recipe])
    if args.data:
        recipe["data"] = args.data
    exp_dir = Path(args.exp_dir) if args.exp_dir else recipe["exp_dir"]
    exp_dir = exp_dir if exp_dir.is_absolute() else cfg.PROJECT_ROOT / exp_dir
    classes = recipe["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(exp_dir / "train_log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    df = load_frame(recipe["data"], classes)
    df["fold"] = assign_folds(df)
    df[["filepath", "class", "source", "source_type", "group_id", "grade", "sev_kind", "fold"]].to_csv(exp_dir / "folds.csv", index=False)
    real = df[df.source_type.isin(REAL_SOURCE_TYPES)]
    log(f"recipe={args.recipe} data={recipe['data']} device={device} rows={len(df)} classes={classes}")
    log("real photos per fold: " + str(real.groupby("fold")["class"].value_counts().unstack().fillna(0).astype(int).to_dict("index")))

    folds = range(HP["n_folds"]) if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    for k in folds:
        tr = df[df["fold"] != k]
        if args.limit:
            tr = tr.sample(args.limit, random_state=0)
        ho = df[(df["fold"] == k) & (df["source_type"].isin(REAL_SOURCE_TYPES))].reset_index(drop=True)
        model = train_one_fold(k, tr.reset_index(drop=True), device, args.epochs, recipe, log)
        # Deploy weights are stored fp16; calibrate on exactly those weights.
        state16 = {n: (t.half() if t.is_floating_point() else t) for n, t in model.state_dict().items()}
        model.load_state_dict({n: t.float() for n, t in state16.items()})

        tr_out = run_outputs(model, tr.reset_index(drop=True), device, tta=False)
        maha = fit_maha(tr_out["feat"], tr["label"].to_numpy(), HP["pca_dims"])
        ho_out = run_outputs(model, ho, device, tta=True)
        ho_out["maha"] = maha_score(ho_out["feat"], maha)
        m = quick_metrics(ho, ho_out)
        log(f"fold {k} held-out real photos n={len(ho)}: {json.dumps({a: round(b, 4) for a, b in m.items()})}")

        torch.save(
            {"model_state": state16, "maha": {a: torch.from_numpy(b) for a, b in maha.items()},
             "fold": k, "hp": HP, "classes": classes, "recipe": args.recipe,
             "created_utc": datetime.now(timezone.utc).isoformat()},
            exp_dir / f"fold{k}.pt",
        )
        np.savez_compressed(
            exp_dir / f"oof_fold{k}.npz",
            filepath=ho["filepath"].to_numpy(), label=ho["label"].to_numpy(), grade=ho["grade"].to_numpy(),
            sev_kind=ho["sev_kind"].to_numpy(), source=ho["source"].to_numpy(),
            cls=ho_out["cls"], presence=ho_out["presence"], sev=ho_out["sev"], maha=ho_out["maha"],
        )
        del model
        torch.cuda.empty_cache()

    train_data = {"v2": "splits/train.csv + splits/val.csv (test.csv never read)",
                  "v3": "splits_v3/train_pool.csv (external_test.csv and splits/test.csv never read)",
                  "v3-noext": "splits_v3/train_pool.csv without external sources",
                  "v4": "splits_v4/train_pool.csv = v3 pool + reviewed insect_damage photos "
                        "(splits_v3/external_test.csv, splits_v4/insect_test.csv and splits/test.csv never read)"}[recipe["data"]]
    (exp_dir / "config.json").write_text(json.dumps({
        "experiment": exp_dir.name, "recipe": args.recipe, "data": recipe["data"], "classes": classes,
        "realworld_aug": recipe["realworld"], "scene_aug": recipe["scene"],
        "bucket_shares": recipe["bucket_shares"], "hp": HP, "train_data": train_data,
        "severity_labels": str(GRADES_CSV.relative_to(cfg.PROJECT_ROOT))}, indent=2))
    log("done")


if __name__ == "__main__":
    main()
