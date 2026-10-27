"""Web-probe fix: refit ONLY the diagnosis + presence heads with classic black rot photos.

Why (outputs/web_probe/): classic black rot (yellow V-shaped lesions from the leaf
edge) is called downy mildew - 0 of 13 web photos right, 4 of them "confident".
In training, VegNet's black rot is a few blue-grey leaves on red soil, cleaf's is
small detached leaves, and yellow-edged leaves are all downy mildew. No public
dataset adds real classic black rot (the Multi-Crop set is 621/640 copies of our
photos), so the fix uses the few web photos whose source names cauliflower black rot.

What changes: the linear diagnosis and presence heads of each fold model. The
backbone, the severity head and the out-of-scope (Mahalanobis) statistics stay
bit-identical, so measured severity, OOD and affected-% cannot change. The new
heads are anchored to the old ones (L2 to the old weights) and trained on the
same class x source balance as before, plus a small "web black rot" share.

Honest evaluation with tiny data:
  * web black-rot photos are grouped by source site; `cv` scores every web photo
    with heads fitted WITHOUT its site (leave-one-site-out), for every class
  * Shutterstock photos (watermarked) are never trained on: a watermark must not
    become a black-rot cue; they are always held out
  * the frozen external test (splits_v3/external_test.csv) is scored for the gate

  python src/refit_diagnosis.py feats          # cache fold features (train pool, external test, web)
  python src/refit_diagnosis.py cv             # leave-one-site-out + external-test check, prints a table
  python src/refit_diagnosis.py write          # fit on all allowed web photos, write experiments/<DST>

RESULT (6 Oct 2026): NOT SHIPPED. Leave-one-site-out, 11 close-ups from 11 sites:
web black rot 11% -> 56-67% but healthy web photos 81% -> 55-65% (+1 confident false
alarm) in every setting (anchor 0.3-30, web share 0.15-0.3). Adding healthy web photos
as a counterweight fixes healthy (84-87%) but drops the frozen external test 94.1% ->
93.0-93.8% (subtle head rot pulled to healthy). Too few photos to move only the
black-rot boundary. Rerun `feats` + `cv` once real, labelled classic black-rot photos exist.
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
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from train_v2 import bucket, load_frame
from v2_cache import cache_path_for
from v2_model import CLASS_ORDER, DISEASES, MultiTaskNet, eval_transforms

ROOT = cfg.PROJECT_ROOT
SRC = ROOT / "experiments" / "experiment_006_v3_measured_severity"
DST = ROOT / "experiments" / "experiment_007_web_blackrot_heads"
FEAT_DIR = ROOT / "cache" / "refit_diag"
WEB_TRAIN = ROOT / "new_pics" / "web_train.csv"
PROBE = ROOT / "new_pics" / "manifest.csv"
N_VIEWS = 24          # augmented views per web training photo
WEB_SHARE = 0.15      # share of black-rot sampling weight given to web photos
ANCHOR = 1e-2         # L2 pull towards the old head weights
STEPS, LR = 300, 2e-2
BR = CLASS_ORDER.index("black_rot")
# Wide field views: the lesions are a tiny part of the frame, so their crops are mostly
# healthy leaves labelled black rot. A first run with them taught "leafy plant = black
# rot" (healthy web photos 83% -> 50%). Only lesion-dominated close-ups are trained on.
TRAIN_EXCLUDE = {"brt_e43", "brt_e45", "brt_o2", "black_rot_0011"}


def _model(ckpt_path: Path, device) -> MultiTaskNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = MultiTaskNet(pretrained=False, num_classes=len(CLASS_ORDER))
    m.load_state_dict({n: t.float() for n, t in ckpt["model_state"].items()})
    return m.to(device).eval()


@torch.inference_mode()
def _feats(model, ims: list[Image.Image], device) -> torch.Tensor:
    tfm = eval_transforms(384)
    out = []
    for i in range(0, len(ims), 64):
        x = torch.stack([tfm(im) for im in ims[i:i + 64]]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            f = (model.embed(x) + model.embed(torch.flip(x, dims=[3]))) / 2
        out.append(f.float().cpu())
    return torch.cat(out)


VIEW = T.Compose([T.RandomResizedCrop(512, scale=(0.7, 1.0), ratio=(0.75, 1.33)), T.RandomHorizontalFlip(),
                  T.RandomRotation(15), T.ColorJitter(0.25, 0.25, 0.2, 0.03)])


def web_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(trainable web photos, all probe photos). Trainable = web_train.csv + probe black-rot photos
    that are not watermarked stock and not copies of known photos."""
    probe = pd.read_csv(PROBE)
    probe = probe[probe["is_copy"] != True].copy()  # noqa: E712
    probe["site"] = probe["page_url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0]
    extra = pd.read_csv(WEB_TRAIN)
    trainable = pd.concat([
        extra.assign(id=extra["file"].map(lambda f: Path(f).stem), in_probe=False),
        probe[(probe["cls"] == "black_rot") & ~probe["evidence"].str.contains("WATERMARK")]
        [["id", "file", "cls", "site"]].assign(in_probe=True),
    ], ignore_index=True)
    trainable = trainable[~trainable["id"].isin(TRAIN_EXCLUDE)].reset_index(drop=True)
    return trainable, probe


def feats() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pool = load_frame("v3", CLASS_ORDER)
    ext = pd.read_csv(ROOT / "splits_v3" / "external_test.csv")
    trainable, probe = web_sets()
    rng = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    for ck in sorted(SRC.glob("fold*.pt")):
        k = int(ck.stem[4:])
        out = FEAT_DIR / f"fold{k}.pt"
        if out.is_file():
            continue
        m = _model(ck, device)
        f_pool = torch.cat([_feats(m, [Image.open(cache_path_for(p)).convert("RGB") for p in pool["filepath"][i:i + 512]], device)
                            for i in range(0, len(pool), 512)])
        f_ext = torch.cat([_feats(m, [Image.open(cache_path_for(p)).convert("RGB") for p in ext["filepath"][i:i + 512]], device)
                           for i in range(0, len(ext), 512)])
        f_probe = _feats(m, [Image.open(ROOT / f).convert("RGB") for f in probe["file"]], device)
        views, owner = [], []
        for j, f in enumerate(trainable["file"]):
            im = Image.open(ROOT / f).convert("RGB")
            views += [im] + [VIEW(im) for _ in range(N_VIEWS)]
            owner += [j] * (N_VIEWS + 1)
        f_web = _feats(m, views, device)
        torch.save({"pool": f_pool, "ext": f_ext, "probe": f_probe, "web": f_web, "web_owner": torch.tensor(owner),
                    "cls_w": m.cls_head.weight.detach().cpu(), "cls_b": m.cls_head.bias.detach().cpu(),
                    "pres_w": m.presence_head.weight.detach().cpu(), "pres_b": m.presence_head.bias.detach().cpu()}, out)
        print(f"fold {k}: pool {tuple(f_pool.shape)} ext {tuple(f_ext.shape)} web {tuple(f_web.shape)}")
    pool[["filepath", "class", "source", "source_type"]].to_csv(FEAT_DIR / "pool.csv", index=False)
    trainable.to_csv(FEAT_DIR / "web_trainable.csv", index=False)
    probe.to_csv(FEAT_DIR / "probe.csv", index=False)


def _weights(pool: pd.DataFrame, n_web: int, web_share: float) -> torch.Tensor:
    """make_sampler's class x bucket balance as loss weights, plus web rows inside black_rot."""
    shares = {"real": 0.45, "aug": 0.2, "cutout": 0.15, "external": 0.2}
    b = pool.apply(bucket, axis=1).to_numpy()
    c = pool["class"].to_numpy()
    w = np.zeros(len(pool) + n_web)
    for cls in CLASS_ORDER:
        present = {k: s for k, s in shares.items() if ((c == cls) & (b == k)).any()}
        total = sum(present.values())
        scale = (1 - web_share) if (cls == "black_rot" and n_web) else 1.0
        for k, s in present.items():
            m = (c == cls) & (b == k)
            w[:len(pool)][m] = scale * s / total / m.sum()
    if n_web:
        w[len(pool):] = web_share / n_web
    return torch.tensor(w / w.sum() * len(w), dtype=torch.float32)


def fit(f: torch.Tensor, y: torch.Tensor, w: torch.Tensor, d: dict, device) -> dict:
    f, y, w = f.to(device), y.to(device), w.to(device)
    W0, b0 = d["cls_w"].to(device), d["cls_b"].to(device)
    P0, c0 = d["pres_w"].to(device), d["pres_b"].to(device)
    W, b, P, c = (t.clone().requires_grad_(True) for t in (W0, b0, P0, c0))
    pres_t = torch.zeros(len(y), len(DISEASES), device=device)
    dis = y < len(DISEASES)
    pres_t[dis, y[dis]] = 1.0
    opt = torch.optim.Adam([W, b, P, c], lr=LR)
    for _ in range(STEPS):
        lc = (F.cross_entropy(f @ W.T + b, y, reduction="none") * w).mean()
        lp = (F.binary_cross_entropy_with_logits(f @ P.T + c, pres_t, reduction="none").mean(1) * w).mean()
        reg = ANCHOR * (((W - W0) ** 2).sum() + ((P - P0) ** 2).sum())
        opt.zero_grad()
        (lc + 0.5 * lp + reg).backward()
        opt.step()
    return {k: v.detach().cpu() for k, v in zip(("cls_w", "cls_b", "pres_w", "pres_b"), (W, b, P, c))}


def _probs(f: torch.Tensor, h: dict, temp: float) -> torch.Tensor:
    return torch.softmax((f @ h["cls_w"].T + h["cls_b"]) / temp, 1)


def cv() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp = json.loads((SRC / "calibration.json").read_text())["cls_temperature"]
    pool = pd.read_csv(FEAT_DIR / "pool.csv")
    folds = pd.read_csv(SRC / "folds.csv")[["filepath", "fold"]]
    fold_of = pool.merge(folds, on="filepath", how="left")["fold"].to_numpy()
    trainable = pd.read_csv(FEAT_DIR / "web_trainable.csv")
    probe = pd.read_csv(FEAT_DIR / "probe.csv")
    ext = pd.read_csv(ROOT / "splits_v3" / "external_test.csv")
    y_pool = torch.tensor(pool["class"].map(CLASS_ORDER.index).to_numpy())
    y_ext = ext["class"].map(CLASS_ORDER.index).to_numpy()
    sites = sorted(trainable["site"].unique())
    D = {k: torch.load(FEAT_DIR / f"fold{k}.pt") for k in range(5)}

    base_probe = sum(_probs(D[k]["probe"], D[k], temp) for k in D) / 5
    base_ext = sum(_probs(D[k]["ext"], D[k], temp) for k in D) / 5

    def fit_fold(k, keep_web: np.ndarray, share: float):
        d = D[k]
        tr = fold_of != k
        own = d["web_owner"].numpy()
        sel = keep_web[own]
        f = torch.cat([d["pool"][tr], d["web"][sel]])
        y = torch.cat([y_pool[tr], torch.full((int(sel.sum()),), BR)])
        w = _weights(pool[tr].reset_index(drop=True), int(sel.sum()), share)
        return fit(f, y, w, d, device)

    for share in (0.0, WEB_SHARE, 0.3):
        # frozen external test: heads fitted on all trainable web photos
        all_web = np.ones(len(trainable), bool)
        heads = {k: fit_fold(k, all_web, share) for k in D}
        p_ext = sum(_probs(D[k]["ext"], heads[k], temp) for k in D) / 5
        p_probe_all = sum(_probs(D[k]["probe"], heads[k], temp) for k in D) / 5
        # leave-one-site-out for probe photos whose site contributed training photos
        p_probe = p_probe_all.clone()
        for s in sites:
            keep = (trainable["site"] != s).to_numpy()
            hs = {k: fit_fold(k, keep, share) for k in D}
            rows = np.where(probe["site"].to_numpy() == s)[0]
            if len(rows):
                p_probe[rows] = sum(_probs(D[k]["probe"][rows], hs[k], temp) for k in D) / 5
        yp = probe["cls"].map(CLASS_ORDER.index).to_numpy()
        pred, conf = p_probe.argmax(1).numpy(), p_probe.max(1).values.numpy()
        bpred = base_probe.argmax(1).numpy()
        acc_ext = (p_ext.argmax(1).numpy() == y_ext).mean()
        print(f"\nweb share {share}: external test argmax acc {acc_ext:.4f} (baseline {(base_ext.argmax(1).numpy() == y_ext).mean():.4f})")
        for c in CLASS_ORDER:
            m = yp == CLASS_ORDER.index(c)
            print(f"  probe {c:20s} n={m.sum():2d} acc {np.mean(pred[m] == yp[m]):.2f} (baseline {np.mean(bpred[m] == yp[m]):.2f})"
                  f"  conf>=0.8 & wrong {int(((pred != yp) & (conf >= 0.8) & m).sum())}")
        ext_cm = pd.crosstab(pd.Series(y_ext, name="true").map(dict(enumerate(CLASS_ORDER))),
                             pd.Series(p_ext.argmax(1).numpy(), name="pred").map(dict(enumerate(CLASS_ORDER))))
        print(ext_cm.to_string())


def write(share: float = WEB_SHARE) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = pd.read_csv(FEAT_DIR / "pool.csv")
    folds = pd.read_csv(SRC / "folds.csv")[["filepath", "fold"]]
    fold_of = pool.merge(folds, on="filepath", how="left")["fold"].to_numpy()
    trainable = pd.read_csv(FEAT_DIR / "web_trainable.csv")
    y_pool = torch.tensor(pool["class"].map(CLASS_ORDER.index).to_numpy())
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    for k in range(5):
        d = torch.load(FEAT_DIR / f"fold{k}.pt")
        tr = fold_of != k
        f = torch.cat([d["pool"][tr], d["web"]])
        y = torch.cat([y_pool[tr], torch.full((len(d["web"]),), BR)])
        h = fit(f, y, _weights(pool[tr].reset_index(drop=True), len(d["web"]), share), d, device)
        ck_path = DST / f"fold{k}.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        ck["model_state"]["cls_head.weight"] = h["cls_w"].half()
        ck["model_state"]["cls_head.bias"] = h["cls_b"].half()
        ck["model_state"]["presence_head.weight"] = h["pres_w"].half()
        ck["model_state"]["presence_head.bias"] = h["pres_b"].half()
        ck["diagnosis_heads"] = "refit on frozen features + web classic black rot (src/refit_diagnosis.py)"
        torch.save(ck, ck_path)
        # held-out logits for calibration: same rows as the OOF file, new heads
        oof = dict(np.load(DST / f"oof_fold{k}.npz", allow_pickle=True))
        idx = pool.reset_index().set_index("filepath").loc[list(oof["filepath"]), "index"].to_numpy()
        fo = d["pool"][idx]
        oof["cls"] = (fo @ h["cls_w"].half().float().T + h["cls_b"].half().float()).numpy()
        oof["presence"] = (fo @ h["pres_w"].half().float().T + h["pres_b"].half().float()).numpy()
        np.savez_compressed(DST / f"oof_fold{k}.npz", **oof)
        print(f"fold {k} written")
    conf = json.loads((DST / "config.json").read_text())
    conf["experiment"] = DST.name
    conf["diagnosis_heads"] = (f"refit on frozen fold features with {len(trainable)} web classic-black-rot photos "
                               f"x {N_VIEWS + 1} views, web share {share}, anchor L2 {ANCHOR} (src/refit_diagnosis.py)")
    (DST / "config.json").write_text(json.dumps(conf, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write" and len(sys.argv) > 2:
        write(float(sys.argv[2]))
    else:
        {"feats": feats, "cv": cv, "write": write}[cmd]()
