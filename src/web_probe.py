"""Web probe: real cauliflower photos found online, used to look for failure modes.

Photos live in new_pics/<class>/ (git-ignored: other people's photos, kept for
private evaluation only). new_pics/manifest.csv records where each photo came
from and why it carries its label (the source page must name cauliflower and the
disease; every photo is also looked at before it is kept).

  python src/web_probe.py add --cls black_rot --url <image url> --page <page url> --evidence "..."
  python src/web_probe.py check      # flag copies of training / external-test photos
  python src/web_probe.py run        # run the current app pipeline, write outputs/web_probe/results.csv

Copies of photos we already have (train pool or frozen external test) are kept
in the manifest but excluded from the scores: they say nothing new.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = cfg.PROJECT_ROOT
PICS = ROOT / "new_pics"
MANIFEST = PICS / "manifest.csv"
OUT = ROOT / "outputs" / "web_probe"
CLASSES = ("no_disease", "downy_mildew", "black_rot", "bacterial_spot_rot")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
EMB_CACHE = ROOT / "cache" / "v3_all_emb.npz"


def _manifest() -> pd.DataFrame:
    if MANIFEST.is_file():
        df = pd.read_csv(MANIFEST, dtype={"is_copy": str})
        # NaN = not checked yet; older writes stored the flag as 1.0/0.0, newer as True/False
        df["is_copy"] = df["is_copy"].map({"True": True, "1.0": True, "False": False, "0.0": False}).astype(object)
        return df
    return pd.DataFrame(columns=["id", "file", "cls", "img_url", "page_url", "evidence", "phash", "sha256",
                                 "width", "height", "added_at", "is_copy", "copy_of", "copy_inliers", "notes"])


def add(cls: str, url: str, page: str, evidence: str, notes: str = "", file: str = "") -> None:
    assert cls in CLASSES, cls
    if file:  # already downloaded (and looked at) from `url`
        data = Path(file).read_bytes()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": page or url})
        data = urllib.request.urlopen(req, timeout=60).read()
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    sha = hashlib.sha256(data).hexdigest()
    ph = str(imagehash.phash(im))
    df = _manifest()
    if (df["sha256"] == sha).any():
        print("SKIP exact duplicate of", df.loc[df["sha256"] == sha, "id"].iloc[0])
        return
    for _, r in df.iterrows():
        if imagehash.hex_to_hash(r["phash"]) - imagehash.hex_to_hash(ph) <= 6:
            print("SKIP near-duplicate of", r["id"])
            return
    n = int(df["id"].str.split("_").str[-1].astype(int).max()) + 1 if len(df) else 1
    pid = f"{cls}_{n:04d}"
    path = PICS / cls / f"{pid}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=95)
    row = {"id": pid, "file": str(path.relative_to(ROOT)).replace("\\", "/"), "cls": cls, "img_url": url,
           "page_url": page, "evidence": evidence, "phash": ph, "sha256": sha, "width": im.width,
           "height": im.height, "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "is_copy": np.nan, "copy_of": "", "copy_inliers": np.nan, "notes": notes}
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(MANIFEST, index=False)
    print(f"ADDED {pid} {im.width}x{im.height} -> {path.relative_to(ROOT)}")


def _known() -> pd.DataFrame:
    tr = pd.read_csv(ROOT / "splits_v3" / "train_pool.csv").assign(split="train_pool")
    te = pd.read_csv(ROOT / "splits_v3" / "external_test.csv").assign(split="external_test")
    return pd.concat([tr, te], ignore_index=True)


def check() -> None:
    import torch
    from external_data import _backbone, _embed, inliers, sift_feats, DUP_INLIERS
    from v2_cache import cache_path_for

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known = _known()
    model = _backbone(device)
    if EMB_CACHE.is_file() and list(np.load(EMB_CACHE, allow_pickle=True)["filepath"]) == known["filepath"].tolist():
        emb = np.load(EMB_CACHE, allow_pickle=True)["emb"]
    else:
        embs = []
        for i in range(0, len(known), 256):
            ims = [Image.open(cache_path_for(f)).convert("RGB") for f in known["filepath"][i:i + 256]]
            embs.append(_embed(model, ims, device))
        emb = np.concatenate(embs)
        np.savez_compressed(EMB_CACHE, filepath=known["filepath"].to_numpy(), emb=emb)
    df = _manifest()
    todo = df[df["is_copy"].isna()].index
    if not len(todo):
        print("nothing new to check")
        return
    ims = [Image.open(ROOT / df.at[i, "file"]).convert("RGB") for i in todo]
    sims = _embed(model, ims, device) @ emb.T
    for k, i in enumerate(todo):
        fa = sift_feats(ims[k])
        top = np.argsort(-sims[k])[:20]
        scores = [inliers(fa, sift_feats(Image.open(cache_path_for(known.at[j, "filepath"])))) for j in top]
        b = int(np.argmax(scores))
        df.at[i, "copy_inliers"] = scores[b]
        df.at[i, "is_copy"] = scores[b] >= DUP_INLIERS
        df.at[i, "copy_of"] = f"{known.at[top[b], 'split']}:{known.at[top[b], 'filepath']}"
        print(df.at[i, "id"], "COPY" if df.at[i, "is_copy"] else "new", scores[b], df.at[i, "copy_of"][:90])
    df.to_csv(MANIFEST, index=False)


def run(exp_dir: str = "", out_name: str = "results.csv") -> pd.DataFrame:
    from plant_runtime import PlantEngine, plant_public_result

    eng = PlantEngine(Path(exp_dir)) if exp_dir else PlantEngine()
    df = _manifest()
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        im = Image.open(ROOT / r["file"]).convert("RGB")
        res = eng.predict_pil(im, with_overlay=True)
        p = plant_public_result(res)
        plant = p.get("plant") or {}
        if p.get("overlay_jpeg_b64"):
            import base64
            (OUT / "overlays").mkdir(exist_ok=True)
            (OUT / "overlays" / f"{r['id']}.jpg").write_bytes(base64.b64decode(p["overlay_jpeg_b64"]))
        guess = p["prediction"] if p["reliable"] else p.get("unreliable_guess", {}).get("prediction")
        rows.append({
            "id": r["id"], "cls": r["cls"], "is_copy": r["is_copy"], "status": p["status"],
            "prediction": p.get("prediction"), "confidence": p.get("confidence"), "guess": guess,
            "correct": p["reliable"] and p.get("prediction") == r["cls"],
            "confident_wrong": p["status"] == "confident" and p.get("prediction") != r["cls"],
            "stage": plant.get("stage"), "affected_pct": plant.get("affected_pct"),
            "parts": ";".join(f"{g['type']}:{g['disease']}:{g.get('measured_affected_pct')}" for g in plant.get("regions", [])),
            "message": p.get("message"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / out_name, index=False)
    ev = out[out["is_copy"] != True]  # noqa: E712
    print(f"\n{len(out)} photos ({len(ev)} not copies of known photos)")
    print(ev.groupby("cls").agg(n=("id", "size"), correct=("correct", "sum"), confident_wrong=("confident_wrong", "sum"),
                                answered=("status", lambda s: (s.isin(["confident", "uncertain"])).sum())))
    print(f"overall correct {ev['correct'].mean():.1%}, confident-wrong {ev['confident_wrong'].mean():.1%}")
    bad = ev[~ev["correct"]]
    if len(bad):
        print("\nNot correct:")
        print(bad[["id", "cls", "status", "prediction", "confidence", "guess", "stage", "parts"]].to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--cls", required=True, choices=CLASSES)
    a.add_argument("--url", required=True)
    a.add_argument("--page", default="")
    a.add_argument("--evidence", required=True)
    a.add_argument("--notes", default="")
    a.add_argument("--file", default="", help="local copy of --url, already downloaded")
    b = sub.add_parser("add-batch", help="TSV: cls, file, img_url, page_url, evidence")
    b.add_argument("tsv")
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--exp", default="", help="model folder to test (default: the app's model)")
    r.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    if args.cmd == "add":
        add(args.cls, args.url, args.page, args.evidence, args.notes, args.file)
    elif args.cmd == "add-batch":
        for line in Path(args.tsv).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                cls, file, url, page, evidence = line.split("\t")[:5]
                add(cls, url, page, evidence, file=file)
    elif args.cmd == "check":
        check()
    else:
        run(args.exp, args.out)


if __name__ == "__main__":
    main()
