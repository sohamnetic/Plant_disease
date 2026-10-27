"""Step I-2, bootstrap round k: let a trained outline model draft the holes on the
insect TRAINING photos whose rule-based draft was not good enough (review "D"), so a
model that sees shape and texture can catch the tinted holes the colour rule missed.

  python src/insect_bootstrap.py drafts --weights experiments/seg_002_insect/weights/best.pt --round 2
      -> cache/insect_prelabel/round2/*.npz (leaf, hole), outputs/insect_seg/round2_drafts.csv,
         review sheets outputs/insect_seg/review_sheets/round2_*.jpg
  (review: write outputs/insect_seg/round2_review.txt with a "K idx idx ..." line)
  python src/insect_bootstrap.py apply --round 2
      -> outputs/insect_seg/round2_picks.csv (read by insect_seg_dataset.py --round 2)

Only training photos are bootstrapped: test photos keep their rule-based reference,
so a model is never scored against its own drafts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from insect_prelabel import CACHE, OUT_DIR, cache_path, eaten_pct, load_work, sheets
from seg_runtime import SegEngine


def drafts(weights: Path, k: int, device: str) -> None:
    rev = pd.read_csv(OUT_DIR / "review.csv")
    rows = rev[(rev["review"] == "D") & (rev["split"] == "insect_train")].reset_index(drop=True)
    eng = SegEngine(weights, device=device)
    mdir = CACHE / f"round{k}"
    mdir.mkdir(parents=True, exist_ok=True)
    pct = []
    for _, r in rows.iterrows():
        rgb = load_work(r["filepath"])
        m, _ = eng.measure_all(rgb)
        hole = m["_holes"]
        leaf = m["_tissue"] & ~hole
        mk = {"leaf": leaf, "hole": hole}
        np.savez_compressed(mdir / cache_path(r["filepath"]).name, **mk)
        pct.append(eaten_pct(mk))
    rows = rows.assign(eaten_pct=pct)
    rows.to_csv(OUT_DIR / f"round{k}_drafts.csv", index=False)
    sheets(rows, f"round{k}", mask_dir=mdir)
    print(f"round {k}: {len(rows)} drafts, holes found on {(rows['eaten_pct'] > 0).sum()}")


def apply(k: int) -> None:
    rows = pd.read_csv(OUT_DIR / f"round{k}_drafts.csv")
    keep: set[int] = set()
    for line in (OUT_DIR / f"round{k}_review.txt").read_text().splitlines():
        if line.startswith("K "):
            keep |= {int(i) for i in line.split()[1:]}
    unknown = keep - set(rows["idx"])
    assert not unknown, f"review lists photos not in this round: {sorted(unknown)[:5]}"
    rows["review"] = np.where(rows["idx"].isin(keep), "K", "D")
    rows.to_csv(OUT_DIR / f"round{k}_picks.csv", index=False)
    print(f"round {k}: {len(keep)} of {len(rows)} drafts kept")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["drafts", "apply"])
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    if a.cmd == "drafts":
        drafts(Path(a.weights), a.round, a.device)
    else:
        apply(a.round)


if __name__ == "__main__":
    main()
