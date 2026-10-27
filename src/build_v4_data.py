"""Step I-3: v4 training pool = the v3 pool + reviewed insect-damage photos.

Insect photos (cleaf "Insect Hole", splits_insect/insect_pool.csv) whose visual
review (outputs/insect_seg/review.csv) found insect damage (K or D) get the class
"insect_damage"; photos with no visible damage (N) and the 2 excluded ones are
dropped. The insect split from build_insect_data.py is kept:
  insect_train -> splits_v4/train_pool.csv
  insect_test  -> splits_v4/insect_test.csv (never trained on)
splits_v3/ (and its frozen external test) is read, never written.

  python src/build_v4_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

V3 = cfg.PROJECT_ROOT / "splits_v3"
OUT = cfg.PROJECT_ROOT / "splits_v4"
REVIEW = cfg.OUTPUTS_DIR / "insect_seg" / "review.csv"
COLS = ["filepath", "class", "source", "source_type", "group_id", "subset"]


def main() -> None:
    v3 = pd.read_csv(V3 / "train_pool.csv")
    frozen = pd.read_csv(V3 / "external_test.csv")
    rev = pd.read_csv(REVIEW)
    ins = rev[rev["review"].isin(["K", "D"])].assign(
        **{"class": "insect_damage", "source": "cleaf", "source_type": "external", "subset": "insect"})
    assert not set(ins["filepath"]) & (set(v3["filepath"]) | set(frozen["filepath"]))
    train = pd.concat([v3[COLS], ins.loc[ins["split"] == "insect_train", COLS]], ignore_index=True)
    test = ins.loc[ins["split"] == "insect_test", COLS + ["review"]].reset_index(drop=True)
    leak = set(train["group_id"].astype(str)) & (set(test["group_id"].astype(str)) | set(frozen["group_id"].astype(str)))
    assert not leak, f"group leakage: {sorted(leak)[:5]}"
    OUT.mkdir(exist_ok=True)
    train.to_csv(OUT / "train_pool.csv", index=False)
    test.to_csv(OUT / "insect_test.csv", index=False)
    summary = [
        "V4 TRAIN POOL (class x source) = splits_v3/train_pool.csv + insect_damage",
        pd.crosstab(train["source"] + "/" + train["source_type"], train["class"], margins=True).to_string(),
        "",
        f"INSECT TEST (never trained on): {len(test)} photos, review {test['review'].value_counts().to_dict()}",
        f"dropped from the category: {int((rev['review'] == 'N').sum())} photos with no visible insect damage",
        "frozen external test: splits_v3/external_test.csv (unchanged)",
    ]
    (OUT / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
