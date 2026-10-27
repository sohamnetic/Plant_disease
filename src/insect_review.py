"""Step I-1: apply the visual review of the insect-hole drafts.

Reads outputs/insect_seg/drafts.csv and review_pass1.txt (K / N lines; everything
else is D) and writes outputs/insect_seg/review.csv with one decision per photo:
  K  holes outlined well enough to train the outline model on
  D  insect damage visible, but the draft misses or mis-marks some of it
  N  no insect damage visible: drop from the "insect damage" category

  python src/insect_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

SEG = cfg.OUTPUTS_DIR / "insect_seg"
PREDS = cfg.OUTPUTS_DIR / "insect_probe" / "insect_hole_predictions.csv"


def read_picks(path: Path) -> dict[int, str]:
    picks: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if line[:2] in ("K ", "N "):
            for idx in line.split()[1:]:
                if int(idx) in picks and picks[int(idx)] != line[0]:
                    raise SystemExit(f"idx {idx} marked both {picks[int(idx)]} and {line[0]}")
                picks[int(idx)] = line[0]
    return picks


def main() -> None:
    drafts = pd.read_csv(SEG / "drafts.csv")
    picks = read_picks(SEG / "review_pass1.txt")
    unknown = set(picks) - set(drafts["idx"])
    assert not unknown, f"review lists unknown idx: {sorted(unknown)[:5]}"
    drafts["review"] = drafts["idx"].map(picks).fillna("D")
    drafts["drafted"] = drafts["eaten_pct"] > 0
    assert not (drafts["drafted"] & ~drafts["review"].isin(["K", "D", "N"])).any()
    preds = pd.read_csv(PREDS)
    preds["filepath"] = preds["filepath"].map(lambda f: str(Path(f).resolve().relative_to(cfg.PROJECT_ROOT)).replace("\\", "/"))
    preds["model_answer"] = preds["status"].where(preds["status"] != "confident", "confident:" + preds["prediction"].fillna(""))
    drafts = drafts.merge(preds[["filepath", "model_answer"]], on="filepath", how="left")
    drafts.to_csv(SEG / "review.csv", index=False)

    out = [
        "I-1 REVIEW (K = outline usable for training, D = damage visible / outline not good enough, N = no damage visible)",
        pd.crosstab(drafts["split"], drafts["review"], margins=True).to_string(),
        "",
        "BY DRAFT (hole drafted vs not)",
        pd.crosstab(drafts["drafted"], drafts["review"], margins=True).to_string(),
        "",
        "% EATEN ON K PHOTOS (drafted holes)",
        drafts.loc[drafts["review"] == "K", "eaten_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_string(),
        "",
        "CURRENT MODEL'S ANSWER BY REVIEW",
        pd.crosstab(drafts["model_answer"], drafts["review"], margins=True).to_string(),
    ]
    (SEG / "REVIEW_SUMMARY.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
