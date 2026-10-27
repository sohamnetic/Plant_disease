# Phase 3 report — measuring affected area with YOLO11-seg

**Status:** complete. All 4 plan targets pass. Not connected to the API yet: that's Phase 4, which has not been started.
**Model:** `experiments/seg_001_yolo11s/weights/best.pt` (YOLO11s-seg, 10 M parameters, 20 MB, 0.15 s per photo on CPU).
**What it does:** outlines every **leaf** and **head (curd)** plus every **lesion**, then computes, for each leaf and head, the % of tissue affected, the number of lesions and a draft stage. The plant's stage comes from its worst leaf or head.

## How the labels were made

1. **550 cauliflower photos selected.** This includes all 40 early-stage (grade 1) VegNet photos, plus photos from all 4 external cauliflower sources. 103 were held back for testing, drawn from test splits only.
2. **Machine pre-labels.** SAM2 (segment-anything) drew 4 candidate leaf/head outlines per photo. Colour rules then marked lesions inside them: yellowing, brown/black necrosis, bleached tissue, and rot on the head that is darker *and* browner, so shadows between florets don't count. Healthy photos get no lesions.
3. **Visual review of all 550.** For each photo I picked the best candidate or rejected it. **372 kept** (299 train, 73 held out). The picks are in `outputs/seg/review_picks.txt`.
   - Rejected: shadows marked as rot, only part of the plant outlined, soil marked as lesion, and all 15 healthy-leaf crops (several partial leaves in each).
   - **These labels are machine-drawn and AI-reviewed, not drawn by a person or an expert.**
4. **Independent expert check.** 90 PlantSeg cauliflower images with lesion masks drawn by human experts. Everything else in the 1.7 GB download was deleted.

## Results on the 73 held-out photos (never trained on)

| Target (from the plan) | Result | |
|---|---:|:-:|
| Affected-% error (worst leaf/head) ≤ 5 points | **2.7 points** (overall plant: 1.6) | ✅ |
| Lesion outline overlap (Dice) ≥ 0.6 | **0.75** | ✅ |
| Early cases caught ≥ 80% | **4/4** grade-1 photos; **13/14** photos measured at 0.5–5% | ✅ |
| False lesion area on healthy photos < 1% | **0.33%** | ✅ |

More detail:
- **Tissue outline overlap (IoU):** 0.90.
- **Measured % agrees with label %:** Spearman 0.91.
- **Stage matches the label's stage:** 73%. All misses are one stage off.
- **Error by source (points):**

| healthy leaves | SPPU field | white-bg leaves | 2025 set | VegNet |
|---:|---:|---:|---:|---:|
| 0.9 | 1.4 | 1.9 | 2.3 | 4.0 |

- **Against the earlier severity grades (93 VegNet test photos):**
  - Median measured % by grade:

    | Grade | 0 | 1 | 2 | 3 | 4 |
    |---|---:|---:|---:|---:|---:|
    | Median % | 0 | 1.4 | 6.3 | 10.5 | 43.1 |

  - The order is right, but grades 2 and 3 overlap a lot: Spearman 0.52 on diseased photos.
- **Your field photo:** the rotten head is **70% affected, critical**. The surrounding leaves are at 10% (a few spots).

## Honest caveats

1. **Healthy photos often get an "early" flag.** 10 of the 31 held-out healthy photos (32%) show 2–5 tiny false spots, about 1% of the leaf. That's enough to tip the draft rule into "early". The false *area* is small (0.33% on average), but for early warning this is a real false-alarm problem.
   - I did **not** tune the threshold on the held-out set, because that would make the test meaningless.
   - The natural fix is Phase 4: the classifier, which calls these photos healthy at 97–100% confidence, can overrule tiny spots.
2. **The expert check is weak: Dice 0.31.**
   - Precision is 73%: what it marks is mostly real.
   - Recall is 27%: it misses most of the expert-marked lesion area.
   - These photos show Alternaria and soft rot, which are **not** in our training labels, and experts outline whole lesion zones, while our colour-based labels mark only clearly discoloured pixels.
   - So the model measures our 3 diseases the way our labels define them. It will **under-count** diseases it hasn't seen.
3. **A bug was fixed after looking at held-out results.** Leaves covered entirely by lesion measured 0%, because there was no healthy leaf outline to measure against. Fixing it brought the error down from 7.5 to 2.7 points. It is a logic fix, not a tuned number, and there is now a test for it.
4. **Labels are machine-drawn and AI-reviewed.** Measured % is only as good as that review. Grade 2 vs 3 separation and any bacterial spot rot photos where rot looks like shadow are the weakest.
5. **Stage thresholds are drafts.** They live in `src/stages.json` with `expert_verified: false`:

   | Stage | Rule |
   |---|---|
   | early | under 5%, or 1–3 lesions and under 10% |
   | moderate | 5–25% |
   | severe | 25–50% |
   | critical | over 50% |

6. **Small, same-season data:** 299 training photos. No Raspberry Pi photos yet.

## Files

| What | Where |
|---|---|
| Pre-labelling (SAM2 + colour rules) | `src/seg_prelabel.py` |
| Review picks / label set | `outputs/seg/review_picks.txt`, `outputs/seg/label_set.csv` |
| YOLO dataset builder | `src/seg_dataset.py` → `data_external/seg_yolo/` (git-ignored) |
| Training | `src/seg_train.py` → `experiments/seg_001_yolo11s/` |
| Runtime (standalone) | `src/seg_runtime.py`: `python src/seg_runtime.py photo.jpg --out overlay.png` |
| Stage thresholds | `src/stages.json` |
| Evaluation | `src/seg_eval.py` → `outputs/seg/seg_eval.json`, `val_predictions.csv/png`, `user_photo_1_overlay.png` |
| Tests | `tests/test_seg.py` (4 tests; whole suite 15 pass + 1 known limitation) |

Reproduce: `python src/seg_prelabel.py select && python src/seg_prelabel.py candidates && python src/seg_dataset.py && python src/seg_train.py && python src/seg_eval.py`. Training takes ~18 min on an RTX 4060 laptop GPU.
