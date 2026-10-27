# Step I-1: insect-hole outlines (drafted and reviewed)

Date: 2026-10-26. **Nothing was trained, and the app and model were not changed.** This step only produces labels for step I-2: training the outline model to find insect holes and measure % eaten. Plan: [INSECT_HOLE_REPORT.md](INSECT_HOLE_REPORT.md), Section 4.

## Summary

- **168 photos have hole outlines good enough to train on.** On those, a median of 2.2% of the leaf is eaten (range 0.2–21%).
- **378 photos show insect damage that the drafts don't capture fully.** These go into the "insect damage" category, but they can't be used yet to teach the % measurement.
- **91 photos (14%) show no insect damage I can see.** They are labelled "Insect Hole" in the dataset and should be **dropped** from the category, otherwise the model learns to call clean leaves "insect damage".

| Review result | Training split | Insect test split | Total |
|---|---|---|---|
| **K**: holes outlined well enough to train on | 106 | 62 | **168** |
| **D**: insect damage visible, draft misses or mis-marks some of it | 250 | 128 | **378** |
| **N**: no insect damage visible | 56 | 35 | **91** |
| Total | 412 | 225 | 637 |

## How the drafts were made (`src/insect_prelabel.py`)

The photos are leaves cut out onto white, so a hole shows what was under the leaf: the white backdrop or a pale grey surface. A pixel is drafted as **hole** if it lies inside the leaf outline and meets all three of these:

- it is white, or a **neutral** grey with no green or yellow tint
- it is pale
- it is **much brighter than this leaf's own tissue**

Holes that open onto the leaf edge through a narrow mouth are included, but only if leaf tissue rings at least 75% of them. Photo-frame cut-offs are ignored.

**Draft % eaten = hole area ÷ (remaining leaf + hole area).**

**Control check:** the same rule on 120 healthy and 120 black-rot leaves from the same source (which should read about 0%):

| | Median | 90th percentile | Max |
|---|---|---|---|
| Healthy | 0.0% | 0.17% | 2.1% |
| Black Rot | 0.0% | 0.25% | 6.0% |

Most of the higher control values are **real holes**: some leaves in the Healthy and Black Rot folders do have insect holes (`outputs/insect_seg/review_sheets/control_worst_*.jpg`). The earlier colour-only rule read up to 10% holes on healthy leaves, because it caught midribs, waxy patches and glare. Requiring a neutral grey removed those.

## Review

I checked every one of the 637 photos visually on side-by-side sheets (original and draft): 17 sheets of photos with a drafted hole and 8 sheets of photos without one. The decisions are in `outputs/insect_seg/review_pass1.txt`, and `src/insect_review.py` applies them to produce `review.csv` and `REVIEW_SUMMARY.txt`.

**K** means the main holes are outlined and nothing large is wrongly marked. Some tiny holes may be missed, so the % on K photos **slightly under-counts**. Examples: `review_sheets/sample_K_00.jpg`. These labels were drawn by machine and checked by AI, not by a person, the same standard as Phase 3.

**Why so many D (378):**
1. **Tinted grey holes are missed.** In many photos the surface seen through a hole has a yellow-green tint, the same colour as midribs and waxy leaf patches. I tested looser thresholds: they didn't catch these holes, and they made the controls worse. **No colour rule can separate them.**
2. **Edge bites** (154 of the 234 photos with no draft): chewed notches in the margin can't be told from the natural leaf outline by a fixed rule.
3. **"Windowpane" feeding:** insects scraped the surface without eating through, leaving pale translucent patches. That's insect damage, but not a hole.
4. **Curled or folded leaves:** gaps between overlapping parts get marked as holes.

**N (91):** photos where I couldn't see a hole, bite or scraped patch at review size. Most are intact leaves. A few have brown spots that look more like disease than insect feeding.

## How the current model answers, by review result

| Current app answer | K | D | N |
|---|---|---|---|
| Confident black rot | 31 (18%) | 77 (20%) | 8 (9%) |
| Confident no disease | 43 | 130 | 45 (49%) |
| Uncertain | 75 | 151 | 33 |
| No reliable diagnosis | 19 | 20 | 5 |

Photos with visible damage get a confident black-rot answer about twice as often (18–20%) as photos with none (9%). This matches the earlier finding that holes push the model towards black rot. The N photos mostly get "no disease", which suggests their dataset label really is wrong.

## What this means for I-2 (outline model)

- **168 K photos** (106 train / 62 test) plus healthy leaves as 0% examples are enough for a **first round** of the outline model.
- For the 378 D photos, I recommend bootstrapping: train round 1 on K, let it draft holes on the D photos, review those, add the good ones, and retrain. A trained model sees shape and texture, not just colour, so it should catch the tinted grey holes the rule missed. Each round would get its own report.
- **Edge bites** stay approximate. The original leaf edge is gone, so the first version measures holes only and says so.
- The **"insect damage" category** for the classifier (I-3) would use **K + D = 546 photos** and drop the 91 N photos.

## Files

| What | Where |
|---|---|
| Draft rule and sheets | `src/insect_prelabel.py` (`drafts`, `controls`) |
| Apply review | `src/insect_review.py` |
| Draft masks | `cache/insect_prelabel/*.npz` (one per photo: `leaf`, `hole`) |
| Per-photo drafts + review | `outputs/insect_seg/drafts.csv`, `outputs/insect_seg/review.csv` |
| Decisions | `outputs/insect_seg/review_pass1.txt` |
| Summary tables | `outputs/insect_seg/REVIEW_SUMMARY.txt` |
| Review sheets | `outputs/insect_seg/review_sheets/` (`holes_*`, `nohole_*`, `control_worst_*`, `sample_K_00`, `sample_D_00`) |
