# Insect damage: 5th result category + % of the leaf eaten (steps I-2 to I-5)

Date: 2026-10-27. Follows `INSECT_HOLE_REPORT.md` (the data) and `INSECT_I1_REPORT.md` (the reviewed hole outlines).

## In short

- **The app now has an "Insect damage" result.**
  - It gives a stage and **"% of the worst leaf eaten"** with a hole count.
  - Eaten holes are drawn in orange on the photo, and it shows insect advice.
  - When holes are found next to a disease, the message says so.
- **The disease results are unchanged.**
  - The frozen outside test (1,142 photos), the web photos and your field photo give exactly the same answers as before.
  - Every "never worse" check passes (Section 5).
- **On held-out insect-eaten leaves:**
  - **77% are now called insect damage** (the old app: 0%).
  - Any disease answer fell from 30% to 18%.
  - The remaining 17% are *confident* black-rot answers from the 4-category classifier, which the new setup never overrides (Section 7).
  - The % eaten is within **0.6 points** of the reviewed outlines on average (median 0.13), on white and on field backgrounds.
- **How it is built:**
  - The new 5-category classifier is **not used alone**: it called 6.9% of black-rot leaves "insect damage" and failed the outside-test check.
  - It runs as a **second opinion** next to the unchanged 4-category classifier (Section 4).
  - Measuring the eaten % also needs a second outline model (Section 3).

## 1. What was built

| Part | Folder | Role in the app |
|---|---|---|
| 4-category classifier (v3, unchanged) | `experiments/experiment_006_v3_measured_severity` | decides every disease, as before |
| **5-category classifier (v4, new)** | `experiments/experiment_007_v4_insect` | second opinion: may only answer "insect damage" (Section 4) |
| Outline model seg_001 (unchanged) | `experiments/seg_001_yolo11s` | leaves, head, disease spots, % affected, stage |
| **Outline model seg_002 (new)** | `experiments/seg_002_insect` | insect holes and % eaten only |
| Outline model seg_003 (round 2) | `experiments/seg_003_insect_r2` | **not used** (Section 3) |

### Training data

**v4 classifier:**
- The v3 training pool plus **356 insect-hole photos** from the training side of the insect split, as class `insect_damage`. That gives 6,120 photos in total.
- `src/build_v4_data.py` builds it into `splits_v4/`.
- Insect rows carry no disease-severity label: their severity loss is masked.
- The frozen outside test and the 190-photo insect test were never trained on.
- Training ran 5 folds × 18 epochs, about 2 h 55 min.

**seg_002:**
- The Phase 3 outline data plus 168 reviewed insect training photos (step I-1).
- Each photo also appears twice on field backgrounds, with the holes see-through.
- Holes are class 3, `insect_hole`.

**seg_003 (round 2):**
- seg_002 drafted holes on the photos the colour rule had missed.
- I reviewed the drafts and kept 45 more training photos.
- seg_003 is seg_002 fine-tuned on that larger set.

## 2. The 5-category classifier on its own

Held-out folds (training-pool photos the fold never saw):

| | v3 (4 classes) | v4 (5 classes) |
|---|---|---|
| Accuracy, held-out real photos (mean of 5 folds) | 95.6% | 91.5% |
| Answered "confident" | 91.6% | 83.8% |
| Accuracy when confident | 97.9% | 97.0% |
| Insect leaves called insect damage with high confidence | – | 86.5% |

On the insect test (190 held-out insect-hole photos: 62 with reviewed outlines "K", 128 harder "D") and the 91 "no visible damage" photos (`src/eval_insect.py`):

| | v3 | v4 alone |
|---|---|---|
| Insect test: answer "insect damage" | 0% | **96.3%** |
| Insect test: confident disease (black rot) | 17.4% | **0%** |
| "No visible damage" photos called insect damage | 0% | 45% |

Frozen outside test, 1,142 photos (`src/eval_external.py`, gate: not worse than v3):

| | v3 | v4 alone | Gate |
|---|---|---|---|
| Correct | 92.7% | 89.5% | ✗ |
| Wrong answers | 5.3% | 7.7% | ✗ |
| Healthy false alarms | 3.3% | 3.4% | |
| cleaf black rot correct | 91.2% | 82.7% | |
| Black rot called insect damage | – | 6.9% (cleaf), 8.7% (c2h) | |

**v4 alone fails the gate.** Dead, torn black-rot tissue looks like feeding damage to it.

The "no visible damage" result means something too. Almost half of the photos the dataset files as "Insect Hole", but where we saw no hole, are called insect damage. The model has partly learnt the photo set rather than the damage.

Neither result is acceptable for a deployed answer. That is why v4 is used only as a second opinion.

## 3. Measuring the % eaten (outline models)

Held-out insect test (`src/insect_seg_eval.py`). "Found" means holes were found on the leaf. The error is in % points of the leaf, against the reviewed outlines.

| | seg_002 (round 1) | seg_003 (round 2) | seg_001 leaf + seg_002 holes | **App (seg_002 measures holes on its own leaf outline)** |
|---|---|---|---|---|
| Holes found, reviewed insect leaves (62) | 90.3% | 90.3% | 61.3% | **90.3%** |
| % eaten: mean error / median error | 0.59 / 0.13 | – | 2.50 / 1.14 | **0.59 / 0.13** |
| Spearman vs reviewed % | 0.94 | – | 0.69 | **0.94** |
| Same leaves on field backgrounds: found / mean error | 90.3% / 0.61 | – | 82.3% / 1.55 | **90.3% / 0.61** |
| Harder insect photos (128): found | 30.5% | 34.4% | 24.2% | 30.5% |
| "No visible damage" (91): found | 0% | 0% | 0% | 0% |
| Healthy, white background (325): false holes | 1.2% | 0.6% | 0.9% | 1.2% |
| Healthy, field photos (241): false holes | 0.8% | 0.8% | 2.1% | 0.8% |

Disease measurement, Phase 3 checks (`src/seg_eval.py`; outputs in `outputs/insect_seg/seg_00x_phase3_eval/`):

| | seg_001 | seg_002 | seg_003 |
|---|---|---|---|
| % affected error (gate ≤ 5) | 2.02 | 2.10 | 2.96 |
| Disease-spot outline overlap, dice (gate ≥ 0.60) | 0.750 | 0.749 | 0.666 |
| Early-disease recall (gate ≥ 0.80) | pass | 0.93 | **0.79 ✗** |
| Healthy false spot area (gate ≤ 1%) | 0.33% | 0.51% | 0.32% |
| **Your field photo: rotten head measured** | **> 25%** | **0% ✗** | 0% ✗ |

Decisions:

1. **seg_003 is not used.** Its holes are only slightly better, it fails the early-disease check, and its spot outlines are worse.
2. **seg_002 does not replace seg_001.** It passes the four Phase 3 checks, but it measures the rotten head on your field photo at 0%. The app's test suite caught this.
3. **Both models run, each doing only its own job.**
   - seg_001 measures disease exactly as before.
   - seg_002 measures holes on *its own* leaf outlines. seg_001's leaf outlines cut the holes out, so pairing them with seg_002's holes found holes on only 61% of insect leaves.
   - Each leaf/head found by seg_001 takes the eaten % of the seg_002 leaf it overlaps most.

What the % means: **hole area ÷ leaf area, with the holes counted as leaf.**
- Bites taken from the leaf edge are not counted, because the original edge is gone. The app says this in every insect message.
- The insect stage thresholds are drafts, not checked by an expert:

| Stage | Share of the worst leaf eaten |
|---|---|
| early | under 5% |
| moderate | 5–15% |
| severe | 15–30% |
| critical | 30% or more |

  Below 0.3% no stage is given.

## 4. How the app decides "insect damage" (the second opinion)

`PlantEngine._insect_second_opinion` in `src/plant_runtime.py`. The answer becomes insect damage only if **v4 is confident it is insect damage** AND one of these holds for v3:
- v3 answered "no disease", or
- v3 gave no reliable answer, or
- v3 named a disease but was unsure, and v4 gives insect damage at least 80% probability.

A confident v3 disease answer is never changed.

Scores of the rules tried (`src/insect_hybrid_eval.py`, per photo in `outputs/insect_seg/hybrid/`):

| Rule | Outside test: correct / wrong / healthy false alarm | Insect test K / D: called insect | K / D: called a disease | "No visible damage": insect |
|---|---|---|---|---|
| v3 alone (old app) | 92.7% / 5.3% / 3.3% | 0% / 0% | 34% / 29% | 0% |
| v4 alone | 89.5% / 7.7% / 3.4% ✗ | 100% / 95% | 0% / 1% | 45% |
| Planned rule: v4 may turn "no disease" or no answer into insect | 92.0% / 6.0% / 4.6% ✗ | 66% / 67% | 34% / 29% | 34% |
| **Adopted: as above, v4 must be confident; may replace an unsure v3 disease answer** | **92.7% / 5.3% / 3.3% ✓** | **81% / 75%** | **19% / 17%** | **20%** |

"Called a disease" counts any disease answer, confident or not. Under the adopted rule, every remaining disease answer on the insect test is a confident v3 black-rot answer: 17.4% of insect photos, the same as before.

**Be aware: the confidence requirement was chosen after looking at the frozen outside test.** The planned rule changed 13 outside-test answers, and every one came from an "unsure" v4 answer. So the outside-test score of the adopted rule is optimistic. The independent checks are:
- **Held-out training folds:** v4 is "sure insect" (≥ 80%, insect top answer) on 1.5% of non-insect photos. That is 3.5% of black rot, 0.7% of healthy and 0% of the other two diseases. In the app this only matters where v3 itself is unsure or says healthy.
- **Web photos** (81 photos never used anywhere): no answer changed and nothing was called insect damage (Section 5).

## 5. "Never worse" checks on the final app

| Check | Before | After | |
|---|---|---|---|
| Frozen outside test through the full app (`src/eval_phase4.py`) | 92.7% correct, 5.3% wrong, 2.5% confident-wrong | **92.7%, 5.3%, 2.5%: identical** | ✓ all 3 Phase 4 checks pass |
| Healthy false alarms (outside test) | 3.3% | 3.3% | ✓ |
| Your field photo (head: bacterial spot rot, leaves not flagged) | ✓ | ✓ (same message: bacterial spot rot, critical) | ✓ |
| Web photos, 81 (`src/web_probe.py`) | 44.4% correct, 22.2% confident-wrong | **identical**; no photo called insect damage; 1 healthy photo gets a false "holes also found" note (Section 7) | ✓ |
| Test suite (`pytest tests`) | – | **47 passed**, 1 expected failure (unchanged), including the new insect tests | ✓ |
| Time per photo, CPU (your field photo) | 3.75 s* | 5.30 s* (+41%) | |

\* Old and new setups measured back to back on this laptop, which was slower this morning than when the 1.03 s Phase 4 figure was taken. The ratio (+41%) is the meaningful number: one extra classifier ensemble on the whole photo plus one extra outline model.

Checked in the browser: an insect test leaf (IMG (51)) gave the following.
- **Result:** "Insect damage, early stage: 3.4% of the worst leaf eaten (43 holes; bites from the leaf edge are not counted)" at 98.9% confidence.
- **Photo:** the holes are drawn in orange.
- **Parts table:** the leaf shows "insect damage – eaten by insects; the disease check also suggests black rot".
- **Advice:** the insect advice is shown.

The large bite on that leaf opens onto the edge, so it is not counted.

## 6. What the app shows

| Case | Message |
|---|---|
| Insect damage diagnosed, holes measured | "Insect damage, early stage: 3.4% of the worst leaf eaten (43 holes; bites from the leaf edge are not counted)." |
| Insect damage diagnosed, no hole measurable | "Insect damage (feeding damage seen, but the eaten area could not be measured: bites from the leaf edge and scraped patches are not measured)." |
| A disease diagnosed, holes also found | "... Insect feeding holes also found: X% of the worst leaf eaten (N holes)." plus a second advice block, "Insect holes: what to do" |

- `/health` lists `insect_model` and `insect_hole_model`.
- The model fingerprint now covers both classifiers and both outline models.
- In the parts table, the column header "Affected" is renamed "Spot area". On an eaten leaf, seg_001 can mark the bare bite as "spots": 16% on IMG (51). That figure is the disease-spot measurement, not the eaten area, and the "Eaten" column sits next to it.

The insect advice draws on UC IPM guides (diamondback moth, cabbage looper):
- check undersides and the growing point
- hand-pick
- re-check in 3–4 days
- spare natural enemies

There are no chemical doses. It is a draft that no expert has checked.

## 7. Limits

1. **All insect training photos come from one source (cleaf):** single leaves cut out onto white. Field backgrounds were simulated, and no real field photo of insect damage has been tested. Collecting 20–30 real field photos of eaten cauliflower leaves would be the most useful next step.
2. **"No visible damage" photos are called insect damage 20% of the time.** These are photos the dataset files as insect-damaged but where we saw no hole.
3. **About 1 in 6 insect-eaten leaves (17.4%) still gets a confident black-rot answer**, the same rate as the old app. The second opinion never overrides a confident v3 answer: that is what keeps the disease results unchanged. Fixing this needs a classifier that tells holes from black rot, for example v4 retrained with more black-rot photos showing torn tissue.
4. **Edge bites and scraped "window" feeding are not measured.** The % is a lower bound for leaves eaten from the edge.
5. **Insect stage thresholds and advice are drafts.** No expert has checked them.
6. **False holes at the stalk notch.** On 1 of 31 healthy web photos, the dark background showing between the leaf base and the stalk was measured as a hole (0.6%). The diagnosis stayed "no disease", but the message adds "Insect feeding holes also found". The colour-rule drafts had the same weakness, which the I-1 review fixed for training labels; the model still makes it now and then.
7. **Two classifiers and two outline models run per photo.** CPU time and memory go up (Section 5).

## 8. Files

- **Code:**
  - `src/build_v4_data.py`, `src/train_v2.py` (recipe "v4")
  - `src/insect_seg_dataset.py`, `src/insect_bootstrap.py`, `src/seg_train.py`
  - `src/insect_seg_eval.py`, `src/eval_insect.py`, `src/insect_hybrid_eval.py`
  - `src/seg_runtime.py` (`measure_all`, hole model), `src/plant_runtime.py` (second opinion), `src/api.py`
  - `tests/test_insect.py`
- **Results:** in `outputs/insect_seg/`:
  - `external_eval_v4.json`, `eval_insect.json`
  - `hybrid/`
  - `seg_002_eval/`, `seg_003_eval/`, `seg_app_eval/`, `seg_combined_eval/`
  - `seg_00x_phase3_eval/`, `phase4_app_v4/`
  - web photos: `outputs/web_probe/results_v4_app.csv`
- **Nothing was trained on** `splits_v3/external_test.csv` or `splits_v4/insect_test.csv`.
