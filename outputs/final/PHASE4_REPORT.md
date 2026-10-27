# Phase 4 report — combining diagnosis with per-part measurement

**Status:** complete. All 3 plan targets pass. Phases 5–7 have not been started.
**What changed for the user:** the app (API, web page, CLI) now returns one combined answer. It gives the disease and a calibrated confidence as before, plus, for every leaf and head it finds, the % of tissue affected, the spot count and a verdict. The plant's stage comes from its worst part, and an overlay image highlights the lesions.

## The main finding: parts should measure, not diagnose

The plan was for each leaf and head to be cropped and diagnosed separately, with the segmentation model allowed to overrule the classifier. The first build did exactly that, and it was **worse** on the outside test:

| (outside test, 1,142 photos) | Classifier alone (v3) | First design: parts decide |
|---|---:|---:|
| Correct | 92.7% | 90.0% |
| Wrong answer given | 5.3% | 9.1% |
| Photos fixed / broken vs v3 | — | 18 fixed / **51 broken** |

**Why:** 47 of the 51 broken photos were diseased plants where the segmentation model found **no lesions**. These were mostly faint bacterial spot on field heads and early black rot on leaves. The rule "no lesions → healthy" then overruled a correct classifier. A second rule ("measured stage and estimated grade disagree → uncertain") marked 97 correct answers as uncertain and caught nothing.

**Avoiding tuning on the test set:** after seeing those results I did not adjust rules and re-score on the same photos. Instead I compared rule sets on a **separate validation set** of 1,962 real photos that neither model had seen:
- training-pool photos not used for segmentation labels;
- each classified by the one fold model that held it out.

| Validation (1,962 photos) | Correct | Wrong | Uncertain | Healthy called diseased |
|---|---:|---:|---:|---:|
| **Classifier decides, parts measure (chosen)** | **93.7%** | **4.1%** | 6.7% | 2.6% |
| Parts decide, lesion-free part is vetoed healthy | 91.7% | 6.9% | 11.1% | 3.0% |
| Parts decide, no veto | 93.5% | 5.4% | 9.2% | 4.7% |
| … each of the above + grade check | same | same | +8 pts | same |

**Adopted design:** the whole-photo classifier decides the disease, confidence and status. The segmentation model measures each part and adds per-part notes. It can no longer turn a correct diagnosis into a wrong one. (The "classifier decides" candidate was added after the first validation run showed every "parts decide" option losing to the classifier alone. The choice is recorded in `src/phase4_policy.json`.)

## Final results (outside and in-dataset tests, run once with the chosen design)

| Target (plan) | Result | |
|---|---|:-:|
| Your photo: head → bacterial spot rot, leaves not flagged | head **bacterial spot rot, 70% affected, critical**; leaves **healthy** (2 tiny unidentified spots, 0.6%) | ✅ |
| Outside test not worse than v3 | **92.7% correct, 5.3% wrong**: identical to v3 | ✅ |
| Wrong answers not higher | 5.3% (same) | ✅ |

Also measured:
- **Healthy photos given a disease stage:** 3.1% (612 healthy test photos). Segmentation alone flagged 32% of healthy photos in Phase 3.
- **Early cases:** every grade-1 (early) diseased test photo was measured as diseased.
- **Measured stage vs the reviewed severity grade:** within one stage 79%, exact 44% (61 correctly diagnosed diseased VegNet test photos).
- **Speed:** about 1.0 s per photo on a laptop CPU, versus about 0.9 s for the classifier alone.

## Example (your field photo)

> Bacterial spot rot, critical stage: 70% of the worst part affected (head 70%; 3 spots). Small unidentified spots on: leaf (keep an eye on them).

| Part | Result | Affected | Spots |
|---|---|---:|---:|
| head | bacterial spot rot | 70.4% | 3 |
| leaf | healthy (small unidentified spots) | 0.6% | 2 |

## Bugs found and fixed along the way

1. **The head was counted twice.** When the leaves formed a ring around the head, filling holes in the leaf outline swallowed the head and its rot, so the leaves showed "10% affected". Each lesion now belongs to one part, and a leaf excludes the parts inside it. This also improved Phase 3's own score: the affected-% error went from 2.7 to **2.0 points**, and all Phase 3 targets still pass.
2. **The leaves inherited the head's diagnosis.** A part covering most of the photo reused the whole-photo answer. It is now classified with the other diagnosed parts blanked out.

## Limitations

- **Diagnosis accuracy is unchanged.** Phase 4 adds measurement, not better diagnosis. Weak spots from v3 remain: black rot on some sources, and leaves of other plants (`pic.jpg` is still "black rot, 100% affected").
- **"Small unidentified spots"** appears on 20% of healthy photos. It never changes the diagnosis, but it is a noisy note.
- **Faint lesions can't be measured.** When the classifier names a disease but no spots are found, the stage falls back to the classifier's grade and the result shows "spots too small or faint to measure".
- **Stage thresholds and segmentation labels are drafts** (not expert-verified), as reported in Phase 3.

## Files

| What | Where |
|---|---|
| Combined runtime (analyse + decide) | `src/plant_runtime.py` |
| Chosen policy | `src/phase4_policy.json` |
| Policy selection on validation | `src/select_phase4_policy.py` → `outputs/final/phase4_policy_selection.json` |
| Final evaluation | `src/eval_phase4.py` → `outputs/final/phase4_eval.json`, `phase4_predictions.csv` |
| Measurement fix | `src/seg_runtime.py` (`measure`) |
| API / web page / CLI | `src/api.py`, `src/predict.py` |
| Tests | `tests/test_plant.py` (5 new; suite: 20 pass + 1 known limitation) |
