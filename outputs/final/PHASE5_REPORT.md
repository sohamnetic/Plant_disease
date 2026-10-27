# Phase 5 report — measured severity grades

**Status:** complete. The plan target passes. Phases 6–7 have not been started.
**Goal:** replace the severity grades I had assigned by eye with grades **measured** from the lesion outlines, then retrain the model's severity output on them.

## What was done

1. **Measured all 618 original dataset photos:**
   - the reviewed Phase 3 outlines where they exist (164 photos), otherwise the YOLO11-seg model;
   - % affected = worst leaf or head, turned into a grade on the same scale the app shows:

     | Grade | 1 early | 2 moderate | 3 severe | 4 critical |
     |---|---|---|---|---|
     | % affected | under 5%, or a few small spots | 5–25% | 25–50% | over 50% |

2. **Compared the measured grades with the by-eye grades** on the 396 diseased photos that could be measured. They agree exactly 41% of the time and within one grade 90% of the time. I reviewed all 41 photos that differed by two or more grades.
3. **Found that the measurement is only trustworthy for leaf diseases:**

   | Median measured % by by-eye grade | 1 | 2 | 3 | 4 |
   |---|---:|---:|---:|---:|
   | Downy mildew | 1.4 | 8.6 | 26.6 | 67.2 |
   | Black rot | 0.6 | 2.2 | 7.1 | — |
   | Bacterial spot rot | 2.4 | 5.2 | 7.4 | 8.0 |

   - **Downy mildew:** tracks severity well.
   - **Black rot:** the outlines are right. The by-eye grades were **too high** (leaves I called moderate are really 2–7% affected).
   - **Bacterial spot rot:** the measurement stays flat around 5–8% from grade 2 to grade 4. The segmentation model treats dark purple-brown rotten head tissue as background, so it misses most of the rot.
4. **Final grades** (`outputs/severity/severity_grades.csv`, with the old grade kept as `grade_visual` and the source in `grade_source`):

   | Source | Photos |
   |---|---:|
   | Measured (black rot, downy mildew) | 219 |
   | Disputed and reviewed (33 kept by-eye, 8 set by hand) | 41 |
   | Bacterial spot rot, kept by-eye (measurement untrusted) | 136 |
   | Could not be measured (no lesions found), kept by-eye | 16 |
   | Healthy = 0 | 206 |

5. **Retrained only the severity output.**
   - For each of the 5 models, the severity layer was refit on the new grades using that model's frozen image features.
   - The backbone and disease output are untouched, so **diagnoses are exactly the same** (checked).
   - It took 3.5 minutes instead of a 65-minute retrain.
6. **Grading tool updated.** `src/grading_app.py` now shows each photo next to its measured overlay, the measured %, the earlier by-eye grade, and where the current grade came from.
7. **App updated.** The grade descriptions now match the measured scale, and the app uses the new model: `experiments/experiment_006_v3_measured_severity/`.

## Results

Scored against the new grades.

| | Old severity output | **Refit** |
|---|---:|---:|
| Test split: exact grade (62 correctly diagnosed diseased photos) | 53% | **73%** |
| Test split: within ±1 grade | 98% | **100%** |
| Test split, measured labels only: exact / within ±1 | 28% / 97% | **69% / 100%** |
| Out-of-fold training photos: exact / within ±1 | 57% / 98% | **65% / 98%** |

- **By disease** (test, exact): black rot 22% → **56%**, downy mildew 37% → **74%**, bacterial spot rot 81% → 77%. Bacterial spot rot barely changes because its labels mostly didn't change.
- **Plan target** (within ±1 grade of the measured grade, at least 95%): **100% on the test split, 98% out-of-fold** ✅.

## Limitations

- **Bacterial spot rot severity is still by-eye.** Fixing it needs the head's dark rot to be outlined as lesion in the segmentation labels. That's the manual outline-fixing task (#3 in the list of teammate tasks) or an expert.
- **No grade has been checked by an expert.** Measured grades inherit the Phase 3 outline quality, and the 41 disputes were decided by me.
- **The two severity numbers can disagree.** Example: a black rot training photo measures "early" (1%), but the model's grade says "moderate". The app's headline stage uses the measurement; the model's grade is a second opinion in `severity`.
- **The measured scale's "early" band is narrower than the old one** (under 5% instead of under 10%), which is better for early warning. Old reports used the old scale.

## Files

| What | Where |
|---|---|
| Measure + apply grades | `src/measure_severity.py` → `outputs/severity/measured_grades.csv`, `severity_grades.csv` (old copy: `severity_grades_visual_v1.csv`) |
| Dispute decisions | `outputs/severity/measured_review.txt` |
| Severity refit | `src/refit_severity.py` → `experiments/experiment_006_v3_measured_severity/` |
| Evaluation | `src/eval_phase5.py` → `outputs/final/phase5_eval.json` |
| Grading tool with overlays | `src/grading_app.py` |
