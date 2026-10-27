# Model v2 report — diagnosis + severity + trustworthy confidence

**Deployed model:** `experiments/experiment_004_v2_multitask_cv/` — ensemble of 5 EfficientNet-B0 fold models (`fold0-4.pt`, fp16, 8.9 MB each) + `calibration.json`.
Same dataset as v1: no photos added, removed or relabelled. `splits/test.csv` was never used for training, calibration or any choice; it was evaluated once at the end.

## What each output means

| Field | Meaning |
|---|---|
| `prediction`, `confidence` | Diagnosis and its **temperature-calibrated** probability |
| `prediction_set_95` | All diagnoses plausible at the 95% level (split conformal) |
| `severity.grade` / `stage` | 0 none · 1 early (<10% tissue) · 2 moderate (10–25%) · 3 severe (25–50%) · 4 critical (>50% / collapse) |
| `severity.grade_range_90` | Grade interval that contains the true grade 90% of the time (conformal) |
| `severity.severity_score_0_100` | Expected grade ÷ 4 × 100 (a continuous score, not % area) |
| `disease_presence`, `other_findings` | Independent per-disease probabilities; a second disease ≥ 0.5 is listed |
| `status` | `confident` · `uncertain` · `out_of_scope` · `retake_photo` — act on this first |

## Fixes for each v1 limitation

| v1 limitation | v2 fix |
|---|---|
| Always answers one of 4 classes, even for junk input | Image-quality gate (blur/exposure/size) → `retake_photo`; feature-space Mahalanobis OOD detector (99th-pct threshold) → `out_of_scope` |
| Single label only | Presence head (3 sigmoids) trained with CutMix composites; `other_findings` + conformal prediction set |
| Underconfident softmax (label smoothing) | No label smoothing; temperature fitted on out-of-fold data; 5-model ensemble + flip TTA |
| 224 px + centre crop loses small / edge lesions | 384 px, letterboxed full frame |
| Only last block fine-tuned | Full backbone, layer-wise LR decay (0.8/stage), AdamW, warmup + cosine |
| No class/source balancing | Sampler balances class × source (40% real photos per epoch vs 14% before) |
| No severity at all | Ordinal CORAL head (grades 0–4) trained on 618 graded originals |

## Severity labels (approach A → B)

1. **A – colour-rule pre-fill** (`src/severity_prefill.py`): lesion-pixel fraction inside the plant/curd mask → grade. Agreement with the reviewed grades: 27% exact, 81% within 1 → good for pre-filling, **not** usable as the severity model.
2. **Review:** all 412 diseased originals were graded from contact sheets (photo + A overlay); 206 healthy photos = grade 0. Labels: `outputs/severity/severity_grades.csv`, `grade_source = claude_visual_review`. **These grades come from an AI review, not an agronomist.** `src/grading_app.py` lets an expert confirm or overwrite them (saved as `grade_source = human`); then retrain.
3. **B – ordinal head** learns from these grades (augmented diseased photos only supervise "grade ≥ 1").

Grade distribution (originals): BSR 27/87/46/13 · black rot 2/45/15/0 · downy mildew 11/48/62/56 (grades 1/2/3/4). The model has never seen a grade-4 black rot.

## Results

### Out-of-fold (525 train+val originals; what calibration was fitted on)

Diagnosis accuracy 0.998 · ECE 0.0031 → 0.0006 after temperature (T = 0.78) · severity (diseased) exact 0.717, within ±1 1.000, MAE 0.28 · 90% grade-range coverage 0.906 · OOD false-reject 1.1%.

### Held-out test, original photos (n = 93), evaluated once

| | v1 (Exp 3) | **v2** |
|---|---:|---:|
| Accuracy | 0.989 (92/93) | **1.000 (93/93)** |
| Macro-F1 | 0.991 | **1.000** |
| Mean confidence | 0.908 | 0.999 |
| ECE (lower is better) | 0.082 | **0.001** |
| Severity | — | exact **0.774**, within ±1 **1.000**, MAE 0.23, QWK **0.855** |
| 90% grade-range coverage | — | 0.903 (mean width 1.6 grades) |
| Status | — | 92 confident, 1 out_of_scope |

The one flagged photo is `Bacterial Spot. (166)`, v1's only error (advanced rot with mildew-like fuzz). v2 now gets it right (conf 0.97, grade 3 vs reviewed 4), but its OOD score (367) is above the threshold (298), so it's returned as `out_of_scope` instead of a confident answer.

Severity confusion on the test set (rows = reviewed grade 1–4, cols = predicted): `[[3,2,0,0],[0,19,2,0],[0,4,15,2],[0,0,4,11]]` — every error is one grade.

### Robustness probes (built from test photos; synthetic)

| Probe | n | Flagged (not `confident`) |
|---|---:|---:|
| Gaussian noise | 20 | 100% |
| Flat colour | 20 | 100% |
| Heavily blurred photos | 30 | 100% |
| Very dark photos | 30 | 100% (`retake_photo`) |
| Non-brassica dried leaf (`pic.jpg`) | 1 | flagged (`retake_photo`: overexposed + not brassica-like) |
| Tile-scrambled photos | 30 | 87% |
| Colour-inverted photos | 30 | **60%** — 40% still pass as confident |
| Two-disease side-by-side composites | 30 | 100% (all `out_of_scope`); presence head found **both** diseases in 30/30 |

## Limitations that remain

1. **Severity labels are AI-reviewed, not expert-graded.** Severity metrics measure agreement with those grades. Run `grading_app.py` with an agronomist and retrain before trusting grades for decisions.
2. **Test n = 93 (9 black rot).** 93/93 has a 95% lower bound of about 96%; for black rot 9/9 it's about 70%.
3. **Same collection for train and test;** no Raspberry Pi / other-farm photos. The OOD detector only knows what "in-distribution" looked like here, so a real-camera domain shift may be flagged `out_of_scope` often. That's the safe failure, but re-fit `calibration.json` on real camera photos.
4. **Framing shortcut** (heads vs leaves) can't be fixed without new photos.
5. **OOD is not perfect:** colour-inverted images pass 40% of the time. Real co-infected plants are untested; only synthetic composites were.
6. **Conformal sets are almost always singletons,** because out-of-fold accuracy is 99.8%. Guarantees assume the deployment data looks like this dataset.
7. **Deployment:** 5 models × 2 views ≈ 0.9 s per photo on one local CPU thread; the process uses about 680 MB of RAM locally (CUDA build of PyTorch). Render's 512 MB starter plan may be too small, so test it or use a larger plan.

## Reproduce

```bash
pip install -r requirements-dev.txt
python src/v2_cache.py            # resized image cache (gitignored)
python src/severity_prefill.py    # approach A + review sheets
python src/train_v2.py            # 5 folds, ~45 min on an RTX 4060 laptop
python src/calibrate_v2.py        # temperature / conformal / OOD / quality thresholds
python src/eval_v2_test.py        # one-time held-out evaluation
```
