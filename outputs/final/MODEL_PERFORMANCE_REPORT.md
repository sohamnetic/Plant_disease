# Model performance report

Cauliflower / brassica leaf-disease classifier (4 classes).  
**Selected model:** Experiment 3 — EfficientNet-B0, combined original+augmented train, original-only validation.

This document consolidates the dataset audit, Experiments 1–3 (validation only during selection), and the one-time held-out test evaluation of the frozen Experiment 3 checkpoint. **No numbers here come from a new training run.** Splits and checkpoints were not modified.

---

## 1. Dataset summary (from the original audit)

Counts are from disk (`outputs/eda/class_counts_table.csv`, `quality_summary.json`), not assumed.

| Class | Original | Augmented | Total | % of dataset |
|---|---:|---:|---:|---:|
| bacterial_spot_rot | 173 | 935 | 1108 | 25.9% |
| black_rot | 62 | 766 | 828 | 19.3% |
| downy_mildew | 177 | 921 | 1098 | 25.6% |
| no_disease | 206 | 1041 | 1247 | 29.1% |
| **Total** | **618** | **3663** | **4281** | 100% |

Layout: nested publisher folders  
`dataset/Original Dataset/Original Dataset/{class}/` and  
`dataset/Augmented Dataset/Augmented Dataset/{class}/`.

**Frozen split** (`splits/`, seed 42, group-atomic; do not regenerate):

| Split | All images | Original | Augmented |
|---|---:|---:|---:|
| train | 2997 | 432 | 2565 |
| val | 642 | 93 | 549 |
| test | 642 | 93 | 549 |

Original-only per class (train / val / test): BSR 121/26/26, black_rot **44/9/9**, downy_mildew 123/27/27, no_disease 144/31/31.

Quality screen: 0 corrupt / blank / grayscale / extreme-brightness files; 11 exact-MD5 duplicate groups (24 original images, different filenames); 172 near-duplicate groups (422 images); 41 augmented files flagged as saturation/contrast artifact suspects.

### Class-shortcut risk (flagged, not “fixed”)

Field photos, not isolated lab leaves:

- `bacterial_spot_rot` and `no_disease` are dominated by **curd (head) + wrapper leaves**.
- `black_rot` and `downy_mildew` are dominated by **single-leaf close-ups on soil**.

A model can learn “curd vs leaf shot” as a proxy between those pairs. That remains a deployment risk for a Pi camera that may frame either way. Cracked red-brown soil is a possible **background shortcut** for the leaf classes. Some frames carry photographer/numeric watermarks.

Inside `no_disease`, most originals are healthy heads; a minority are **healthy leaf-on-soil** shots (distribution shift within the negative class).

### Label-conflict / pairing risks (flagged, nothing relabeled)

- Mixed-class hash groups (see `outputs/review/potential_mislabels.csv`):  
  - `g003594`: same head labeled `bacterial_spot_rot` and `no_disease` (original–original). **Strong label conflict.**  
  - `g002984`: `no_disease` (augmented) vs `bacterial_spot_rot` (original). **Strong label conflict.**  
  - `g001701`: downy-mildew leaf vs no-disease curd — likely a **hash false positive**.
- `Downy Mildew. (105)` flagged as possibly shot-hole/pest damage rather than classic downy mildew — **not changed**.
- Filename numbers **do not** join original↔augmented (`Bacterial Spot. (1).jpg` ≠ `Bacterial spot rot._ (1).png`). Only 25 hash groups contain both an original and an augmented file. Residual leakage is possible but not proven; the split still keeps detected groups together.
- Augmented images commonly show rotation **black triangles**, edge smear, and heavy color jitter — **not** extra independent real photos. Experiment 2’s validation score is therefore treated as same-domain / inflated.

---

## 2. Shared training methodology (all three experiments)

- Architecture: EfficientNet-B0, ImageNet-1K init, 4-class head, dropout 0.4.
- Loss: cross-entropy, `label_smoothing=0.1`, **no class weights**.
- Train augs: rotate 18° (ImageNet-mean fill) → RandomResizedCrop 224 (scale 0.8–1.0) → flips → ColorJitter 0.25. **Eval:** Resize 256, CenterCrop 224, ImageNet normalize.
- Two-stage: 8 epochs frozen backbone (head lr 1e-3); then unfreeze `features[7]`, backbone 1e-5 / head 1e-4, AdamW wd 1e-4, patience 8 on val macro-F1.
- Experiment 1 ran on CPU (batch 8). Experiments 2–3 ran on CUDA (RTX 4060 Laptop, batch 32). Same recipe otherwise.
- **Test set was not used for selection.** Training scripts refuse `splits/test.csv`.

---

## 3. Validation results — three experiments

**Comparability:** Experiment 1 and Experiment 3 are **directly comparable** (same original-only validation set, n=93). Experiment 2 is **not** comparable to 1 or 3: it trains and validates on augmented images only (n=549). Its score is flagged as **same-domain / inflated**.

| | Experiment 1 | Experiment 2 | Experiment 3 **(selected)** |
|---|---|---|---|
| Train | original-only (432) | augmented-only (2565) | combined (2997 = 432 orig + 2565 aug) |
| Val | **original-only n=93** | augmented-only n=549 | **original-only n=93** |
| Directly comparable to Exp 1? | — | **No** (different val domain) | **Yes** |
| Val accuracy | 0.968 (90/93) | 0.998 (548/549) | **1.000 (93/93)** |
| Val macro-F1 | **0.973** | **0.998** (inflated) | **1.000** |
| Val macro P / R | 0.973 / 0.974 | 0.998 / 0.998 | 1.000 / 1.000 |
| Val weighted F1 | 0.968 | 0.998 | 1.000 |
| Best epoch (ckpt) | 14 | 29 | 13 |
| Val errors | 3 | 1 | 0 |

See `outputs/final/exp_macro_f1_comparison.png`.

### Per-class validation (support in parentheses)

**Experiment 1 — original val (n=93)**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bacterial_spot_rot | 0.926 | 0.962 | 0.943 | 26 |
| black_rot | 1.000 | 1.000 | 1.000 | **9** |
| downy_mildew | 0.964 | 1.000 | 0.982 | 27 |
| no_disease | 1.000 | 0.935 | 0.967 | 31 |

Errors (3): `No disease. (16).jpg` and `(15).jpg` → bacterial_spot_rot (conf 0.70 / 0.63); `Bacterial Spot. (83).jpg` → downy_mildew (conf 0.45).

**Experiment 2 — augmented val (n=549) — do not treat as a real-camera score**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bacterial_spot_rot | 0.993 | 1.000 | 0.996 | 140 |
| black_rot | 1.000 | 1.000 | 1.000 | 115 |
| downy_mildew | 1.000 | 0.993 | 0.996 | 138 |
| no_disease | 1.000 | 1.000 | 1.000 | 156 |

Error (1): `DownyMildew. (185).png` → bacterial_spot_rot (conf 0.45).

**Experiment 3 — original val (n=93) — selected**

All four classes: precision/recall/F1 = 1.000. Combined training **helped** vs Experiment 1 on the same val set (+0.027 macro-F1); the three Exp 1 errors became correct (two of them remain low-confidence on val). Val macro-F1 1.000 on n=93 is a **small-set ceiling**, not proof of zero error in the wild.

Low-confidence val predictions (conf &lt; 0.60, all still correct):  
`No disease. (16).jpg` 0.470; `No disease. (2).jpeg` 0.477; `Bacterial Spot. (83).jpg` 0.523; `No disease. (7).jpeg` 0.559; `Bacterial Spot. (92).jpg` 0.600. Contact sheet: `outputs/review/exp003_low_confidence.png`.

---

## 4. Final held-out test (Experiment 3 checkpoint, original-only)

**Protocol:** load `experiments/experiment_003_effnetb0_combined/best_model.pt` with **no retraining**. Infer on `splits/test.csv` where `source_type == "original"` (n=93), matching how Experiment 3 was validated. This is the realistic real-camera **proxy inside this dataset**. After this run, metrics were not used to tune anything.

| Metric | Value |
|---|---|
| N | 93 (BSR 26, black_rot **9**, downy_mildew 27, no_disease 31) |
| Accuracy | **0.989** (92/93) |
| Macro-F1 | **0.991** |
| Macro precision / recall | **0.991 / 0.990** |
| Weighted F1 | **0.989** |
| Confidence min / max / mean | 0.501 / 0.993 / 0.908 |

### Per-class test metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bacterial_spot_rot | 1.000 | 0.962 | 0.980 | 26 |
| black_rot | 1.000 | 1.000 | 1.000 | **9** |
| downy_mildew | 0.964 | 1.000 | 0.982 | 27 |
| no_disease | 1.000 | 1.000 | 1.000 | 31 |

### Confusion matrix (rows = true, columns = predicted)

Order: bacterial_spot_rot, black_rot, downy_mildew, no_disease.

```
[[25,  0,  1,  0],
 [ 0,  9,  0,  0],
 [ 0,  0, 27,  0],
 [ 0,  0,  0, 31]]
```

Figure: `outputs/final/exp003_test_confusion_matrix.png`  
(also `experiments/experiment_003_effnetb0_combined/test_confusion_matrix.png`).

**The one error:** `Bacterial Spot. (166).jpg` — true `bacterial_spot_rot` → predicted `downy_mildew`, confidence **0.501** (lowest confidence on the whole original test set; not an overconfident miss). Visual: advanced curd rot with dark necrosis **and** white fuzzy growth, plus necrotic wrapper-leaf margins. True-class Grad-CAM stays on the rotten curd; predicted-class CAM pulls toward **leaf-margin necrosis**. Mixed symptoms / disease-identity confusion, not a leaf-vs-head framing shortcut.

Informational only (not a selection metric): same frozen checkpoint on **augmented-only** test (n=549) scored accuracy 1.000 / macro-F1 1.000. Same-domain synthetic images; do not optimize against this.

---

## 5. Known limitations

1. **black_rot small-n.** Original support is 62 images total, **9 val and 9 test**. Perfect precision/recall/F1 on those 9+9 images is a **small-n ceiling**, not a robust guarantee that black rot will be recognized under new lighting, cultivars, or framing.

2. **Single test error.** `Bacterial Spot. (166).jpg`: BSR → downy_mildew at conf 0.50. Advanced mixed rot vs mildew-like fuzz and necrotic leaf edges. Expect similar failures when two diseases co-occur or look alike on a collapsed curd.

3. **Grad-CAM soft spot on `no_disease` leaf-only shots.** Healthy **head** photos usually attend to curd texture. Healthy **leaf-on-soil** photos (e.g. val `No disease. (7).jpeg`, `(2).jpeg`; test `No disease. (4).jpeg`) often have **lower confidence** and heat on **leaf–soil edges / soil**, not the blade interior. Residual background/framing cue for the negative class.

4. **Dataset-internal evaluation only.** Train, val, and test are partitions of the **same published collection**. Real Raspberry Pi camera performance (optics, exposure, distance, farm conditions) has **not** been tested and is the **next required step**.

Additional caveats that still apply: two strong original/augmented label-conflict groups were never relabeled; Experiment 2 must not be cited as field accuracy; val/test original n=93 can hit 1.000 by chance on an easy draw; label smoothing makes raw softmax “confidence” slightly conservative.

---

## 6. Grad-CAM findings (across experiments)

**What confirmed symptom learning**

- On **original** images, disease-class CAMs typically sit on lesions: brown/purple curd rot for bacterial_spot_rot; necrotic holes/margin damage for black_rot; chlorosis and angular spots for downy_mildew. Soil and wrapper-leaf silhouettes are usually cool.
- Experiment 1 errors were **not** a clean “head vs leaf” shortcut: the two no_disease→BSR misses attended to plausible curd edges/tint; the BSR→downy miss attended to a real mushy rot mass and confused **identity**.
- Experiment 3 on the same original val set kept lesion-focused maps and corrected those three cases (still low-conf on two of them).
- Experiment 3 **test** CAMs (7 correct per class + the one error) remain consistent with that pattern on originals.

**Residual risks**

- `no_disease` leaf-only shots: edge/soil attention and low confidence (see §5.3).
- Experiment 2: most augmented CAMs still on tissue, but `DownyMildew. (42).png` showed **border-smear artifact attention** — evidence that combined training *could* have used aug artifacts, even though original-only val/test Grad-CAM did not show that pattern dominating.
- The test miss: predicted-class CAM on necrotic **leaf margins** while the diagnostic rot is on the **curd**.
- Shortcut risk from the audit (curd vs leaf class pairing, soil background, watermarks) is reduced by these maps but **not eliminated**, especially off this dataset.

Overlays: `outputs/review/gradcam/` (Exp 1 val), `gradcam_exp002/`, `gradcam_exp003/` (Exp 3 val), `gradcam_exp003_test/` (Exp 3 original test).

---

## 7. Why Experiment 3 was selected (and then frozen)

On the **same original-only val set**, combined training beat original-only training (macro-F1 1.000 vs 0.973) without replacing Exp 1’s lesion-focused CAMs with artifact maps. Experiment 2’s 0.998 was discarded as a decision metric because val matched the augmented domain. After selection, the Experiment 3 checkpoint was evaluated **once** on original-only test. That test number is **not** a license to retrain.

Inference entry point: `src/predict.py` (default checkpoint = Experiment 3).

Training curves for the selected run: `outputs/final/exp003_training_curves.png`.

---

## 8. Final model artifacts (paths)

All paths relative to the project root.

### Selected checkpoint (Experiment 3)

| Artifact | Path |
|---|---|
| Checkpoint | `experiments/experiment_003_effnetb0_combined/best_model.pt` |
| Config | `experiments/experiment_003_effnetb0_combined/config.json` |
| Training history | `experiments/experiment_003_effnetb0_combined/training_history.csv` |
| Val metrics | `experiments/experiment_003_effnetb0_combined/validation_metrics.json` |
| Val predictions | `experiments/experiment_003_effnetb0_combined/validation_predictions.csv` |
| Val low-confidence list | `experiments/experiment_003_effnetb0_combined/val_low_confidence.csv` |
| **Test metrics (primary)** | `experiments/experiment_003_effnetb0_combined/test_metrics.json` |
| **Test predictions** | `experiments/experiment_003_effnetb0_combined/test_predictions.csv` |
| Test confusion matrix (exp folder) | `experiments/experiment_003_effnetb0_combined/test_confusion_matrix.png` |
| Test error / low-conf tables | `experiments/experiment_003_effnetb0_combined/test_highest_confidence_wrong.csv`, `test_lowest_confidence.csv` |
| Augmented test (informational) | `test_metrics_augmented_informational.json`, `test_predictions_augmented_informational.csv` |

### Experiment 1 (original-only)

`experiments/experiment_001_effnetb0_original/` — `best_model.pt`, `config.json`, `training_history.csv`, `validation_metrics.json`, `validation_predictions.csv`, `highest_confidence_wrong.csv`, `lowest_confidence.csv`, `confusion_pairs.csv`.

### Experiment 2 (augmented-only)

`experiments/experiment_002_effnetb0_augmented/` — same filename set as Exp 1 (`best_model.pt`, `config.json`, `training_history.csv`, `validation_metrics.json`, `validation_predictions.csv`, error CSVs).

### Splits (frozen)

`splits/train.csv`, `splits/val.csv`, `splits/test.csv`, `splits/SPLIT_MANIFEST.json`, `splits/README.md`.

### Audit / review

`outputs/eda/class_counts_table.csv`, `quality_summary.json`, `image_inventory.csv`, `duplicate_groups.csv`; `outputs/review/MANUAL_REVIEW_NOTES.md`, `potential_mislabels.csv`; Grad-CAM folders listed in §6.

### This consolidation

`outputs/final/MODEL_PERFORMANCE_REPORT.md`, `outputs/final/exp_macro_f1_comparison.png`, `outputs/final/exp003_test_confusion_matrix.png`, `outputs/final/exp003_training_curves.png`.  
Inference: `src/predict.py`.
