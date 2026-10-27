# Plant disease classifier

Cauliflower photos → **disease**, **calibrated confidence**, **measured % of tissue affected per leaf / head**, and **stage**.

Classes: `bacterial_spot_rot` · `black_rot` · `downy_mildew` · `no_disease`
Severity: 0 none · 1 early (<10% tissue) · 2 moderate (10–25%) · 3 severe (25–50%) · 4 critical (>50% / collapse)

**Deployed model (v3 + measured severity):** `experiments/experiment_006_v3_measured_severity/` (= `experiment_005_v3_multisource/` with the severity head refit on measured grades in Phase 5; diagnosis identical), an ensemble of 5 EfficientNet-B0 fold models with diagnosis, per-disease presence and ordinal severity heads, plus `calibration.json` (temperature, conformal, OOD and evidence thresholds fitted on out-of-fold data). Trained on VegNet plus 4 other cauliflower datasets, copy-checked against each other.
On a frozen **external** test of 1,142 cauliflower photos it never trained on: **92.7% correct, 5.3% wrong** (v2: 15.1% / 34.4%). In-dataset test: 98.9%.
Write-ups: [`MODEL_V3_REPORT.md`](outputs/final/MODEL_V3_REPORT.md) (current) · [`MODEL_V2_REPORT.md`](outputs/final/MODEL_V2_REPORT.md) · v1 [`MODEL_PERFORMANCE_REPORT.md`](outputs/final/MODEL_PERFORMANCE_REPORT.md).
Known gaps: black rot is the weakest class; leaves of other plants are not recognised as "not cauliflower".

**Affected-area measurement (Phase 3–4):** a YOLO11-seg model (`experiments/seg_001_yolo11s/`) outlines every leaf, head and lesion; the API reports % affected, spot count and a draft stage (early <5% · moderate 5–25% · severe 25–50% · critical >50%; `src/stages.json`, not expert-verified) plus an overlay image. The disease itself is still decided by the whole-photo classifier: letting the separate parts decide was tested and was less accurate (validation: 5.4% vs 4.1% wrong). Reports: [`PHASE3_SEG_REPORT.md`](outputs/final/PHASE3_SEG_REPORT.md), [`PHASE4_REPORT.md`](outputs/final/PHASE4_REPORT.md).

**Insect damage (2026-10-27):** a 5th result, "insect damage", with a stage and **% of the worst leaf eaten** (holes ÷ leaf, holes counted as leaf; bites from the leaf edge are not counted).
- **Second-opinion classifier:** a 5-category classifier (`experiments/experiment_007_v4_insect/`, trained with 356 insect-hole photos) runs next to the unchanged v3. It can turn "no disease", no answer, or an *unsure* disease answer into insect damage, and only when it is confident. On its own it called black-rot leaves insect damage and failed the external-test gate.
- **Hole model:** a second outline model (`experiments/seg_002_insect/`) measures the holes only. Disease % and stage still come from seg_001.
- **Results:** external test, web photos and the field photo are unchanged. 77% of held-out insect-eaten leaves are called insect damage, and the % eaten is within 0.6 points of reviewed outlines.
- **Report:** [`INSECT_MODEL_V4_REPORT.md`](outputs/final/INSECT_MODEL_V4_REPORT.md).

Severity labels are in `outputs/severity/severity_grades.csv`. Since Phase 5, black rot and downy mildew grades are **measured** (% of the worst leaf affected, from the segmentation model or reviewed outlines); bacterial spot rot grades are still **by-eye** because the measurement misses dark rot on the head (`grade_source` says which). None are expert-checked: `src/grading_app.py` shows each photo with its measured overlay so an expert can confirm or correct, then `python src/refit_severity.py`. Report: [`PHASE5_REPORT.md`](outputs/final/PHASE5_REPORT.md).

The image **dataset is not in this repo** (too large). Training/eval expect a local `dataset/` folder. The API never retrains: it loads the fold checkpoints once into RAM and reuses them, so the same photo always gets the same scores.

## Inference API

**Deploying on Render free (512 MB):** the same models run as ONNX without PyTorch (`PLANT_BACKEND=onnx`, about 340 MB peak, identical answers on 100 test photos). Steps and the API guide for integrating it are in [`DEPLOY.md`](DEPLOY.md).

```bash
pip install -r requirements.txt
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
```

| What | URL |
|---|---|
| Upload form | http://127.0.0.1:8000/ |
| Swagger docs | http://127.0.0.1:8000/docs |
| Health | `GET` http://127.0.0.1:8000/health |
| Predict | `POST` http://127.0.0.1:8000/predict  (multipart field `file`) |

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/leaf.jpg"
```

Check `status` first:

| status | what to do |
|---|---|
| `confident` | one diagnosis at the 95% level; use `prediction` + `severity` |
| `uncertain` | several diagnoses plausible (`prediction_set_95`) or confidence < 0.80; get a second look |
| `not_cauliflower` | the photo does not show a cauliflower plant, leaf or head (checked first by a CLIP gate, `src/cauliflower_gate.py`; nothing else runs) |
| `out_of_scope` | photo doesn't resemble the training photos (odd framing, collage, not a plant) |
| `no_evidence` | no class stands out (e.g. a blank or featureless image) |
| `retake_photo` | blurry, too dark, overexposed, blank or too small |

`reliable` is `false` for `not_cauliflower`, `out_of_scope`, `no_evidence` and `retake_photo`. In that case `prediction`, `confidence` and `severity` are `null`, and the model's top class is only under `unreliable_guess` (debugging, not a diagnosis). `message` is a one-line human-readable result.

**Advice and tracking (Phase 6):** confident diagnoses include `advice` (actions for that disease and stage from `src/advice.json`, draft, not expert-verified, no chemical doses). Send an optional `plant_id` form field to store the result and get a `trend` (change since the last check, alerts when a disease appears or gets worse); `GET /plants/{plant_id}/history` lists past checks. For bacterial spot rot the stage comes from the model's grade because the measured % under-counts dark head rot. Report: [`PHASE6_REPORT.md`](outputs/final/PHASE6_REPORT.md).

Measurement fields (when reliable): `plant.stage`, `plant.affected_pct` (worst leaf/head; `null` if the spots were too faint to measure, then `stage_source` = `classifier_grade`), `plant.lesion_count`, `plant.regions[]` (per leaf/head: `type`, `disease`, `verdict`, `measured_affected_pct`, `lesion_count`), `plant.other_part_findings`, and `overlay_jpeg_b64` (pink = affected tissue).

Other fields: `confidence` (calibrated), `class_probabilities`, `severity.{grade, stage, grade_range_90, grade_confidence, severity_score_0_100}`, `disease_presence`, `other_findings` (second disease ≥ 0.5), `ood_score`, `image_quality`.

CLI (no server):

```bash
python src/predict.py --image path/to/leaf.jpg
python src/predict.py --image path/to/leaf.jpg --json
python src/predict.py --folder path/to/images
```

## Severity grading tool

```bash
python -m uvicorn src.grading_app:app --port 8010
```

Keys 0–4 set the grade, ←/→ navigate, `h` shows only images not yet graded by a human.

## Reproduce / train (v3)

The full command list (downloads, copy detection, data build, training, evaluation) is in [`MODEL_V3_REPORT.md`](outputs/final/MODEL_V3_REPORT.md#reproduce). In short:

```bash
pip install -r requirements-dev.txt
python src/build_v3_data.py
python src/train_v2.py --recipe v3
python src/calibrate_v2.py --exp-dir experiments/experiment_005_v3_multisource
python src/eval_external.py experiments/experiment_005_v3_multisource
python -m pytest tests -q
```

Frozen splits: `splits/` (do not regenerate). `splits_v3/external_test.csv` is the frozen outside test set: never train on it. `python src/train_v2.py` with no `--recipe` still reproduces v2.
