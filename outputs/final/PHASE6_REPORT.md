# Phase 6 report — advice and plant tracking (plus the bacterial spot rot interim fix)

**Status:** complete. Phase 7 has not been started.

## 1. Bacterial spot rot interim fix

**Problem (Phase 5):** the segmentation model misses dark rot on the head, so the measured % under-states bacterial spot rot.

**Check on non-test photos** (117 bacterial spot rot photos), each stage source compared with the reviewed grades:

| Stage source | Exact | Within ±1 |
|---|---:|---:|
| Measured % | 30% | 80% |
| Disease model's severity grade | **78%** | **100%** |

**Fix:** for bacterial spot rot, the app's stage now comes from the model's grade (`STAGE_FROM_GRADE` in `src/plant_runtime.py`). The measured % is still shown, with a note that it can under-count dark rot.

**Confirmed once on the test split** (25 photos): exact **48% → 80%**, within ±1 **72% → 100%** (`src/eval_bsr_stage.py` → `outputs/final/bsr_stage_eval.json`).

Black rot and downy mildew keep the measured stage. The real fix is still hand-drawn rot outlines for bacterial spot rot.

## 2. Advice table

**Where it lives:**
- `src/advice.json`: for each disease and stage (early, moderate, severe, critical) plus healthy, 2–3 concrete actions, general hygiene rules, a one-line explanation of the disease, and source links;
- `src/advice.py`: returns the advice for a diagnosis and stage.

**What it contains:**
- **Sources:** university extension fact sheets (Pacific Northwest Handbooks, NC State, Wisconsin, Illinois, Utah State).
- **Examples:**
  - *black rot, early:* pick off affected leaves when dry, check neighbours, switch to base watering;
  - *bacterial spot rot, moderate:* harvest affected heads now and cut beyond the discoloured area with a clean knife;
  - *critical:* remove and destroy the plant, and plan a 2–3 year rotation.

**Safety rules:**
- `expert_verified: false` is shown with every piece of advice.
- **No chemical products or doses.** It says "ask a local extension officer" instead, and a test checks this.
- Advice is given **only for confident diagnoses.** For uncertain answers the app says "take a closer photo or ask an expert", because acting on a guess could mean destroying healthy plants.

## 3. Plant tracking (optional)

**How to use it:** send a `plant_id` with a photo (web page box, or the `plant_id` form field in the API). Each reliable result is stored in `outputs/tracking.db`, a local database that git ignores.

**The response adds a `trend` block:**
- the previous check;
- the change in affected %, per day;
- alerts for: a new disease on a healthy plant, a changed diagnosis, a stage that got worse, or the area growing by 5 or more percentage points.

**Other details:**
- `GET /plants/{plant_id}/history` lists every stored check for a plant.
- Plant IDs are validated (letters, digits, `.` `_` `-`, up to 64 characters). Unreliable photos are never stored.

**End-to-end demo** (same plant ID, three photos):
- healthy → "First check stored";
- black rot early → "New disease on a plant that was healthy: black rot. Stage got worse: none → early";
- bacterial spot rot critical → "Diagnosis changed … Stage got worse: early → critical".

The demo records were deleted afterwards.

## Tests

`tests/test_phase6.py` (6 new tests):
- every disease and stage has advice;
- the advice contains no doses or products;
- tracking alerts and the per-day rate are correct;
- plant IDs are validated;
- bacterial spot rot gets its stage from the model's grade;
- unreliable photos get no advice.

Whole suite: **26 pass, plus 1 known limitation.**

## Limitations

- **Advice is a draft.** It comes from general extension guides and is not verified for your region, varieties or local rules.
- **Tracking accuracy depends on consistent photos.** Different framing or distance changes the measured % between checks, so small changes (under 5 points) are not alerted.
- **Alerts can be triggered by misdiagnoses.** A one-off misdiagnosis also shows as "diagnosis changed"; repeated checks make it clearer.
- **The bacterial spot rot stage is the model's estimate**, trained on my by-eye grades, not a measurement.
