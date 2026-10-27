# Insect-hole photos: how the current model sees them, data preparation, integration plan

Date: 2026-10-26. **Nothing was trained and the app and model were not changed.** The deployed model is still `experiments/experiment_006_v3_measured_severity/`.

## 1. The dataset

The Kaggle link (`jocelyndumlao/cauliflower-leaf-diseases-datase`) is a re-upload of Mendeley [x995snz7p3](https://data.mendeley.com/datasets/x995snz7p3/1). The original authors are Durjoy, Shikder, Shoib and Bijoy (Daffodil International University, 2025). The original license is CC BY 4.0; Kaggle wrongly lists CC0. We have used this source since v3 under the name **"cleaf"**.

| Folder | Photos | Used before today |
|---|---|---|
| Black Rot | 1,088 | 703 in the training pool, 376 in the frozen outside test |
| Healthy | 934 | 597 in the training pool, 325 in the frozen outside test |
| **Insect Hole** | **639** | not used ("not one of our 4 classes") |

Every photo is a single picked leaf cut out onto white, 3000×3000 px. The Kaggle bundle also has an "Original Image" folder (5.3 GB, the photos before cut-out), which we do not have. The same leaves are in our copy, so we don't need it.

## 2. What the current app says about the 639 insect-hole photos

Run: `python src/predict.py --folder ".../Insect Hole"` (the full app path: cauliflower check, 5-model ensemble, calibration, segmentation).
Per-photo results: `outputs/insect_probe/insect_hole_predictions.csv`.

| App answer | Photos | Share |
|---|---|---|
| Confident: no disease | 220 | 34.4% |
| **Confident: black rot** (all "early stage") | **116** | **18.2%** |
| Uncertain (best guess: no disease 169, black rot 90) | 259 | 40.5% |
| No reliable diagnosis | 44 | 6.9% |
| Not cauliflower / out of scope / retake photo | 0 | 0% |

Another 57 photos list black rot as a second finding.

- **Insect damage is never named.** That's expected: the model has no such class or flag.
- **The cauliflower check and the "doesn't look like our training photos" check accept all 639 photos.** That's correct, because they are cauliflower leaves in a style the model has seen.
- **Holes push the model towards black rot.** Its top answer is black rot for 226 of 639 (35%). For comparison, on the 325 healthy test photos from the same source and same setup, the top answer is black rot for only 19 (5.8%). The average black-rot probability also rises with the measured hole size:

| Enclosed holes (measured, % of leaf) | Photos | Confident black rot | Avg. black-rot probability |
|---|---|---|---|
| none found | 169 | 19 (11%) | 0.33 |
| under 0.5% | 163 | 39 (24%) | 0.45 |
| 0.5–2% | 150 | 26 (17%) | 0.43 |
| over 2% | 157 | 32 (20%) | 0.46 |

Looking at the confident black-rot photos (`outputs/insect_probe/sheet_model_conf_black_rot.jpg`), most are green leaves with clean holes, so the black-rot answer is a **false alarm**. A minority also have yellow edges or small dark specks and may really have disease as well; nobody has checked these. Holes and black rot also appear together in this source's own Black Rot folder (Section 3).

**What this means for farmers today:** about 1 in 5 photos of insect-eaten leaves gets a confident early-black-rot diagnosis and black-rot advice.

## 3. Preparing the data (`src/build_insect_data.py`)

This step reads the photos and `splits_v3/` and writes `splits_insect/`. The frozen outside test is not changed.

| Check | Result |
|---|---|
| Exact duplicate files | 0 |
| Copies of our VegNet photos (v3 copy check) | 1 flagged (32 matching points with a downy-mildew training photo, most likely a chance match on veins) → **excluded**, same rule as v3 |
| Same leaf also in this source's other folders | 2: 1 in Black Rot, 1 in Healthy. The one also in the Healthy folder shows no visible hole → **excluded** (contradictory label). None of the 2 is in the frozen outside test. |
| Near-duplicate groups | 628 groups for 637 photos, so almost every photo is a different leaf |
| **Split by group** (35% test, seed 42) | **412 training candidates · 225 insect test · 2 excluded** |

**Measuring holes:** a hole shows the pale grey surface under the leaf, not the white cut-out background, so it's detected as low-saturation, bright pixels enclosed by the leaf. The `tier` column records the result: "clear_holes" means at least 0.5% of the leaf is missing through enclosed holes.

| Folder | Has an enclosed hole | "Clear" tier | Median % missing |
|---|---|---|---|
| Insect Hole | 470 / 639 (74%) | 307 (48%) | 0.43% |
| Black Rot | 387 / 1,088 (36%) | 156 (14%) | 0 |
| Healthy | 302 / 934 (32%) | 125 (13%) | 0 |

The measurement is only **a sorting aid, not a label**:
- On healthy leaves it fires 13% of the time, mostly on glare and pale veins.
- It misses holes that are open to the leaf edge, and it can't measure bites taken from the edge.

**Label quality:** in the sample sheets (`outputs/insect_probe/sheet_edge_or_subtle.jpg`), a noticeable share of the "edge_or_subtle" insect photos show no hole I can see at thumbnail size. The authors may have labelled whole plants rather than single leaves. I estimate roughly 1 in 5 photos, which needs a proper review.

**Black Rot folder:** real holes are visible in some of its photos, so a black-rot leaf can also be insect-damaged. This supports treating insect damage as a separate yes/no flag rather than a 5th disease.

Outputs:
- `splits_insect/insect_pool.csv`, with one row per photo: group, split, tier, exclude reason, hole measurements and the cross-folder match.
- `splits_insect/SUMMARY.txt`
- `outputs/insect_probe/cleaf_hole_metrics.csv`, covering all 2,661 cleaf photos.
- Sample sheets: `outputs/insect_probe/sheet_*.jpg`

## 4. Revised plan (2026-10-26): "Insect damage" as its own category + % of the leaf eaten

Your request: the app shows **Insect damage** as a category of its own (next to bacterial spot rot, black rot and downy mildew) and reports **what % of the leaf the insects have eaten**. This replaces the yes/no flag design in Section 5.

### Can the % be measured from colour alone? Tested: no

I tried measuring the eaten area with a fixed colour rule: grey pixels enclosed by the leaf count as holes, and a large shape-smoothing step estimates bites from the edge. I ran it on 120 photos per folder and checked overlays (`outputs/insect_probe/damage_trial.csv`, `sheet_damage_colour_rule.jpg`: red = holes, blue = bites).

| Folder | Holes: median / 90th percentile | Edge bites: median |
|---|---|---|
| Insect Hole | 0.6% / 5.4% | 2.0% |
| Healthy (should be 0) | 0.04% / 1.7% | 1.1% |
| Black Rot | 0.02% / 1.2% | 0.6% |

- It **catches clear round holes**.
- It also **marks pale midribs, bluish waxy leaf patches and glare as holes**: one healthy leaf read 10% holes.
- It **reads natural leaf lobes as bites** (up to 23% on a healthy leaf), and it **misses grey holes** that look like the leaf around them.

This is the same lesson as Phase 5, where colour rules failed on dark head rot. The % has to come from a **trained outline model**, not a fixed rule.

### How it would work

1. **The classifier gets a 5th output, "insect_damage".** The app shows it as the category "Insect damage" when the main problem is holes. If a leaf has both a disease and holes, the disease stays the main result and "insect damage, X% eaten" is shown as a second finding. A new per-category presence output makes this possible.
2. **The outline model (YOLO11-seg from Phase 3) gets a 4th outline type, "insect_hole".** % eaten = hole area ÷ (remaining leaf + hole area), for the worst leaf, the same way lesion % is measured today.
3. **App:** "Insect damage, 4% of the leaf eaten (early)", an overlay with holes in a second colour, and general pest advice with no chemical doses.

### Steps (each one waits for your go-ahead; a report after each)

| Step | What | Time |
|---|---|---|
| I-1 Hole outlines | Machine drafts for the 637 insect photos: the colour rule plus a texture check (leaf tissue is textured, the surface seen through a hole is smooth) plus SAM2 to tighten outlines. Then a visual review on overlay sheets, as in Phase 3, keeping about 250 good ones. Photos where no hole can be seen are dropped from the category. Healthy leaves serve as 0% examples. | ~half a day; your check of the sheets is optional |
| I-2 Outline model | Add "insect_hole" to `data_external/seg_yolo`. Background swap so the holes show soil or other leaves, not only grey. Retrain YOLO and check the measured % against the reviewed outlines on held-out photos. Leaf, head and lesion results must not get worse. | ~20 min training + checks |
| I-3 Classifier | 5 classes. Insect photos from the training split are labelled insect_damage. Fix the background-swap step so grey holes are swapped too. One-fold trial, then 5 folds + calibration. | ~1.5 h |
| I-4 Checks (ship only if all pass) | Outside test ≥ 92.7% correct and ≤ 5.3% wrong (as v3); healthy false alarms and web photos no worse; insect test: "Insect damage" for ≥ 80%, confident black rot well below today's 18%; measured % within ±2 points of the reviewed outlines | |
| I-5 App | Category, %, stage, overlay, advice, tests, report | |

**Open points:**
- **Bites from the edge** can only be estimated, because the original leaf edge is gone. The first version will count holes plus clear bites and say "approximately".
- **Stage cut-offs** for insect damage (for example light < 5%, moderate 5–15%, heavy > 15%) would be a draft, like the disease stages, and are not expert-checked.
- **Field photos:** every insect training photo is on a white background, so field performance needs about 150 field photos to check (Section 5, step B).

## 5. Earlier integration plan: yes/no flag (replaced by Section 4)

**Design:** keep the 4 diagnoses and add one **"insect damage" yes/no output** next to the existing per-disease presence outputs. The app would report it separately ("Insect feeding damage seen"), so a leaf can be "black rot + insect damage" or "no disease + insect damage".

| Step | What | Why |
|---|---|---|
| A. Review the insect labels (about 1 hour, `grading_app`-style page) | For each of the 412 training + 225 test photos: (1) are holes visible? (2) any disease signs? | Removes photos with no visible damage. Insect photos marked "no disease signs" can then also be trained as **no_disease**, which is what removes the black-rot false alarms. Without this, the diagnosis would not learn that holes aren't black rot. |
| B. Field check set (about 150 photos) | Mark holes yes/no on field photos we already have (SPPU healthy, web photos) | Every insect photo is on white. This set tests whether the flag works in a field and how often it fires falsely there. |
| C. Code | `v2_model.py`: presence output 3 → 4 (old checkpoints still load). `train_v2.py`: new recipe, insect label 1 for insect rows, 0 for cleaf Healthy rows, "unknown" (not trained) everywhere else, and diagnosis trained only where step A gave a clean answer. Fix `scene_aug.py`, which currently keeps the grey hole pixels when it swaps the background, leaving grey blobs on soil. CutMix: don't train the insect flag on mixed images. | |
| D. Trial: 1 fold first (about 15 min on the GPU) | Compare with `experiments/ablation_cur_ref_fold0` | Cheap early stop if diagnosis gets worse |
| E. Full run: 5 folds + calibration (about 1–1.5 h) | Also fit the insect-flag threshold on out-of-fold data | Same process as v3 |
| F. Gates (ship only if all pass) | Outside test ≥ 92.7% correct and ≤ 5.3% wrong (as v3); healthy false alarm and web photos no worse; **insect test: flag found in ≥ 80%, confident black rot well below today's 18%**; insect flag false on the 325 frozen cleaf healthy test photos and the field check set ≤ 10% | Same "never worse" rule as before |
| G. App | `insect_damage` field in the API and page, a line in the message, general pest advice in `advice.json` (no chemical doses), tests, report | |

**Limits that stay:**
- Holes only. It won't identify the insect, and it won't catch aphids or other damage that leaves no holes.
- All insect training photos come from one source, so field performance depends on step B.
- The measured % of leaf missing could later become an "insect stage", but that's not part of this plan.
