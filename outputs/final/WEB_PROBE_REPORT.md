# Web-photo test: report (night of 5–6 Oct 2026)

**Task:** collect real cauliflower photos from the web, rotating through healthy, downy mildew, black rot and bacterial spot rot. Run the model on each one, find out why it is wrong, and fix what can be fixed. The model must not get worse.

**Result in one line:** the model was **not changed**. Every fix that could have changed its answers was tested, and each one either failed or made other answers worse. Three safe changes to the app's display and advice were made. The main weakness is now known precisely, and it needs new training data.

## 1. The photo set (`new_pics/`, git-ignored)

**Rules applied to every photo:**
- **Cauliflower only.** The source page or caption must name cauliflower and the disease.
- **Checked by eye** on a contact sheet before it was kept.
- **Not a copy** of any photo we already have. Each photo was checked against all 6,907 training and external-test photos, using the same embedding + SIFT copy check as before.

**Totals:**
- 92 photos kept, from about 53 websites in 8 languages (English, Spanish, Portuguese, Turkish, Italian, French, Japanese, Chinese), plus Hindi-language Indian farm sites.
- Photos marked "unauthorized use prohibited" (the Japanese Zennokyo library) were not used.
- 11 of them are copies of photos we already have, so they are excluded from the scores.
- **81 independent test photos** remain.
- 11 photos have stock watermarks. They are tagged in the manifest.
- The non-English searches found two official plant-pathology sources (Universidad Nacional de Luján, Argentina, the Québec Iriis phytoprotection site, Kagawa Prefecture and Takii Seed in Japan) with labelled cauliflower photos of downy mildew, black rot and wet rot.

**Rejected, as examples of the checks working:**
- Commons photos filed under cauliflower that are actually cabbage heads.
- AI-generated "garden" images.
- Collages.
- Brassica leaves that may not be cauliflower: mustard or Chinese-cabbage types, collard, kale.
- Kitchen spoilage photos.
- Stalk rot (not head rot).

`new_pics/manifest.csv` records each photo's image URL, source page, the evidence for its label, the copy-check result and any notes.

**The label for head rot is confirmed by the copies.** The web calls bacterial rot of the head "bacterial soft rot". Six such web photos (from US, Iranian and Japanese sites) turned out to be photos that **VegNet and c2h file as bacterial spot rot**. So soft rot of the head on the web is the same thing our datasets call bacterial spot rot, and these photos are scored as bacterial spot rot.

## 2. Current model on the 81 independent photos

| Class | n | Correct | Declined to answer | Confident and WRONG |
|---|---:|---:|---:|---:|
| Bacterial spot rot | 9 | **9** | 0 | 0 |
| Black rot | 26 | **0** | 6 | **11** (all → downy mildew, up to 99.5%) |
| Downy mildew, leaves/seedlings | 8 | 6 | 2 | 0 |
| Downy mildew, **heads** | 7 | **0** | 1 | **6** (all → bacterial spot rot, 84–99.6%) |
| Healthy | 31 | 21 | 7 | 1 (→ bacterial spot rot, 80%) |

Overall: 44% correct, 22% confident and wrong. 17 of the 18 confident-wrong answers fall into the two patterns in section 3.

**The 18th is `no_disease_0075`,** an Indian field photo of a creamy head, called bacterial spot rot at 80%, right at the confidence threshold. I see no symptoms, but slight discolouration can't be ruled out.

**What works:**
- bacterial spot rot (9/9, including Japanese seed-company photos);
- downy mildew on leaves (6/8, none wrong);
- healthy photos (one borderline false alarm in 31).

Full table: `outputs/web_probe/results.csv`. Overlays: `outputs/web_probe/overlays/`.

**Affected %:**
- **Black rot leaves:** the lesion outlines sit correctly on the yellow V-lesions, giving 13–21% on the clear cases. **The measurement works; the disease name is what's wrong.**
- **Healthy heads:** outlined correctly.
- **Head rot:** under-measured. Subtle rot gave 1% on two Alamy heads; badly rotten Takii heads gave 1% or no measurement at all. **The stage is still right ("critical")**, because the bacterial spot rot fix takes the stage from the model's grade, not from the %. Dark rot needs the hand-drawn outlines (pending).

## 3. Why it fails

**1. Classic black rot is called downy mildew (0 of 26 correct, from 14 sites in 6 languages, including the Kagawa Prefecture government and Takii Seed in Japan).** Classic black rot is yellow V-shaped lesions from the leaf edge with blackened veins. The training data almost never shows it:
- **VegNet "black rot" (704 photos):** a handful of blue-grey leaves on red soil, with holes and small spots.
- **cleaf "black rot" (703 photos):** small detached leaves on a white background.
- **Training "downy mildew":** full of yellow-edged leaves.

So the model learned "yellow leaf edge = downy mildew". This is a gap in the training data, not a code bug.

**2. Downy mildew on the head is called bacterial spot rot (6 of 7, all confident; the 7th was declined; sources in Argentina, Québec and the US).** The training set has essentially no downy mildew on heads, so grey-brown patches on a head look like rot to the model. The expert review (section 6) should also check whether some training "bacterial spot rot" heads are really downy mildew.

**Possible label overlap in training (hypothesis for the expert, not verified).** SPPU's "Bacterial Spot" heads (62 in training, mapped to bacterial spot rot) fall into two groups:
- **Small grey-brown surface spots** (e.g. BS-010, BS-015, BS-016 in the review sheets). These look much like the university's downy-mildew-on-head photos.
- **Nearly healthy creamy heads** (e.g. BS-002, BS-004, BS-005, BS-006, BS-017).

If some of these are really downy mildew, or healthy, that would explain pattern 2 and also the creamy healthy head called rot (`no_disease_0075`, 80%). An expert check of these sheets is the cheapest possible fix: relabel, then retrain.

**3. 16 photos were declined (out of scope or no clear evidence): 7 healthy, 6 black rot and 3 downy mildew.** These were wide field views, whole plants, seedlings, a text banner and low-resolution photos. Declining is safe, but it is a coverage gap.

## 4. What was tried and NOT used (each would have made something worse)

**Multi-Crop Disease Dataset (Mendeley 6243z8r6t6, CC BY 4.0, 978 MB).**
- **Why it looked useful:** it claims original Tamil Nadu field photos covering exactly our four cauliflower classes.
- **Copy check:** **621 of its 640 cauliflower images are copies of our photos.** 612 copy training photos and 9 copy external-test photos. Some of them are relabelled; for example, a VegNet black rot leaf appears as "downy mildew".
- **Decision:** only 19 photos are new, so a retrain is pointless. Not used. Record: `outputs/web_probe/multicrop_copy_check.csv`.

**Retraining only the decision layer on web black rot photos** (`src/refit_diagnosis.py`). This retrains the diagnosis and presence heads while the backbone stays frozen.
- **Data:** 6 close-up cauliflower black rot photos from 5 sites. Each test photo was scored by a head trained without photos from its own site; Shutterstock photos were never trained on.
- **Black rot:** 0% → 46–62% (raw top answer).
- **Healthy web photos:** 81% → 65%, with an extra confident-wrong answer, in every setting tried.
- **Stronger anchoring:** also cost the external test 2–5 points.
- **First version:** it included wide field photos, which taught "leafy plant = black rot".
- **Rerun with 11 close-ups from 11 sites (after the multilingual searches):**
  - black rot 11% → 56–67%, with confident-wrong 6 → 0;
  - healthy 81% → 55–65%, with confident false alarms 1 → 2.
  - Same pattern.
- **Adding healthy web photos as a counterweight:**
  - healthy 81% → 84–87%, with no confident false alarm;
  - black rot → 22–39%;
  - but the **frozen external test drops 94.1% → 93.0–93.8%**, because subtle rotten heads get pulled to "healthy" (bacterial spot rot → healthy: 9 → 17–21).
- **Decision: not shipped.** Every version trades one class's mistakes for another's. Calling healthy plants diseased, or missing head rot, is worse than the current failure, which the new "check before acting" text now warns about.

**"Lesions at the leaf edge = black rot" rule.** Real downy mildew photos in our data also have edge-touching lesions (median edge share 0.91–0.98), so the rule would flag most downy mildew. Not used. Data: `outputs/web_probe/edge_frac.csv`.

**"Head disease but no head outlined → uncertain" rule.**
- **External test:** affects 3 of 64 confident bacterial spot rot answers. All 3 were correct, and in all 3 the head is clearly visible; the segmentation just missed it.
- **Labelled data:** it catches no wrong answer.
- **Decision:** not used (`outputs/web_probe/head_check.csv`). A display-only version was used instead; see section 5.

**A stricter out-of-scope limit for "confident".** Wrong confident web answers have the same out-of-scope scores as correct ones. On the external test it would cost 17–106 correct confident answers to remove 3–5 wrong ones. Not used. Data: `outputs/web_probe/ood_scores.csv`.

## 5. What was changed (none of it changes a diagnosis)

**1. Page bug fixed.** Results where nothing could be measured (affected % empty) crashed the page, and it showed raw JSON instead. This was your young-head photo.

**2. Lookalike checks in the advice.** For all three diseases, "What to do" now starts with **"Check before acting"**, which tells the farmer how to rule out the lookalike the model confuses it with:
- **Black rot vs downy mildew:** black rot gives V-shapes from the leaf edge with black veins. Downy mildew gives angular patches between veins with grey fuzz underneath.
- **Bacterial spot rot vs downy mildew on the head:** bacterial rot is soft, wet and smelly. Downy mildew on the head is dry grey-brown discolouration, with downy mildew on the leaves nearby.
- **The warning:** each check states that the model can confuse the two.
- **Why:** these target exactly the two confident-wrong patterns found tonight, which together cover 17 of the 18 confident-wrong answers. Defined in `src/advice.json`, returned as `advice.check_first`.

**3. Head disease without an outlined head.** This applies when the photo is judged bacterial spot rot but no head was outlined, as on your young-head photo.
- **Before:** the leaves were listed as "bacterial spot rot".
- **Now:** the leaves are listed as "the photo is judged bacterial spot rot (a head disease); the head was not outlined, so the rot could not be measured". The message also suggests a closer photo of the head.
- **Unchanged:** diagnosis, confidence and status.

**Other:**
- New test tool: `src/web_probe.py` (`add`, `check`, `run`).
- `src/eval_external.py` gained `--out`.
- `.gitignore` now covers `new_pics/`.
- 2 new tests. Whole suite: **28 passed + 1 known limitation**.

**Unchanged:** the model files (`experiment_006`), its answers, confidences, affected % and stages. So the external test (1,142 photos), the in-dataset test, and the Phase 3–5 numbers are exactly as before.

## 6. What would actually fix it (needs people, not code)

1. **About 50–100 real photos of classic black rot on cauliflower** (how to take and label them: `PHOTO_COLLECTION_GUIDE.md`): yellow V-lesions from the leaf edge, taken in the field. Teammates can photograph them, or an extension officer or expert can confirm them. Then rerun `python src/refit_diagnosis.py feats` and `cv`, or retrain fully. The test photos from tonight are ready to measure the gain.
2. **About 20–30 photos of downy mildew on the head.**
3. **Expert review (the old Phase 8):** check whether some VegNet "downy mildew" leaves with yellow V-edges are actually black rot. If labels change, retrain.
   - **Ready for review:** `outputs/web_probe/expert_review/` has numbered contact sheets of all 150 VegNet downy mildew originals and 53 black rot originals, plus the 141 SPPU and c2h bacterial spot rot heads in training (bacterial rot vs downy mildew on the head vs healthy).
   - **How to record answers:** `review_list.csv` has an empty `expert_label` column to fill in.
   - **Also check whether each photo is cauliflower at all.** Some VegNet downy mildew "originals" carry Bugwood/UGA image-number stamps (e.g. DO-033, DO-034, DO-035), so they were taken from the web and their host plant is not verified.
4. **Whole-plant and top-down photos with small heads,** to cut the "declined" answers and help the segmentation find small heads.

## Files

- `new_pics/<class>/`, `new_pics/manifest.csv`: web test photos with source and evidence.
- `new_pics/web_train/`, `new_pics/web_train.csv`: 4 black rot photos used only in the rejected head-refit experiment.
- `outputs/web_probe/`: results, overlays, copy-check records, measurement tables.
- `data_external/raw/6243z8r6t6/`: the Multi-Crop download (978 MB zip plus 640 extracted images). Git-ignored and unused; it can be deleted.
- `cache/refit_diag/`: features for the refit experiment. Git-ignored and can be deleted.
