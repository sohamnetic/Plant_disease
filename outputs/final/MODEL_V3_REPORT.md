# Model v3 report — trained on more cauliflower sources, tested on photos it never saw

**Deployed model:** `experiments/experiment_005_v3_multisource/` (5-fold EfficientNet-B0 ensemble, same heads as v2: diagnosis, per-disease presence, severity grade 0–4).
**Why v3:** v2 scored 99% on its own test split but failed on outside photos. The user's own bacterial spot rot photo came back "downy mildew", because v2 had learned photo style (pale leaves = downy mildew, cracked soil = black rot) more than symptoms.

## What changed

| Change | Why |
|---|---|
| 4 new **cauliflower** sources, each copy-checked against our photos (SIFT + RANSAC) | more variety; copies of our photos never cross the train/test line |
| 53 healthy-leaf crops cut from our own healthy photos | v2 had almost no healthy leaves |
| Real-world photo augmentation: low-res, JPEG, white balance, shadows | web / phone / Pi images |
| Scene augmentation: background swap for white-backdrop photos, zoom-out | breaks "soil = black rot", and "subject fills frame" |
| Quality check judges exposure on the **subject**, not a white backdrop | white-backdrop leaves were wrongly rejected as "overexposed" |
| Sharpness bug fixed (PIL left the image border unfiltered) | flat images looked "sharp" |
| New status `no_evidence` + "image is blank" check | flat / junk images must not get a diagnosis |

Training pool: 3,639 → **5,764** images (black rot 704 → 1,479). Nothing from the external test set, or from `splits/test.csv`, was trained on.

## Data sources (cauliflower only, CC BY 4.0)

| Source | Used | Dropped, and why |
|---|---|---|
| [Cauliflower Diseases Identification 2025](https://data.mendeley.com/datasets/c2hrwntzgh/2) ("c2h") | all 4 classes | — (249 of its 400 images turned out to be **our photos with the background cut away**; they follow the photo they copy: train → train, test → test) |
| [SPPU Cauliflower](https://data.mendeley.com/datasets/x26px3xnmy/1) | Bacterial Spot, Downy Mildew, Healthy | Alternaria Leaf Spot (182/182 checkable images were our downy mildew photos relabelled), Soft Rot (our BSR photos relabelled), Black Spot (curd spots, not leaf black rot), Purple Tinges (includes plain heads), Alternaria Brassicae (45 copies of one photo); 28 "healthy" photos with visible yellowing (`outputs/review/external_review_excluded.csv`) |
| [Cauliflower Leaf Diseases 2025](https://data.mendeley.com/datasets/x995snz7p3/1) ("cleaf") | Black Rot, Healthy | Insect Hole (not one of our classes) |
| [Vegetable Leaf Collection](https://data.mendeley.com/datasets/rh8c5tgg5k/1) | Cauliflower Healthy leaf | all non-cauliflower folders |

A cabbage dataset was downloaded, tested, then **removed** (cauliflower-only decision).

## Frozen external test (1,142 images; 35% of every new source, by near-duplicate group)

| | v2 (previous) | **v3** |
|---|---:|---:|
| Correct | 15.1% | **92.7%** |
| Abstained (no diagnosis given) | 50.5% | 1.9% |
| **Wrong answer given** | 34.4% | **5.3%** |
| Confidently wrong | 34.4% | **2.5%** |
| Accuracy when it answers | 30.4% | **94.6%** |
| Healthy photos given a disease | 19.6% | **3.3%** |
| In-dataset test (93 originals) | 97.8% | **98.9%** |
| User photo 1 (true: bacterial spot rot) | "no reliable diagnosis" (guess: downy mildew) | **bacterial spot rot, confident (0.96)** |

By source (v3): field photos (SPPU) 96.7% correct · cauliflower leaves on white (cleaf) 92.0% · healthy leaves 100% · background-removed VegNet (c2h) 70.7%.

**Ablation** (one model each, same external set, before the quality-check fix): v2 recipe 10.8% wrong answers · new augmentation only 12.4% · augmentation **+ new data 1.9%**. The new data, not the augmentation, is what fixed real-world accuracy.

Plan gates: accuracy ≥ 85% ✅ · healthy false alarm ≤ 5% ✅ · in-dataset ≥ 97% ✅ · confidently wrong ≤ 2% ❌ (2.5%). Thresholds were **not** tuned on the test set to pass it.

## Known limitations

1. **Black rot is still the weakest class.** On c2h black rot photos 43% get a wrong answer (mostly downy mildew); on cleaf black rot leaves 8%.
2. **Other plants are not recognised as "not cauliflower".** A dried leaf from another plant on white paper (`pic.jpg`) is called black rot with 99.6% confidence. Fixing this needs non-cauliflower training images, which was out of scope (cauliflower only). Pinned as a known-failure test.
3. **Junk rejection is weaker than v2.** Flat colours are caught by the new blank / no-evidence checks, but random noise only just clears the out-of-scope limit (scores 311–353 vs 314).
4. **The external test is 61% cleaf** (young leaves photographed on white paper). Field photos are 26% (SPPU). Real Raspberry Pi photos are still untested.
5. **Severity grades are unchanged:** AI-reviewed, and only for the 618 VegNet originals. Measured severity is Phase 3 (YOLO11-seg).

## Reproduce

```bash
pip install -r requirements-dev.txt
python src/fetch_mendeley.py c2hrwntzgh --version 2 --only "Original Cauliflower Dataset.zip"
python src/fetch_mendeley.py x26px3xnmy --version 1
python src/fetch_mendeley.py x995snz7p3 --version 1 --only "Processed Image.zip"
python src/fetch_mendeley.py rh8c5tgg5k --version 1 --only "Cauliflower Healthy leaf.zip"
python src/external_data.py               # copy detection per source
python src/healthy_leaf_crops.py
python src/background_bank.py
python src/build_v3_data.py               # splits_v3/train_pool.csv + external_test.csv
python src/v2_cache.py splits_v3/train_pool.csv splits_v3/external_test.csv
python src/train_v2.py --recipe v3        # ~65 min on an RTX 4060 laptop
python src/calibrate_v2.py --exp-dir experiments/experiment_005_v3_multisource
python src/eval_external.py experiments/experiment_004_v2_multitask_cv experiments/experiment_005_v3_multisource
python -m pytest tests -q
```
