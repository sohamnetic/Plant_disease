# Photo collection guide for teammates

The web test showed exactly which photos the model is missing; see `WEB_PROBE_REPORT.md`. These photos are what will actually fix it. **Cauliflower only**: no cabbage, broccoli, kale or other crops.

## What to photograph (in priority order)

| # | What | How many | What it must show |
|---|---|---:|---|
| 1 | **Black rot, classic** | 50–100 | Yellow or tan **V-shaped** areas starting at the **leaf edge** and pointing inward along a vein. Veins inside turn **black**; hold the leaf against the light to see them. Early cases matter most: one or two small Vs. |
| 2 | **Downy mildew on the head** | 20–30 | Grey-brown or dark discolouration on the **curd surface**. If possible, add one photo of a leaf from the same plant showing downy mildew, and say it's the same plant. |
| 3 | **Downy mildew on leaves, with the underside** | 20–30 | Angular yellow patches between veins on top, plus a photo of the **underside** showing grey-white fuzz (early morning is best). |
| 4 | **Healthy whole plants, top view, small heads** | 20–30 | Young plants with a small head among the leaves, like the user's second photo. These help the model avoid false alarms and help the segmentation find small heads. |

## How to take each photo

- **One plant per photo.** The affected leaf or head should fill most of the frame.
- **For each plant:** one close-up, plus one photo of the whole plant from about 1 m above.
- **Light:** daylight, no flash, no filters. Sharp focus on the lesion.
- **No text, stickers or watermarks** on the photos.
- **Any phone is fine;** keep the original files, don't send them through WhatsApp compression.

## How to label

**Folder layout:** one folder per class:
```
field_photos/black_rot/
field_photos/downy_mildew/
field_photos/no_disease/
field_photos/bacterial_spot_rot/
```

**A small sheet** `field_photos/labels.csv` with columns:
`file, class, part (leaf/head/whole plant), same_plant_id, date, place, who_confirmed, notes`

**If unsure, put the photo in `field_photos/unsure/`.** A wrong label hurts more than a missing photo. Ideally an extension officer or plant pathologist confirms the black rot and downy mildew photos (`who_confirmed`).

## What happens with them

1. **Copy check:** the photos are checked against everything we already have (the same check used tonight).
2. **Test set:** about one third becomes a new frozen test set, never trained on.
3. **Retraining:** the rest is used for a full retrain, or for the head-refit tool (`src/refit_diagnosis.py`).
4. **Release rule:** the new model only replaces the current one if, on the frozen external test, the web test photos and the new field test set:
   - black rot and downy-mildew-on-head results improve;
   - no other class gets worse;
   - healthy false alarms don't rise.
