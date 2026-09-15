# Dataset structure

Fresh inspection of `D:\project-1`. Counts are computed from disk, not assumed.

## Project layout (after this audit)

```
project-1/
  dataset/
    Original Dataset/Original Dataset/{class}/   # originals
    Augmented Dataset/Augmented Dataset/{class}/ # augmented
  src/
    config.py
    eda.py
  outputs/eda/
  outputs/review/
  splits/
  DATASET_STRUCTURE.md
```

There was **no existing code, notebooks, requirements, or README** in the
repository at inspection time — only the `dataset/` tree.

## Nested folders

The publisher wrapped each subset twice:

- Originals: `dataset/Original Dataset/Original Dataset`
- Augmented: `dataset/Augmented Dataset/Augmented Dataset`

Class folders (exact names on disk → canonical ids):

- `Bacterial spot rot` → `bacterial_spot_rot`
- `Black Rot` → `black_rot`
- `Downy Mildew` → `downy_mildew`
- `No disease` → `no_disease`

No other class folders were found. Files whose parent folder is not in this
map would be labeled `UNMAPPED` in the inventory.

## File formats

| location | typical format | notes |
|---|---|---|
| original / Bacterial spot rot | `.jpg` | single pattern `Bacterial Spot. (N).jpg` |
| original / Black Rot | mostly `.jpeg`, 1× `.jpg`, 1× `.jfif` | `Black rot(N).*`, numbers 2–63 (no 1) |
| original / Downy Mildew | `.jpg` and `.jpeg` | same `Downy Mildew. (N)` stem can exist in both extensions |
| original / No disease | `.jpg` and `.jpeg` | same `No disease. (N)` stem can exist in both extensions |
| all augmented classes | `.png` | two naming styles in some classes (spaced vs concatenated) |

## Original vs augmented independence

Augmented filename numbers extend well beyond original counts (e.g. sequential IDs in the augmented folders), so filename numbers are NOT a safe original<->augmented join key. Grouping uses perceptual hashes (phash/dhash) plus exact MD5. Same-number original files with both .jpg and .jpeg are treated as possible duplicates and hashed.

Per-class filename stats:

```json
{
  "bacterial_spot_rot": {
    "n_original": 173,
    "n_augmented": 935,
    "orig_num_min": 1,
    "orig_num_max": 173,
    "aug_num_min": 1,
    "aug_num_max": 597,
    "orig_unique_numbers": 173,
    "aug_unique_numbers": 597,
    "number_overlap_count": 173,
    "orig_filename_patterns": {
      "Bacterial Spot. (N).jpg": 173
    },
    "aug_filename_patterns": {
      "BacterialSpotRot. (N).png": 597,
      "Bacterial spot rot._ (N).png": 338
    },
    "ratio_aug_per_orig": 5.404624277456647
  },
  "black_rot": {
    "n_original": 62,
    "n_augmented": 766,
    "orig_num_min": 2,
    "orig_num_max": 63,
    "aug_num_min": 1,
    "aug_num_max": 522,
    "orig_unique_numbers": 62,
    "aug_unique_numbers": 522,
    "number_overlap_count": 62,
    "orig_filename_patterns": {
      "Black rot(N).jpeg": 60,
      "Black rot(N).jpg": 1,
      "Black rot(N).jfif": 1
    },
    "aug_filename_patterns": {
      "BlackRot. (N).png": 522,
      "Black Rot. (N).png": 244
    },
    "ratio_aug_per_orig": 12.35483870967742
  },
  "downy_mildew": {
    "n_original": 177,
    "n_augmented": 921,
    "orig_num_min": 1,
    "orig_num_max": 146,
    "aug_num_min": 1,
    "aug_num_max": 609,
    "orig_unique_numbers": 146,
    "aug_unique_numbers": 609,
    "number_overlap_count": 146,
    "orig_filename_patterns": {
      "Downy Mildew. (N).jpg": 146,
      "Downy Mildew. (N).jpeg": 31
    },
    "aug_filename_patterns": {
      "DownyMildew. (N).png": 609,
      "Downy Mildew. (N).png": 312
    },
    "ratio_aug_per_orig": 5.203389830508475
  },
  "no_disease": {
    "n_original": 206,
    "n_augmented": 1041,
    "orig_num_min": 1,
    "orig_num_max": 188,
    "aug_num_min": 1,
    "aug_num_max": 1041,
    "orig_unique_numbers": 188,
    "aug_unique_numbers": 1041,
    "number_overlap_count": 188,
    "orig_filename_patterns": {
      "No disease. (N).jpg": 188,
      "No disease. (N).jpeg": 18
    },
    "aug_filename_patterns": {
      "No disease(N).png": 1041
    },
    "ratio_aug_per_orig": 5.053398058252427
  }
}
```

Therefore: **do not join original N to augmented N by filename number.**
Leakage control uses MD5 exact-duplicates and perceptual-hash connected
components (`group_id`). All members of a `group_id` stay in one split.

## Inventory size

- Total files under `dataset/`: **4281** (not ~5000)
- Readable images: **4281**
- Corrupt / unreadable: **0**

```
Class                 Original   Augmented   Total
---------------------------------------------------
bacterial_spot_rot         173         935     1108
black_rot                   62         766      828
downy_mildew               177         921     1098
no_disease                 206        1041     1247
Total                      618        3663     4281
```

`black_rot` originals are the scarce class (62). Augmented black_rot is ~12× the
original count; other classes are ~5×.

## Leakage grouping (used for the frozen split)

- Filename numbers are **independent sequences**. Do not join original N to augmented N.
- `.jpg` and `.jpeg` with the same number in Downy Mildew / No disease are **different images**.
- Grouping: exact MD5 + phash/dhash connected components (`group_id` in `outputs/eda/image_inventory.csv`).
- 172 multi-image groups (422 images); 25 original+augmented paired groups; 3 mixed-class hash groups.
- Canonical split: `splits/train.csv`, `splits/val.csv`, `splits/test.csv` (seed 42, group-atomic).

## Image geometry (readable only)

- Width: 259–4032 (median 512.0)
- Height: 194–4032 (median 512.0)
- Aspect ratio: 0.5020–2.1429 (median 1.0000)
- Unique resolutions: 195
- Color modes: `{'RGB': 4281}`
- Extensions: `{'.png': 3663, '.jpg': 508, '.jpeg': 109, '.jfif': 1}`

## What this audit does **not** do

- No files were deleted, moved, or relabeled.
- No model training was started.
- `splits/test.csv` is written for later use but must not be evaluated yet.

