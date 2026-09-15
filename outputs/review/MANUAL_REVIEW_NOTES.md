# Manual visual review notes

Flags only. **Nothing was deleted or relabeled.**

Grids inspected: `outputs/eda/grid_*_{original,augmented}.png` and
`outputs/review/*.png`.

## Domain

Images are **cauliflower / brassica** field photos, not isolated lab leaves.

- `bacterial_spot_rot` and `no_disease` are dominated by **curd (head) + wrapper leaves**.
- `black_rot` and `downy_mildew` are dominated by **single-leaf close-ups** on soil.

A model can learn “curd vs leaf shot” as a shortcut between those class pairs.
That is a real deployment risk for a Pi camera that may frame either way.

## Original grids

- **bacterial_spot_rot:** mix of whole-head field shots and tighter curd crops; symptoms range from speckling to collapsed brown curd. At least one photographer watermark (`V. Krejzar`) and phone-camera watermarks appear. Some frames look like the same plant from different angles.
- **black_rot:** more consistent leaf-on-soil framing. Large necrotic patches and shot-holes. Several near-duplicate leaf poses (same leaf, small viewpoint change). Background is often cracked red-brown soil (possible background shortcut).
- **downy_mildew:** high intra-class variance (mild yellow mottling through fully necrotic/shriveled leaves). Numeric watermarks on some frames (e.g. `5512793`). Pair `(68)/(69)` and `(42)/(44)` look like near-duplicates (hash grouping caught many original–original clones; see `duplicate_groups.csv`). File `(105)` looks more like shot-hole/pest damage than classic downy mildew — **possible mislabel, not changed**.
- **no_disease:** healthy curds, but white-balance varies from cool white to strongly yellow. Yellow healthy curds could later confuse with early rot. No obvious disease in the 20-image original sample.

## Augmented grids — artifacts (unrealistic for a Pi camera)

Very common across all four classes:

- Rotation with **black triangular corners**
- Rotation/shear with **edge-smear / clamp padding** (streaked borders)
- Occasional **reflect/mirror padding**
- Color jitter that goes **too blue / too warm / oversaturated**
- Aggressive crops that leave little context

These must not be treated as extra independent real photos. If combined-training
beats original-only by a lot, suspect artifact learning (planned Experiment C).

## Possible mislabels / odd samples (flagged, not edited)

- Hash mixed-class groups (see `outputs/review/potential_mislabels.png`):
  - `g003594`: same cauliflower head labeled `bacterial_spot_rot` and `no_disease` (original–original). **Strong label conflict.**
  - `g002984`: same head labeled `no_disease` (augmented) vs `bacterial_spot_rot` (original). **Strong label conflict.**
  - `g001701`: downy-mildew leaf vs no-disease curd — likely a **hash false positive**, not a true duplicate.
- `No disease(416).png`: healthy **leaf on soil**, while most `no_disease` images are **curd shots**. Not necessarily wrong, but distribution shift inside the class.
- Downy mildew originals with watermarks: model could latch onto watermark digits if they are class-specific.

## Original vs augmented pairing quality

Filename numbers **do not** identify the source photo (`Bacterial Spot. (1).jpg` is not
the same plant as `Bacterial spot rot._ (1).png`). Same-number `.jpg` vs `.jpeg`
originals in downy mildew / no_disease are **different photos**, not dual encodings.

Perceptual hash (phash≤8 or phash≤12 & dhash≤10):

- 172 multi-image groups, 422 images in those groups
- only **25** groups contain both an original and an augmented file
- nearest original→augmented phash Hamming is typically **14–18 bits** (same as
  “another same-class plant”, not a transformed copy)

Interpretation: most augmented files are **not near-duplicates** of these originals
under standard hashes. They may still come from a related collection after heavy
geometry/color transforms that break pHash. Residual leakage is therefore
**possible but not proven**. The frozen split still keeps every detected group
together. Do not join on filename index.

Exact byte duplicates (MD5): **11 groups / 24 images**, all among originals with
*different* filenames (copied files). Grouped correctly.

## Quality heuristics

No corrupt, blank, grayscale, or extreme-brightness files.
41 augmented images flagged `augmentation_artifact_suspect` (high saturation /
very high contrast) — see `outputs/review/suspicious_images.png`.
