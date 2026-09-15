"""Shared paths and constants for the leaf-disease project.

Do not hardcode dataset counts. Discover files from DISK at runtime.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"

ORIGINAL_ROOT = DATASET_ROOT / "Original Dataset" / "Original Dataset"
AUGMENTED_ROOT = DATASET_ROOT / "Augmented Dataset" / "Augmented Dataset"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
REVIEW_DIR = OUTPUTS_DIR / "review"
SPLITS_DIR = PROJECT_ROOT / "splits"

RANDOM_SEED = 42

# Folder name on disk -> canonical class id used in splits / reports
FOLDER_TO_CLASS = {
    "Bacterial spot rot": "bacterial_spot_rot",
    "Black Rot": "black_rot",
    "Downy Mildew": "downy_mildew",
    "No disease": "no_disease",
}

CLASS_DISPLAY = {
    "bacterial_spot_rot": "bacterial_spot_rot",
    "black_rot": "black_rot",
    "downy_mildew": "downy_mildew",
    "no_disease": "no_disease",
}

CLASS_ORDER = [
    "bacterial_spot_rot",
    "black_rot",
    "downy_mildew",
    "no_disease",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp", ".tif", ".tiff", ".webp"}

# Perceptual-hash grouping (64-bit phash / dhash)
PHASH_NEAR_DUP_MAX = 8
DHASH_NEAR_DUP_MAX = 8
# Slightly looser corroborating rule for stronger augmentations
PHASH_LOOSE_MAX = 12
DHASH_LOOSE_MAX = 10

# Quality heuristics (pixel values on 0-255)
BLANK_STD_MAX = 5.0
EXTREME_DARK_MEAN_MAX = 15.0
EXTREME_BRIGHT_MEAN_MIN = 240.0
VERY_LOW_RES_MIN_SIDE = 64
UNUSUAL_ASPECT_MAX = 3.0
HIGH_SATURATION_MEAN = 180.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
