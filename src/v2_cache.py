"""Build a resized image cache for v2 training (local only, gitignored).

Every image in splits/{train,val,test}.csv is EXIF-transposed, converted to RGB
and downscaled so the long side is at most CACHE_MAX_SIDE. Aspect ratio is kept
(letterboxing happens later in the transforms). Nothing in dataset/ is modified.

  python src/v2_cache.py                       # the frozen splits
  python src/v2_cache.py splits_v3/all.csv     # any manifest with a filepath column
"""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

CACHE_DIR = cfg.PROJECT_ROOT / "cache" / "img640"
CACHE_MAX_SIDE = 640


def cache_path_for(filepath: str) -> Path:
    key = hashlib.md5(filepath.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.jpg"


def _one(filepath: str) -> str:
    out = cache_path_for(filepath)
    if out.is_file():
        return "skip"
    with Image.open(cfg.PROJECT_ROOT / filepath) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((CACHE_MAX_SIDE, CACHE_MAX_SIDE), Image.Resampling.LANCZOS)
        im.save(out, "JPEG", quality=95)
    return "ok"


def cache_files(paths: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_one, paths))
    print(f"cached {results.count('ok')} new, {results.count('skip')} existing -> {CACHE_DIR}")


def main() -> None:
    manifests = [Path(a) for a in sys.argv[1:]]
    if not manifests:
        manifests = [cfg.SPLITS_DIR / f"{split}.csv" for split in ("train", "val", "test")]
    paths: list[str] = []
    for m in manifests:
        paths += pd.read_csv(m)["filepath"].tolist()
    cache_files(sorted(set(paths)))


if __name__ == "__main__":
    main()
