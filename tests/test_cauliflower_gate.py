"""The app must refuse photos that are not cauliflower (src/cauliflower_gate.py).

  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plant_runtime import PlantEngine, plant_public_result  # noqa: E402

USER_BSR = ROOT / "outputs" / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"
USER_YOUNG_HEAD = ROOT / "outputs" / "review" / "user_photos" / "2_young_head_label_unknown.jpg"
WALLPAPER = Path(r"C:/Users/lenovo/Downloads/wp8926127-aot-4k-wallpapers.jpg")  # the photo that exposed the bug


@pytest.fixture(scope="module")
def eng() -> PlantEngine:
    return PlantEngine()


def library_images() -> list[Path]:
    out = []
    try:
        import skimage
        d = Path(skimage.__file__).parent / "data"
        out += [d / n for n in ("astronaut.png", "chelsea.png", "coffee.png", "rocket.jpg", "grass.png", "ihc.png")]
    except ImportError:
        pass
    try:
        import sklearn
        out += [Path(sklearn.__file__).parent / "datasets" / "images" / n for n in ("china.jpg", "flower.jpg")]
    except ImportError:
        pass
    return [p for p in out if p.exists()]


def assert_refused(r: dict) -> None:
    assert r["status"] == "not_cauliflower"
    assert not r["reliable"] and r["prediction"] is None and r["confidence"] is None
    assert "plant" not in r and "overlay_jpeg_b64" not in r and "advice" not in r
    assert "not a cauliflower photo" in r["message"]


@pytest.mark.parametrize("path", library_images(), ids=lambda p: p.name)
def test_everyday_images_are_not_cauliflower(eng, path):
    with Image.open(path) as im:
        assert_refused(plant_public_result(eng.predict_pil(im)))


def test_random_noise_is_not_cauliflower(eng):
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    assert_refused(plant_public_result(eng.predict_pil(noise)))


@pytest.mark.skipif(not WALLPAPER.exists(), reason="user's wallpaper not on this machine")
def test_anime_wallpaper_is_not_cauliflower(eng):
    with Image.open(WALLPAPER) as im:
        assert_refused(plant_public_result(eng.predict_pil(im)))


@pytest.mark.parametrize("path", [USER_BSR, USER_YOUNG_HEAD], ids=lambda p: p.name)
def test_real_cauliflower_photos_pass_the_gate(eng, path):
    with Image.open(path) as im:
        r = plant_public_result(eng.predict_pil(im))
    assert r["status"] != "not_cauliflower"
    assert r["cauliflower_score"] >= r["cauliflower_threshold"]
