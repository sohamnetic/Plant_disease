"""Regression tests for the deployed model through the real API runtime.

They pin behaviour that was fixed on purpose, so a future retrain cannot silently
undo it. Run from the project root (the model files must be present):

  python -m pytest tests -q

Photos from dataset/ are optional (the dataset is not in git); those tests skip
when it is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_runtime import DEFAULT_EXP, RELIABLE_STATUSES, V2Engine, public_result  # noqa: E402

ORIGINALS = ROOT / "dataset" / "Original Dataset" / "Original Dataset"
USER_BSR = ROOT / "outputs" / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"


@pytest.fixture(scope="module")
def engine() -> V2Engine:
    return V2Engine(DEFAULT_EXP)


def predict(engine: V2Engine, im: Image.Image) -> dict:
    return public_result(engine.predict_pil(im))


def test_user_photo_head_with_leaves_is_bacterial_spot_rot(engine):
    """User's field photo: rotten head surrounded by healthy leaves. v2 said downy mildew."""
    r = predict(engine, Image.open(USER_BSR))
    assert r["reliable"], r["message"]
    assert r["prediction"] == "bacterial_spot_rot"


def test_noise_is_never_a_diagnosis(engine):
    rng = np.random.default_rng(0)
    for _ in range(3):
        r = predict(engine, Image.fromarray(rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)))
        assert not r["reliable"]
        assert r["prediction"] is None and r["severity"] is None


def test_flat_colour_is_never_a_diagnosis(engine):
    for colour in [(60, 130, 50), (200, 200, 200), (120, 80, 40)]:
        r = predict(engine, Image.new("RGB", (400, 400), colour))
        assert not r["reliable"]


def test_blank_white_is_overexposed(engine):
    r = predict(engine, Image.new("RGB", (400, 400), (255, 255, 255)))
    assert r["status"] == "retake_photo"
    assert "overexposed" in r["status_reason"]


def test_very_dark_photo_asks_for_retake(engine):
    r = predict(engine, Image.fromarray(np.full((400, 400, 3), 6, np.uint8)))
    assert r["status"] == "retake_photo"


def test_unreliable_answers_hide_the_guess(engine):
    r = predict(engine, Image.new("RGB", (400, 400), (0, 0, 255)))
    assert not r["reliable"]
    assert r["unreliable_guess"]["note"].startswith("NOT a diagnosis")


@pytest.mark.skipif(not ORIGINALS.is_dir(), reason="dataset/ not present")
@pytest.mark.parametrize(
    "rel, expected",
    [
        ("Bacterial spot rot/Bacterial Spot. (10).jpg", "bacterial_spot_rot"),
        ("Black Rot/Black rot(10).jpeg", "black_rot"),
        ("Downy Mildew/Downy Mildew. (46).jpg", "downy_mildew"),
        ("No disease/No disease. (99).jpg", "no_disease"),
    ],
)
def test_known_dataset_photos(engine, rel, expected):
    r = predict(engine, Image.open(ORIGINALS / rel))
    assert r["status"] in RELIABLE_STATUSES
    assert r["prediction"] == expected
    if expected == "no_disease":
        assert r["severity"]["grade"] == 0
    else:
        assert 1 <= r["severity"]["grade"] <= 4


@pytest.mark.skipif(not ORIGINALS.is_dir(), reason="dataset/ not present")
def test_healthy_leaf_on_white_backdrop_is_not_overexposed(engine):
    """A well-lit subject on a white backdrop must not be rejected as overexposed."""
    with Image.open(ORIGINALS / "No disease/No disease. (99).jpg") as im:
        im = im.convert("RGB")
        im.thumbnail((300, 300))
    canvas = Image.new("RGB", (600, 600), (255, 255, 255))
    canvas.paste(im, (150, 150))
    r = predict(engine, canvas)
    assert "overexposed" not in (r.get("status_reason") or "")


@pytest.mark.xfail(strict=True, reason="known limitation: no non-cauliflower training data, so a dried "
                   "leaf of another plant on white paper looks like black rot (OOD score ~95 vs limit ~314)")
def test_non_cauliflower_dried_leaf_is_rejected(engine):
    r = predict(engine, Image.open(ROOT / "pic.jpg"))
    assert not r["reliable"]
