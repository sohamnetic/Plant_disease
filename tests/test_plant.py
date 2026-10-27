"""Regression tests for Phase 4 region-based diagnosis (classifier + segmentation).

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
from seg_runtime import measure  # noqa: E402

ORIGINALS = ROOT / "dataset" / "Original Dataset" / "Original Dataset"
USER_BSR = ROOT / "outputs" / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"


@pytest.fixture(scope="module")
def eng() -> PlantEngine:
    return PlantEngine()


def test_head_rot_not_counted_on_surrounding_leaves():
    """A ring of leaves around a head must not absorb the head (or its rot) when holes are filled."""
    leaf = np.ones((100, 100), bool)
    leaf[30:70, 30:70] = False
    curd = np.zeros((100, 100), bool)
    curd[30:70, 30:70] = True
    lesion = np.zeros((100, 100), bool)
    lesion[35:55, 35:55] = True
    res = measure([("leaf", leaf), ("curd", curd)], lesion)
    leaf_r, curd_r = res["regions"]
    assert leaf_r["affected_pct"] == 0.0
    assert curd_r["affected_pct"] == pytest.approx(25.0, abs=1)


def test_user_photo_head_is_bsr_and_leaves_not_flagged(eng):
    """The Phase 4 gate: v2 called this photo downy mildew because of the leaves."""
    r = plant_public_result(eng.predict_pil(Image.open(USER_BSR)))
    assert r["reliable"] and r["prediction"] == "bacterial_spot_rot"
    parts = r["plant"]["regions"]
    assert any(p["type"] == "curd" and p["disease"] == "bacterial_spot_rot" for p in parts)
    assert all(p["disease"] == "no_disease" for p in parts if p["type"] == "leaf")
    assert r["plant"]["stage"] in ("severe", "critical")
    assert r["overlay_jpeg_b64"]


def test_unreliable_photo_has_no_plant_section(eng):
    r = plant_public_result(eng.predict_pil(Image.new("RGB", (400, 400), (0, 0, 255))))
    assert not r["reliable"]
    assert "plant" not in r and "overlay_jpeg_b64" not in r


@pytest.mark.skipif(not ORIGINALS.is_dir(), reason="dataset/ not present")
def test_healthy_head_is_healthy_with_stage_none(eng):
    r = plant_public_result(eng.predict_pil(Image.open(ORIGINALS / "No disease/No disease. (99).jpg")))
    assert r["prediction"] == "no_disease"
    assert r["plant"]["stage"] == "none"


@pytest.mark.skipif(not ORIGINALS.is_dir(), reason="dataset/ not present")
def test_early_black_rot_still_detected(eng):
    r = plant_public_result(eng.predict_pil(Image.open(ORIGINALS / "Black Rot/Black rot(10).jpeg")))
    assert r["prediction"] == "black_rot"
    assert r["plant"]["stage"] == "early"


def test_head_disease_without_outlined_head_does_not_blame_the_leaves(eng):
    # user photo 2 (young head among leaves; true label unknown): the head is not outlined, the
    # photo-level verdict is bacterial spot rot. The leaves must not be reported as rotten.
    p = ROOT / "outputs" / "review" / "user_photos" / "2_young_head_label_unknown.jpg"
    r = plant_public_result(eng.predict_pil(Image.open(p), with_overlay=False))
    if r["reliable"] and r["prediction"] == "bacterial_spot_rot":
        assert all(g["disease"] != "bacterial_spot_rot" for g in r["plant"]["regions"] if g["type"] == "leaf")
        assert "head was not outlined" in r["message"]
