"""Regression tests for the Phase 3 segmentation runtime (affected-area measurement).

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

from seg_runtime import DEFAULT_WEIGHTS, SegEngine, measure, stage_for  # noqa: E402

USER_BSR = ROOT / "outputs" / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"


@pytest.fixture(scope="module")
def seg() -> SegEngine:
    return SegEngine(DEFAULT_WEIGHTS)


def test_stage_rules():
    assert stage_for(0.0, 0) == "none"
    assert stage_for(3.0, 5) == "early"
    assert stage_for(8.0, 2) == "early"      # few lesions, still small
    assert stage_for(8.0, 6) == "moderate"
    assert stage_for(30.0, 1) == "severe"    # one huge lesion is not "early"
    assert stage_for(60.0, 4) == "critical"


def test_fully_diseased_leaf_is_not_zero_percent():
    """A leaf covered entirely by lesion (no healthy leaf pixels) must count as ~100%."""
    lesion = np.zeros((100, 100), bool)
    lesion[20:80, 20:80] = True
    res = measure([], lesion)
    assert res["worst_region_pct"] > 95


def test_user_photo_rotten_head_is_measured(seg):
    r = seg.predict_pil(Image.open(USER_BSR), with_overlay=False)
    curd = [x for x in r["regions"] if x["type"] == "curd"]
    assert curd, "head not found"
    assert max(x["affected_pct"] for x in curd) > 25
    assert r["stage"] in ("severe", "critical")


def test_blank_image_has_no_lesions(seg):
    r = seg.predict_pil(Image.new("RGB", (400, 400), (60, 130, 50)), with_overlay=False)
    assert r["stage"] == "none"
