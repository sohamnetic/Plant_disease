"""Insect damage (steps I-2..I-5): hole measurement, the insect category and its message.

Model-based tests use the deployed defaults (classifier DEFAULT_EXP, outline DEFAULT_WEIGHTS)
and photos from the insect test split, which no model was trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advice import advice_for  # noqa: E402
from plant_runtime import PlantEngine, plant_public_result  # noqa: E402
from seg_runtime import SegEngine, insect_stage_for, measure  # noqa: E402

INSECT_TEST = ROOT / "splits_v4" / "insect_test.csv"
REVIEW = ROOT / "outputs" / "insect_seg" / "review.csv"


def test_eaten_pct_counts_holes_inside_the_leaf_only():
    leaf = np.zeros((100, 100), bool)
    leaf[10:90, 10:90] = True                     # 6400 px leaf, holes included
    holes = np.zeros_like(leaf)
    holes[30:40, 30:40] = True                    # 100 px hole on the leaf
    holes[0:5, 0:5] = True                        # outside the leaf: ignored
    m = measure([("leaf", leaf)], np.zeros_like(leaf), holes)
    r = m["regions"][0]
    assert r["eaten_pct"] == pytest.approx(100 * 100 / 6400, abs=0.01)
    assert r["hole_count"] == 1
    assert m["insect_stage"] == "early"
    assert r["affected_pct"] == 0.0               # holes are not disease lesions


def test_no_holes_means_no_insect_stage():
    leaf = np.zeros((50, 50), bool)
    leaf[5:45, 5:45] = True
    m = measure([("leaf", leaf)], np.zeros_like(leaf))  # older outline weights: no hole class
    assert m["worst_region_eaten_pct"] == 0.0 and m["insect_stage"] == "none"


def test_insect_stage_rules():
    assert insect_stage_for(0.1, 3) == "none"
    assert insect_stage_for(2.0, 0) == "none"
    assert insect_stage_for(2.0, 2) == "early"
    assert insect_stage_for(10.0, 5) == "moderate"
    assert insect_stage_for(20.0, 5) == "severe"
    assert insect_stage_for(40.0, 5) == "critical"


def test_insect_advice_has_no_pathogen_hygiene():
    a = advice_for("insect_damage", "moderate")
    assert a["actions"] and a["general"] == [] and a["sources"]


def _cls(pred: str, status: str = "confident", p_insect: float = 0.9) -> dict:
    return {"prediction": pred, "status": status, "class_probabilities": {"insect_damage": p_insect}}


def test_insect_second_opinion_never_overrides_a_confident_disease():
    rule = PlantEngine._insect_second_opinion
    assert not rule({"whole": _cls("black_rot"), "insect_whole": _cls("insect_damage")}, "confident")
    a = {"whole": _cls("black_rot", "uncertain"), "insect_whole": _cls("insect_damage")}
    assert rule(a, "uncertain")                         # unsure disease, very sure insect
    a = {"whole": _cls("black_rot", "uncertain"), "insect_whole": _cls("insect_damage", p_insect=0.7)}
    assert not rule(a, "uncertain")
    a = {"whole": _cls("no_disease"), "insect_whole": _cls("insect_damage")}
    assert rule(a, "confident")
    assert not rule(a, "retake_photo")
    a = {"whole": _cls("downy_mildew", "out_of_scope"), "insect_whole": _cls("insect_damage")}
    assert rule(a, "out_of_scope")                      # abstained: the insect answer may stand
    a = {"whole": _cls("no_disease"), "insect_whole": _cls("insect_damage", "uncertain")}
    assert not rule(a, "confident")                     # an unsure insect answer changes nothing
    assert not rule({"whole": _cls("no_disease"), "insect_whole": None}, "confident")


@pytest.fixture(scope="module")
def eng():
    if not INSECT_TEST.is_file():
        pytest.skip("insect test split not built")
    e = PlantEngine()
    if ("insect_damage" not in e.classes and e.insect_cls is None) or not e.seg.has_holes:
        pytest.skip("deployed models predate the insect category")
    return e


def _clear_hole_photos(n: int) -> list[str]:
    rev = pd.read_csv(REVIEW)
    test = rev[(rev["split"] == "insect_test") & (rev["review"] == "K")]
    return test.sort_values("eaten_pct", ascending=False)["filepath"].head(n).tolist()


def test_clear_insect_holes_are_measured(eng):
    hits = 0
    for f in _clear_hole_photos(5):
        r = eng.predict_pil(Image.open(ROOT / f), with_overlay=False)
        if r.get("plant") and r["plant"]["insect"]["found"]:
            hits += 1
            assert r["plant"]["insect"]["eaten_pct"] > 0
    assert hits >= 4, "the outline model should find holes on clearly eaten held-out leaves"


def test_insect_damage_message(eng):
    for f in _clear_hole_photos(5):
        r = eng.predict_pil(Image.open(ROOT / f), with_overlay=False)
        p = plant_public_result(r)
        if p.get("prediction") == "insect_damage" and p["reliable"]:
            assert p["message"].startswith(("Insect damage", "Not sure. Insect damage"))
            assert "eaten" in p["message"]
            return
    pytest.fail("no clearly eaten held-out leaf was called insect damage")
