"""Tests for Phase 6: advice table and plant tracking, plus the BSR stage interim fix.

  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advice import ADVICE, advice_for  # noqa: E402
from plant_runtime import PlantEngine, plant_public_result  # noqa: E402
from tracking import history, record_and_trend, valid_plant_id  # noqa: E402

USER_BSR = ROOT / "outputs" / "review" / "user_photos" / "1_bacterial_spot_rot.jpg"


def test_every_disease_and_stage_has_advice():
    for d in ("black_rot", "downy_mildew", "bacterial_spot_rot"):
        for stage in ("early", "moderate", "severe", "critical"):
            a = advice_for(d, stage)
            assert a["actions"] and a["sources"] and a["expert_verified"] is False
    assert advice_for("no_disease", "none")["actions"]
    assert advice_for("unknown", "early") is None


def test_lookalike_diseases_say_how_to_tell_them_apart():
    # web probe: classic black rot is called downy mildew, downy mildew on heads is called bacterial
    # spot rot; each answer must tell the user how to check against its lookalike
    for d, other in (("black_rot", "downy mildew"), ("downy_mildew", "black rot"), ("bacterial_spot_rot", "downy mildew")):
        assert other in advice_for(d, "early")["check_first"].lower()
    assert advice_for("no_disease", "none")["check_first"] is None


def test_advice_has_no_chemical_doses():
    text = str(ADVICE).lower()
    for word in ("ml/l", "g/l", "kg/ha", "ppm", "mancozeb", "copper oxychloride", "streptomycin"):
        assert word not in text


def _result(pred, stage, pct, spots=0):
    return {"prediction": pred, "confidence": 0.95, "status": "confident",
            "plant": {"stage": stage, "affected_pct": pct, "lesion_count": spots}}


def test_tracking_alerts(tmp_path):
    db = tmp_path / "t.db"
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    first = record_and_trend("p1", _result("no_disease", "none", 0.0), "a.jpg", db, t0)
    assert first["previous"] is None and first["alerts"] == []
    second = record_and_trend("p1", _result("black_rot", "early", 2.0, 3), "b.jpg", db, t0 + timedelta(days=2))
    assert any("New disease" in a for a in second["alerts"])
    third = record_and_trend("p1", _result("black_rot", "moderate", 12.0, 9), "c.jpg", db, t0 + timedelta(days=4))
    assert any("Stage got worse" in a for a in third["alerts"])
    assert any("grew by 10" in a for a in third["alerts"])
    assert third["affected_pct_change_per_day"] == pytest.approx(5.0)
    assert len(history("p1", db)) == 3 and history("other", db) == []


def test_plant_id_validation():
    assert valid_plant_id("row3-plant_12.a")
    assert not valid_plant_id("../etc/passwd") and not valid_plant_id("") and not valid_plant_id("a" * 65)


@pytest.fixture(scope="module")
def eng() -> PlantEngine:
    return PlantEngine()


def test_bsr_stage_comes_from_model_grade_and_has_advice(eng):
    r = plant_public_result(eng.predict_pil(Image.open(USER_BSR), with_overlay=False))
    assert r["prediction"] == "bacterial_spot_rot"
    assert r["plant"]["stage_source"] == "classifier_grade"
    assert r["advice"]["disease"] == "bacterial_spot_rot" and r["advice"]["actions"]


def test_unreliable_photo_gets_no_advice(eng):
    r = plant_public_result(eng.predict_pil(Image.new("RGB", (400, 400), (0, 0, 255))))
    assert not r["reliable"] and "advice" not in r
