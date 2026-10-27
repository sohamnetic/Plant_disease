"""ONNX deployment runtime (src/onnx_runtime.py): same answers as the PyTorch app.

Skipped when deploy_models/ has not been exported (python src/onnx_runtime.py export).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_runtime import MODELS_DIR, OnnxClassifier, OnnxSeg, _nms, build_engine  # noqa: E402
from plant_runtime import PlantEngine, plant_public_result  # noqa: E402

INSECT_PHOTO = ROOT / "data_external/raw/x995snz7p3/Processed Image/Insect Hole/IMG (51).jpg"
needs_models = pytest.mark.skipif(not (MODELS_DIR / "classifier" / "calibration.json").is_file(),
                                  reason="ONNX models not exported")


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], np.float32)
    keep = _nms(boxes, np.array([0.9, 0.8, 0.7]), 0.7)
    assert list(keep) == [0, 2]


@needs_models
def test_classifier_matches_pytorch():
    from model_runtime import V2Engine

    im = Image.open(INSECT_PHOTO).convert("RGB")
    t, o = V2Engine(), OnnxClassifier(MODELS_DIR / "classifier")
    rt, ro = t.forward_images([im]), o.forward_images([im])
    assert np.abs(rt["cls"].numpy() - ro["cls"]).max() < 1e-3
    assert t.interpret(rt, 0, [])["prediction"] == o.interpret(ro, 0, [])["prediction"]


@needs_models
def test_outline_masks_match_ultralytics():
    from seg_runtime import WORK_SIDE, SegEngine

    im = Image.open(INSECT_PHOTO).convert("RGB")
    im.thumbnail((WORK_SIDE, WORK_SIDE))
    rgb = np.asarray(im)
    a = SegEngine()._predict(SegEngine().model, rgb)
    b = OnnxSeg(MODELS_DIR / "seg.onnx", None)._predict(MODELS_DIR / "seg.onnx", rgb)
    assert [n for n, _ in a] == [n for n, _ in b]
    for (_, x), (_, y) in zip(a, b):
        assert (x & y).sum() / max((x | y).sum(), 1) > 0.99


@needs_models
def test_onnx_app_gives_the_same_answer():
    im = Image.open(INSECT_PHOTO)
    a = plant_public_result(PlantEngine().predict_pil(im, with_overlay=False))
    b = plant_public_result(build_engine().predict_pil(im, with_overlay=False))
    assert (a["status"], a["prediction"]) == (b["status"], b["prediction"]) == ("confident", "insect_damage")
    assert abs(a["plant"]["insect"]["eaten_pct"] - b["plant"]["insect"]["eaten_pct"]) < 0.5
