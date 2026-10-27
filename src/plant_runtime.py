"""Phase 4: region-based diagnosis (classifier v3 + YOLO11-seg).

Two steps:
  analyse(photo)  classifier on the whole photo; segmentation finds every leaf /
                  head (curd) and lesion and measures % affected per part; each
                  large part is cropped and classified on its own (a part that
                  spans most of the photo is classified with the other diagnosed
                  parts blanked out, so a rotten head does not decide the leaves)
  decide(...)     per-part verdicts + one plant result, under a POLICY

Policy options (compared on a validation set, see src/select_phase4_policy.py):
  seg_veto     True: a part with no measured lesions is healthy whatever the
               classifier says. False: the classifier may still name a disease
               there ("spots too small or faint to measure").
  grade_check  True: measured stage and classifier grade 2+ stages apart -> uncertain.
Fixed rules (both policies):
  * spots under early_max_pct of a part count as disease only if the classifier
    names a disease for that part; otherwise "small unidentified spots"
  * larger lesions the classifier cannot name -> "unknown" -> uncertain
  * parts naming different diseases -> uncertain, all listed

Before all of this, the cauliflower gate (src/cauliflower_gate.py, CLIP) checks that
the photo shows a cauliflower plant, leaf or head at all; if not, the result is
status "not_cauliflower" and neither the classifier nor YOLO runs.

Insect damage (step I-5): with a v4 classifier "insect_damage" is a diagnosis like the
diseases, and with seg_002 weights the outline model measures insect holes.
The v4 classifier alone called too many black rot leaves insect damage (external test
gate failed), so by default it runs as a second opinion (`insect_exp`): the 4-class
classifier still decides, and the diagnosis becomes insect damage only when the 5-class
one is confident it is insect damage and the 4-class answer is "no disease", an
abstention, or a disease it is unsure of (p(insect) >= 0.8). A confident 4-class disease
answer is never changed (outputs/insect_seg/hybrid/, src/insect_hybrid_eval.py). The
plant result always carries `plant.insect` (% of the worst leaf eaten, hole count,
draft stage), so holes are reported next to a disease too. For an insect_damage
diagnosis the stage comes from that measurement (edge bites are not measured).

Output keeps every V2Engine field (public_result keeps working) plus `plant` and
`overlay_jpeg_b64`.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_runtime import (
    DEFAULT_EXP,
    RELIABLE_STATUSES,
    V2Engine,
    _pretty,
    public_result,
    quality_issues,
    quality_metrics,
)
from seg_runtime import DEFAULT_HOLE_WEIGHTS, DEFAULT_WEIGHTS, STAGES, WORK_SIDE, SegEngine, overlay
from advice import advice_for
from v2_common import FILL, GRADE_LABELS

POLICY_FILE = Path(__file__).resolve().parent / "phase4_policy.json"
DEFAULT_POLICY = {"seg_veto": True, "grade_check": True}
MIN_CLASSIFY_FRAC = 0.03       # parts smaller than this share of the photo are not cropped
WHOLE_PHOTO_FRAC = 0.70        # a part whose box covers this much of the photo is not cropped
MAX_CROPS = 6
CROP_PAD = 0.08
STAGE_INDEX = {s: i for i, s in enumerate(STAGES["stages"])}
# Diseases whose plant stage comes from the classifier's severity grade instead of the
# measured %: the segmentation model misses dark rot on the head (Phase 5 finding).
STAGE_FROM_GRADE = {"bacterial_spot_rot"}  # none=0 ... critical=4 (same scale as grades)
INSECT = "insect_damage"
DEFAULT_INSECT_EXP = DEFAULT_EXP.parent / "experiment_007_v4_insect"
INSECT_OVERRIDE_MIN_PROB = 0.8  # a confident insect answer this sure also replaces an unsure disease answer


def load_policy() -> dict:
    if POLICY_FILE.is_file():
        return {**DEFAULT_POLICY, **json.loads(POLICY_FILE.read_text())["policy"]}
    return dict(DEFAULT_POLICY)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class PlantEngine:
    def __init__(self, exp_dir: Path = DEFAULT_EXP, seg_weights: Path = DEFAULT_WEIGHTS,
                 device: "torch.device | None" = None, seg_device: str = "cpu", policy: dict | None = None,
                 use_gate: bool = True, insect_exp: Path | None = DEFAULT_INSECT_EXP):
        from cauliflower_gate import CauliflowerGate

        self.cls = V2Engine(exp_dir, device=device)
        # second opinion only when the main classifier has no insect class
        use_insect = insect_exp is not None and INSECT not in self.cls.classes and (Path(insect_exp) / "calibration.json").is_file()
        self.insect_cls = V2Engine(insect_exp, device=self.cls.device) if use_insect else None
        self.gate = CauliflowerGate(self.cls.device) if use_gate else None
        self.seg = SegEngine(seg_weights, device=seg_device)
        self.classes = self.cls.classes
        self.checkpoints = (self.cls.checkpoints + (self.insect_cls.checkpoints if self.insect_cls else [])
                            + [f"seg:{Path(seg_weights).parent.parent.name}"]
                            + ([f"holes:{DEFAULT_HOLE_WEIGHTS.parent.parent.name}"] if self.seg.hole_model is not None else []))
        self.device = self.cls.device
        self.policy = policy or load_policy()

    @classmethod
    def from_engines(cls, classifier, seg, gate=None, insect_cls=None, policy: dict | None = None) -> "PlantEngine":
        """Same decisions from engines built elsewhere (the ONNX deployment, src/onnx_runtime.py)."""
        self = cls.__new__(cls)
        self.cls, self.seg, self.gate, self.insect_cls = classifier, seg, gate, insect_cls
        self.classes = classifier.classes
        self.checkpoints = (classifier.checkpoints + (insect_cls.checkpoints if insect_cls else []) + seg.checkpoints)
        self.device = "cpu"
        self.policy = policy or load_policy()
        return self

    def _classify(self, ims: list[Image.Image]) -> list[dict]:
        raw = self.cls.forward_images(ims)
        return [self.cls.interpret(raw, i, []) for i in range(len(ims))]

    # ------------------------------------------------------------------ analyse
    def analyse(self, im: Image.Image) -> dict:
        im = ImageOps.exif_transpose(im).convert("RGB")
        q = quality_metrics(im)
        issues = quality_issues(q, self.cls.cal["quality"])
        work = im.copy()
        work.thumbnail((WORK_SIDE, WORK_SIDE))
        rgb = np.asarray(work)
        m, lesion = self.seg.measure_all(rgb)
        H, W = rgb.shape[:2]
        k = im.width / W

        order = sorted(range(len(m["regions"])), key=lambda i: -m["regions"][i]["tissue_px"])
        crops, crop_of, large = [im], {}, []
        for i in order:
            r = m["regions"][i]
            x0, y0, x1, y1 = _bbox(r["_mask"])
            if r["tissue_px"] < MIN_CLASSIFY_FRAC * W * H or len(crops) > MAX_CROPS:
                continue
            if (x1 - x0) * (y1 - y0) / (W * H) >= WHOLE_PHOTO_FRAC:
                large.append(i)
                continue
            pw, ph = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
            box = (max(0, (x0 - pw) * k), max(0, (y0 - ph) * k), min(im.width, (x1 + pw) * k), min(im.height, (y1 + ph) * k))
            crop_of[i] = len(crops)
            crops.append(im.crop(tuple(int(v) for v in box)))
        # A part spanning most of the photo (e.g. all the leaves around a head) would
        # inherit the whole-photo answer, which the head dominates: blank the other parts.
        if crop_of:
            others = np.zeros((H, W), bool)
            for j in crop_of:
                others |= m["regions"][j]["_mask"]
            for i in large:
                blank = others & ~m["regions"][i]["_mask"]
                if blank.any():
                    a = rgb.copy()
                    a[blank] = FILL
                    crop_of[i] = len(crops)
                    crops.append(Image.fromarray(a))
        results = self._classify(crops)
        insect_whole = None
        if self.insect_cls is not None:
            raw = self.insect_cls.forward_images([im])
            insect_whole = self.insect_cls.interpret(raw, 0, [])
        parts = []
        for i, r in enumerate(m["regions"]):
            c = results[crop_of[i]] if i in crop_of else results[0]
            decided_by = ("photo_with_other_parts_blanked" if i in large else "region_crop") if i in crop_of else "whole_photo"
            parts.append({"type": r["type"], "measured_affected_pct": r["affected_pct"], "lesion_count": r["lesion_count"],
                          "measured_stage": r["stage"], "cls": c, "decided_by": decided_by,
                          "eaten_pct": r["eaten_pct"], "hole_count": r["hole_count"], "insect_stage": r["insect_stage"],
                          "tissue_share_of_photo_pct": round(100 * r["tissue_px"] / (W * H), 1), "_mask": r["_mask"]})
        return {"whole": results[0], "insect_whole": insect_whole, "issues": issues, "quality": q, "parts": parts,
                "crop_results": results[1:], "_rgb": rgb, "_m": m, "_lesion": lesion}

    # ------------------------------------------------------------------ decide
    def decide(self, a: dict, policy: dict | None = None) -> dict:
        pol = policy or self.policy
        whole, issues = a["whole"], a["issues"]
        whole_status = "retake_photo" if issues else whole["status"]
        regions = []
        for p in a["parts"]:
            c = p["cls"]
            names_disease = c["status"] in RELIABLE_STATUSES and c["prediction"] != "no_disease"
            pct, stage = p["measured_affected_pct"], p["measured_stage"]
            if names_disease and c["prediction"] == INSECT:
                # insect damage is measured as area eaten through, not as lesions
                pct, stage = p["eaten_pct"], p["insect_stage"]
            if stage == "none":
                if names_disease and not pol["seg_veto"]:
                    disease, verdict = c["prediction"], "disease recognised by the classifier; spots too small or faint to measure"
                else:
                    disease, verdict = "no_disease", "healthy: no lesions found"
            elif names_disease:
                disease, verdict = c["prediction"], "diseased"
            elif pct < STAGES["early_max_pct"]:
                disease, verdict = "no_disease", "small unidentified spots: no disease recognised on this part"
            else:
                disease, verdict = "unknown", "lesions found but the classifier cannot name the disease"
            regions.append({
                "type": p["type"], "disease": disease, "verdict": verdict,
                "classifier_prediction": c["prediction"], "classifier_confidence": c["confidence"],
                "classifier_status": c["status"], "decided_by": p["decided_by"],
                "affected_pct": pct if disease != "no_disease" else 0.0,
                "measured_affected_pct": pct, "lesion_count": p["lesion_count"],
                "stage": (stage if stage != "none" else "unmeasured") if disease != "no_disease" else "none",
                "tissue_share_of_photo_pct": p["tissue_share_of_photo_pct"],
                "eaten_pct": p["eaten_pct"], "hole_count": p["hole_count"], "insect_stage": p["insect_stage"],
            })

        out = dict(whole)
        diseased = [r for r in regions if r["disease"] != "no_disease"]
        named = sorted({r["disease"] for r in diseased if r["disease"] != "unknown"})
        region_reliable = any(res["status"] == "confident" for res in a["crop_results"])
        reasons = []
        if whole_status == "retake_photo":
            status = "retake_photo"
        elif whole_status in ("out_of_scope", "no_evidence") and not region_reliable:
            status = whole_status
        elif whole_status != "confident":
            status = "uncertain"
            reasons.append("the whole photo is unusual; judged from the individual leaves/head" if whole_status != "uncertain"
                           else whole["status_reason"])
        else:
            status = "confident"

        if pol.get("diagnosis") == "classifier":
            if self._insect_second_opinion(a, whole_status):
                whole, whole_status = a["insect_whole"], a["insect_whole"]["status"]
            return self._classifier_decides(a, whole, whole_status, regions)

        # No parts found at all: nothing to measure, the classifier alone decides.
        if not regions and not pol["seg_veto"] and whole["prediction"] != "no_disease":
            named = [whole["prediction"]]
            diseased = [{"disease": whole["prediction"], "affected_pct": None, "stage": "unmeasured",
                         "lesion_count": 0, "classifier_confidence": whole["confidence"], "decided_by": "whole_photo"}]

        if not diseased:
            prediction = "no_disease"
            confs = [r["classifier_confidence"] for r in regions if r["classifier_prediction"] == "no_disease"]
            conf = max(confs + ([whole["confidence"]] if whole["prediction"] == "no_disease" else []), default=whole["confidence"])
            if whole["prediction"] != "no_disease" and whole_status == "confident":
                status = "uncertain"
                reasons.append(f"the classifier sees {_pretty(whole['prediction'])} but no lesions were found")
            stage, pct, count = "none", 0.0, 0
        else:
            if len(named) == 1:
                prediction = named[0]
            elif named:
                prediction = max((r for r in diseased if r["disease"] != "unknown"),
                                 key=lambda r: r["affected_pct"] or 0)["disease"]
                status = "uncertain"
                reasons.append("different parts show different diseases: " + ", ".join(_pretty(d) for d in named))
            else:
                prediction = whole["prediction"] if whole["prediction"] != "no_disease" else "unknown"
                status = "uncertain"
                reasons.append("lesions found but the disease could not be named")
            if any(r["disease"] == "unknown" for r in diseased) and named:
                status = "uncertain"
                reasons.append("some affected parts could not be diagnosed")
            conf = max([r["classifier_confidence"] for r in diseased if r["disease"] == prediction], default=whole["confidence"])
            if prediction in self.classes and conf < self.cls.cal["confident_min_prob"]:
                status = "uncertain"
                reasons.append(f"confidence below {self.cls.cal['confident_min_prob']:.2f}")
            measured = [r for r in diseased if r["affected_pct"] is not None and r["stage"] != "unmeasured"]
            grade = whole["severity"]["grade"] if whole["prediction"] != "no_disease" else None
            if measured:
                worst = max(measured, key=lambda r: r["affected_pct"])
                stage, pct = worst["stage"], worst["affected_pct"]
                if pol["grade_check"] and grade is not None and abs(STAGE_INDEX[stage] - grade) >= 2:
                    status = "uncertain"
                    reasons.append(f"measured stage ({stage}) and estimated grade ({grade}) disagree")
            else:
                # Classifier sees the disease but the spots could not be measured: use its grade.
                stage, pct = (GRADE_LABELS[grade] if grade else "early"), None
            count = int(sum(r["lesion_count"] for r in diseased))

        if status in ("out_of_scope", "no_evidence", "retake_photo"):
            out["status"] = status
            if status == "retake_photo":
                out["status_reason"] = "; ".join(issues)
            return out | {"plant": None}

        decider = next((r for r in diseased if r["disease"] == prediction and r.get("decided_by") != "whole_photo"), None)
        if decider is not None:
            i = next(j for j, r in enumerate(regions) if r is decider)
            out["class_probabilities"] = a["parts"][i]["cls"]["class_probabilities"]
        out.update({
            "status": status,
            "status_reason": "; ".join(reasons) if reasons else "single diagnosis; affected parts measured",
            "prediction": prediction,
            "confidence": round(float(conf), 4),
            "prediction_set_95": sorted(set([prediction] + named + (whole["prediction_set_95"] if status != "confident" else []))),
            "plant": {
                "stage": stage,
                "affected_pct": pct,
                "stage_source": "measured" if pct is not None else "classifier_grade",
                "lesion_count": count,
                "diseases_found": named,
                "regions": regions,
                "stage_thresholds_expert_verified": STAGES["expert_verified"],
                "whole_photo_prediction": whole["prediction"],
                "whole_photo_confidence": whole["confidence"],
                "policy": pol,
            },
        })
        return out

    def _classifier_decides(self, a: dict, whole: dict, whole_status: str, regions: list[dict]) -> dict:
        """Diagnosis, confidence and status exactly as the whole-photo classifier (best on
        validation); the parts only add the measurement and per-part notes."""
        out = dict(whole)
        if whole_status in ("retake_photo", "out_of_scope", "no_evidence"):
            out["status"] = whole_status
            if whole_status == "retake_photo":
                out["status_reason"] = "; ".join(a["issues"])
            return out | {"plant": None}
        prediction = whole["prediction"]
        grade = whole["severity"]["grade"]
        insect = self._insect(a)
        if prediction == INSECT:
            if insect["found"]:
                stage, pct, count, source = insect["stage"], insect["eaten_pct"], insect["hole_count"], "measured"
            else:
                # feeding damage the outline model cannot measure (edge bites, scraped windows)
                stage, pct, count, source = "early", None, 0, "classifier_grade"
            for r in regions:
                if r["hole_count"] and r["insect_stage"] != "none":
                    # The part check (4-class model) reads holes as black rot (insect report,
                    # Section 2): an eaten part is reported as eaten, a named disease only as a note.
                    note = f"; the disease check also suggests {_pretty(r['disease'])}" if r["disease"] not in ("no_disease", INSECT) else ""
                    r.update({"disease": INSECT, "affected_pct": r["eaten_pct"], "stage": r["insect_stage"],
                              "verdict": f"eaten by insects{note}"})
                elif r["disease"] not in ("no_disease", INSECT):
                    r["verdict"] = f"possible {_pretty(r['disease'])} on this part (the plant is judged insect-damaged overall)"
        elif prediction == "no_disease":
            stage, pct, count, source = "none", 0.0, 0, "measured"
            for r in regions:  # the diagnosis says healthy: parts are reported, not counted
                if r["disease"] != "no_disease":
                    r["verdict"] = f"possible {_pretty(r['disease'])} on this part (the plant is judged healthy overall)"
        else:
            same = [r for r in regions if r["disease"] == prediction and r["stage"] != "unmeasured"]
            if same:
                worst = max(same, key=lambda r: r["affected_pct"])
                stage, pct, source = worst["stage"], worst["affected_pct"], "measured"
                count = int(sum(r["lesion_count"] for r in same))
            else:
                stage, pct, count, source = GRADE_LABELS[max(grade, 1)], None, 0, "classifier_grade"
            if prediction in STAGE_FROM_GRADE:
                # Measured % under-counts dark curd rot; the model's grade is far closer to
                # reviewed grades (non-test photos: 78% vs 30% exact). % kept for information.
                stage, source = GRADE_LABELS[max(grade, 1)], "classifier_grade"
                if not any(r["type"] == "curd" for r in regions):
                    # A head disease, but no head was outlined (young/small heads, cut-outs): the
                    # leaves only inherit the photo's verdict, so do not report them as rotten.
                    for r in regions:
                        if r["disease"] == prediction and r["decided_by"] == "whole_photo":
                            r["disease"], r["affected_pct"], r["stage"] = "no_disease", 0.0, "none"
                            r["verdict"] = (f"the photo is judged {_pretty(prediction)} (a head disease); "
                                            "the head was not outlined, so the rot could not be measured")
        other = sorted({r["disease"] for r in regions if r["disease"] not in ("no_disease", "unknown", INSECT, prediction)})
        out.update({
            "status": whole_status,
            "plant": {
                "stage": stage, "affected_pct": pct, "stage_source": source, "lesion_count": count,
                "diseases_found": sorted({prediction} - {"no_disease", INSECT} | set(other)),
                "other_part_findings": other,
                "insect": insect,
                "regions": regions,
                "stage_thresholds_expert_verified": STAGES["expert_verified"],
                "whole_photo_prediction": prediction, "whole_photo_confidence": whole["confidence"],
                "policy": {"diagnosis": "classifier"},
            },
        })
        return out

    @staticmethod
    def _insect_second_opinion(a: dict, whole_status: str) -> bool:
        """Insect damage from the 5-class classifier, only when it is confident and the 4-class
        answer is "no disease", an abstention, or a disease it is itself unsure of."""
        iw = a.get("insect_whole")
        if iw is None or whole_status == "retake_photo":
            return False
        if iw["status"] != "confident" or iw["prediction"] != INSECT:
            return False
        if a["whole"]["prediction"] == "no_disease" or whole_status not in RELIABLE_STATUSES:
            return True
        return whole_status != "confident" and iw["class_probabilities"][INSECT] >= INSECT_OVERRIDE_MIN_PROB

    @staticmethod
    def _insect(a: dict) -> dict:
        """Insect holes measured on the worst leaf/head (all zero for outline weights without holes)."""
        m = a["_m"]
        found = m["insect_stage"] != "none"
        return {"found": found, "eaten_pct": m["worst_region_eaten_pct"] if found else 0.0,
                "hole_count": m["hole_count"] if found else 0, "stage": m["insect_stage"],
                "edge_bites_measured": False,
                "stage_thresholds_expert_verified": STAGES["expert_verified"]}

    # ------------------------------------------------------------------ main
    def check_cauliflower(self, im: Image.Image) -> dict | None:
        """Gate result ({is_cauliflower, cauliflower_score, cauliflower_threshold}); None without a gate."""
        return self.gate.check(im) if self.gate is not None else None

    @staticmethod
    def not_cauliflower(g: dict) -> dict:
        return {"status": "not_cauliflower", "status_reason": "the photo does not show a cauliflower plant, leaf or head",
                "cauliflower_score": g["cauliflower_score"], "cauliflower_threshold": g["cauliflower_threshold"],
                "plant": None}

    def predict_pil(self, im: Image.Image, with_overlay: bool = True, policy: dict | None = None) -> dict:
        g = self.check_cauliflower(im)
        if g is not None and not g["is_cauliflower"]:
            return self.not_cauliflower(g)  # neither the classifier nor YOLO runs
        a = self.analyse(im)
        out = self.decide(a, policy)
        if g is not None:
            out["cauliflower_score"], out["cauliflower_threshold"] = g["cauliflower_score"], g["cauliflower_threshold"]
        q = a["quality"]
        out["image_quality"] = {"ok": not a["issues"], "issues": a["issues"], **{kk: round(v, 3) for kk, v in q.items()}}
        if with_overlay and out.get("plant"):
            out["overlay_jpeg_b64"] = self._overlay(a, out["plant"]["regions"])
        return out

    @staticmethod
    def _overlay(a: dict, regions: list[dict]) -> str:
        ov = Image.fromarray(overlay(a["_rgb"], a["_m"], a["_lesion"]))
        d = ImageDraw.Draw(ov)
        for p, r in zip(a["parts"], regions):
            if r["tissue_share_of_photo_pct"] < 1:
                continue
            x0, y0, _, _ = _bbox(p["_mask"])
            if r["disease"] == "no_disease":
                label = "healthy" if not r.get("eaten_pct") else f"eaten {r['eaten_pct']:.0f}%"
            else:
                label = f"{_pretty(r['disease'])} {(r['affected_pct'] or 0):.0f}%"
            d.rectangle((x0, y0, x0 + 7 * len(label), y0 + 12), fill=(0, 0, 0))
            d.text((x0 + 2, y0), label, fill=(255, 255, 255))
        buf = io.BytesIO()
        ov.save(buf, "JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def _part(kind: str) -> str:
    return "head" if kind == "curd" else kind


def plant_public_result(r: dict) -> dict:
    """public_result + a message built from the measured stage and the per-part verdicts."""
    p = public_result(r)
    if not p["reliable"] or r.get("plant") is None:
        p.pop("overlay_jpeg_b64", None)
        p.pop("plant", None)
        return p
    plant = r["plant"]
    insect = plant.get("insect") or {"found": False}
    if p["prediction"] == INSECT:
        if plant["affected_pct"] is None:
            msg = ("Insect damage (feeding damage seen, but the eaten area could not be measured: "
                   "bites from the leaf edge and scraped patches are not measured).")
        else:
            msg = (f"Insect damage, {plant['stage']} stage: {plant['affected_pct']:.1f}% of the worst leaf eaten "
                   f"({plant['lesion_count']} holes; bites from the leaf edge are not counted).")
    elif p["prediction"] == "no_disease":
        msg = "No disease detected."
    elif plant["affected_pct"] is None:
        msg = (f"{_pretty(p['prediction']).capitalize()}, {plant['stage']} stage (estimated; "
               "the spots are too small or faint to measure).")
    elif plant["stage_source"] == "classifier_grade":
        msg = (f"{_pretty(p['prediction']).capitalize()}, {plant['stage']} stage (estimated by the model; "
               f"measured visible rot {plant['affected_pct']:.0f}%, which can under-count dark rot on the head).")
    else:
        parts = [reg for reg in plant["regions"] if reg["disease"] != "no_disease" and reg["stage"] != "unmeasured"]
        where = ", ".join(f"{_part(reg['type'])} {reg['affected_pct']:.0f}%" for reg in parts)
        msg = (f"{_pretty(p['prediction']).capitalize()}, {plant['stage']} stage: "
               f"{plant['affected_pct']:.0f}% of the worst part affected ({where}; {plant['lesion_count']} spots).")
    possible = sorted({f"{_pretty(reg['disease'])} on the {_part(reg['type'])}" for reg in plant["regions"]
                       if reg["disease"] not in ("no_disease", "unknown", INSECT, p["prediction"])})
    if possible:
        msg += f" Also check: possible {'; '.join(possible)}."
    if insect["found"] and p["prediction"] != INSECT:
        msg += (f" Insect feeding holes also found: {insect['eaten_pct']:.1f}% of the worst leaf eaten "
                f"({insect['hole_count']} holes).")
    spots = [_part(reg["type"]) for reg in plant["regions"] if reg["verdict"].startswith("small unidentified")]
    if spots:
        msg += f" Small unidentified spots on: {', '.join(sorted(set(spots)))} (keep an eye on them)."
    if p["prediction"] in STAGE_FROM_GRADE and not any(reg["type"] == "curd" for reg in plant["regions"]):
        msg += " The head was not outlined in this photo; a closer photo of the head gives a measured affected %."
    if p["status"] == "uncertain":
        msg = "Not sure. " + msg + f" Reason: {r['status_reason']}."
    p["message"] = msg
    # Advice only for a confident diagnosis: acting on a guess could mean removing healthy plants.
    p["advice"] = advice_for(p["prediction"], plant["stage"]) if p["status"] == "confident" else None
    p["insect_advice"] = (advice_for(INSECT, insect["stage"])
                          if p["status"] == "confident" and insect["found"] and p["prediction"] != INSECT else None)
    return p
