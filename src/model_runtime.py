"""Inference runtime for the v2 ensemble (Experiment 4). No pandas/sklearn/matplotlib.

Pipeline for one photo:
  1. image-quality gate (size, blur, exposure)            -> "retake_photo"
  2. 5 fold models x TTA (identity + hflip), logits averaged
  3. out-of-distribution check: feature-space Mahalanobis -> "out_of_scope"
  4. temperature-scaled diagnosis probabilities + conformal prediction set
  5. per-disease presence probabilities (co-infection)
  6. ordinal severity grade 0-4 with calibrated probabilities + conformal range
Thresholds and temperatures come from calibration.json (fit on out-of-fold data
by src/calibrate_v2.py); nothing is tuned on the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat

from v2_common import CLASS_ORDER, DISEASES, GRADE_LABELS, GRADE_MEANING

try:  # the ONNX deployment (src/onnx_runtime.py) runs without PyTorch
    import torch
except ImportError:
    torch = None

RELIABLE_STATUSES = {"confident", "uncertain"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP = PROJECT_ROOT / "experiments" / "experiment_006_v3_measured_severity"
MODEL_NAME = DEFAULT_EXP.name


def pick_device() -> "torch.device":
    # Render has no GPU. Prefer CPU even if a CUDA wheel is present.
    return torch.device("cpu")


def quality_metrics(im: Image.Image) -> dict[str, float]:
    """Cheap photo checks on a 512-px copy: sharpness (Laplacian variance) and exposure."""
    small = im.copy()
    small.thumbnail((512, 512))
    gray = small.convert("L")
    lap = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    # PIL leaves the 1-px border unfiltered; including it made flat images look sharp.
    lap = lap.crop((1, 1, lap.width - 1, lap.height - 1))
    arr = np.asarray(gray, dtype=np.float32)
    # Subject = everything except a plain white backdrop. Overexposure is judged on
    # the subject, so a well-lit leaf on white paper is not "overexposed".
    rgb = np.asarray(small.convert("RGB"))
    subject = rgb.min(2) <= 235
    return {
        "min_side": float(min(im.size)),
        "sharpness": float(ImageStat.Stat(lap).var[0]),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "clipped_dark": float((arr < 8).mean()),
        "clipped_bright": float((arr > 247).mean()),
        "subject_frac": float(subject.mean()),
        "subject_brightness": float(arr[subject].mean()) if subject.any() else 255.0,
    }


MIN_SUBJECT_FRAC = 0.05
MIN_CONTRAST = 5.0  # grey-level std; same "blank" bar the dataset audit used (config.BLANK_STD_MAX)


def quality_issues(q: dict[str, float], th: dict[str, float]) -> list[str]:
    issues = []
    if q["min_side"] < th["min_side"]:
        issues.append("image too small")
    if q["contrast"] < MIN_CONTRAST:
        issues.append("image is blank")
    if q["sharpness"] < th["min_sharpness"]:
        issues.append("image too blurry")
    if q["brightness"] < th["min_brightness"] or q["clipped_dark"] > th["max_clipped"]:
        issues.append("image too dark")
    if q["subject_frac"] < MIN_SUBJECT_FRAC or q["subject_brightness"] > th["max_brightness"]:
        issues.append("image overexposed")
    return issues


class V2Engine:
    def __init__(self, exp_dir: Path = DEFAULT_EXP, device: torch.device | None = None):
        from v2_model import MultiTaskNet, eval_transforms

        self.exp_dir = Path(exp_dir)
        self.device = device or pick_device()
        self.cal = json.loads((self.exp_dir / "calibration.json").read_text())
        self.tfm = eval_transforms(self.cal["image_size"])
        self.models: list[MultiTaskNet] = []
        self.maha: list[dict[str, torch.Tensor]] = []
        self.classes: list[str] = list(self.cal.get("classes", CLASS_ORDER))
        for path in sorted(self.exp_dir.glob("fold*.pt")):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if list(ckpt.get("classes", CLASS_ORDER)) != self.classes:
                raise ValueError(f"{path.name} classes {ckpt.get('classes')} != calibration classes {self.classes}")
            model = MultiTaskNet(pretrained=False, num_classes=len(self.classes))
            model.load_state_dict({k: v.float() for k, v in ckpt["model_state"].items()})
            model.to(self.device).eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.models.append(model)
            self.maha.append({k: v.float().to(self.device) for k, v in ckpt["maha"].items()})
        if not self.models:
            raise FileNotFoundError(f"No fold*.pt checkpoints in {self.exp_dir}")
        self.checkpoints = [p.name for p in sorted(self.exp_dir.glob("fold*.pt"))]

    # ---- raw network outputs -------------------------------------------------
    def forward_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (N,3,H,W) normalised. Returns ensemble x TTA averaged outputs + OOD score."""
        with torch.inference_mode():
            return self._forward_batch(x)

    def _forward_batch(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(self.device)
        views = torch.cat([x, torch.flip(x, dims=[3])])
        n = len(x)
        cls, pres, sev, maha = [], [], [], []
        for model, st in zip(self.models, self.maha):
            out = model(views)
            cls.append((out["cls"][:n] + out["cls"][n:]) / 2)
            pres.append((out["presence"][:n] + out["presence"][n:]) / 2)
            sev.append((out["sev"][:n] + out["sev"][n:]) / 2)
            feat = (out["feat"][:n] + out["feat"][n:]) / 2
            z = (feat - st["mu"]) @ st["comps"].T
            d = z[:, None, :] - st["means"][None]
            maha.append(torch.einsum("nci,ij,ncj->nc", d, st["prec"], d).min(1).values)
        return {
            "cls": torch.stack(cls).mean(0),
            "presence": torch.stack(pres).mean(0),
            "sev": torch.stack(sev).mean(0),
            "maha": torch.stack(maha).mean(0),
        }

    # ---- calibrated interpretation ------------------------------------------
    def interpret(self, raw: dict, i: int, issues: list[str]) -> dict:
        return interpret_probs(self.cal, self.classes, *self._probs(raw, i), issues)

    def _probs(self, raw: dict[str, torch.Tensor], i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Calibrated (diagnosis, presence, severity) probabilities and the OOD score of image i."""
        from v2_model import coral_to_probs

        c = self.cal
        with torch.inference_mode():
            probs = torch.softmax(raw["cls"][i] / c["cls_temperature"], dim=0).cpu().numpy()
            pres = torch.sigmoid(raw["presence"][i] / c["presence_temperature"]).cpu().numpy()
            sev_p = coral_to_probs(raw["sev"][i : i + 1] / c["sev_temperature"])[0].cpu().numpy()
        return probs, pres, sev_p, float(raw["maha"][i])

    def forward_images(self, ims: list[Image.Image]) -> dict:
        return self.forward_batch(torch.stack([self.tfm(im) for im in ims]))

    def predict_pil(self, im: Image.Image) -> dict:
        im = ImageOps.exif_transpose(im).convert("RGB")
        q = quality_metrics(im)
        issues = quality_issues(q, self.cal["quality"])
        raw = self.forward_batch(self.tfm(im).unsqueeze(0))
        out = self.interpret(raw, 0, issues)
        out["image_quality"] = {"ok": not issues, "issues": issues, **{k: round(v, 3) for k, v in q.items()}}
        return out



def interpret_probs(c: dict, classes: list[str], probs: np.ndarray, pres: np.ndarray, sev_p: np.ndarray,
                    ood: float, issues: list[str]) -> dict:
    """Status, prediction set, severity and findings from calibrated probabilities
    (shared by the PyTorch engine and the ONNX engine, src/onnx_runtime.py)."""
    sev_p = sev_p / sev_p.sum()
    pred = int(probs.argmax())
    pred_name = classes[pred]
    pred_set = [classes[j] for j in range(len(classes)) if probs[j] >= 1 - c["cls_conformal_qhat"]]
    if pred_name not in pred_set:
        pred_set.insert(0, pred_name)

    if pred_name == "no_disease":
        grade, grade_conf, grade_range = 0, float(sev_p[0]), [0, 0]
    else:
        # Diagnosed diseased: grade 0 is excluded, renormalise over 1-4.
        sp = sev_p.copy()
        sp[0] = 0.0
        sp = sp / max(sp.sum(), 1e-9)
        grade = int(sp.argmax())
        grade_conf = float(sp[grade])
        members = [g for g in range(1, 5) if sp[g] >= 1 - c["sev_conformal_qhat"]] or [grade]
        grade_range = [min(members + [grade]), max(members + [grade])]
        sev_p = sp
    expected = float((sev_p * np.arange(len(sev_p))).sum())
    severity = {
        "grade": grade,
        "stage": GRADE_LABELS[grade],
        "meaning": GRADE_MEANING[grade],
        "grade_confidence": round(grade_conf, 4),
        "grade_range_90": grade_range,
        "expected_grade": round(expected, 2),
        "severity_score_0_100": round(expected / 4 * 100, 1),
        "grade_probabilities": {GRADE_LABELS[g]: round(float(p), 4) for g, p in enumerate(sev_p)},
    }

    others = [
        {"disease": d, "probability": round(float(pres[j]), 4)}
        for j, d in enumerate(DISEASES)
        if d != pred_name and pres[j] >= c["presence_threshold"]
    ]

    confidence = float(probs[pred])
    if issues:
        status = "retake_photo"
    elif ood > c["ood_threshold"]:
        status = "out_of_scope"
    elif confidence < c.get("min_evidence_prob", 0.0):
        # No class stands out: real photos essentially never look like this.
        status = "no_evidence"
    elif len(pred_set) > 1 or confidence < c["confident_min_prob"]:
        status = "uncertain"
    else:
        status = "confident"

    retake_reason = "; ".join(issues)
    if issues and ood > c["ood_threshold"]:
        retake_reason += "; the subject also does not look like a cauliflower/brassica photo"
    return {
        "status": status,
        "status_reason": {
            "retake_photo": retake_reason,
            "out_of_scope": "photo does not look like the cauliflower/brassica photos the model was trained on",
            "no_evidence": "the model sees no clear sign of any disease or of a healthy plant",
            "uncertain": "more than one diagnosis is plausible at the 95% level" if len(pred_set) > 1
            else f"confidence below {c['confident_min_prob']:.2f}",
            "confident": "single diagnosis at the 95% level",
        }[status],
        "prediction": pred_name,
        "confidence": round(confidence, 4),
        "prediction_set_95": pred_set,
        "class_probabilities": {n: round(float(p), 4) for n, p in zip(classes, probs)},
        "severity": severity,
        "disease_presence": {d: round(float(p), 4) for d, p in zip(DISEASES, pres)},
        "other_findings": others,
        "ood_score": round(ood, 2),
        "ood_threshold": round(float(c["ood_threshold"]), 2),
    }


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def public_result(r: dict) -> dict:
    """What callers (API, CLI) should see.

    For out_of_scope / retake_photo the model's top class is not a diagnosis, so
    prediction, confidence and severity are withheld (null) and the raw guess is
    moved under `unreliable_guess` for debugging only.
    """
    status = r["status"]
    if status == "not_cauliflower":
        return {
            "reliable": False,
            "message": ("This is not a cauliflower photo. Please upload a photo of a cauliflower plant, "
                        "leaf or head."),
            **{k: r[k] for k in ("status", "status_reason", "cauliflower_score", "cauliflower_threshold") if k in r},
            "prediction": None, "confidence": None, "prediction_set_95": [], "severity": None,
            "other_findings": [], "unreliable_guess": None,
        }
    if status in RELIABLE_STATUSES:
        s = r["severity"]
        if status == "confident":
            if r["prediction"] == "no_disease":
                msg = "No disease detected."
            else:
                msg = f"{_pretty(r['prediction']).capitalize()}, {s['stage']} stage (grade {s['grade']}/4: {s['meaning']})."
        else:
            options = " or ".join(_pretty(p) for p in r["prediction_set_95"])
            msg = (f"Not sure: most likely {_pretty(r['prediction'])}, but could be {options}. "
                   "Take a closer photo of the affected part or ask an expert.")
        if r["other_findings"]:
            msg += " Also possible: " + ", ".join(_pretty(o["disease"]) for o in r["other_findings"]) + "."
        return {"reliable": True, "message": msg, **r}

    if status == "retake_photo":
        msg = f"No diagnosis. Photo problem: {r['status_reason']}. Please retake the photo."
    elif status == "no_evidence":
        msg = ("No reliable diagnosis. The model sees no clear sign of any disease or of a healthy plant. "
               "Take a close-up of a cauliflower head or leaf.")
    else:
        msg = ("No reliable diagnosis. This photo does not look like the photos the model was trained on. "
               "Take a close-up where the affected head or leaf fills most of the frame.")
    keep = ("status", "status_reason", "ood_score", "ood_threshold", "image_quality")
    return {
        "reliable": False,
        "message": msg,
        **{k: r[k] for k in keep if k in r},
        "prediction": None,
        "confidence": None,
        "prediction_set_95": [],
        "severity": None,
        "other_findings": [],
        "unreliable_guess": {
            "note": "NOT a diagnosis. Shown for debugging only.",
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "class_probabilities": r["class_probabilities"],
            "disease_presence": r["disease_presence"],
        },
    }
