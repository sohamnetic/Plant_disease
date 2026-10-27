"""Phase 3 runtime: measure affected area with the YOLO11-seg model.

For one photo it returns, per leaf / head (curd) the model finds:
  * affected_pct  = lesion pixels / tissue pixels of that leaf or head
  * lesion_count  = separate lesions on it
and for the plant: the worst region's % (drives the stage), the overall %,
the total lesion count, a draft stage from src/stages.json, and an overlay PNG.

Step I-2 weights (seg_002_insect) also outline insect holes (class "insect_hole"):
per leaf/head `eaten_pct` = hole pixels / tissue pixels (tissue includes the holes,
the model learns leaves "over" their holes) and `hole_count`, staged with the
`insect` thresholds in stages.json. Weights without that class get the insect
measurement from a second model (`hole_weights`, default seg_002): % eaten is measured
on that model's own leaf outlines (seg_001's leaf outlines cut the holes out: holes
found on 61% instead of 90% of held-out insect leaves), and each seg_001 part takes
the figures of the seg_002 part it overlaps most. Lesions, disease % and stage stay seg_001's.

Tissue of a region = the region mask plus lesions touching it, holes filled
(the model learns leaves "under" their lesions, so leaf masks have holes there).

Standalone for now; wiring into the API is Phase 4.

  python src/seg_runtime.py path/to/photo.jpg [--out overlay.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "experiments" / "seg_001_yolo11s" / "weights" / "best.pt"
# seg_002 = seg_001 classes + insect_hole (step I-2). It measured the rotten head of the
# user's field photo at 0% (seg_001: >25%), so it is used for the holes only.
DEFAULT_HOLE_WEIGHTS = PROJECT_ROOT / "experiments" / "seg_002_insect" / "weights" / "best.pt"
STAGES = json.loads((Path(__file__).resolve().parent / "stages.json").read_text())
CLASSES = ["leaf", "curd", "lesion", "insect_hole"]  # fallback; the weights' own names are used
WORK_SIDE = 640
MIN_REGION_FRAC = 0.01


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = np.pad(mask.astype(np.uint8), 1)
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return (m | (1 - flood)).astype(bool)[1:-1, 1:-1]


def stage_for(pct: float, lesions: int) -> str:
    s = STAGES
    if pct < s["min_pct_for_disease"] or lesions == 0:
        return "none"
    if pct < s["early_max_pct"] or (lesions <= s["early_max_lesions"] and pct < s["early_lesion_rule_max_pct"]):
        return "early"
    if pct < s["moderate_max_pct"]:
        return "moderate"
    if pct < s["severe_max_pct"]:
        return "severe"
    return "critical"


def insect_stage_for(pct: float, holes: int) -> str:
    s = STAGES["insect"]
    if pct < s["min_pct"] or holes == 0:
        return "none"
    if pct < s["early_max_pct"]:
        return "early"
    if pct < s["moderate_max_pct"]:
        return "moderate"
    if pct < s["severe_max_pct"]:
        return "severe"
    return "critical"


def measure(regions: list[tuple[str, np.ndarray]], lesion: np.ndarray, holes: np.ndarray | None = None) -> dict:
    """regions: [(type, mask)], lesion: union of lesion masks, holes: union of insect-hole
    masks (None = model without that class). Shared by runtime and eval."""
    holes = np.zeros_like(lesion) if holes is None else holes & ~lesion
    n_les, les_lab = cv2.connectedComponents(lesion.astype(np.uint8), connectivity=8)
    # A leaf that is diseased all over has no healthy "leaf" pixels left, so the
    # model outputs only lesion. A large lesion touching no region is that tissue.
    regions = list(regions)
    any_region = np.zeros_like(lesion)
    for _, m in regions:
        any_region |= cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    for i in range(1, n_les):
        comp = les_lab == i
        if not (comp & any_region).any() and comp.mean() >= MIN_REGION_FRAC:
            regions.append(("leaf", comp))
    # Each lesion belongs to ONE part: the one whose (slightly grown) mask it
    # overlaps most, so a head's rot is not also counted on the leaves around it.
    near = [cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for _, m in regions]
    owner = np.full(n_les, -1)
    for i in range(1, n_les):
        comp = les_lab == i
        overlaps = [int((comp & nm).sum()) for nm in near]
        if overlaps and max(overlaps) > 0:
            owner[i] = int(np.argmax(overlaps))
    out_regions = []
    tissue_all = np.zeros_like(lesion)
    for r_i, (kind, m) in enumerate(regions):
        ids = np.nonzero(owner == r_i)[0]
        own_les = np.isin(les_lab, ids)
        # Filling holes closes a ring of leaves around a head: remove other parts again.
        others = np.zeros_like(lesion)
        for o_i, (_, om) in enumerate(regions):
            if o_i != r_i:
                others |= om
        others |= lesion & ~own_les
        tissue = _fill_holes(m | own_les) & ~(others & ~m)
        les_px = int((own_les & tissue).sum())
        pct = 100.0 * les_px / max(int(tissue.sum()), 1)
        own_holes = holes & tissue
        n_holes = cv2.connectedComponents(own_holes.astype(np.uint8), connectivity=8)[0] - 1
        eaten = 100.0 * int(own_holes.sum()) / max(int(tissue.sum()), 1)
        out_regions.append({"type": kind, "tissue_px": int(tissue.sum()), "lesion_px": les_px,
                            "affected_pct": round(pct, 2), "lesion_count": int(len(ids)),
                            "stage": stage_for(pct, len(ids)),
                            "eaten_pct": round(eaten, 2), "hole_count": int(n_holes),
                            "insect_stage": insect_stage_for(eaten, n_holes), "_mask": tissue})
        tissue_all |= tissue
    total_les = int((lesion & tissue_all).sum())
    overall = 100.0 * total_les / max(int(tissue_all.sum()), 1)
    worst = max(out_regions, key=lambda r: r["affected_pct"], default=None)
    ids_all = np.unique(les_lab[lesion & tissue_all])
    count = int((ids_all > 0).sum())
    eaten_worst = max(out_regions, key=lambda r: r["eaten_pct"], default=None)
    return {
        "worst_region_eaten_pct": eaten_worst["eaten_pct"] if eaten_worst else 0.0,
        "hole_count": int(sum(r["hole_count"] for r in out_regions)),
        "insect_stage": eaten_worst["insect_stage"] if eaten_worst else "none",
        "_holes": holes & tissue_all,
        "regions": out_regions,
        "worst_region_pct": worst["affected_pct"] if worst else 0.0,
        "overall_pct": round(overall, 2),
        "lesion_count": count,
        "stage": stage_for(worst["affected_pct"], worst["lesion_count"]) if worst else "none",
        "_tissue": tissue_all,
    }


def overlay(rgb: np.ndarray, res: dict, lesion: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32) * 0.5
    t = res["_tissue"]
    out[t] = rgb[t] * 0.95
    les = lesion & t
    out[les] = out[les] * 0.3 + np.array([255, 0, 200]) * 0.7
    holes = res.get("_holes")
    if holes is not None and holes.any():
        out[holes] = out[holes] * 0.25 + np.array([255, 140, 0]) * 0.75  # orange = eaten by insects
    for r in sorted(res["regions"], key=lambda r: r["type"] == "curd"):  # heads last, on top of leaf borders
        edge = cv2.morphologyEx(r["_mask"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        out[edge] = (255, 220, 0) if r["type"] == "curd" else (0, 255, 255)
    return out.clip(0, 255).astype(np.uint8)


class SegEngine:
    def __init__(self, weights: Path = DEFAULT_WEIGHTS, device: str = "cpu", conf: float = 0.25,
                 hole_weights: Path | None = DEFAULT_HOLE_WEIGHTS):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device, self.conf = device, conf
        self.names = self._names(self.model)
        self.hole_model = None
        if "insect_hole" not in self.names and hole_weights is not None and Path(hole_weights).is_file():
            self.hole_model = YOLO(str(hole_weights))
        self.has_holes = "insect_hole" in self.names or self.hole_model is not None

    @staticmethod
    def _names(model) -> list[str]:
        names = getattr(model, "names", None) or dict(enumerate(CLASSES))
        return [names[i] for i in sorted(names)]

    def _predict(self, model, rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
        h, w = rgb.shape[:2]
        r = model.predict(rgb[..., ::-1], imgsz=WORK_SIDE, conf=self.conf, device=self.device,
                          retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        names = self._names(model)
        out = []
        for cls, m in zip(r.boxes.cls.cpu().numpy().astype(int), r.masks.data.cpu().numpy() > 0.5):
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            out.append((names[cls], m))
        return out

    def measure_all(self, rgb: np.ndarray) -> tuple[dict, np.ndarray]:
        """-> (measure() result, lesion union): disease from the main model, insect holes
        (eaten %, hole count, stage) from the hole model when there is one."""
        regions, lesion, holes = self.raw_masks_all(rgb)
        if self.hole_model is None:
            return measure(regions, lesion, holes), lesion
        m = measure(regions, lesion)
        h_regions, h_lesion, h_holes = self._split(self._predict(self.hole_model, rgb), rgb.shape[:2], True)
        hm = measure(h_regions, h_lesion, h_holes)
        for r in m["regions"]:
            best = max(hm["regions"], key=lambda q: int((q["_mask"] & r["_mask"]).sum()), default=None)
            if best is not None and (best["_mask"] & r["_mask"]).any():
                r.update({k: best[k] for k in ("eaten_pct", "hole_count", "insect_stage")})
        m.update({k: hm[k] for k in ("worst_region_eaten_pct", "hole_count", "insect_stage", "_holes")})
        return m, lesion

    def raw_masks(self, rgb: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
        regions, lesion, _ = self.raw_masks_all(rgb)
        return regions, lesion

    def raw_masks_all(self, rgb: np.ndarray) -> tuple[list[tuple[str, np.ndarray]], np.ndarray, np.ndarray | None]:
        """-> (leaf/curd regions, lesion union, insect-hole union or None for older weights)."""
        found = self._predict(self.model, rgb)
        if self.hole_model is not None:
            found += [(n, m) for n, m in self._predict(self.hole_model, rgb) if n == "insect_hole"]
        return self._split(found, rgb.shape[:2], self.has_holes)

    @staticmethod
    def _split(found: list[tuple[str, np.ndarray]], shape: tuple[int, int], with_holes: bool):
        h, w = shape
        regions, lesion = [], np.zeros((h, w), bool)
        holes = np.zeros((h, w), bool) if with_holes else None
        for name, m in found:
            if name == "lesion":
                lesion |= m
            elif name == "insect_hole":
                holes |= m
            elif m.mean() >= MIN_REGION_FRAC:
                regions.append((name, m))
        return regions, lesion, holes

    def predict_pil(self, im: Image.Image, with_overlay: bool = True) -> dict:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE))
        rgb = np.asarray(im)
        res, lesion = self.measure_all(rgb)
        ov = overlay(rgb, res, lesion) if with_overlay else None
        public = {k: v for k, v in res.items() if not k.startswith("_")}
        public["regions"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res["regions"]]
        public["stage_thresholds_expert_verified"] = STAGES["expert_verified"]
        public["_overlay"] = ov
        return public


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    args = ap.parse_args()
    eng = SegEngine(Path(args.weights))
    res = eng.predict_pil(Image.open(args.image))
    ov = res.pop("_overlay")
    print(json.dumps(res, indent=2))
    if args.out:
        Image.fromarray(ov).save(args.out)
        print("overlay ->", args.out)


if __name__ == "__main__":
    main()
