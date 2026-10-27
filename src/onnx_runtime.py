"""PyTorch-free runtime for small servers (Render free: 512 MB RAM, 0.1 CPU).

Same models, same decisions as the PyTorch app: the classifiers, outline models and
cauliflower gate are exported to ONNX and run with onnxruntime; everything after the
networks (calibration, statuses, measurement, the plant decision) is the app's own
code (model_runtime.interpret_probs, seg_runtime.measure, plant_runtime.PlantEngine).

Memory: no model stays loaded. Each request loads one network at a time, runs it and
frees it; the CLIP gate is split in two halves (layers 1-12, 13-24) run one after the
other, so the largest piece loaded is ~0.17 GB. A request is slow (~7-20 s per photo on
one CPU, minutes on 0.1 CPU) but peaks at ~0.35 GB, inside 512 MB.

Differences from the PyTorch app:
  * the CLIP gate's image encoder is 8-bit quantised in blocks of 32 weights (MatMul
    weights; 1.2 GB -> 0.33 GB), so its score moves slightly (mean 0.003); `gate-eval`
    re-runs the gate test. Rejected: per-channel 8-bit and 4-bit (blocks of 16/32):
    smaller, but they moved scores ~15x more and made gate mistakes (outputs/deploy/).
  * everything else is float32 ONNX: outputs match PyTorch to ~1e-5

  python src/onnx_runtime.py export              # (needs PyTorch) -> deploy_models/
  python src/onnx_runtime.py compare [--n 200]   # ONNX app vs PyTorch app on test photos
  python src/onnx_runtime.py gate-eval           # cauliflower gate test (as cauliflower_gate.py eval) with the 8-bit CLIP
Serve it: set PLANT_BACKEND=onnx (and PLANT_MODELS_DIR, default deploy_models/) for src/api.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_runtime import DEFAULT_EXP, interpret_probs
from plant_runtime import DEFAULT_INSECT_EXP, PlantEngine
from seg_runtime import DEFAULT_HOLE_WEIGHTS, DEFAULT_WEIGHTS, WORK_SIDE, SegEngine
from v2_common import IMAGENET_MEAN, IMAGENET_STD, coral_to_probs_np, letterbox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", PROJECT_ROOT / "deploy_models"))
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
CLIP_SIDE = 224
CLIP_HALVES = ("clip_vision_a.onnx", "clip_vision_b.onnx")  # 8-bit, run one after the other
CLIP_SPLIT = "/m/vision_model/encoder/layers.11/Add_1_output_0"  # hidden state after layer 12 of 24
CHUNK = 2  # photos per classifier run (4 views with the flip): keeps activations small


def _session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _run(path: Path, feeds: dict) -> list[np.ndarray]:
    s = _session(path)
    try:
        return s.run(None, feeds)
    finally:
        del s
        gc.collect()


# ---------------------------------------------------------------- classifier
class OnnxClassifier:
    """V2Engine without PyTorch: fold models are loaded one at a time per call."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.cal = json.loads((self.dir / "calibration.json").read_text())
        self.classes = list(self.cal["classes"])
        self.folds = sorted(self.dir.glob("fold*.onnx"))
        z = np.load(self.dir / "maha.npz")
        self.maha = [{k: z[f"{i}_{k}"] for k in ("mu", "comps", "means", "prec")} for i in range(len(self.folds))]
        self.checkpoints = [p.with_suffix(".pt").name for p in self.folds]
        self.size = int(self.cal["image_size"])
        self.device = "cpu"

    def tfm(self, im: Image.Image) -> np.ndarray:
        """= v2_model.eval_transforms: letterbox, bilinear resize, ImageNet normalisation (CHW float32)."""
        x = np.asarray(letterbox(im).resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.0
        return ((x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)).transpose(2, 0, 1)

    def forward_images(self, ims: list[Image.Image]) -> dict[str, np.ndarray]:
        x = np.stack([self.tfm(im) for im in ims]).astype(np.float32)
        n = len(x)
        per_fold = []
        for path, st in zip(self.folds, self.maha):
            s = _session(path)
            outs = {k: [] for k in ("cls", "presence", "sev", "feat")}
            for c0 in range(0, n, CHUNK):
                xc = x[c0 : c0 + CHUNK]
                views = np.concatenate([xc, xc[..., ::-1]])  # identity + horizontal flip
                k = len(xc)
                for name, v in zip(("cls", "presence", "sev", "feat"), s.run(None, {"x": np.ascontiguousarray(views)})):
                    outs[name].append((v[:k] + v[k:]) / 2)
            del s
            gc.collect()
            o = {k: np.concatenate(v) for k, v in outs.items()}
            z = (o["feat"] - st["mu"]) @ st["comps"].T
            dlt = z[:, None, :] - st["means"][None]
            o["maha"] = np.einsum("nci,ij,ncj->nc", dlt, st["prec"], dlt).min(1)
            per_fold.append(o)
        return {k: np.stack([o[k] for o in per_fold]).mean(0) for k in ("cls", "presence", "sev", "maha")}

    def _probs(self, raw: dict, i: int):
        c = self.cal

        def softmax(v):
            e = np.exp(v - v.max())
            return e / e.sum()

        probs = softmax(raw["cls"][i] / c["cls_temperature"]).astype(np.float32)
        pres = (1 / (1 + np.exp(-raw["presence"][i] / c["presence_temperature"]))).astype(np.float32)
        sev_p = coral_to_probs_np(raw["sev"][i : i + 1] / c["sev_temperature"])[0]
        return probs, pres, sev_p, float(raw["maha"][i])

    def interpret(self, raw: dict, i: int, issues: list[str]) -> dict:
        return interpret_probs(self.cal, self.classes, *self._probs(raw, i), issues)


# ---------------------------------------------------------------- outline models
def _letterbox_rect(rgb: np.ndarray, side: int = 640, stride: int = 32):
    """= ultralytics LetterBox(auto=True): long side -> 640, pad to a multiple of 32 with grey 114."""
    h, w = rgb.shape[:2]
    r = min(side / h, side / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (side - nw) % stride / 2, (side - nh) % stride / 2
    if (w, h) != (nw, nh):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter) <= iou]
    return np.array(keep, int)


class OnnxSeg(SegEngine):
    """SegEngine with ultralytics' predict replaced by ONNX + the same pre/post-processing
    (rect letterbox, per-class NMS iou 0.7, retina masks cropped to their boxes)."""

    IOU, MAX_DET, MAX_WH = 0.7, 300, 7680

    def __init__(self, main: Path, holes: Path | None, conf: float = 0.25):
        self.conf = conf
        meta = json.loads((Path(main).parent / "seg_names.json").read_text())
        self.model, self.hole_model = Path(main), (Path(holes) if holes else None)
        self.names = meta[self.model.name]
        self.has_holes = "insect_hole" in self.names or self.hole_model is not None
        self.checkpoints = [f"seg:{meta['_from'][self.model.name]}"] + (
            [f"holes:{meta['_from'][self.hole_model.name]}"] if self.hole_model else [])
        self._meta = meta

    def _names(self, model) -> list[str]:
        return self._meta[Path(model).name]

    def _predict(self, model, rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
        h, w = rgb.shape[:2]
        img = _letterbox_rect(rgb)
        x = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        pred, proto = _run(Path(model), {"images": np.ascontiguousarray(x)})
        names = self._names(model)
        nc = len(names)
        p = pred[0].T  # (anchors, 4 + nc + 32)
        cls_conf = p[:, 4 : 4 + nc]
        conf, j = cls_conf.max(1), cls_conf.argmax(1)
        keep = conf > self.conf
        p, conf, j = p[keep], conf[keep], j[keep]
        if not len(p):
            return []
        xy, wh = p[:, :2], p[:, 2:4] / 2
        boxes = np.concatenate([xy - wh, xy + wh], 1)
        k = _nms(boxes + j[:, None] * self.MAX_WH, conf, self.IOU)[: self.MAX_DET]
        boxes, j, coef = boxes[k], j[k], p[k, 4 + nc :]
        # boxes back to the photo (ultralytics ops.scale_boxes)
        H1, W1 = img.shape[:2]
        gain = min(H1 / h, W1 / w)
        pad_x, pad_y = round((W1 - round(w * gain)) / 2 - 0.1), round((H1 - round(h * gain)) / 2 - 0.1)
        boxes = boxes - [pad_x, pad_y, pad_x, pad_y]
        boxes /= gain
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        # masks (ultralytics ops.process_mask_native: crop padding, bilinear up, > 0, crop to box)
        c, mh, mw = proto[0].shape
        logits = (coef @ proto[0].reshape(c, -1)).reshape(-1, mh, mw)
        g = min(mh / h, mw / w)
        pw, ph = (mw - round(w * g)) / 2, (mh - round(h * g)) / 2
        top, left = round(ph - 0.1), round(pw - 0.1)
        bottom, right = mh - round(ph + 0.1), mw - round(pw + 0.1)
        rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
        out = []
        for lg, b, ci in zip(logits, boxes, j):
            m = cv2.resize(np.ascontiguousarray(lg[top:bottom, left:right]), (w, h), interpolation=cv2.INTER_LINEAR) > 0
            m &= (cols >= b[0]) & (cols < b[2]) & (rows >= b[1]) & (rows < b[3])
            if m.any():
                out.append((names[int(ci)], m))
        return out


# ---------------------------------------------------------------- cauliflower gate
class OnnxGate:
    """cauliflower_gate.CauliflowerGate with the 8-bit CLIP image encoder, run in two halves."""

    def __init__(self, d: Path, keep_loaded: bool = False):
        from cauliflower_gate import NEGATIVE, POSITIVE, THRESHOLD  # constants only (torch-free import below)

        self.paths = [Path(d) / f for f in CLIP_HALVES]
        self.name = "openai/clip-vit-large-patch14 (image encoder, ONNX 8-bit, blocks of 32)"
        z = np.load(Path(d) / "cauliflower_gate.npz")
        if list(z["prompts"]) != POSITIVE + NEGATIVE:
            raise RuntimeError("cauliflower_gate.npz does not match the gate prompts")
        self.text, self.scale, self.n_pos, self.threshold = z["text"].astype(np.float32), float(z["logit_scale"]), len(POSITIVE), THRESHOLD
        self._sess = [_session(p) for p in self.paths] if keep_loaded else None  # evaluations only: the server loads per photo

    @staticmethod
    def pixels(im: Image.Image) -> np.ndarray:
        """= CLIPImageProcessor: shortest side -> 224 (bicubic), centre crop 224, CLIP normalisation."""
        w, h = im.size
        s = CLIP_SIDE / min(w, h)
        nw, nh = (CLIP_SIDE, int(h * s)) if w <= h else (int(w * s), CLIP_SIDE)
        im = im.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - CLIP_SIDE) // 2, (nh - CLIP_SIDE) // 2
        x = np.asarray(im.crop((left, top, left + CLIP_SIDE, top + CLIP_SIDE)), np.float32) / 255.0
        return ((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)[None].astype(np.float32)

    def score(self, im: Image.Image) -> float:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE))
        h = self.pixels(im)
        for k, (name, path) in enumerate(zip(("px", CLIP_SPLIT), self.paths)):
            h = (self._sess[k].run(None, {name: h}) if self._sess else _run(path, {name: h}))[0]
        e = h[0]
        e = e / np.linalg.norm(e)
        lg = self.scale * self.text @ e
        p = np.exp(lg - lg.max())
        p /= p.sum()
        return float(p[: self.n_pos].sum())

    def check(self, im: Image.Image) -> dict:
        s = self.score(im)
        return {"is_cauliflower": s >= self.threshold, "cauliflower_score": round(s, 4), "cauliflower_threshold": self.threshold}


def build_engine(d: Path = MODELS_DIR) -> PlantEngine:
    d = Path(d)
    insect = OnnxClassifier(d / "insect") if (d / "insect" / "calibration.json").is_file() else None
    hole = d / "seg_holes.onnx"
    return PlantEngine.from_engines(OnnxClassifier(d / "classifier"), OnnxSeg(d / "seg.onnx", hole if hole.is_file() else None),
                                    gate=OnnxGate(d), insect_cls=insect)


# ---------------------------------------------------------------- export / compare (need PyTorch)
def export(d: Path) -> None:
    import shutil

    import torch

    from cauliflower_gate import EMB_FILE
    from model_runtime import V2Engine

    d.mkdir(parents=True, exist_ok=True)

    class Heads(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m(x)
            return o["cls"], o["presence"], o["sev"], o["feat"]

    for sub, exp in (("classifier", DEFAULT_EXP), ("insect", DEFAULT_INSECT_EXP)):
        eng = V2Engine(exp, device=torch.device("cpu"))
        out = d / sub
        out.mkdir(exist_ok=True)
        size = eng.cal["image_size"]
        maha = {}
        for i, (m, st) in enumerate(zip(eng.models, eng.maha)):
            torch.onnx.export(Heads(m).eval(), torch.randn(2, 3, size, size), out / f"fold{i}.onnx", input_names=["x"],
                              output_names=["cls", "presence", "sev", "feat"], dynamic_axes={"x": {0: "n"}},
                              opset_version=17, dynamo=False)
            maha.update({f"{i}_{k}": v.cpu().numpy() for k, v in st.items()})
        np.savez(out / "maha.npz", **maha)
        (out / "calibration.json").write_text(json.dumps({**eng.cal, "classes": eng.classes, "exp": exp.name}, indent=1))
        print("exported", sub, exp.name, len(eng.models), "folds")

    from ultralytics import YOLO
    names, frm = {}, {}
    for fname, w in (("seg.onnx", DEFAULT_WEIGHTS), ("seg_holes.onnx", DEFAULT_HOLE_WEIGHTS)):
        y = YOLO(str(w))
        shutil.move(y.export(format="onnx", imgsz=640, dynamic=True, simplify=False), d / fname)
        names[fname] = [y.names[i] for i in sorted(y.names)]
        frm[fname] = Path(w).parent.parent.name
    (d / "seg_names.json").write_text(json.dumps({**names, "_from": frm}, indent=1))

    export_clip(d)
    shutil.copy2(EMB_FILE, d / "cauliflower_gate.npz")
    for f in sorted(d.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(d)}  {f.stat().st_size / 2**20:.1f} MB")


def export_clip(d: Path) -> None:
    import onnx
    import torch
    from onnx.utils import extract_model
    from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    from transformers import CLIPVisionModelWithProjection

    from cauliflower_gate import CLIP_ID

    class Clip(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, px):
            return self.m(pixel_values=px).image_embeds

    clip = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, local_files_only=True).eval()
    fp32 = d / "clip_vision_fp32.onnx"
    torch.onnx.export(Clip(clip), torch.randn(1, 3, CLIP_SIDE, CLIP_SIDE), fp32, input_names=["px"], output_names=["emb"],
                      opset_version=17, dynamo=False)
    # split first (shape inference does not know the quantised op), then quantise each half:
    # block-wise quantisation is per weight block, so the halves equal the quantised whole
    for name, ins, outs in zip(CLIP_HALVES, (["px"], [CLIP_SPLIT]), ([CLIP_SPLIT], ["emb"])):
        part = d / f"{name}.fp32"
        extract_model(str(fp32), str(part), ins, outs)
        q = MatMulNBitsQuantizer(onnx.load(str(part)), algo_config=DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8))
        q.process()
        q.model.save_model_to_file(str(d / name), use_external_data_format=False)
        part.unlink()
    for f in d.glob("clip_vision_fp32.onnx*"):
        f.unlink()


def compare(d: Path, n: int) -> None:
    """Full app, PyTorch vs ONNX, on frozen external test + insect test photos (the app's real path)."""
    import pandas as pd

    from plant_runtime import plant_public_result

    fr = pd.read_csv(PROJECT_ROOT / "splits_v3" / "external_test.csv")
    it = pd.read_csv(PROJECT_ROOT / "splits_v4" / "insect_test.csv")
    files = (fr.sample(min(n, len(fr)), random_state=0)["filepath"].tolist()
             + it.sample(min(n // 4, len(it)), random_state=0)["filepath"].tolist())
    torch_eng, onnx_eng = PlantEngine(), build_engine(d)
    rows = []
    for f in files:
        im = Image.open(PROJECT_ROOT / f)
        a, b = plant_public_result(torch_eng.predict_pil(im, with_overlay=False)), plant_public_result(onnx_eng.predict_pil(im, with_overlay=False))
        rows.append({"file": f, "torch_status": a["status"], "onnx_status": b["status"],
                     "torch_pred": a.get("prediction"), "onnx_pred": b.get("prediction"),
                     "torch_conf": a.get("confidence"), "onnx_conf": b.get("confidence"),
                     "torch_gate": a.get("cauliflower_score"), "onnx_gate": b.get("cauliflower_score"),
                     "torch_msg": a.get("message"), "onnx_msg": b.get("message")})
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "outputs" / "deploy" / "onnx_vs_torch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    same = (df.torch_status == df.onnx_status) & (df.torch_pred.fillna("") == df.onnx_pred.fillna(""))
    conf = (df.torch_conf.fillna(0) - df.onnx_conf.fillna(0)).abs()
    print(f"{len(df)} photos: same status + diagnosis {same.mean():.1%}; same message {(df.torch_msg == df.onnx_msg).mean():.1%}; "
          f"confidence max diff {conf.max():.4f}; gate score max diff {(df.torch_gate - df.onnx_gate).abs().max():.4f}")
    print(df[~same][["file", "torch_status", "onnx_status", "torch_pred", "onnx_pred"]].to_string())
    print("->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "compare", "gate-eval"])
    ap.add_argument("--dir", default=str(MODELS_DIR))
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "export":
        export(Path(a.dir))
    elif a.cmd == "compare":
        compare(Path(a.dir), a.n)
    else:
        from cauliflower_gate import evaluate
        evaluate(OnnxGate(Path(a.dir), keep_loaded=True), PROJECT_ROOT / "outputs" / "deploy" / "cauliflower_gate_eval_onnx.json")


if __name__ == "__main__":
    main()
