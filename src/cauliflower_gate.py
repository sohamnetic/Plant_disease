"""Cauliflower gate: refuse any photo that is not a cauliflower plant, leaf or head.

Why: the disease classifier and YOLO only know cauliflower, so they always answer
something. Their own out-of-scope check (Mahalanobis distance on the classifier's
features) let an anime wallpaper through as "no disease, 86%" and YOLO outlined 92%
of it as a leaf. The classifier's features cannot tell a dark blue sky from a dark
blue-green leaf; a general image model can.

How: CLIP ViT-L/14 (OpenAI, already in the local Hugging Face cache; nothing is
trained) compares the photo with short text descriptions. The gate score is CLIP's
probability mass on the cauliflower descriptions vs. the other descriptions. Below
THRESHOLD the photo is "not a cauliflower photo" and nothing else runs.

  POSITIVE  descriptions of cauliflower plants, leaves, heads, diseased leaves
  NEGATIVE  generic non-plant content (people, drawings, sky, buildings, text, noise,
            microscope images, ...). Other vegetables (cabbage, broccoli) are NOT
            negatives: with them, up to 1 in 4 real cauliflower leaf photos were
            rejected (a lone cauliflower leaf looks like a cabbage leaf).

Threshold 0.10, chosen on the cauliflower TRAINING images (rejects 2 of 2,650, 0.075%),
then checked on photos no model trained on (outputs/final/cauliflower_gate_eval.json).

  python src/cauliflower_gate.py build      # text embeddings -> src/cauliflower_gate.npz
  python src/cauliflower_gate.py evaluate   # pass / reject rates -> outputs/final/cauliflower_gate_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # the ONNX deployment imports the prompts and threshold only (src/onnx_runtime.py)
    import torch
except ImportError:
    torch = None

HERE = Path(__file__).resolve().parent
CLIP_ID = "openai/clip-vit-large-patch14"
EMB_FILE = HERE / "cauliflower_gate.npz"
THRESHOLD = 0.10
WORK_SIDE = 640  # the photo is shrunk to this before CLIP's own 224 px resize / centre crop

POSITIVE = ["a photo of a cauliflower plant", "a photo of a cauliflower head", "a photo of a cauliflower leaf",
            "a close-up photo of a diseased cauliflower leaf", "a photo of cauliflower curd",
            "a photo of a rotten cauliflower", "a photo of cauliflower leaves in a field"]
NEGATIVE = ["a photo of a person", "an anime illustration", "a painting", "a cartoon", "a landscape",
            "a photo of the sky", "a photo of a building", "a photo of a car", "a photo of an animal",
            "a photo of food on a plate", "a document with text", "a screenshot", "a photo of a flower",
            "a photo of a tree", "a photo of grass", "a photo of another vegetable", "a photo of a fruit",
            "a photo of an object", "a dark image", "a blank image", "random noise", "a photo of bare soil",
            "a photo of a wall", "a logo", "a photo of space", "a microscope image", "a medical scan",
            "a close-up of a texture or pattern", "a diagram or chart"]


def _quiet() -> None:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()


class CauliflowerGate:
    def __init__(self, device: torch.device | str = "cpu"):
        _quiet()
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        z = np.load(EMB_FILE)
        if list(z["prompts"]) != POSITIVE + NEGATIVE:
            raise RuntimeError(f"{EMB_FILE.name} is out of date: run python src/cauliflower_gate.py build")
        self.device = torch.device(device)
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, local_files_only=True).to(self.device).eval()
        self.proc = CLIPImageProcessor.from_pretrained(CLIP_ID, local_files_only=True)
        self.text = torch.from_numpy(z["text"]).float().to(self.device)
        self.scale = float(z["logit_scale"])
        self.threshold = THRESHOLD

    def score(self, im: Image.Image) -> float:
        """Probability (0-1) that the photo shows a cauliflower plant, leaf or head."""
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((WORK_SIDE, WORK_SIDE))
        with torch.inference_mode():
            x = self.proc(images=im, return_tensors="pt")["pixel_values"].to(self.device)
            e = torch.nn.functional.normalize(self.model(pixel_values=x).image_embeds.float(), dim=-1)
            p = torch.softmax(self.scale * e @ self.text.T, dim=-1)[0]
        return float(p[:len(POSITIVE)].sum())

    def check(self, im: Image.Image) -> dict:
        s = self.score(im)
        return {"is_cauliflower": s >= self.threshold, "cauliflower_score": round(s, 4),
                "cauliflower_threshold": self.threshold}


def build() -> None:
    _quiet()
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_ID, local_files_only=True).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID, local_files_only=True)
    prompts = POSITIVE + NEGATIVE
    with torch.inference_mode():
        t = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True))
        t = t if torch.is_tensor(t) else t.pooler_output
        t = torch.nn.functional.normalize(t.float(), dim=-1).numpy()
    np.savez(EMB_FILE, text=t, prompts=np.array(prompts), logit_scale=float(model.logit_scale.exp()))
    print("written", EMB_FILE, t.shape)


def eval_sets() -> dict[str, list[Path]]:
    """Cauliflower photos no model trained on, and local non-cauliflower images."""
    import glob
    import pandas as pd
    sys.path.insert(0, str(HERE))
    import config as cfg
    root = cfg.PROJECT_ROOT
    sets: dict[str, list[Path]] = {}
    sets["external_test"] = [root / f for f in pd.read_csv(root / "splits_v3/external_test.csv")["filepath"]]
    t = pd.read_csv(cfg.SPLITS_DIR / "test.csv")
    sets["vegnet_test"] = [root / f for f in t[t["source_type"] == "original"]["filepath"]]
    w = pd.read_csv(root / "new_pics/manifest.csv", dtype={"is_copy": str})
    sets["web_photos"] = [root / f for f in w["file"]]
    sets["user_photos"] = sorted((cfg.OUTPUTS_DIR / "review/user_photos").glob("*.jpg"))
    import matplotlib
    import skimage
    import sklearn
    sk = Path(skimage.__file__).parent / "data"
    neg = [Path(p) for p in glob.glob(r"C:/Windows/Web/**/*.jpg", recursive=True) + glob.glob(r"C:/Windows/Web/**/*.png", recursive=True)]
    neg += [sk / n for n in ("astronaut.png", "brick.png", "camera.png", "chelsea.png", "coffee.png", "coins.png", "color.png",
                             "grass.png", "gravel.png", "horse.png", "hubble_deep_field.jpg", "ihc.png", "logo.png", "moon.png",
                             "motorcycle_left.png", "page.png", "retina.jpg", "rocket.jpg", "text.png", "cell.png",
                             "clock_motion.png", "chessboard_RGB.png", "phantom.png", "microaneurysms.png")]
    neg += [Path(sklearn.__file__).parent / "datasets/images" / n for n in ("china.jpg", "flower.jpg")]
    neg += [Path(matplotlib.get_data_path()) / "sample_data" / n for n in ("grace_hopper.jpg", "Minduka_Present_Blue_Pack.png")]
    user_wallpaper = Path(r"C:/Users/lenovo/Downloads/wp8926127-aot-4k-wallpapers.jpg")
    sets["not_cauliflower"] = [p for p in [user_wallpaper] + neg if p.exists()]

    return sets


def evaluate(gate=None, dst: Path | None = None) -> None:
    """Pass rate on cauliflower photos no model trained on; reject rate on local non-cauliflower images."""
    sys.path.insert(0, str(HERE))
    import config as cfg
    gate = gate or CauliflowerGate("cuda" if torch.cuda.is_available() else "cpu")
    sets = eval_sets()

    out: dict = {"threshold": THRESHOLD, "model": getattr(gate, "name", CLIP_ID), "sets": {}}
    for name, files in sets.items():
        s = []
        for f in files:
            with Image.open(f) as im:
                s.append(gate.score(im))
        s = np.array(s)
        passed = s >= THRESHOLD
        out["sets"][name] = {"n": int(len(s)), "passed_as_cauliflower": int(passed.sum()),
                             "rejected": int((~passed).sum()), "lowest_score": round(float(s.min()), 4),
                             "highest_score": round(float(s.max()), 4)}
        if name == "not_cauliflower":
            out["sets"][name]["passed_files"] = [str(f) for f, ok in zip(files, passed) if ok]
        else:
            out["sets"][name]["rejected_files"] = [str(f) for f, ok in zip(files, passed) if not ok]
        print(name, out["sets"][name], flush=True)
    rng = np.random.default_rng(0)
    synth = {"noise": Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)),
             "black": Image.new("RGB", (640, 480)), "white": Image.new("RGB", (640, 480), (245, 245, 240)),
             "plain_green": Image.new("RGB", (640, 480), (30, 120, 40))}
    out["synthetic_scores"] = {k: round(gate.score(v), 4) for k, v in synth.items()}
    print("synthetic", out["synthetic_scores"])
    dst = dst or cfg.OUTPUTS_DIR / "final" / "cauliflower_gate_eval.json"
    dst.write_text(json.dumps(out, indent=2))
    print("written", dst)


if __name__ == "__main__":
    {"build": build, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "build"]()
