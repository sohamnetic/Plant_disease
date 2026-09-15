"""HTTP API for the frozen Experiment 3 EfficientNet-B0 checkpoint.

The trained weights live on disk as best_model.pt (not a .pkl). They are
loaded once into RAM and reused for every request. Nothing is retrained.
If the process slept (Render free tier) and RAM was cleared, the same .pt
file is loaded again so scores stay identical for the same image.

Run from the project root:

  python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import hashlib
import io
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import build_transforms
from predict import DEFAULT_CKPT, load_checkpoint, pick_device, predict_pil

ENGINE_LOCK = threading.Lock()
ENGINE: dict = {
    "model": None,
    "tfm": None,
    "device": None,
    "checkpoint": str(DEFAULT_CKPT),
    "checkpoint_sha256": None,
    "loaded": False,
}

UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Plant disease classifier</title>
  <style>
    body { font-family: Segoe UI, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
    pre { background: #111; color: #eee; padding: 1rem; overflow: auto; }
    button { padding: 0.5rem 1rem; }
  </style>
</head>
<body>
  <h1>Cauliflower / brassica disease classifier</h1>
  <p>Frozen Experiment 3 weights (<code>best_model.pt</code>). Loaded once, reused for every photo. Same image → same scores.</p>
  <form id="f">
    <input type="file" name="file" accept="image/*" required/>
    <button type="submit">Predict</button>
  </form>
  <pre id="out">Upload an image to see predicted class and disease scores (0–100).</pre>
  <script>
    document.getElementById("f").addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target);
      const out = document.getElementById("out");
      out.textContent = "Running...";
      const r = await fetch("/predict", { method: "POST", body: fd });
      const t = await r.text();
      try { out.textContent = JSON.stringify(JSON.parse(t), null, 2); }
      catch { out.textContent = t; }
    });
  </script>
</body>
</html>
"""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model_loaded() -> None:
    """Load best_model.pt once. Reload only if RAM was emptied (process restart)."""
    if ENGINE["loaded"] and ENGINE["model"] is not None:
        return
    with ENGINE_LOCK:
        if ENGINE["loaded"] and ENGINE["model"] is not None:
            return
        ckpt = DEFAULT_CKPT
        if not ckpt.is_file():
            raise RuntimeError(
                f"Frozen checkpoint missing: {ckpt}. "
                "The API does not train a new model."
            )
        device = pick_device()
        ENGINE["device"] = device
        ENGINE["tfm"] = build_transforms(train=False)
        ENGINE["model"] = load_checkpoint(ckpt, device)
        ENGINE["model"].eval()
        for p in ENGINE["model"].parameters():
            p.requires_grad_(False)
        ENGINE["checkpoint"] = str(ckpt)
        ENGINE["checkpoint_sha256"] = _file_sha256(ckpt)
        ENGINE["loaded"] = True
        print("Frozen checkpoint loaded (stays in memory):", ckpt)


def format_prediction(filename: str | None, row: dict) -> dict:
    scores = {name: round(float(row[f"prob_{name}"]) * 100.0, 2) for name in cfg.CLASS_ORDER}
    predicted = row["predicted_class"]
    disease_score = scores[predicted]
    present = round(100.0 - scores["no_disease"], 2)
    return {
        "filename": filename,
        "predicted_class": predicted,
        "disease_score": disease_score,
        "disease_score_meaning": (
            f"Model probability of '{predicted}' as a 0-100 score "
            "from the frozen Experiment 3 checkpoint. Same photo → same score."
        ),
        "disease_present_score": present,
        "class_scores": scores,
        "confidence": round(float(row["confidence"]), 6),
        "checkpoint": "experiment_003_effnetb0_combined/best_model.pt",
        "checkpoint_sha256": ENGINE["checkpoint_sha256"],
        "model_kept_in_memory": True,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_model_loaded()
    yield
    # Do not delete best_model.pt. Only drop the in-process handle on shutdown.


app = FastAPI(
    title="Plant disease classifier",
    description="Frozen EfficientNet-B0. Weights persist on disk; loaded once per process.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    try:
        ensure_model_loaded()
    except Exception as exc:
        return {"status": "error", "loaded": False, "detail": str(exc)}
    return {
        "status": "ok",
        "loaded": True,
        "model": "experiment_003_effnetb0_combined",
        "checkpoint": ENGINE["checkpoint"],
        "checkpoint_sha256": ENGINE["checkpoint_sha256"],
        "classes": list(cfg.CLASS_ORDER),
        "device": str(ENGINE["device"]),
        "persists_on_disk": True,
        "retrains": False,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(503, f"Could not load frozen checkpoint: {exc}") from exc
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc
    row = predict_pil(ENGINE["model"], im, ENGINE["tfm"], ENGINE["device"])
    return format_prediction(file.filename, row)
