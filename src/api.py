"""HTTP API for the selected Experiment 3 EfficientNet-B0 checkpoint.

Run from the project root:

  python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

Then open:
  http://127.0.0.1:8000/          (upload form)
  http://127.0.0.1:8000/docs      (Swagger)
  POST http://127.0.0.1:8000/predict
"""

from __future__ import annotations

import io
import sys
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

STATE: dict = {}

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
  <p>Experiment 3 EfficientNet-B0. Returns class + confidence. This model does not estimate severity.</p>
  <form id="f">
    <input type="file" name="file" accept="image/*" required/>
    <button type="submit">Predict</button>
  </form>
  <pre id="out">Upload an image to see the JSON response.</pre>
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    device = pick_device()
    ckpt = DEFAULT_CKPT
    if not ckpt.is_file():
        raise RuntimeError(f"Checkpoint missing: {ckpt}")
    STATE["device"] = device
    STATE["model"] = load_checkpoint(ckpt, device)
    STATE["tfm"] = build_transforms(train=False)
    print("API ready. Checkpoint:", ckpt)
    yield
    STATE.clear()


app = FastAPI(
    title="Plant disease classifier",
    description="Frozen Experiment 3 EfficientNet-B0. Four classes. No severity score.",
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
    return {
        "status": "ok",
        "model": "experiment_003_effnetb0_combined",
        "classes": list(cfg.CLASS_ORDER),
        "device": str(STATE.get("device", "")),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if STATE.get("model") is None:
        raise HTTPException(503, "Model not loaded")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = Image.open(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc
    row = predict_pil(STATE["model"], im, STATE["tfm"], STATE["device"])
    probs = {name: row[f"prob_{name}"] for name in cfg.CLASS_ORDER}
    return {
        "filename": file.filename,
        "predicted_class": row["predicted_class"],
        "confidence": round(row["confidence"], 6),
        "probabilities": {k: round(v, 6) for k, v in probs.items()},
        "severity": None,
        "note": "This checkpoint classifies disease only. confidence is not a severity score.",
    }
