"""HTTP API for the frozen v2 ensemble (Experiment 4).

Five fold checkpoints (fold*.pt) plus calibration.json are loaded once into RAM
and reused for every request. Nothing is retrained. If the process slept (Render
free tier) and RAM was cleared, the same files are loaded again, so scores for the
same image stay identical.

Every response carries a status the caller should act on:
  confident     - single diagnosis at the 95% conformal level
  uncertain     - more than one diagnosis plausible, or low calibrated confidence
  not_cauliflower - the photo does not show a cauliflower plant, leaf or head (CLIP gate,
                    src/cauliflower_gate.py); nothing else runs
  out_of_scope  - photo does not resemble the brassica photos the model knows
  retake_photo  - blurry / too dark / overexposed / too small

Run from the project root:

  python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1

Small servers (Render free, 512 MB): PLANT_BACKEND=onnx runs the same models with
onnxruntime and no PyTorch, loading one network at a time (src/onnx_runtime.py,
DEPLOY.md). Photos are then analysed one at a time, at most MAX_SIDE px.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_runtime import DEFAULT_EXP, MODEL_NAME, V2Engine, pick_device, public_result
from plant_runtime import DEFAULT_INSECT_EXP, PlantEngine, plant_public_result
from seg_runtime import DEFAULT_HOLE_WEIGHTS as HOLE_WEIGHTS
from seg_runtime import DEFAULT_WEIGHTS as SEG_WEIGHTS
from v2_common import GRADE_LABELS, GRADE_MEANING

BACKEND = os.environ.get("PLANT_BACKEND", "torch")  # "onnx": no PyTorch (src/onnx_runtime.py)
if BACKEND == "torch":
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
MAX_UPLOAD_MB = 20
MAX_SIDE = int(os.environ.get("MAX_SIDE", "0" if BACKEND == "torch" else "2048"))  # 0 = keep full size
PREDICT_LOCK = threading.Lock()  # one photo at a time (memory); /health stays responsive meanwhile

ENGINE_LOCK = threading.Lock()
ENGINE: dict = {"engine": None, "fingerprint": None, "loaded": False}
# Previous model, loaded lazily and only for the /compare page.
COMPARE_EXP = DEFAULT_EXP.parent / "experiment_004_v2_multitask_cv"
COMPARE: dict = {"engine": None}

UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Cauliflower disease check</title>
  <style>
    body { font-family: Segoe UI, sans-serif; max-width: 680px; margin: 2rem auto; padding: 0 1rem; color: #1b1b1b; }
    pre { background: #111; color: #eee; padding: 1rem; overflow: auto; font-size: 12px; }
    button { padding: 0.5rem 1rem; }
    #preview { max-width: 100%; max-height: 260px; margin-top: 1rem; display: none; border-radius: 6px; }
    .card { border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; border-left: 6px solid; }
    .confident { background: #e8f5e9; border-color: #2e7d32; }
    .uncertain { background: #fff8e1; border-color: #f9a825; }
    .out_of_scope, .retake_photo, .no_evidence, .not_cauliflower { background: #fdecea; border-color: #c62828; }
    .card h2 { margin: 0 0 .4rem; font-size: 1.25rem; }
    .muted { color: #555; font-size: .9rem; }
    .bar { background: #ddd; border-radius: 4px; height: 10px; margin: 2px 0 8px; }
    .bar span { display: block; height: 10px; border-radius: 4px; background: #2e7d32; }
    #overlay { max-width: 100%; border-radius: 6px; margin: .5rem 0; }
    table { border-collapse: collapse; width: 100%; font-size: .9rem; }
    td, th { text-align: left; padding: 4px 6px; border-bottom: 1px solid #ddd; }
  </style>
</head>
<body>
  <h1>Cauliflower disease check</h1>
  <p class="muted">For best results, take a close-up where the affected head or leaf fills most of the frame.
  <a href="/compare">Compare with the previous model</a>.</p>
  <form id="f">
    <input type="file" name="file" accept="image/*" required id="file"/>
    <button type="submit">Predict</button>
  </form>
  <img id="preview" alt="selected photo"/>
  <div id="result"></div>
  <details><summary class="muted">Full JSON response</summary><pre id="out"></pre></details>
  <script>
    const pct = v => (v * 100).toFixed(1) + "%";
    const pretty = s => s.replaceAll("_", " ");
    const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
    document.getElementById("file").addEventListener("change", e => {
      const f = e.target.files[0], img = document.getElementById("preview");
      if (f) { img.src = URL.createObjectURL(f); img.style.display = "block"; }
    });
    function render(r) {
      const titles = { confident: "Diagnosis", uncertain: "Uncertain", out_of_scope: "No reliable diagnosis", retake_photo: "Please retake the photo", no_evidence: "No reliable diagnosis", not_cauliflower: "Not a cauliflower photo" };
      let html = `<div class="card ${r.status}"><h2>${titles[r.status] || r.status}</h2><div>${esc(r.message)}</div>`;
      if (r.reliable) {
        html += `<p><b>${pretty(r.prediction)}</b> &middot; confidence ${pct(r.confidence)}</p>`;
        for (const [k, v] of Object.entries(r.class_probabilities))
          html += `<div class="muted">${pretty(k)} ${pct(v)}</div><div class="bar"><span style="width:${v*100}%"></span></div>`;
        if (r.plant) {
          const p = r.plant;
          const ins = p.insect || {found: false};
          if (r.prediction === "insect_damage")
            html += `<p><b>Stage:</b> ${esc(p.stage)} &middot; ${p.affected_pct == null ? "eaten area could not be measured (edge bites / scraped patches)" : p.affected_pct.toFixed(1) + "% of the worst leaf eaten"} &middot; ${p.lesion_count} holes <span class="muted">(draft thresholds, not expert-checked; bites from the leaf edge are not counted)</span></p>`;
          else if (r.prediction !== "no_disease")
            html += `<p><b>Stage:</b> ${esc(p.stage)} &middot; ${p.affected_pct == null ? "affected area too small or faint to measure" : p.affected_pct.toFixed(1) + "% of the worst part affected"} &middot; ${p.lesion_count} spots <span class="muted">(draft thresholds, not expert-checked)</span></p>`;
          if (ins.found && r.prediction !== "insect_damage")
            html += `<p><b>Insect holes:</b> ${ins.eaten_pct.toFixed(1)}% of the worst leaf eaten &middot; ${ins.hole_count} holes &middot; ${esc(ins.stage)}</p>`;
          if (r.overlay_jpeg_b64)
            html += `<img id="overlay" src="data:image/jpeg;base64,${r.overlay_jpeg_b64}" alt="lesions highlighted"/><div class="muted">Pink = affected tissue; orange = eaten by insects; cyan = leaf; yellow = head.</div>`;
          html += `<table><tr><th>Part</th><th>Result</th><th>Spot area</th><th>Spots</th><th>Eaten</th></tr>`;
          for (const g of p.regions)
            html += `<tr><td>${esc(g.type === "curd" ? "head" : g.type)}</td><td>${g.disease === "no_disease" ? "healthy" : esc(pretty(g.disease))}<div class="muted">${esc(g.verdict)}</div></td><td>${g.measured_affected_pct == null ? "-" : g.measured_affected_pct.toFixed(1) + "%"}</td><td>${g.lesion_count}</td><td>${g.eaten_pct ? g.eaten_pct.toFixed(1) + "%" : "-"}</td></tr>`;
          html += `</table>`;
        }
        const adviceBlock = (a, title) => {
          let h = `<h3 style="margin-bottom:.2rem">${title}</h3>`;
          if (a.about) h += `<div class="muted">${esc(a.about)}</div>`;
          if (a.check_first) h += `<p><b>${esc(a.check_first)}</b></p>`;
          h += `<ul>${a.actions.concat(a.general).map(x => `<li>${esc(x)}</li>`).join("")}</ul>`;
          h += `<div class="muted">Draft advice from university extension guides, not yet checked by an expert. ${a.sources.map((u, i) => `<a href="${esc(u)}" target="_blank" rel="noopener">source ${i + 1}</a>`).join(" ")}</div>`;
          return h;
        };
        if (r.advice) {
          html += adviceBlock(r.advice, "What to do");
          if (r.insect_advice) html += adviceBlock(r.insect_advice, "Insect holes: what to do");
        } else if (r.status === "uncertain") {
          html += `<p class="muted">No treatment advice while the diagnosis is uncertain: take a closer photo or ask an expert.</p>`;
        }
      } else if (r.status !== "not_cauliflower") {
        html += `<p class="muted">The model's internal guess is hidden because it is not trustworthy for this photo (see JSON below if needed).</p>`;
      }
      document.getElementById("result").innerHTML = html + "</div>";
    }
    document.getElementById("f").addEventListener("submit", async (e) => {
      e.preventDefault();
      const out = document.getElementById("out"), res = document.getElementById("result");
      res.innerHTML = "<p>Running... (on the free server this takes 1-3 minutes)</p>"; out.textContent = "";
      let r = await fetch("/jobs", { method: "POST", body: new FormData(e.target) });
      if (r.ok) {  // a ticket: ask for the result every 5 s
        const id = (await r.json()).job_id;
        let j = { status: "queued" };
        while (j.status === "queued" || j.status === "running") {
          await new Promise(ok => setTimeout(ok, 5000));
          r = await fetch("/jobs/" + id);
          if (!r.ok) break;
          j = await r.json();
        }
        if (r.ok && j.status === "error") r = new Response(JSON.stringify({ detail: j.detail }), { status: 500 });
        else if (r.ok) r = new Response(JSON.stringify(j.result));
      }
      const t = await r.text();
      try { const j = JSON.parse(t); out.textContent = JSON.stringify(j, null, 2); if (r.ok) render(j); else res.innerHTML = `<div class="card retake_photo">${esc(j.detail || t)}</div>`; }
      catch { res.innerHTML = ""; out.textContent = t; }
    });
  </script>
</body>
</html>
"""


COMPARE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Model comparison</title>
  <style>
    body { font-family: Segoe UI, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1b1b1b; }
    button { padding: 0.5rem 1rem; }
    #preview { max-width: 100%; max-height: 240px; margin-top: 1rem; display: none; border-radius: 6px; }
    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    @media (max-width: 640px) { .cols { grid-template-columns: 1fr; } }
    .card { border-radius: 8px; padding: 1rem 1.2rem; border-left: 6px solid; }
    .confident { background: #e8f5e9; border-color: #2e7d32; }
    .uncertain { background: #fff8e1; border-color: #f9a825; }
    .out_of_scope, .retake_photo, .no_evidence, .not_cauliflower { background: #fdecea; border-color: #c62828; }
    .card h2 { margin: 0 0 .2rem; font-size: 1.05rem; }
    .muted { color: #555; font-size: .85rem; }
    .bar { background: #ddd; border-radius: 4px; height: 8px; margin: 2px 0 6px; }
    .bar span { display: block; height: 8px; border-radius: 4px; background: #2e7d32; }
  </style>
</head>
<body>
  <h1>Previous vs current model</h1>
  <p class="muted">Upload one photo; both models answer it. <a href="/">Back to the normal page</a>.
  The first comparison takes a few seconds longer while the previous model loads.</p>
  <form id="f">
    <input type="file" name="file" accept="image/*" required id="file"/>
    <button type="submit">Compare</button>
  </form>
  <img id="preview" alt="selected photo"/>
  <div class="cols" id="result"></div>
  <script>
    const pct = v => (v * 100).toFixed(1) + "%";
    const pretty = s => s.replaceAll("_", " ");
    const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
    document.getElementById("file").addEventListener("change", e => {
      const f = e.target.files[0], img = document.getElementById("preview");
      if (f) { img.src = URL.createObjectURL(f); img.style.display = "block"; }
    });
    function card(title, r) {
      let h = `<div class="card ${r.status}"><div class="muted">${esc(title)} (${esc(r.model)})</div>`;
      h += `<h2>${r.reliable ? esc(pretty(r.prediction)) + " &middot; " + pct(r.confidence) : "No diagnosis"}</h2><div>${esc(r.message)}</div>`;
      const probs = r.reliable ? r.class_probabilities : (r.unreliable_guess ? r.unreliable_guess.class_probabilities : {});
      if (!r.reliable && r.unreliable_guess) h += `<p class="muted">Internal guess (not shown to users): ${esc(pretty(r.unreliable_guess.prediction))}</p>`;
      for (const [k, v] of Object.entries(probs))
        h += `<div class="muted">${pretty(k)} ${pct(v)}</div><div class="bar"><span style="width:${v*100}%"></span></div>`;
      return h + "</div>";
    }
    document.getElementById("f").addEventListener("submit", async (e) => {
      e.preventDefault();
      const res = document.getElementById("result");
      res.innerHTML = "<p>Running both models...</p>";
      const r = await fetch("/compare", { method: "POST", body: new FormData(e.target) });
      const j = await r.json();
      res.innerHTML = r.ok ? card("Previous", j.previous) + card("Current", j.current)
                           : `<div class="card retake_photo">${esc(j.detail)}</div>`;
    });
  </script>
</body>
</html>
"""


def _fingerprint(exp_dir: Path) -> str:
    h = hashlib.sha256()
    if BACKEND == "onnx":
        from onnx_runtime import MODELS_DIR
        files = sorted(p for p in MODELS_DIR.rglob("*") if p.is_file())
    else:
        insect = sorted(DEFAULT_INSECT_EXP.glob("fold*.pt")) + [DEFAULT_INSECT_EXP / "calibration.json"] if (DEFAULT_INSECT_EXP / "calibration.json").is_file() else []
        files = sorted(exp_dir.glob("fold*.pt")) + [exp_dir / "calibration.json"] + insect + [SEG_WEIGHTS] + ([HOLE_WEIGHTS] if HOLE_WEIGHTS.is_file() else [])
    for path in files:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def ensure_model_loaded() -> PlantEngine:
    """Load classifier ensemble + segmentation model once. Reload only after a process restart."""
    if ENGINE["loaded"]:
        return ENGINE["engine"]
    with ENGINE_LOCK:
        if ENGINE["loaded"]:
            return ENGINE["engine"]
        if BACKEND == "onnx":
            from onnx_runtime import MODELS_DIR, build_engine
            if not (MODELS_DIR / "classifier" / "calibration.json").is_file():
                raise RuntimeError(f"ONNX models missing in {MODELS_DIR} (see DEPLOY.md). The API does not train a model.")
            ENGINE["engine"] = build_engine(MODELS_DIR)
        elif not (DEFAULT_EXP / "calibration.json").is_file():
            raise RuntimeError(f"Frozen ensemble missing in {DEFAULT_EXP}. The API does not train a model.")
        else:
            ENGINE["engine"] = PlantEngine(DEFAULT_EXP, SEG_WEIGHTS, device=pick_device(), seg_device="cpu")
        ENGINE["fingerprint"] = _fingerprint(DEFAULT_EXP)
        ENGINE["loaded"] = True
        print("Frozen ensemble ready:", DEFAULT_EXP, f"(backend {BACKEND})")
        return ENGINE["engine"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_model_loaded()
    yield


app = FastAPI(
    title="Plant disease classifier",
    description="Frozen EfficientNet-B0 ensembles + YOLO11-seg: diagnosis per leaf/head, calibrated confidence, measured affected area and stage; insect damage with the % of the leaf eaten.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return UPLOAD_PAGE


@app.get("/health")
def health() -> dict:
    try:
        engine = ensure_model_loaded()
    except Exception as exc:
        return {"status": "error", "loaded": False, "detail": str(exc)}
    return {
        "status": "ok",
        "loaded": True,
        "model": MODEL_NAME,
        "segmentation_model": str(SEG_WEIGHTS.relative_to(DEFAULT_EXP.parents[1])),
        "insect_hole_model": str(HOLE_WEIGHTS.relative_to(DEFAULT_EXP.parents[1])) if engine.seg.hole_model is not None else None,
        "checkpoints": engine.checkpoints,
        "fingerprint_sha256": ENGINE["fingerprint"],
        "classes": engine.classes + (["insect_damage"] if engine.insect_cls is not None else []),
        "insect_model": DEFAULT_INSECT_EXP.name if engine.insect_cls is not None else None,
        "severity_grades": {g: f"{GRADE_LABELS[g]}: {GRADE_MEANING[g]}" for g in range(len(GRADE_LABELS))},
        "device": str(engine.device),
        "backend": BACKEND,
        "memory_mb": _memory(),
        "retrains": False,
    }


def _memory() -> dict | None:
    if BACKEND != "onnx":
        return None
    from onnx_runtime import memory_mb
    return memory_mb()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Diagnose one photo. (Plant tracking, src/tracking.py, is parked and not exposed.)"""
    try:
        engine = ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(503, f"Could not load frozen ensemble: {exc}") from exc
    im = await read_image(file)
    result = await run_in_threadpool(_predict_locked, engine, im)
    return {"filename": file.filename, **result, "model": MODEL_NAME, "fingerprint_sha256": ENGINE["fingerprint"]}


# ---------------------------------------------------------------- jobs (slow servers)
# On Render free a photo takes 1-3 minutes, and a request left open that long can be cut
# on the way back. POST /jobs answers at once with a job_id; GET /jobs/{job_id} returns
# the result when it is ready. Jobs live in memory: a restart or sleep clears them.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_KEEP_S = 3600  # finished results are kept 1 hour
MAX_WAITING = 5  # photos in line (each held in memory until its turn)


def _job_set(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields)


def _run_job(job_id: str, engine: PlantEngine, im: Image.Image, filename: str | None) -> None:
    try:
        with PREDICT_LOCK:
            _job_set(job_id, status="running")
            out = plant_public_result(engine.predict_pil(im, with_overlay=True))
            print("job done, memory MB:", _memory(), flush=True)
        result = {"filename": filename, **out, "model": MODEL_NAME, "fingerprint_sha256": ENGINE["fingerprint"]}
        _job_set(job_id, status="done", result=result, finished=time.time())
    except Exception as exc:
        _job_set(job_id, status="error", detail=f"Prediction failed: {exc}", finished=time.time())


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile = File(...)) -> dict:
    """Start diagnosing one photo; returns a job_id at once. Poll GET /jobs/{job_id}."""
    try:
        engine = ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(503, f"Could not load frozen ensemble: {exc}") from exc
    im = await read_image(file)
    now = time.time()
    with JOBS_LOCK:
        for k in [k for k, j in JOBS.items() if now - j.get("finished", now) > JOB_KEEP_S]:
            del JOBS[k]
        if sum(j["status"] in ("queued", "running") for j in JOBS.values()) > MAX_WAITING:
            raise HTTPException(429, "Too many photos in line. Try again in a few minutes.")
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"job_id": job_id, "status": "queued", "filename": file.filename}
    threading.Thread(target=_run_job, args=(job_id, engine, im, file.filename), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "result_url": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """status: queued | running | done (result = same JSON as /predict) | error (detail)."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job. Results are kept 1 hour, and a server restart clears them.")
        return {k: v for k, v in job.items() if k != "finished"}


def _predict_locked(engine: PlantEngine, im: Image.Image, with_overlay: bool = True) -> dict:
    with PREDICT_LOCK:
        out = plant_public_result(engine.predict_pil(im, with_overlay=with_overlay))
        print("predict done, memory MB:", _memory(), flush=True)
        return out


async def read_image(file: UploadFile) -> Image.Image:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    if len(raw) > MAX_UPLOAD_MB * 2**20:
        raise HTTPException(413, f"Photo larger than {MAX_UPLOAD_MB} MB")
    try:
        im = Image.open(io.BytesIO(raw))
        if MAX_SIDE:
            im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode at reduced size, saves memory
            im = ImageOps.exif_transpose(im)
            im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.load()
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}") from exc
    return im


def compare_engine() -> V2Engine:
    if COMPARE["engine"] is None:
        with ENGINE_LOCK:
            if COMPARE["engine"] is None:
                if not (COMPARE_EXP / "calibration.json").is_file():
                    raise HTTPException(404, f"Previous model not found: {COMPARE_EXP}")
                COMPARE["engine"] = V2Engine(COMPARE_EXP, device=pick_device())
    return COMPARE["engine"]


@app.get("/compare", response_class=HTMLResponse)
def compare_page() -> str:
    return COMPARE_PAGE


@app.post("/compare")
async def compare(file: UploadFile = File(...)) -> dict:
    """Same photo through the previous (v2) and current model, for side-by-side testing."""
    if BACKEND == "onnx":
        raise HTTPException(404, "The comparison with the previous model is not available on this server.")
    im = await read_image(file)
    engine = ensure_model_loaded()
    g = engine.check_cauliflower(im)
    if g is not None and not g["is_cauliflower"]:
        refused = public_result(engine.not_cauliflower(g))
        return {"filename": file.filename, "previous": {"model": COMPARE_EXP.name, **refused},
                "current": {"model": MODEL_NAME, **refused}}
    old = compare_engine()
    return {
        "filename": file.filename,
        "previous": {"model": COMPARE_EXP.name, **public_result(old.predict_pil(im))},
        "current": {"model": MODEL_NAME, **plant_public_result(ensure_model_loaded().predict_pil(im, with_overlay=False))},
    }
