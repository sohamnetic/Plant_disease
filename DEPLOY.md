# Deploying the API on Render (free plan) and using it

## Part 1: Deploy (one-time, about 15 minutes)

The free plan has **512 MB of memory and 0.1 CPU**. The normal app (PyTorch, everything kept in memory) needs about 2.4 GB, so the server runs a lightweight version instead:
- **The same models, exported to ONNX** (`src/onnx_runtime.py`), with no PyTorch installed.
- **One network loaded at a time** for each photo.
- **Peak memory is about 340 MB** (measured with 4 photos). The biggest piece is the cauliflower check (CLIP), which is compressed to 8-bit and loaded in two halves.

**Result:** the same answers as the normal app (checks in Part 3), but slower. A photo takes about 7–12 s on a laptop core, and **roughly 1–2 minutes on Render free**.

**Steps**

1. **Upload the models as a GitHub release** (Render downloads them while building).
   - On GitHub open **sohamnetic/Plant_disease → Releases → Draft a new release**.
   - Tag: `models-v1` (exactly this, because the download link uses it). Title: anything.
   - Attach the file `deploy_models.zip` (500 MB, in the project folder).
   - Click **Publish release**.

   The file is then public, like the repo.
2. **Push the code to GitHub.** Render deploys from a branch, either `main` or this branch.
3. **Create the service on Render.**
   - Open https://dashboard.render.com and choose **New → Blueprint**.
   - Connect the GitHub repo. Render reads `render.yaml`: free plan, the build and start commands, and the settings.
   - Click **Apply**. The first build takes about 5–10 minutes. It installs the small packages, then downloads and checks the models.
4. **Check it is running.** Open `https://<your-service>.onrender.com/health`. It should show `"status": "ok"` and `"backend": "onnx"`.
   - The upload page is `https://<your-service>.onrender.com/`.
   - The API documentation is at `/docs`.

**Free-plan behaviour to expect**
- **It sleeps after 15 minutes without requests.** The next request wakes it, which takes about 1 minute on top of the photo itself.
- **It handles one photo at a time.** A second photo waits in line.
- **Photos larger than 2048 px are shrunk to 2048 px.** Uploads over 20 MB are refused.

**To update the models later:**
1. Run `python src/onnx_runtime.py export` (needs PyTorch, on the laptop).
2. Run `python src/fetch_models.py --pack`.
3. Upload the new zip as release `models-v2`.
4. Set `MODELS_URL` and `MODELS_SHA256` in the Render dashboard to the new link and the printed checksum.

## Part 2: For the teammate — using the API

**Base URL:** `https://<your-service>.onrender.com`. Locally it is `http://localhost:8000`.

| Method | Path | What it does |
|---|---|---|
| `POST` | `/jobs` | **Use this on Render.** Send one photo (multipart form field `file`). Answers at once with a `job_id`. |
| `GET` | `/jobs/{job_id}` | The job's `status`: `queued`, `running`, `done` (then `result` holds the diagnosis) or `error` (then `detail`). Ask every 5 seconds. |
| `POST` | `/predict` | Same diagnosis in one request. Fine locally, but on Render the request stays open 1–3 minutes and the reply can be cut off on the way back. |
| `GET` | `/health` | Shows whether the server is up and which models it runs. Use it to wake the server before sending a photo. |
| `GET` | `/docs` | Interactive documentation. |

**Why jobs:** a photo takes 1–3 minutes on the free plan (more after a sleep). Testing on Render showed that replies of requests left open that long were cut off, so the photo is sent in one short request and the result fetched in another.
- Results are kept **1 hour**. A restart or sleep of the server clears them: if `GET /jobs/{job_id}` answers `404`, send the photo again.
- At most 5 photos wait in line; more get `429` (try again later).

### Example requests

```bash
curl -F "file=@leaf.jpg" https://<your-service>.onrender.com/jobs
# -> {"job_id": "94016ec5...", "status": "queued", "result_url": "/jobs/94016ec5..."}
curl https://<your-service>.onrender.com/jobs/94016ec5...
# -> {"job_id": "...", "status": "done", "filename": "leaf.jpg", "result": { ...diagnosis... }}
```

```python
import time, requests
URL = "https://<your-service>.onrender.com"
job = requests.post(f"{URL}/jobs", files={"file": open("leaf.jpg", "rb")}, timeout=120).json()
while True:
    time.sleep(5)
    j = requests.get(f"{URL}/jobs/{job['job_id']}", timeout=60).json()
    if j["status"] in ("done", "error"):
        break
print(j["result"]["message"] if j["status"] == "done" else j["detail"])
```

**Express (Node 18+) backend**, forwarding a photo from your frontend:

```javascript
import express from "express";
import multer from "multer";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 20 * 1024 * 1024 } });
const ML_URL = process.env.ML_URL || "https://<your-service>.onrender.com";
const sleep = ms => new Promise(r => setTimeout(r, ms));

app.post("/api/diagnose", upload.single("photo"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "photo is required" });
  const form = new FormData();
  form.append("file", new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname);
  try {
    let r = await fetch(`${ML_URL}/jobs`, { method: "POST", body: form, signal: AbortSignal.timeout(120_000) });
    if (!r.ok) return res.status(r.status === 413 ? 413 : 502).json({ error: `ML service error ${r.status}` });
    const { job_id } = await r.json();
    for (let i = 0; i < 120; i++) {            // up to 10 minutes
      await sleep(5000);
      r = await fetch(`${ML_URL}/jobs/${job_id}`, { signal: AbortSignal.timeout(60_000) });
      if (!r.ok) return res.status(502).json({ error: "The result was lost (server restarted). Please try again." });
      const j = await r.json();
      if (j.status === "done") return res.json(j.result);   // same JSON as /predict
      if (j.status === "error") return res.status(500).json({ error: j.detail });
    }
    res.status(504).json({ error: "The diagnosis took too long. Please try again." });
  } catch (e) {
    res.status(502).json({ error: "Could not reach the ML service. Please try again." });
  }
});
```

If the frontend should not wait that long on one request, return the `job_id` to it instead and let it ask your backend for the result every few seconds.

### What comes back (the fields to use)

| Field | Meaning |
|---|---|
| `message` | A ready-to-show sentence for the user, e.g. "Insect damage, early stage: 3.7% of the worst leaf eaten (51 holes; …)". |
| `status` | One of the values below. **Always check this first.** |
| `reliable` | `true` only for `confident` and `uncertain`. |
| `prediction` | `bacterial_spot_rot`, `black_rot`, `downy_mildew`, `no_disease`, `insect_damage`, or `null` when not reliable. |
| `confidence` | 0–1. |
| `plant.stage` | `none`, `early`, `moderate`, `severe` or `critical`. |
| `plant.affected_pct` | % of the worst leaf or head affected. For insect damage, the % eaten. |
| `plant.insect` | `found`, `eaten_pct`, `hole_count` and `stage`. Holes are reported even when a disease is the diagnosis. |
| `plant.regions` | One entry per leaf or head found in the photo. |
| `advice` | `about`, `check_first` and the `actions` list for the diagnosis. Draft guidance, not checked by an expert. |
| `insect_advice` | Extra advice when a disease is diagnosed **and** insect holes are found. Otherwise `null`. |
| `overlay_jpeg_b64` | The photo with affected areas coloured, as a base64 JPEG. Show it with `<img src="data:image/jpeg;base64,...">`. |
| `class_probabilities` | Probability of each result. |

The `status` values:
- `confident`: show the result.
- `uncertain`: show it as "not sure"; the `message` already says so.
- `not_cauliflower`: the photo is not a cauliflower; ask for another photo.
- `retake_photo`: blurry, dark or too small; `status_reason` says which.
- `out_of_scope` / `no_evidence`: no diagnosis.

Errors:
- `400`: empty or unreadable image.
- `413`: photo larger than 20 MB.
- `503`: models not loaded.

## Part 3: Checks that the server gives the same answers

Run `python src/onnx_runtime.py compare` and `python src/onnx_runtime.py gate-eval`. Results are in `outputs/deploy/`.

| Check | Normal app (PyTorch) | Server version (ONNX) |
|---|---|---|
| Full app on 100 test photos (80 outside test, 20 insect test): status, diagnosis, confidence, message | – | **identical on all 100** |
| Cauliflower check, 1,329 real photos (outside test, VegNet test, web, yours) | all pass | **all pass** (lowest score 0.136 vs 0.142) |
| Cauliflower check, 66 non-cauliflower images | all refused | **all refused** (highest score 0.096 vs 0.092; cutoff 0.10) |
| Peak memory | about 2.4 GB | **about 340 MB** |
| Tests (`pytest tests`) | – | 51 passed, including 4 for the ONNX version |

**Compression choices for the cauliflower check** (CLIP is 1.2 GB uncompressed):

| Option | Size | Result | Why |
|---|---|---|---|
| 4-bit | 185–230 MB | rejected | It refused 1 real photo and let 1–2 non-cauliflower images through. |
| Per-channel 8-bit | about 290 MB | rejected | Further from the original. |
| **8-bit in blocks of 32, two halves** | 330 MB | **used** | Average score shift 0.003, no wrong decisions, and only half is in memory at a time. |

The **only** differences from the normal app:
- **Cauliflower scores:** they move by at most about 0.02.
- **Large photos:** those over 2048 px are shrunk first, which can change a measured % slightly (e.g. 3.6% instead of 3.4% eaten).
