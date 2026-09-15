# Plant disease classifier

EfficientNet-B0 (ImageNet init) for 4 cauliflower / brassica classes:

`bacterial_spot_rot` · `black_rot` · `downy_mildew` · `no_disease`

**Selected checkpoint:** Experiment 3 (combined original+augmented train, original-only val).  
Held-out original-only test: accuracy **0.989**, macro-F1 **0.991** (n=93). Full writeup: [`outputs/final/MODEL_PERFORMANCE_REPORT.md`](outputs/final/MODEL_PERFORMANCE_REPORT.md).

The image **dataset is not in this repo** (too large). Training/eval still expect a local `dataset/` folder.

The trained network is **`experiments/experiment_003_effnetb0_combined/best_model.pt`** (PyTorch weights, not a `.pkl`). The API never retrains. It loads that file **once into RAM** and reuses it for every request. If the host sleeps (Render free plan), the `.pt` file is still on disk and is loaded again — scores for the same photo stay the same.

`disease_score` is the predicted class probability on a **0–100** scale (softmax). That is **not** agronomic severity (percent leaf area infected).

## Inference API

From the project root (after `pip install -r requirements.txt`):

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 1
```

| What | URL |
|---|---|
| Upload form | http://127.0.0.1:8000/ |
| Swagger docs | http://127.0.0.1:8000/docs |
| Health | `GET` http://127.0.0.1:8000/health |
| Predict | `POST` http://127.0.0.1:8000/predict  (multipart field `file`) |

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/leaf.jpg"
```

CLI (no server):

```bash
python src/predict.py --image path/to/leaf.jpg
```

## Reproduce / train

Frozen splits: `splits/` (do not regenerate).  
Training scripts refuse `splits/test.csv`. Use `src/predict.py` / `src/api.py` for new photos.
