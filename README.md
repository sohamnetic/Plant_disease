# Plant disease classifier

EfficientNet-B0 (ImageNet init) for 4 cauliflower / brassica classes:

`bacterial_spot_rot` · `black_rot` · `downy_mildew` · `no_disease`

**Selected checkpoint:** Experiment 3 (combined original+augmented train, original-only val).  
Held-out original-only test: accuracy **0.989**, macro-F1 **0.991** (n=93). Full writeup: [`outputs/final/MODEL_PERFORMANCE_REPORT.md`](outputs/final/MODEL_PERFORMANCE_REPORT.md).

The image **dataset is not in this repo** (too large). Training/eval still expect a local `dataset/` folder.

This model **does not output disease severity**. API `confidence` is softmax certainty, not how severe the infection is.

## Inference API

From the project root (after `pip install -r requirements.txt`):

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
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
