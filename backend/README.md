# EcoLens AI — Python ML classification service

FastAPI service that classifies waste photos into the app's six categories
(organic, recyclable, hazardous, e-waste, residual, uncertain) and returns the
recommended bin and disposal steps.

## Engine order

1. **Local PyTorch model** — MobileNetV3-Small, transfer learning from ImageNet,
   trained on a labelled public waste dataset. Active when `models/best.pt` +
   `models/config.json` exist (set `ML_MODEL_DIR` to override the directory).
2. **Vision-model fallback** — an OpenAI-compatible Chat Completions API
   (Lovable AI Gateway by default). Used **only when explicitly configured**
   with `AI_API_KEY`. `/health` always reports which engine is active.

No silent fallback: if neither is available, `/classify` returns HTTP 503, and
if the configured engine fails the error surfaces to the user. No prediction is
random, fabricated or hardcoded.

## Dataset

| Dataset | Source | Licence | Classes | Coverage |
|---|---|---|---|---|
| TrashNet (default) | github.com/garythung/trashnet | MIT (c) 2017 Gary Thung — verified in repo LICENSE | cardboard, glass, metal, paper, plastic, trash (2,527 images) | recyclable, residual only |
| Garbage Classification 12 classes | kaggle.com/datasets/mostafaabla/garbage-classification | check the Kaggle page before redistribution | 12 classes, 15,150 images | adds biological (organic), battery (hazardous), e-waste |

**Category mapping** (`prepare_dataset.py`): TrashNet glass/paper/cardboard/plastic/metal →
recyclable, trash → residual. Garbage12 additionally maps biological → organic,
battery → hazardous, e-waste → ewaste, clothes/shoes/trash → residual.

**Limitations (honest scope):**
- TrashNet contains no organic, hazardous or e-waste images — a TrashNet-only
  model can never genuinely predict those categories. Use `--source kaggle`
  for fuller coverage.
- "Uncertain" is never a dataset class; it is produced solely by the 55%
  confidence threshold (`UNCERTAIN_THRESHOLD`, mirrored in
  `src/lib/scan-schema.ts`).
- Dataset photos are studio-like; expect lower accuracy on real bin photos.
- `item` labels are dataset class names (the finest label the model has) —
  the service never invents a product name.

## Commands

```bash
# 1. install (Python 3.11+)
pip install -r backend/requirements.txt

# 2. dataset + stratified train/val/test split (no leakage, split fixed once)
python backend/prepare_dataset.py --source hf --data-dir backend/data      # TrashNet via Hugging Face
# or: python backend/prepare_dataset.py --source dir --dir /path/to/classes
# or: python backend/prepare_dataset.py --kaggle  (needs `pip install kaggle` + Kaggle account)

# 3. train (MobileNetV3-Small transfer learning, early stopping on val accuracy)
python backend/train.py --split backend/data/split.json --epochs 25 --batch 32

# 4. evaluate on the held-out test split ONLY (accuracy, P/R/F1, per-class,
#    confusion matrix, latency) — writes models/evaluation_report.json
python backend/evaluate.py --split backend/data/split.json

# 5. run the API
cd backend && uvicorn main:app --port 8000
# check: curl localhost:8000/health   -> {"engine": "local-ml:trashnet", ...}

# 6. tests
cd backend && pytest tests/ -q
```

Training and evaluation are **not yet run** for this project — until `train.py`
has completed, `/health` reports `engine: gateway-fallback` (if `AI_API_KEY`
is set) or `none`, and `evaluate.py` refuses to produce metrics. Do not quote
accuracy numbers before step 4 has produced them.

## Deployment (external host — the Lovable app calls this over HTTPS)

1. Train the model (steps 2–4) and keep `models/` alongside the code.
2. Deploy `backend/` (Dockerfile included) to any host: Render, Fly.io,
   Hugging Face Spaces, a VPS — anything that serves HTTPS.
   `docker build -t ecolens-classifier ./backend && docker run -p 8000:8000 ecolens-classifier`
3. In Lovable, set the secret `PYTHON_CLASSIFIER_URL` to the deployed base URL
   (e.g. `https://ecolens-classifier.onrender.com`). The app's authenticated
   `classifyWaste` server function then routes every scan to this service.
   If `PYTHON_CLASSIFIER_URL` is set but unreachable, the scan fails visibly —
   it is never silently replaced by another engine.
4. The frontend validates image type/size before upload; this service
   re-validates magic bytes and the 6 MB limit server-side.
