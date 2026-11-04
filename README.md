# EcoLens AI — Smart Waste Segregation System

**Scan. Sort. Save.**

**Neural Storm 2026 · Track: Green Planet · Problem Statement GP-05: Smart Waste Segregation System**

| | |
|---|---|
| Team name | solo stack |
| Team leader | Jashika Rajasekaran |
| Institution | Rathinam Technical Campus |

Live app: <https://eco-len.lovable.app> _(update to this project's published address after publishing)_

Demo video (2 min): **[ADD VIDEO LINK HERE]**

![Sign-in screen](docs/screenshots/auth.png)

## Problem

Mixed household waste ruins recycling streams and sends hazardous items (batteries, aerosols, e-waste) to landfill. Most people don't know which bin an item belongs in, and segregation guides are easy to forget.

## Solution

EcoLens AI lets anyone photograph a waste item (upload or live camera). A vision AI model identifies the item, assigns it to one of five categories, recommends the correct bin, and gives disposal steps. When the model is not confident, the app reports **Uncertain** and asks for a rescan instead of guessing. Every scan is stored in the user's private history and feeds a personal dashboard.

| Category | Bin |
|---|---|
| Organic | Green Bin (wet waste) |
| Recyclable | Blue Bin (dry recyclable) |
| Hazardous / E-waste | Red Bin (hazardous / e-waste depot) |
| Residual | Black Bin (non-recyclable) |
| Uncertain | No bin — rescan (confidence < 55 %, blurry image, or several mixed items) |

## Screenshots

| | |
|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Scanner](docs/screenshots/scan.png) |
| ![History](docs/screenshots/history.png) | ![Learn Hub](docs/screenshots/learn.png) |
| ![Organic result](docs/screenshots/result-organic.png) | ![Uncertain result](docs/screenshots/result-uncertain-mixed.png) |

## Design

Oceanic Teal theme: bright cyan and mint gradients with white rounded cards. Bin colours stay green, blue, red and black so they match real municipal bins.

## Features

- Email/password accounts (Lovable Cloud auth, hashed passwords, email confirmation)
- Image upload + live camera capture, client-side resize, server-side type/size/magic-byte validation
- Real vision-model classification with confidence, explanation, bin and disposal steps
- Uncertain-result handling; explicit error state — no result is shown or saved if the AI fails
- History stored per user in a database with row-level security; single-scan delete (with confirmation), clear all, delete account, search and filter by category
- Dashboard computed from the user's stored scans: category counts, category split, average confidence, recent scans
- Learn hub with consistent cards for all five waste categories plus the Uncertain rule
- Python FastAPI + PyTorch ML service (see *Python backend* below) — local MobileNetV3 classifier, vision-API fallback only when explicitly configured

## Technology stack

- **Frontend:** React 19, TanStack Start (SSR + server functions), Tailwind CSS v4, shadcn/ui, lucide-react icons
- **Backend:** Lovable Cloud (hosted Postgres with row-level security, email/password auth, server functions)
- **AI:** vision models through the Lovable AI Gateway (`openai/gpt-6-astra`) via the AI SDK
- **Optional service:** Python 3.11+, FastAPI, uvicorn, PyTorch + torchvision (MobileNetV3-Small), scikit-learn
- **Testing:** Vitest (unit), Playwright (end-to-end)

## Architecture

```text
Browser (React, TanStack Start)
   │  upload / camera image (data URL)
   ▼
classifyWaste server function  (auth required, validates image)
   ├─ if PYTHON_CLASSIFIER_URL set ─► Python FastAPI /classify
   │                                     ├─ local MobileNetV3 model (PyTorch, trained weights)  ← primary
   │                                     └─ vision model API — ONLY if AI_API_KEY is configured  ← explicit fallback
   └─ else ─────────────────────────► Lovable AI Gateway (openai/gpt-6-astra vision)
   │  normalizePrediction(): server-owned bin mapping + uncertainty threshold
   ▼
Lovable Cloud Postgres `scans` table (RLS: user can only see own rows)
```

**Production AI architecture as deployed today:** classification runs in the authenticated `classifyWaste` server function and calls the Lovable AI Gateway with the managed app key (`LOVABLE_API_KEY`). The Python service's local PyTorch model is the intended primary engine once trained and deployed (`backend/README.md`); the vision API inside it is a fallback used only when `AI_API_KEY` is set. There is **no silent fallback**: if the Python service is configured and unreachable, the scan fails visibly.

### Frontend (`src/`)
- `src/routes/index.tsx` — UI (auth, scanner, dashboard, history, learn)
- `src/lib/classify.functions.ts` — authenticated server functions (classify + save, delete account)
- `src/lib/scan-schema.ts` — image validation, prediction normalization, bin mapping
- `src/lib/eco-store.ts` — row mapping and dashboard statistics

### Python backend (`backend/`)
- `backend/main.py` — FastAPI service, `POST /classify`, `GET /health`; local ML engine first, vision-API fallback only when `AI_API_KEY` is set
- `backend/prepare_dataset.py` — downloads a labelled public waste dataset (TrashNet, MIT — verified; optional 12-class Kaggle set), maps classes to app categories, writes a leakage-free stratified train/val/test split
- `backend/train.py` — MobileNetV3-Small transfer learning, early stopping, saves weights + preprocessing config
- `backend/evaluate.py` — accuracy, precision/recall/F1 (macro, weighted, per-class), confusion matrix, inference latency — from the held-out test split only
- `backend/waste_model.py`, `backend/common.py` — model factory, preprocessing, shared category/bin/threshold constants
- `backend/tests/` — pytest suite (API contract, uncertainty rule, image sniffing, model shape)
- `backend/Dockerfile`, `backend/requirements.txt`, `backend/README.md` (full ML documentation)

The local CNN predicts dataset classes and maps them to app categories; bins and the 55 % uncertainty rule stay server-owned. `item` labels are dataset class names — never invented product names. The service returns HTTP errors (never fake predictions) on any failure.

## Run locally

### Web app
```bash
cp .env.example .env   # fill in your values
bun install
bun run dev
```

### Python service
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# genuine local ML (dataset is MIT-licensed TrashNet by default):
python prepare_dataset.py --source hf --data-dir data     # download + split (no leakage)
python train.py --split data/split.json --epochs 25       # MobileNetV3 transfer learning
python evaluate.py --split data/split.json                # held-out test metrics ONLY

uvicorn main:app --port 8000                              # /health shows the active engine
```

Deploy it to any Python host (Render, Railway, Fly.io…) with start command `uvicorn main:app --host 0.0.0.0 --port $PORT`. For a local-ML deployment, keep the trained `models/` directory alongside the code (or use the included Dockerfile). Set `AI_API_KEY` only if you want the vision-API fallback. The URL must be publicly reachable over HTTPS. Then set the app secret `PYTHON_CLASSIFIER_URL=https://your-service` to route classifications through it. Without that secret the app uses the built-in AI path.

## Environment variables

| Name | Where | Purpose |
|---|---|---|
| `LOVABLE_API_KEY` | app server (managed by Lovable) | Built-in AI classification |
| `PYTHON_CLASSIFIER_URL` | app server (optional) | Use the FastAPI service instead |
| `AI_API_KEY` | Python service | Vision API key |
| `AI_BASE_URL` | Python service (optional) | Default `https://ai.gateway.lovable.dev/v1` |
| `AI_MODEL` | Python service (optional) | Default `google/gemini-3-flash-preview` |
| `CORS_ORIGINS` | Python service (optional) | Comma-separated origins, default `*` |

No keys are hardcoded in the repository. Copy `.env.example` (root) and `backend/.env.example` to `.env` and fill in values; `.env` files are git-ignored.

## API

`GET /health` → `{"status":"ok","engine":"local-ml:trashnet","local_model":true,"gateway_fallback_configured":false,...}`
The `engine` field reports what is actually serving: `local-ml:<dataset>` after training, `gateway-fallback` when only `AI_API_KEY` is configured, `none` otherwise.

`POST /classify` (multipart, field `file`, JPEG/PNG/WebP ≤ 6 MB) →
```json
{"item":"plastic water bottle","category":"recyclable","confidence":100,
 "reasoning":"...","steps":["..."],"bin":"Blue Bin (Dry Recyclable Waste)"}
```
Both engines return the same fields (`item, category, confidence, reasoning, steps`; the Python service also returns `bin`). The app re-derives `bin` and the uncertain rule itself in `normalizePrediction`, so both paths are stored identically.

Errors: `413` too large, `415` not an image, `422` missing file field, `503` no key, `502` model error/malformed output, `402/429` usage limits.

## Testing

### Unit tests (`bunx vitest run`)
- Bin mapping for all five categories, uncertainty threshold (54 → uncertain, 55 → kept)
- Rejection of malformed model output and non-image uploads, oversized-image rejection
- Dashboard statistics (category counts, e-waste grouped under hazardous, average over confident scans only)

### End-to-end (built-in AI path, run 3 Nov 2026, real UI in a headless browser)

Test photos were AI-generated. Confidence is the model's self-reported estimate — **not classification accuracy**. No accuracy, precision, recall, F1 or latency numbers are claimed because none were measured.

| Case | Result |
|---|---|
| Banana peel | organic, 100 % |
| Single AA battery | hazardous, 100 % |
| Two items (aerosol can + plastic bottle) | uncertain — model reported two items, no single main item |
| Blurry photo | uncertain |
| Non-image upload (text file) | rejected in the browser with "Please choose an image file." |
| Reload after scans | history and dashboard rebuilt from the database |

### Python service checks (3 Nov 2026, local uvicorn run)
- `GET /health` → ok; `POST /classify` with a text file → `415`; oversized image → `413`; missing file field → `422`; valid image without `AI_API_KEY` → `503` with a visible message.

**NOT TESTED (honest gaps):**
- A successful end-to-end classification *through the Python service* (needs an `AI_API_KEY` on a running service — not exercised in the runs above).
- A *deployed* Python service connected via `PYTHON_CLASSIFIER_URL` (no deployment exists).
- Clicking a real email-confirmation link; a real physical camera.

### Mixed-item behaviour
The classification prompt instructs the model to answer `uncertain` when several mixed items appear with no single main item, and in the 3 Nov 2026 runs both a mixed-item photo and a blurry photo were reported as **uncertain**. This rule is enforced in the app's own confidence threshold (< 55 %), but whether the model flags a mixed photo as such depends on the model — it is asked for, not guaranteed.

## Limitations
- Accuracy depends on the third-party vision model; confidence is the model's self-reported estimate, not calibrated accuracy.
- One main item per photo is expected; mixed photos should return Uncertain (see above).
- Bin colours follow common Indian municipal rules; local rules vary.
- Images are stored as compressed data in the database (≤ 768 px).

## Deployment

The web app is published on Lovable at <https://eco-len.lovable.app> (frontend, auth, database and the classification server function all run there — no separate hosting is required). The optional Python service has its own deployment instructions in *Run locally → Python service* above.

## Demo
1. Sign up and confirm your email, then log in.
2. Scan Waste → upload or use the camera → Analyze Waste.
3. Open Dashboard and History to see stored results; try a blurry photo to see the Uncertain state.

## For judges: test account

- Email: **[ADD JUDGE TEST EMAIL]**
- Password: **[ADD JUDGE TEST PASSWORD]**

Or sign up with your own email and click the confirmation link. Each account sees only its own scans.

## AI tools disclosure
- Built with the Lovable AI coding assistant (and ChatGPT for review).
- A custom PyTorch model (MobileNetV3-Small, transfer learning on the MIT-licensed TrashNet dataset) is implemented in `backend/`; **training has not been run yet**, so the deployed app currently classifies with third-party vision models through the Lovable AI Gateway. Metrics will be added from `evaluate.py` after training.
- Test images used in the evaluation tables were AI-generated.

## License

Released under the [MIT License](LICENSE).
