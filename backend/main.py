"""EcoLens AI — Python FastAPI classification service.

POST /classify (multipart "file") -> {item, category, confidence, reasoning, steps, bin}

Engine order:
1. Local PyTorch model (MobileNetV3-Small trained via prepare_dataset.py +
   train.py) when backend/models/best.pt + config.json exist — genuine local
   inference, no external API.
2. Vision-model fallback through an OpenAI-compatible Chat Completions API —
   used ONLY when explicitly configured with AI_API_KEY (e.g. the Lovable AI
   Gateway). Not a silent substitute: /health reports which engine is active.

No predictions are fabricated. On failure the endpoint returns an HTTP error.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from common import ALLOWED, BINS, CATEGORIES, DISPOSAL_STEPS, MAX_BYTES, SAFETY_WARNINGS, UNCERTAIN_THRESHOLD, sniff

AI_BASE_URL = os.getenv("AI_BASE_URL", "https://ai.gateway.lovable.dev/v1").rstrip("/")
AI_API_KEY = os.getenv("AI_API_KEY", "")
# Shared secret required in the X-Classifier-Key header for /classify.
CLASSIFIER_API_KEY = os.getenv("CLASSIFIER_API_KEY", "")
ALLOW_UNAUTHENTICATED = os.getenv("ALLOW_UNAUTHENTICATED", "") == "1"  # local dev only
AI_MODEL = os.getenv("AI_MODEL", "google/gemini-3-flash-preview")
MODELS_DIR = Path(os.getenv("ML_MODEL_DIR", Path(__file__).parent / "models"))
PROMPT = (
    "You are a waste segregation expert. Classify the single main waste item in the image into one of: "
    "organic, recyclable, hazardous, ewaste, residual, or uncertain (blurry, no waste item, mixed items). "
    "confidence = your honest probability 0-100. Reply with ONLY JSON: "
    '{"item": str, "category": str, "confidence": number, "reasoning": str, "steps": [str, ...3-5]}'
)

app = FastAPI(title="EcoLens AI Classifier", version="2.0.0")
# The app calls this service server-to-server, so browsers need no CORS by default.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                       allow_methods=["POST", "GET"], allow_headers=["Content-Type", "X-Classifier-Key"])


def require_key(provided: str | None) -> None:
    if ALLOW_UNAUTHENTICATED and not CLASSIFIER_API_KEY:
        return
    if not CLASSIFIER_API_KEY:
        raise HTTPException(503, "Service misconfigured: CLASSIFIER_API_KEY is not set.")
    if not provided or not hmac.compare_digest(provided.encode(), CLASSIFIER_API_KEY.encode()):
        raise HTTPException(401, "Missing or invalid X-Classifier-Key header.")

# Local model is loaded lazily so the service still starts (and /health works)
# before training has been run.
_local_model = None
_local_config: dict | None = None


def get_local():
    global _local_model, _local_config
    if _local_config is None:
        try:
            from waste_model import load_model
            _local_model, _local_config = load_model(MODELS_DIR)
        except (FileNotFoundError, ImportError) as e:
            _local_config = {}
            print(f"Local ML model not available ({e}); using configured fallback only.")
    return _local_model, _local_config


def engine_name() -> str:
    model, _ = get_local()
    if model is not None:
        return f"local-ml:{_local_config.get('dataset_name', 'unknown')}"
    return "gateway-fallback" if AI_API_KEY else "none"


def predict_local(image_bytes: bytes, mime: str) -> dict:
    model, config = get_local()
    if model is None:
        raise HTTPException(503, "Local ML model is not trained yet. Run prepare_dataset.py and train.py, or configure AI_API_KEY.")
    try:
        image = Image.open(__import__("io").BytesIO(image_bytes))
        image.load()
    except Exception:
        raise HTTPException(415, "Image could not be decoded.")
    if image.width < 32 or image.height < 32:
        raise HTTPException(422, "Image is too small to classify (min 32x32 px).")
    from waste_model import predict_image
    p = predict_image(model, config, image)
    category = p["category"]
    if category not in CATEGORIES:
        raise HTTPException(502, "Model returned an unknown category.")
    return {
        "item": p["item"],
        "category": category,
        "confidence": p["confidence"],
        "reasoning": p["reasoning"],
        "steps": DISPOSAL_STEPS[category],
        "bin": BINS[category],
        "warnings": SAFETY_WARNINGS.get(category, []),
        "engine": engine_name(),
        "latency_ms": p["_latency_ms"],
    }


def predict_gateway(image_bytes: bytes, mime: str) -> dict:
    url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
    body = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [{"type": "text", "text": "Classify this waste item."},
                                          {"type": "image_url", "image_url": {"url": url}}]},
        ],
    }
    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Lovable-API-Key": AI_API_KEY, "X-Lovable-AIG-SDK": "fetch"}
    try:
        with httpx.Client(timeout=90) as client:
            r = client.post(f"{AI_BASE_URL}/chat/completions", json=body, headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Vision API unreachable: {type(e).__name__}")
    if r.status_code in (402, 429):
        raise HTTPException(r.status_code, "AI usage limit reached. Try again later.")
    if r.status_code >= 400:
        raise HTTPException(502, f"Vision API error ({r.status_code}).")

    text = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise HTTPException(502, "Model did not return a classification.")
    try:
        p = json.loads(m.group(0))
        category = str(p["category"]).lower()
        confidence = max(0, min(100, round(float(p["confidence"]))))
        item, reasoning = str(p["item"])[:150], str(p["reasoning"])[:1500]
        steps = [str(s)[:500] for s in p.get("steps", [])][:6]
    except (KeyError, ValueError, TypeError):
        raise HTTPException(502, "Model returned malformed output.")
    if category not in CATEGORIES:
        raise HTTPException(502, "Model returned an unknown category.")
    if confidence < UNCERTAIN_THRESHOLD:
        category = "uncertain"
    return {"item": item, "category": category, "confidence": confidence,
            "reasoning": reasoning, "steps": steps or DISPOSAL_STEPS[category],
            "bin": BINS[category], "warnings": SAFETY_WARNINGS.get(category, []),
            "engine": engine_name()}


@app.get("/health")
def health():
    model, config = get_local()
    info = {}
    if model is not None:
        info = {"trained_dataset": config.get("dataset_name"),
                "classes": config.get("classes"),
                "trained_at": config.get("trained_at")}
    return {"status": "ok", "engine": engine_name(),
            "local_model": model is not None, "local_model_info": info,
            "gateway_fallback_configured": bool(AI_API_KEY),
            "auth_required": not (ALLOW_UNAUTHENTICATED and not CLASSIFIER_API_KEY),
            "uncertain_threshold": UNCERTAIN_THRESHOLD}


@app.post("/classify")
async def classify(file: UploadFile = File(...), x_classifier_key: str | None = Header(default=None)):
    require_key(x_classifier_key)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 6 MB).")
    mime = sniff(data)
    if not data or mime not in ALLOWED:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP image.")

    model, _ = get_local()
    if model is not None:
        return predict_local(data, mime)
    if AI_API_KEY:
        return predict_gateway(data, mime)
    raise HTTPException(503, "No classification engine available: train the local model or configure AI_API_KEY.")
