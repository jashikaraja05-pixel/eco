from fastapi.testclient import TestClient

import common
import main as service


KEY = "test-classifier-key"
AUTH = {"X-Classifier-Key": KEY}


def _client(monkeypatch, predict=None, local=None, api_key=""):
    """Builds a TestClient with controlled engine state and a required classifier key."""
    monkeypatch.setattr(service, "AI_API_KEY", api_key)
    monkeypatch.setattr(service, "CLASSIFIER_API_KEY", KEY)
    monkeypatch.setattr(service, "ALLOW_UNAUTHENTICATED", False)
    monkeypatch.setattr(service, "get_local", lambda: (local, {} if local else None))
    if predict is not None:
        monkeypatch.setattr(service, "predict_local", predict)
    return TestClient(service.app)


PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
TXT = b"not an image"


def test_health_reports_engine(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] in ("gateway-fallback", "none") or body["engine"].startswith("local-ml")
    assert body["uncertain_threshold"] == common.UNCERTAIN_THRESHOLD


def test_classify_rejects_non_images(monkeypatch):
    c = _client(monkeypatch, local=object())
    r = c.post("/classify", files={"file": ("x.txt", TXT, "text/plain")}, headers=AUTH)
    assert r.status_code == 415


def test_classify_without_engine_is_503(monkeypatch):
    c = _client(monkeypatch, local=None, api_key="")
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")}, headers=AUTH)
    assert r.status_code == 503
    assert "train the local model" in r.json()["detail"]


def test_classify_local_payload_contract(monkeypatch):
    def fake_predict(bytes_, mime):
        return {"item": "plastic", "category": "recyclable", "confidence": 92,
                "reasoning": "test", "steps": common.DISPOSAL_STEPS["recyclable"],
                "bin": common.BINS["recyclable"], "engine": "local-ml:test", "latency_ms": 1.0}
    c = _client(monkeypatch, predict=fake_predict, local=object())
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"item", "category", "confidence", "reasoning", "steps", "bin"}
    assert body["category"] == "recyclable"
    assert body["bin"] == common.BINS["recyclable"]


def test_uncertainty_threshold_forces_uncertain(monkeypatch):
    """Below the 55% threshold the service must report uncertain, not a bin guess."""
    seen = {}

    def fake_predict(bytes_, mime):
        seen["called"] = True
        return {"item": "glass", "category": "residual", "confidence": 40,
                "reasoning": "test", "steps": ["x"], "bin": common.BINS["residual"]}
    # gateway path applies the threshold in-process; call it directly with a stubbed HTTP response
    import httpx
    payload = {"choices": [{"message": {"content": '{"item": "glass", "category": "residual", "confidence": 40, "reasoning": "r", "steps": ["a"]}'}}]}

    class FakeResp:
        status_code = 200
        def json(self):
            return payload

    def fake_post(*a, **k):
        return FakeResp()

    monkeypatch.setattr(service, "AI_API_KEY", "key")
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    result = service.predict_gateway(PNG, "image/png")
    assert result["category"] == "uncertain"
    assert result["bin"] == common.BINS["uncertain"]
    assert not seen


def test_gateway_unknown_category_rejected(monkeypatch):
    import httpx
    payload = {"choices": [{"message": {"content": '{"item": "x", "category": "nuclear", "confidence": 99, "reasoning": "r", "steps": []}'}}]}

    class FakeResp:
        status_code = 200
        def json(self):
            return payload

    monkeypatch.setattr(service, "AI_API_KEY", "key")
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: FakeResp())
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        service.predict_gateway(PNG, "image/png")
    assert e.value.status_code == 502


def test_classify_requires_key(monkeypatch):
    c = _client(monkeypatch, local=object())
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")})
    assert r.status_code == 401


def test_classify_rejects_wrong_key(monkeypatch):
    c = _client(monkeypatch, local=object())
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")}, headers={"X-Classifier-Key": "wrong"})
    assert r.status_code == 401


def test_classify_without_configured_key_is_503(monkeypatch):
    c = _client(monkeypatch, local=object())
    monkeypatch.setattr(service, "CLASSIFIER_API_KEY", "")
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")}, headers=AUTH)
    assert r.status_code == 503


def test_health_is_public(monkeypatch):
    c = _client(monkeypatch)
    assert c.get("/health").status_code == 200


def test_oversized_image_is_413(monkeypatch):
    c = _client(monkeypatch, local=object())
    big = PNG + b"0" * (common.MAX_BYTES + 1)
    r = c.post("/classify", files={"file": ("x.png", big, "image/png")}, headers=AUTH)
    assert r.status_code == 413


def test_safety_warnings_never_suggest_opening_devices():
    for category in ("hazardous", "ewaste"):
        text = " ".join(common.SAFETY_WARNINGS[category]).lower()
        assert "do not" in text
