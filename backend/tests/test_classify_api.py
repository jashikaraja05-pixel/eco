from fastapi.testclient import TestClient

import common
import main as service


def _client(monkeypatch, predict=None, local=None, api_key=""):
    """Builds a TestClient with controlled engine state."""
    monkeypatch.setattr(service, "AI_API_KEY", api_key)
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
    assert body["engine"] in ("local-ml", "gateway-fallback", "none")
    assert body["uncertain_threshold"] == common.UNCERTAIN_THRESHOLD


def test_classify_rejects_non_images(monkeypatch):
    c = _client(monkeypatch, local=object())
    r = c.post("/classify", files={"file": ("x.txt", TXT, "text/plain")})
    assert r.status_code == 415


def test_classify_without_engine_is_503(monkeypatch):
    c = _client(monkeypatch, local=None, api_key="")
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")})
    assert r.status_code == 503
    assert "train the local model" in r.json()["detail"]


def test_classify_local_payload_contract(monkeypatch):
    def fake_predict(bytes_, mime):
        return {"item": "plastic", "category": "recyclable", "confidence": 92,
                "reasoning": "test", "steps": common.DISPOSAL_STEPS["recyclable"],
                "bin": common.BINS["recyclable"], "engine": "local-ml:test", "latency_ms": 1.0}
    c = _client(monkeypatch, predict=fake_predict, local=object())
    r = c.post("/classify", files={"file": ("x.png", PNG, "image/png")})
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
