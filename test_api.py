"""End-to-end HTTP tests with the real app, a real SQLite database and a scripted model.

The scripted provider replays fixed completions so the whole pipeline (validation, routing,
tools, persistence, SSE) can be tested without an API key. It plays no part in production.
"""
import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import create_app
from tests.fakes import ScriptedProvider, make_settings, spec

CLIENT = {"X-Client-Id": "device-abcdef123456"}


def jpeg_b64(size=(200, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def factory(provider=None, **settings_kw):
        settings = make_settings(database_path=str(tmp_path / "t.db"), **settings_kw)
        provider = provider or ScriptedProvider(specs=[spec("fake", "m1", vision=True)], script=[("text", "The answer is $2x$.")])
        client = TestClient(create_app(settings, providers=[provider]))
        client.__enter__()
        clients.append(client)
        return client, provider

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


def test_health_and_models(make_client):
    client, _ = make_client()
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["providers"][0]["models"] == ["fake:m1"]
    models = client.get("/api/models").json()
    assert models["default"] == "auto" and models["models"][0]["vision"] is True
    assert models["features"]["math_verification"] is True and models["features"]["pdf_understanding"] is False


def test_chat_matches_the_frontend_contract_and_persists(make_client):
    client, provider = make_client()
    body = {"question": "Differentiate x^2", "subject": "mathematics", "level": "secondary", "mode": "step-by-step",
            "model": "auto", "image": None, "images": [], "conversation_id": "conv-12345",
            "history": [{"role": "user", "content": "hi", "has_images": False}, {"role": "assistant", "content": "hello", "has_images": False}],
            "features": {}, "stream": False, "client": {"name": "Personal Super AI", "locale": "en-US"}}
    r = client.post("/api/chat", json=body, headers=CLIENT)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["answer"] == "The answer is $2x$." and data["conversation_id"] == "conv-12345" and data["persisted"] is True
    assert [m.role for m in provider.calls[0]["messages"]] == ["system", "user", "assistant", "user"]

    listing = client.get("/api/history", headers=CLIENT).json()
    assert listing["total"] == 1 and listing["conversations"][0]["id"] == "conv-12345"
    detail = client.get("/api/history/conv-12345", headers=CLIENT).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]

    assert client.get("/api/history", headers={"X-Client-Id": "other-device-9999"}).json()["total"] == 0  # isolation
    assert client.get("/api/history/conv-12345", headers={"X-Client-Id": "other-device-9999"}).status_code == 404

    assert client.delete("/api/history/conv-12345", headers=CLIENT).json() == {"deleted": True, "conversation_id": "conv-12345"}
    assert client.delete("/api/history/conv-12345", headers=CLIENT).status_code == 404


def test_retrying_the_same_question_replaces_the_previous_turn(make_client):
    client, _ = make_client()
    body = {"question": "same question", "conversation_id": "conv-retry1"}
    client.post("/api/chat", json=body, headers=CLIENT)
    client.post("/api/chat", json=body, headers=CLIENT)
    assert client.get("/api/history/conv-retry1", headers=CLIENT).json()["message_count"] == 2


def test_without_client_id_chat_works_but_nothing_is_stored(make_client):
    client, _ = make_client()
    r = client.post("/api/chat", json={"question": "hi"})
    assert r.status_code == 200 and r.json()["persisted"] is False
    h = client.get("/api/history")
    assert h.status_code == 400 and "X-Client-Id" in h.json()["error"]


def test_validation_errors_use_the_error_shape_the_frontend_reads(make_client):
    client, _ = make_client()
    r = client.post("/api/chat", json={"question": "", "images": []})
    assert r.status_code == 422 and r.json()["error"].startswith("Invalid request")
    assert client.post("/api/chat", json={"question": "x", "conversation_id": "bad id!"}).status_code == 422
    assert client.post("/api/chat", json={"question": "x", "subject": "<script>"}).status_code == 422
    assert client.post("/api/chat", json={"question": "x" * 25000}).status_code == 422


def test_any_subject_and_level_label_is_accepted(make_client):
    client, provider = make_client()
    r = client.post("/api/chat", json={"question": "Explain fermentation", "subject": "food science", "level": "vocational diploma"})
    assert r.status_code == 200
    assert "food science" in provider.calls[0]["messages"][0].content


def test_image_chat_and_invalid_images(make_client):
    client, _ = make_client()
    good = {"name": "p.jpg", "mime_type": "image/jpeg", "data": jpeg_b64()}
    assert client.post("/api/chat", json={"question": "what is this?", "images": [good], "image": good}).status_code == 200
    bad = {"name": "p.jpg", "mime_type": "image/jpeg", "data": base64.b64encode(b"definitely not an image data").decode()}
    r = client.post("/api/chat", json={"question": "x", "images": [bad]})
    assert r.status_code == 422 and "readable image" in r.json()["error"]


def test_image_questions_fail_clearly_without_a_vision_model(make_client):
    text_only = ScriptedProvider(specs=[spec("fake", "m1", vision=False)])
    client, _ = make_client(provider=text_only)
    r = client.post("/api/chat", json={"question": "x", "images": [{"data": jpeg_b64()}]})
    assert r.status_code == 503 and "vision" in r.json()["error"]


def test_analyze_image_multipart_and_json(make_client):
    client, provider = make_client()
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="PNG")
    r = client.post("/api/analyze-image", files={"file": ("shot.png", buf.getvalue(), "image/png")},
                    data={"question": "read the equation", "task": "equation"}, headers=CLIENT)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["task"] == "equation" and out["images"][0]["width"] == 64
    assert "Transcribe every mathematical expression" in provider.calls[0]["messages"][0].content
    r2 = client.post("/api/analyze-image", json={"images": [{"data": jpeg_b64()}], "task": "chart"})
    assert r2.status_code == 200 and r2.json()["task"] == "chart"
    assert client.post("/api/analyze-image", json={"question": "no image"}).status_code == 422


def test_streaming_uses_sse_in_the_shape_script_js_expects(make_client):
    client, _ = make_client()
    with client.stream("POST", "/api/chat", json={"question": "hi", "stream": True}, headers=CLIENT) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    payloads = [ln[5:].strip() for ln in lines]
    assert payloads[-1] == "[DONE]"
    assert "".join(json.loads(p)["delta"] for p in payloads[:-1] if p.startswith("{") and "delta" in json.loads(p)) == "The answer is $2x$."
    assert any(json.loads(p).get("event") == "done" for p in payloads[:-1])


def test_provider_failure_becomes_a_clean_error(make_client):
    from ai.base import ProviderError

    failing = ScriptedProvider(specs=[spec("fake", "m1")], fail_with=ProviderError("x", kind="rate_limit"))
    client, _ = make_client(provider=failing)
    r = client.post("/api/chat", json={"question": "hi"})
    assert r.status_code == 429 and "rate limit" in r.json()["error"] and r.headers["retry-after"]
    authfail = ScriptedProvider(specs=[spec("fake", "m1")], fail_with=ProviderError("x", kind="auth"))
    client2, _ = make_client(provider=authfail)
    assert client2.post("/api/chat", json={"question": "hi"}).status_code == 502


def test_preferences_and_memory_context(make_client):
    client, provider = make_client()
    r = client.put("/api/preferences", json={"preferences": {"units": "metric", "exam": "IB Physics HL"}}, headers=CLIENT)
    assert r.json()["preferences"]["units"] == "metric"
    client.post("/api/chat", json={"question": "hi"}, headers=CLIENT)
    assert "preference units: metric" in provider.calls[0]["messages"][0].content
    assert client.put("/api/preferences", json={"preferences": {"units": None}}, headers=CLIENT).json()["preferences"] == {"exam": "IB Physics HL"}


def test_access_token_and_cors(make_client):
    client, _ = make_client(api_auth_token="s3cret", cors_origins="https://me.hf.space")
    assert client.post("/api/chat", json={"question": "hi"}).status_code == 401
    ok = client.post("/api/chat", json={"question": "hi"}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert client.get("/api/health").status_code == 200  # health stays public
    pre = client.options("/api/chat", headers={"Origin": "https://me.hf.space", "Access-Control-Request-Method": "POST",
                                               "Access-Control-Request-Headers": "content-type,x-client-id"})
    assert pre.headers["access-control-allow-origin"] == "https://me.hf.space"
    bad = client.options("/api/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_compute_endpoint_is_off_by_default_and_works_when_enabled(make_client):
    client, _ = make_client()
    assert client.post("/api/compute", json={"engine": "math", "operation": "evaluate", "args": {"expression": "1+1"}}).status_code == 404
    client2, _ = make_client(enable_compute_api=True)
    r = client2.post("/api/compute", json={"engine": "math", "operation": "solve", "args": {"equations": "x^2=4"}})
    assert r.status_code == 200 and r.json()["result"]["values"] == ["-2", "2"]
    bad = client2.post("/api/compute", json={"engine": "math", "operation": "evaluate", "args": {"expression": "__import__('os')"}})
    assert bad.status_code == 400
