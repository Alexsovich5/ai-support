"""End-to-end checks against a running Compose stack (API + OpenAI stub).

Skipped unless ``STACK_URL`` points at the API, e.g. ``http://api:8000``.
"""

import json
import os
import uuid

import httpx
import pytest

STACK_URL = os.environ.get("STACK_URL")
KEY_FILE = os.environ.get("INGEST_API_KEY_FILE")

pytestmark = pytest.mark.skipif(not STACK_URL, reason="STACK_URL is not set")


def _ingest_headers():
    with open(KEY_FILE, encoding="utf-8") as handle:
        return {"Authorization": f"Bearer {handle.read().strip()}"}


@pytest.fixture(scope="module")
def http():
    with httpx.Client(base_url=STACK_URL, timeout=60.0) as client:
        yield client


def test_ready_reports_seeded_chunks(http):
    response = http.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["kb_chunks"] > 0


def test_chat_about_vpn_cites_article(http):
    response = http.post(
        "/api/v1/chat",
        json={"message": "My VPN keeps disconnecting", "session_id": "it-chat"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "vpn-disconnects.md" in body["source_documents"]
    assert body["message"]


def test_classify_goes_through_stub_not_fallback(http):
    response = http.post(
        "/api/v1/classify",
        json={
            "title": "VPN keeps disconnecting",
            "description": "The VPN tunnel drops every few minutes when working from home.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "vpn"
    assert body["confidence"] != 0.5


def test_websocket_round_trip():
    from websockets.sync.client import connect

    ws_url = STACK_URL.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    with connect(f"{ws_url}/api/v1/chat/ws/it-user", open_timeout=30) as ws:
        first = json.loads(ws.recv(timeout=30))
        assert first["type"] == "session_start"
        ws.send(json.dumps({"type": "user_message", "content": "My VPN keeps disconnecting"}))
        seen = []
        for _ in range(10):
            message = json.loads(ws.recv(timeout=60))
            seen.append(message)
            if message["type"] == "ai_response":
                break
        assert seen[-1]["type"] == "ai_response", seen
        assert "vpn-disconnects.md" in seen[-1]["sources"]
        ws.send(json.dumps({"type": "end_session"}))
        assert json.loads(ws.recv(timeout=30))["type"] == "session_end"


def test_ingest_then_search(http):
    source = f"badge-reader-{uuid.uuid4().hex[:8]}.md"
    ingest = http.post(
        "/api/v1/knowledge/ingest",
        json={
            "content": "Badge readers at the pharmacy door reject badges after a firmware update. "
            "Re-enrol the badge at the security desk.",
            "source": source,
            "category": "hardware",
            "title": "Badge reader rejects badges",
        },
        headers=_ingest_headers(),
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_created"] >= 1

    response = http.post(
        "/api/v1/knowledge/search", json={"query": "pharmacy badge reader rejects badge"}
    )
    assert response.status_code == 200
    sources = [r["source"] for r in response.json()["results"]]
    assert source in sources


def test_ingest_without_the_key_is_refused(http):
    body = {"content": "Some article text here.", "source": "nokey.md", "category": "hardware"}
    assert http.post("/api/v1/knowledge/ingest", json=body).status_code == 401
    wrong = http.post(
        "/api/v1/knowledge/ingest", json=body, headers={"Authorization": "Bearer not-the-key"}
    )
    assert wrong.status_code == 401


def test_cross_origin_post_is_refused(http):
    response = http.post(
        "/api/v1/chat",
        json={"message": "VPN drops", "session_id": "x-origin"},
        headers={"Origin": "http://evil.example"},
    )
    assert response.status_code == 403
