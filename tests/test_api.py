from fastapi.testclient import TestClient

from src.main import create_app
from tests.helpers import INGEST_HEADERS

VPN_TICKET = {
    "title": "VPN keeps disconnecting",
    "description": "The VPN tunnel drops every few minutes when working from home.",
}


def test_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_seeded_chunks(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["kb_chunks"] > 0


def test_ready_is_503_before_startup(settings):
    client = TestClient(create_app(settings))  # no "with": the lifespan never runs
    assert client.get("/health/ready").status_code == 503


def test_unseeded_app_has_empty_knowledge_base(settings):
    with TestClient(create_app(settings.model_copy(update={"kb_seed_dir": None}))) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["kb_chunks"] == 0


def test_chat_returns_vpn_article(client):
    response = client.post(
        "/api/v1/chat",
        json={"message": "My VPN keeps disconnecting", "session_id": "s1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "vpn-disconnects.md" in body["source_documents"]
    assert body["session_id"] == "s1"
    assert body["phi_warning"] is None


def test_chat_warns_when_mrn_present(client):
    response = client.post(
        "/api/v1/chat",
        json={"message": "VPN keeps disconnecting while I chart MRN 12345678", "session_id": "s2"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["phi_warning"]
    assert "12345678" not in body["message"]


def test_chat_empty_message_is_422(client):
    response = client.post("/api/v1/chat", json={"message": "", "session_id": "s3"})
    assert response.status_code == 422


def test_classify_vpn_ticket_uses_fallback(client):
    response = client.post("/api/v1/classify", json=VPN_TICKET)
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "vpn"
    assert body["confidence"] == 0.5
    assert body["requires_escalation"] is True


def test_classify_batch_preserves_order(client):
    tickets = [VPN_TICKET, {"title": "Printer offline", "description": "The ward printer shows offline."}]
    response = client.post("/api/v1/classify/batch", json=tickets)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [c["category"] for c in body["classifications"]] == ["vpn", "printing"]


def test_classify_batch_over_limit_is_400(client):
    response = client.post("/api/v1/classify/batch", json=[VPN_TICKET] * 51)
    assert response.status_code == 400


def test_ingest_then_search_uses_profile_threshold(client):
    ingest = client.post(
        "/api/v1/knowledge/ingest",
        json={
            "content": "Badge readers at the pharmacy door reject badges after a firmware update. "
            "Re-enrol the badge at the security desk.",
            "source": "badge-reader.md",
            "category": "hardware",
            "title": "Badge reader rejects badges",
        },
        headers=INGEST_HEADERS,
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_created"] >= 1

    response = client.post("/api/v1/knowledge/search", json={"query": "pharmacy badge reader rejects badge"})
    assert response.status_code == 200
    body = response.json()
    assert body["total_results"] >= 1
    assert body["results"][0]["source"] == "badge-reader.md"


def test_search_request_min_relevance_defaults_to_none():
    from src.api.models import KnowledgeSearchRequest

    assert KnowledgeSearchRequest(query="vpn").min_relevance is None


def test_explicit_min_relevance_filters_everything(client):
    response = client.post(
        "/api/v1/knowledge/search",
        json={"query": "vpn keeps disconnecting", "min_relevance": 1.0},
    )
    assert response.status_code == 200
    assert response.json()["total_results"] == 0


def test_knowledge_stats(client):
    body = client.get("/api/v1/knowledge/stats").json()
    assert body["total_chunks"] > 0
    assert body["categories"]["vpn"] > 0


def test_system_stats_count_requests(client):
    client.post("/api/v1/chat", json={"message": "My VPN keeps disconnecting", "session_id": "s4"})
    client.post("/api/v1/classify", json=VPN_TICKET)
    body = client.get("/api/v1/stats").json()
    assert body["engine_stats"]["total_messages"] == 1
    assert body["engine_stats"]["prompts_used"] == {"default": 1}
    assert set(body["classifier_stats"]) == {"total_classifications", "avg_classification_time_ms"}
    assert "timestamp" in body


def test_lifespan_cancels_cleanup_task(app):
    with TestClient(app):
        task = app.state.chat_handler.cleanup_task
        assert task is not None and not task.done()
    assert task.cancelled()


def test_services_live_on_app_state(app):
    with TestClient(app):
        assert app.state.ai_engine.knowledge_base is app.state.knowledge_base
        assert app.state.classifier.settings is app.state.settings
