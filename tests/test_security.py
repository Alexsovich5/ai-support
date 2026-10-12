"""Authentication, request guards, bounded state and leak checks for the API."""

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from src.config import load_settings
from src.main import create_app
from src.security import AuthFailureLimiter, ConcurrencyLimiter, LimiterBusy, log_value
from tests.helpers import INGEST_HEADERS, TEST_INGEST_KEY

ARTICLE = {
    "content": "Badge readers at the pharmacy door reject badges after a firmware update.",
    "source": "badge-reader.md",
    "category": "hardware",
    "title": "Badge reader rejects badges",
}
VPN_TICKET = {"title": "VPN keeps disconnecting", "description": "The tunnel drops every few minutes."}
MODEL_ROUTES = [
    ("/api/v1/chat", {"message": "My VPN keeps disconnecting", "session_id": "s1"}),
    ("/api/v1/classify", VPN_TICKET),
    ("/api/v1/classify/batch", [VPN_TICKET]),
    ("/api/v1/knowledge/search", {"query": "vpn keeps disconnecting"}),
    ("/api/v1/knowledge/ingest", ARTICLE),
]


def _with(settings, section, **updates):
    return settings.model_copy(update={section: getattr(settings, section).model_copy(update=updates)})


# --- ingest key ---------------------------------------------------------------

def test_ingest_without_configured_key_is_refused(settings):
    no_key = _with(settings, "security", ingest_api_key=None)
    with TestClient(create_app(no_key)) as client:
        response = client.post("/api/v1/knowledge/ingest", json=ARTICLE)
    assert response.status_code == 503


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-key"}, {"Authorization": f"Basic {TEST_INGEST_KEY}"}, {"Authorization": "Bearer "}],
)
def test_ingest_with_missing_or_wrong_key_is_401(client, headers):
    response = client.post("/api/v1/knowledge/ingest", json=ARTICLE, headers=headers)
    assert response.status_code == 401
    assert "wrong-key" not in response.text
    assert client.post("/api/v1/knowledge/search", json={"query": "pharmacy badge reader"}).json()["total_results"] == 0


def test_ingest_with_the_key_succeeds(client):
    response = client.post("/api/v1/knowledge/ingest", json=ARTICLE, headers=INGEST_HEADERS)
    assert response.status_code == 200


def test_repeated_failures_throttle_that_client_only(settings):
    strict = _with(settings, "security", auth_failure_limit=3)
    with TestClient(create_app(strict)) as client:
        codes = [
            client.post("/api/v1/knowledge/ingest", json=ARTICLE, headers={"Authorization": "Bearer x"}).status_code
            for _ in range(4)
        ]
        assert codes == [401, 401, 401, 429]
        # the same client is throttled even with the right key until the window passes
        assert client.post("/api/v1/knowledge/ingest", json=ARTICLE, headers=INGEST_HEADERS).status_code == 429


def test_failure_limiter_is_per_client_bounded_and_decays():
    now = [0.0]
    limiter = AuthFailureLimiter(limit=2, window_seconds=10, max_clients=100, clock=lambda: now[0])
    limiter.record_failure("attacker")
    limiter.record_failure("attacker")
    assert limiter.blocked("attacker")
    assert not limiter.blocked("operator")
    for i in range(10_000):
        limiter.record_failure(f"10.0.{i // 256}.{i % 256}")
    assert len(limiter) == 100
    now[0] = 11.0
    limiter.record_failure("late")
    assert not limiter.blocked("late")


# --- cross-origin / content-type / body size ----------------------------------

def test_cross_origin_state_change_is_refused(client):
    response = client.post(
        "/api/v1/knowledge/ingest",
        json=ARTICLE,
        headers={**INGEST_HEADERS, "Origin": "http://evil.example"},
    )
    assert response.status_code == 403
    same_origin = client.post(
        "/api/v1/chat",
        json={"message": "VPN drops", "session_id": "s9"},
        headers={"Origin": "http://testserver"},
    )
    assert same_origin.status_code == 200


@pytest.mark.parametrize(
    "content_type", ["application/x-www-form-urlencoded", "text/plain", "multipart/form-data; boundary=x"]
)
def test_form_style_posts_are_refused(client, content_type):
    for path, body in MODEL_ROUTES:
        response = client.post(
            path, content=json.dumps(body), headers={**INGEST_HEADERS, "Content-Type": content_type}
        )
        assert response.status_code == 415, path


def test_oversized_body_is_413_before_parsing(settings):
    small = _with(settings, "limits", max_request_bytes=2048)
    with TestClient(create_app(small)) as client:
        engine = client.app.state.ai_engine
        response = client.post(
            "/api/v1/chat", json={"message": "x" * 4000, "session_id": "big"}
        )
        assert response.status_code == 413
        assert engine.get_stats()["total_messages"] == 0
        chunked = client.post(
            "/api/v1/chat",
            content=iter([b'{"message": "', b"y" * 3000, b'", "session_id": "c"}']),
            headers={"Content-Type": "application/json"},
        )
        assert chunked.status_code == 413


def test_validation_errors_do_not_echo_the_input(client):
    secret = "SSN 123-45-6789 " + "z" * 5000
    response = client.post("/api/v1/chat", json={"message": secret, "session_id": "s1"})
    assert response.status_code == 422
    assert "6789" not in response.text and "zzzz" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {**ARTICLE, "source": "../../etc/passwd"},
        {**ARTICLE, "source": "/etc/passwd"},
        {**ARTICLE, "source": "a\\b.md"},
        {**ARTICLE, "category": "x/../y"},
        {**ARTICLE, "content": "x" * 100_001},
        {**ARTICLE, "unexpected": True},
    ],
)
def test_ingest_rejects_unsafe_or_oversized_fields(client, body):
    response = client.post("/api/v1/knowledge/ingest", json=body, headers=INGEST_HEADERS)
    assert response.status_code == 422


# --- WebSocket ------------------------------------------------------------------

def test_websocket_from_another_origin_is_refused(client):
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/api/v1/chat/ws/u1", headers={"Origin": "http://evil.example"}) as ws:
            ws.receive_json()
    assert excinfo.value.code == 1008


def test_websocket_with_malformed_user_id_is_refused(client):
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/api/v1/chat/ws/" + "u" * 65) as ws:
            ws.receive_json()
    assert excinfo.value.code == 1008


def test_websocket_sessions_per_client_are_capped(settings):
    capped = _with(settings, "chat", max_sessions_per_client=1)
    with TestClient(create_app(capped)) as client:
        with client.websocket_connect("/api/v1/chat/ws/u1") as first:
            assert first.receive_json()["type"] == "session_start"
            with client.websocket_connect("/api/v1/chat/ws/u2") as second:
                assert second.receive_json()["type"] == "error"
                with pytest.raises(WebSocketDisconnect) as excinfo:
                    second.receive_json()
                assert excinfo.value.code == 1013


def test_websocket_oversized_frame_is_rejected_before_parsing(client):
    with client.websocket_connect("/api/v1/chat/ws/u1") as ws:
        ws.receive_json()
        ws.send_text("[" * 200_000)
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "end_session"})
        assert ws.receive_json()["type"] == "session_end"


def test_websocket_escalation_reason_is_redacted(client, caplog):
    caplog.set_level(logging.DEBUG)
    with client.websocket_connect("/api/v1/chat/ws/u1") as ws:
        ws.receive_json()
        ws.send_json({"type": "escalate", "content": "patient MRN 12345678 cannot chart"})
        reply = ws.receive_json()
    assert reply["type"] == "escalation"
    assert "12345678" not in reply["reason"]
    assert "12345678" not in caplog.text


# --- concurrency limiter --------------------------------------------------------

def test_every_model_route_goes_through_the_limiter(client):
    limiter = client.app.state.work_limiter
    while limiter.try_acquire("testclient"):
        pass
    for path, body in MODEL_ROUTES:
        response = client.post(path, json=body, headers=INGEST_HEADERS)
        assert response.status_code == 503, path
    with client.websocket_connect("/api/v1/chat/ws/u1") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "VPN drops"})
        seen = [ws.receive_json() for _ in range(3)]
    assert seen[-1]["type"] == "error" and "busy" in seen[-1]["error"]


def test_limiter_releases_after_each_request(client):
    for _ in range(20):
        assert client.post("/api/v1/classify", json=VPN_TICKET).status_code == 200
    assert client.app.state.work_limiter.in_use == 0


@pytest.mark.asyncio
async def test_limiter_quota_leaves_room_for_other_clients():
    limiter = ConcurrencyLimiter(max_total=4, max_per_client=2)
    assert limiter.try_acquire("noisy") and limiter.try_acquire("noisy")
    assert not limiter.try_acquire("noisy")
    async with limiter.slot("quiet"):
        assert limiter.in_use == 3
    assert limiter.in_use == 2
    with pytest.raises(RuntimeError):
        limiter.release("never-acquired")
    limiter.release("noisy")
    limiter.release("noisy")
    assert limiter.in_use == 0
    with pytest.raises(LimiterBusy):
        tight = ConcurrencyLimiter(max_total=1, max_per_client=1)
        tight.try_acquire("a")
        async with tight.slot("b"):
            pass


# --- bounded session state --------------------------------------------------------

def test_chat_sessions_are_capped_per_client_and_overall(settings):
    small = _with(settings, "chat", max_tracked_sessions=5, max_tracked_sessions_per_client=3)
    with TestClient(create_app(small)) as client:
        engine = client.app.state.ai_engine
        engine._session("other|keep", owner="other")
        for i in range(50):
            client.post("/api/v1/chat", json={"message": "VPN drops", "session_id": f"flood-{i}"})
        assert len(engine._history) <= 5
        assert "other|keep" in engine._history
        assert engine.get_stats()["sessions_evicted"] >= 47


def test_session_ids_are_scoped_to_the_client_address(client):
    client.post("/api/v1/chat", json={"message": "VPN drops at home", "session_id": "shared"})
    engine = client.app.state.ai_engine
    assert engine.get_history("testclient|shared")
    assert not engine.get_history("shared")


def test_websocket_session_history_is_dropped_on_close(client):
    engine = client.app.state.ai_engine
    with client.websocket_connect("/api/v1/chat/ws/u1") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "VPN drops"})
        for _ in range(3):
            ws.receive_json()
        ws.send_json({"type": "end_session"})
        ws.receive_json()
    assert engine.get_stats()["active_sessions"] == 0


# --- PHI on every path ----------------------------------------------------------

def test_search_query_is_redacted_before_echo_and_logs(client, caplog):
    caplog.set_level(logging.DEBUG)
    response = client.post("/api/v1/knowledge/search", json={"query": "vpn for SSN 123-45-6789"})
    assert response.status_code == 200
    assert "6789" not in response.text
    assert "6789" not in caplog.text


def test_ingested_article_is_redacted_before_storage(client):
    body = {**ARTICLE, "content": "Call Jane at 555-123-4567 or jane@example.org about the badge reader."}
    response = client.post("/api/v1/knowledge/ingest", json=body, headers=INGEST_HEADERS)
    assert response.status_code == 200
    assert set(response.json()["phi_redacted"]) == {"phone", "email"}
    stored = client.app.state.knowledge_base._collection.get(include=["documents"])
    joined = " ".join(stored["documents"])
    assert "555-123-4567" not in joined and "jane@example.org" not in joined


def test_knowledge_stats_do_not_expose_the_store_path(client):
    body = client.get("/api/v1/knowledge/stats").json()
    assert "persist_directory" not in body
    assert "/" not in json.dumps(body)


# --- failures are errors, not empty results --------------------------------------

def test_store_failure_is_503_not_an_empty_result(client, monkeypatch):
    collection_type = type(client.app.state.knowledge_base._collection)

    def broken(self, *args, **kwargs):
        raise RuntimeError("disk /var/lib/chroma unavailable")

    monkeypatch.setattr(collection_type, "query", broken)
    monkeypatch.setattr(collection_type, "get", broken)
    search = client.post("/api/v1/knowledge/search", json={"query": "vpn"})
    assert search.status_code == 503
    stats = client.get("/api/v1/knowledge/stats")
    assert stats.status_code == 503
    assert "/var/lib" not in search.text + stats.text


def test_unhandled_error_returns_generic_500(settings, monkeypatch, caplog):
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        async def boom(**kwargs):
            raise RuntimeError("leaky detail /srv/app/secret.txt")

        monkeypatch.setattr(client.app.state.ai_engine, "process_message", boom)
        response = client.post("/api/v1/chat", json={"message": "VPN", "session_id": "s1"})
    assert response.status_code == 500
    assert "leaky" not in response.text and "/srv" not in response.text
    assert "leaky" not in caplog.text


# --- secrets never leak -----------------------------------------------------------

SENTINEL = "S3NTINEL-pw-0123456789"


def test_key_files_are_read_and_kept_out_of_repr(tmp_path):
    ingest_file = tmp_path / "ingest"
    openai_file = tmp_path / "openai"
    ingest_file.write_text(SENTINEL + "-ingest\n")
    openai_file.write_text(SENTINEL + "-openai\n")
    settings = load_settings(
        env={"INGEST_API_KEY_FILE": str(ingest_file), "OPENAI_API_KEY_FILE": str(openai_file)}
    )
    assert settings.security.ingest_api_key.get_secret_value() == SENTINEL + "-ingest"
    assert settings.llm.api_key.get_secret_value() == SENTINEL + "-openai"
    assert SENTINEL not in repr(settings) and SENTINEL not in str(settings.model_dump())


def test_unreadable_key_file_refuses_to_start(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        load_settings(env={"INGEST_API_KEY_FILE": str(tmp_path / "missing")})
    assert "INGEST_API_KEY_FILE" in str(excinfo.value)


def test_ingest_key_is_not_accepted_from_the_settings_file(tmp_path):
    from src.config import DEFAULT_SETTINGS_PATH

    alt = tmp_path / "alt.yml"
    alt.write_text(
        DEFAULT_SETTINGS_PATH.read_text().replace(
            "phi_detection_enabled: true", f"phi_detection_enabled: true\n  ingest_api_key: {SENTINEL}"
        )
    )
    with pytest.raises(ValueError):
        load_settings(env={"APP_CONFIG": str(alt)})


def test_sentinel_secrets_never_reach_logs_or_responses(settings, caplog, capsys):
    """Upstream errors that echo the OpenAI key, and wrong ingest keys, leave no trace."""
    from src.llm_factory import get_chat_model

    caplog.set_level(logging.DEBUG)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"Incorrect API key provided: {SENTINEL}", "type": "invalid_request_error"}},
        )

    openai_settings = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={"provider": "openai", "api_key": SecretStr(SENTINEL), "base_url": "http://upstream/v1"}
            ),
            "security": settings.security.model_copy(update={"ingest_api_key": SecretStr(SENTINEL + "-ingest")}),
        }
    )
    app = create_app(openai_settings.model_copy(update={"llm": settings.llm}))
    with TestClient(app) as client:
        app.state.classifier.llm = get_chat_model(
            openai_settings,
            "classification",
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
        )
        classified = client.post("/api/v1/classify", json=VPN_TICKET)
        rejected = client.post(
            "/api/v1/knowledge/ingest", json=ARTICLE, headers={"Authorization": f"Bearer {SENTINEL}-guess"}
        )
        stats = client.get("/api/v1/stats")

    assert classified.status_code == 200 and classified.json()["confidence"] == 0.5
    assert rejected.status_code == 401
    captured = capsys.readouterr()
    for text in (caplog.text, captured.out, captured.err, classified.text, rejected.text, stats.text):
        assert SENTINEL not in text


def test_log_value_strips_control_characters_and_truncates():
    assert log_value("user\r\nFAKE LOG LINE\x1b[31m") == "user??FAKE LOG LINE?[31m"
    assert log_value("x" * 500, 10) == "x" * 10 + "..."


# --- readiness fails closed ------------------------------------------------------

def test_ready_is_503_when_seeding_had_errors(settings):
    from pathlib import Path

    (Path(settings.kb_seed_dir) / "broken.md").write_bytes(b"\xff\xfe not utf-8")
    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "seed_failed", "seed_errors": 1}
