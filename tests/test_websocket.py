"""WebSocket chat protocol tests, run through the app lifespan."""

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.main as main_module
from src.config import load_prompts
from src.main import create_app
from tests.helpers import RecordingFakeChatModel

WS_PATH = "/api/v1/chat/ws/u1"


def receive_until(ws, msg_type, limit=10):
    """Collect messages until one of ``msg_type`` arrives; return them all."""
    seen = []
    for _ in range(limit):
        message = ws.receive_json()
        seen.append(message)
        if message["type"] == msg_type:
            return seen
    raise AssertionError(f"no {msg_type!r} message in {seen!r}")


def wait_for_active_count(handler, expected, timeout=5.0):
    """Client-side close returns before the server task finishes; poll briefly."""
    deadline = time.monotonic() + timeout
    while handler.get_active_session_count() != expected:
        if time.monotonic() > deadline:
            break
        time.sleep(0.01)
    return handler.get_active_session_count()


def with_chat(settings, **chat_updates):
    return settings.model_copy(
        update={"chat": settings.chat.model_copy(update=chat_updates)}
    )


def test_first_message_is_session_start(client):
    with client.websocket_connect(WS_PATH) as ws:
        first = ws.receive_json()
    assert first["type"] == "session_start"
    assert first["session_id"]
    assert "timestamp" in first


def test_user_message_yields_typing_then_answer_with_sources(client):
    with client.websocket_connect(WS_PATH) as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "My VPN keeps disconnecting"})
        seen = receive_until(ws, "ai_response")
    types = [m["type"] for m in seen]
    assert types == ["typing_start", "typing_stop", "ai_response"]
    assert "vpn-disconnects.md" in seen[-1]["sources"]
    assert seen[-1]["message"]


def test_non_json_yields_error(client):
    with client.websocket_connect(WS_PATH) as ws:
        ws.receive_json()
        ws.send_text("not json at all")
        reply = ws.receive_json()
    assert reply["type"] == "error"


def test_oversize_message_yields_error(client, settings):
    too_long = "x" * (settings.chat.max_message_length + 1)
    with client.websocket_connect(WS_PATH) as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": too_long})
        reply = ws.receive_json()
    assert reply["type"] == "error"
    assert str(settings.chat.max_message_length) in reply["error"]


def test_escalate_has_no_ticket_and_no_compliance_claim(client):
    with client.websocket_connect(WS_PATH) as ws:
        ws.receive_json()
        ws.send_json({"type": "escalate", "content": "printer on fire"})
        reply = ws.receive_json()
    assert reply["type"] == "escalation"
    assert "ticket_created" not in reply
    assert "hipaa" not in str(reply).lower()
    expected = " ".join(load_prompts()["escalation"]["threshold_warning"].split())
    assert reply["message"] == expected
    assert reply["reason"] == "printer on fire"
    for stale in ("ext. 4500", "5-10 minutes"):
        assert stale not in str(reply)


def test_phi_message_yields_system_notice_without_compliance_claim(client):
    with client.websocket_connect(WS_PATH) as ws:
        ws.receive_json()
        ws.send_json(
            {"type": "user_message", "content": "VPN drops while I open MRN: 12345678"}
        )
        seen = receive_until(ws, "system")
    notice = seen[-1]
    assert "patient identifiers" in notice["message"]
    for message in seen:
        assert "hipaa" not in str(message).lower()
        assert "12345678" not in str(message)


def test_role_query_parameter_selects_clinical_prompt(settings, monkeypatch):
    recording = RecordingFakeChatModel(responses=["Restart the client."])
    real_factory = main_module.get_chat_model

    def factory(s, purpose="chat", **kwargs):
        if purpose == "chat":
            return recording
        return real_factory(s, purpose, **kwargs)

    monkeypatch.setattr(main_module, "get_chat_model", factory)
    clinical = load_prompts()["system_prompts"]["clinical_staff"].strip()

    with TestClient(create_app(settings)) as client:
        with client.websocket_connect(WS_PATH + "?role=nurse") as ws:
            ws.receive_json()
            ws.send_json({"type": "user_message", "content": "Printer will not print"})
            receive_until(ws, "ai_response")
        assert clinical in recording.last_prompt_text()

        with client.websocket_connect(WS_PATH) as ws:
            ws.receive_json()
            ws.send_json({"type": "user_message", "content": "Printer will not print"})
            receive_until(ws, "ai_response")
        assert clinical not in recording.last_prompt_text()


def test_end_session_and_active_count_returns_to_zero(client, app):
    handler = app.state.chat_handler
    with client.websocket_connect(WS_PATH) as ws:
        ws.receive_json()
        assert handler.get_active_session_count() == 1
        ws.send_json({"type": "end_session"})
        reply = ws.receive_json()
        assert reply["type"] == "session_end"
    assert wait_for_active_count(handler, 0) == 0


def test_session_limit_is_shared_across_connections(settings):
    app = create_app(with_chat(settings, max_concurrent_sessions=1))
    with TestClient(app) as client:
        with client.websocket_connect(WS_PATH) as first:
            assert first.receive_json()["type"] == "session_start"
            with client.websocket_connect("/api/v1/chat/ws/u2") as second:
                reply = second.receive_json()
                assert reply["type"] == "error"
                with pytest.raises(WebSocketDisconnect) as closed:
                    second.receive_json()
                assert closed.value.code == 1013
            assert app.state.chat_handler.get_active_session_count() == 1
        assert wait_for_active_count(app.state.chat_handler, 0) == 0

        with client.websocket_connect("/api/v1/chat/ws/u3") as third:
            assert third.receive_json()["type"] == "session_start"


def test_idle_connection_receives_session_end(settings):
    app = create_app(with_chat(settings, session_timeout_seconds=0.1))
    with TestClient(app) as client:
        with client.websocket_connect(WS_PATH) as ws:
            assert ws.receive_json()["type"] == "session_start"
            reply = ws.receive_json()
            assert reply["type"] == "session_end"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert wait_for_active_count(app.state.chat_handler, 0) == 0


def test_handler_reads_limits_from_settings(settings):
    app = create_app(with_chat(settings, max_message_length=10))
    with TestClient(app) as client:
        handler = app.state.chat_handler
        assert handler.max_message_length == 10
        assert handler.max_concurrent_sessions == settings.chat.max_concurrent_sessions
        with client.websocket_connect(WS_PATH) as ws:
            ws.receive_json()
            ws.send_json({"type": "user_message", "content": "x" * 11})
            assert ws.receive_json()["type"] == "error"
