"""
WebSocket chat handler.

Manages real-time WebSocket chat sessions for the AI support system.
Handles connection lifecycle, message routing, typing indicators and
idle timeouts. One handler is shared by every connection of an app, so
the concurrent-session limit and the active-session count are app-wide;
one client address may hold at most ``max_sessions_per_client`` of them.
Raw frames are size-checked before JSON parsing, every model call holds a
slot of the app's concurrency limiter, and messages are logged by session id
and length only, never by content. Client-supplied values that reach a log
line are sanitised first.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.config import ChatConfig
from src.phi_filter import redact
from src.security import ConcurrencyLimiter, LimiterBusy, log_value

logger = logging.getLogger("ai_support.chat")


class MessageType(Enum):
    USER_MESSAGE = "user_message"
    AI_RESPONSE = "ai_response"
    TYPING_START = "typing_start"
    TYPING_STOP = "typing_stop"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ESCALATION = "escalation"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class ChatSession:
    session_id: str
    user_id: str
    websocket: WebSocket
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_activity: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message_count: int = 0
    is_active: bool = True
    client: str = "unknown"
    department: Optional[str] = None
    user_name: Optional[str] = None
    role: Optional[str] = None


PHI_NOTICE = (
    "We filtered possible patient identifiers from your message. "
    "Please avoid sharing patient information in support chats."
)
DEFAULT_ESCALATION_TEXT = (
    "This issue needs a human support agent. Please contact the service desk "
    "and mention what you have already tried."
)
BUSY_TEXT = "The assistant is busy right now. Please try again in a moment."
PROCESSING_ERROR_TEXT = (
    "I encountered an issue processing your request. Please try rephrasing "
    "or contact the service desk."
)


class ChatHandler:
    """
    WebSocket chat session manager for real-time AI support interactions.
    Handles multiple concurrent sessions with graceful lifecycle management.
    """

    CLEANUP_INTERVAL_SECONDS = 300

    def __init__(
        self,
        ai_engine,
        chat_config: ChatConfig,
        prompts: Optional[Mapping[str, Any]] = None,
        work_limiter: Optional[ConcurrencyLimiter] = None,
    ):
        self.ai_engine = ai_engine
        self.max_message_length = chat_config.max_message_length
        # A JSON frame carrying a maximal message, with room for escaping.
        self.max_frame_chars = chat_config.max_message_length * 6 + 1024
        self.session_timeout_seconds = chat_config.session_timeout_seconds
        self.max_concurrent_sessions = chat_config.max_concurrent_sessions
        self.max_sessions_per_client = chat_config.max_sessions_per_client
        self.work_limiter = work_limiter or ConcurrencyLimiter(
            chat_config.max_concurrent_sessions, chat_config.max_concurrent_sessions
        )
        escalation = (prompts or {}).get("escalation", {}).get("threshold_warning")
        self.escalation_text = (
            " ".join(escalation.split()) if escalation else DEFAULT_ESCALATION_TEXT
        )
        self._sessions: dict[str, ChatSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info("Chat handler initialized: max_sessions=%d", self.max_concurrent_sessions)

    async def start_cleanup_loop(self) -> None:
        """Start background task to clean up stale sessions."""
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_sessions())

    async def stop_cleanup_loop(self) -> None:
        """Cancel the cleanup task and wait for it to finish."""
        task = self._cleanup_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @property
    def cleanup_task(self) -> Optional[asyncio.Task]:
        return self._cleanup_task

    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: str,
        department: Optional[str] = None,
        user_name: Optional[str] = None,
        role: Optional[str] = None,
        client: str = "unknown",
    ) -> None:
        """Handle a complete WebSocket connection lifecycle."""
        await websocket.accept()

        session_id = str(uuid.uuid4())
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            websocket=websocket,
            department=department,
            user_name=user_name,
            role=role,
            client=client,
        )

        held_by_client = sum(1 for s in self._sessions.values() if s.client == client)
        if (
            len(self._sessions) >= self.max_concurrent_sessions
            or held_by_client >= self.max_sessions_per_client
        ):
            await self._send_message(
                websocket,
                MessageType.ERROR,
                {"error": "Maximum concurrent sessions reached. Please try again later."},
            )
            await websocket.close(code=1013)
            return

        self._sessions[session_id] = session

        # Send session start acknowledgment
        await self._send_message(
            websocket,
            MessageType.SESSION_START,
            {
                "session_id": session_id,
                "message": "Connected to AI IT Support. How can I help you today?",
                "agent_name": "AI Support Assistant",
            },
        )

        logger.info(
            "Chat session started: session_id=%s, user=%s, department=%s",
            session_id,
            log_value(user_id),
            log_value(department),
        )

        try:
            await self._message_loop(session)
        except WebSocketDisconnect:
            logger.info("Client disconnected: session_id=%s", session_id)
        except Exception as e:
            logger.error("Session error: session_id=%s, error=%s", session_id, type(e).__name__)
            try:
                await self._send_message(
                    websocket,
                    MessageType.ERROR,
                    {"error": "An unexpected error occurred. Please reconnect."},
                )
            except Exception:
                pass
        finally:
            session.is_active = False
            self._sessions.pop(session_id, None)
            end_session = getattr(self.ai_engine, "end_session", None)
            if end_session is not None:
                end_session(session_id)
            await self._close(websocket)
            logger.info(
                "Chat session ended: session_id=%s, messages=%d",
                session_id,
                session.message_count,
            )

    async def _message_loop(self, session: ChatSession) -> None:
        """Main message processing loop for a chat session."""
        while session.is_active:
            try:
                raw_data = await asyncio.wait_for(
                    session.websocket.receive_text(),
                    timeout=self.session_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await self._send_message(
                    session.websocket,
                    MessageType.SESSION_END,
                    {"message": "Session closed due to inactivity."},
                )
                break

            if len(raw_data) > self.max_frame_chars:
                await self._send_message(
                    session.websocket,
                    MessageType.ERROR,
                    {"error": f"Message too long. Maximum {self.max_message_length} characters."},
                )
                continue

            try:
                data = json.loads(raw_data)
            except (json.JSONDecodeError, RecursionError):
                await self._send_message(
                    session.websocket,
                    MessageType.ERROR,
                    {"error": "Invalid message format. Expected JSON."},
                )
                continue

            if not isinstance(data, dict):
                await self._send_message(
                    session.websocket,
                    MessageType.ERROR,
                    {"error": "Invalid message format. Expected a JSON object."},
                )
                continue

            message_type = data.get("type", "user_message")
            content = str(data.get("content") or "").strip()

            if message_type == "user_message":
                await self._handle_user_message(session, content)
            elif message_type == "feedback":
                await self._handle_feedback(session, data)
            elif message_type == "escalate":
                await self._handle_escalation(session, content)
            elif message_type == "end_session":
                await self._send_message(
                    session.websocket,
                    MessageType.SESSION_END,
                    {"message": "Session ended. Thank you for using AI IT Support."},
                )
                break

    async def _handle_user_message(
        self, session: ChatSession, content: str
    ) -> None:
        """Process a user message through the AI engine."""
        if not content:
            return

        if len(content) > self.max_message_length:
            await self._send_message(
                session.websocket,
                MessageType.ERROR,
                {"error": f"Message too long. Maximum {self.max_message_length} characters."},
            )
            return

        session.message_count += 1
        session.last_activity = datetime.now(timezone.utc).isoformat()

        # Send typing indicator
        await self._send_message(
            session.websocket,
            MessageType.TYPING_START,
            {},
        )

        try:
            # Process through AI engine
            user_context = {
                "department": session.department,
                "role": session.role,
                "user_id": session.user_id,
            }

            async with self.work_limiter.slot(session.client):
                response = await self.ai_engine.process_message(
                    message=content,
                    session_id=session.session_id,
                    user_context=user_context,
                    owner=session.client,
                )

            # Stop typing indicator
            await self._send_message(
                session.websocket,
                MessageType.TYPING_STOP,
                {},
            )

            # Send AI response
            response_data = {
                "message": response.message,
                "confidence": response.confidence,
                "sources": response.source_documents,
                "response_time_ms": response.response_time_ms,
            }

            if response.phi_detected:
                response_data["warning"] = PHI_NOTICE

            await self._send_message(
                session.websocket,
                MessageType.AI_RESPONSE,
                response_data,
            )

            if response.phi_detected:
                await self._send_message(
                    session.websocket,
                    MessageType.SYSTEM,
                    {"message": PHI_NOTICE},
                )

            # Auto-escalation if confidence is too low
            if response.escalation_recommended:
                await self._send_message(
                    session.websocket,
                    MessageType.ESCALATION,
                    {
                        "message": self.escalation_text,
                        "reason": "low_confidence",
                        "confidence": response.confidence,
                    },
                )

        except LimiterBusy:
            await self._send_message(session.websocket, MessageType.TYPING_STOP, {})
            await self._send_message(session.websocket, MessageType.ERROR, {"error": BUSY_TEXT})
        except Exception as e:
            logger.error(
                "AI processing error: session=%s, error=%s",
                session.session_id,
                type(e).__name__,
            )
            await self._send_message(
                session.websocket,
                MessageType.TYPING_STOP,
                {},
            )
            await self._send_message(
                session.websocket,
                MessageType.ERROR,
                {"error": PROCESSING_ERROR_TEXT},
            )

    async def _handle_feedback(self, session: ChatSession, data: dict) -> None:
        """Handle user feedback on AI responses."""
        feedback_type = data.get("feedback_type", "")  # helpful, unhelpful, incorrect
        message_id = data.get("message_id", "")

        logger.info(
            "Feedback received: session=%s, type=%s, message_id=%s",
            session.session_id,
            log_value(feedback_type, 32),
            log_value(message_id, 64),
        )

        await self._send_message(
            session.websocket,
            MessageType.SYSTEM,
            {"message": "Thank you for your feedback. It helps improve our AI support system."},
        )

    async def _handle_escalation(self, session: ChatSession, reason: str) -> None:
        """Handle manual escalation request; the reason is PHI-redacted and truncated."""
        reason = redact(reason[: self.max_message_length]).text if reason else ""
        logger.info(
            "Manual escalation requested: session=%s, reason_length=%d",
            session.session_id,
            len(reason),
        )

        await self._send_message(
            session.websocket,
            MessageType.ESCALATION,
            {
                "message": self.escalation_text,
                "reason": reason or "user_requested",
            },
        )

    async def _send_message(
        self,
        websocket: WebSocket,
        msg_type: MessageType,
        data: dict,
    ) -> None:
        """Send a typed message through the WebSocket."""
        if websocket.client_state != WebSocketState.CONNECTED:
            return

        message = {
            "type": msg_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning("Failed to send message: %s", type(e).__name__)

    @staticmethod
    async def _close(websocket: WebSocket, code: int = 1000) -> None:
        if websocket.client_state != WebSocketState.CONNECTED:
            return
        try:
            await websocket.close(code=code)
        except Exception as e:
            logger.debug("Close failed: %s", type(e).__name__)

    async def _cleanup_stale_sessions(self) -> None:
        """Periodically clean up inactive sessions."""
        while True:
            await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)

            now = datetime.now(timezone.utc)
            stale_sessions = []

            for session_id, session in self._sessions.items():
                last_activity = datetime.fromisoformat(session.last_activity)
                idle_seconds = (now - last_activity).total_seconds()

                if idle_seconds > self.session_timeout_seconds:
                    stale_sessions.append(session_id)

            for session_id in stale_sessions:
                session = self._sessions.pop(session_id, None)
                if session:
                    await self._send_message(
                        session.websocket,
                        MessageType.SESSION_END,
                        {"message": "Session closed due to inactivity."},
                    )
                    await self._close(session.websocket)
                    session.is_active = False

            if stale_sessions:
                logger.info("Cleaned up %d stale sessions", len(stale_sessions))

    def get_active_session_count(self) -> int:
        """Return number of active chat sessions."""
        return len(self._sessions)

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Return session information for monitoring."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "department": session.department,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "message_count": session.message_count,
            "is_active": session.is_active,
        }
