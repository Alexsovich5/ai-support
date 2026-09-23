#!/usr/bin/env python3
"""
WebSocket Chat Handler
Project: AI-Powered Healthcare IT Support System
Timeline: April 2024 - June 2024

Manages real-time WebSocket chat sessions for the AI support system.
Handles connection lifecycle, message routing, typing indicators,
and session persistence with HIPAA-compliant logging.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

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
    department: Optional[str] = None
    user_name: Optional[str] = None


class ChatHandler:
    """
    WebSocket chat session manager for real-time AI support interactions.
    Handles multiple concurrent sessions with graceful lifecycle management.
    """

    MAX_MESSAGE_LENGTH = 4096
    SESSION_TIMEOUT_SECONDS = 1800  # 30 minutes
    MAX_CONCURRENT_SESSIONS = 200

    def __init__(self, ai_engine):
        self.ai_engine = ai_engine
        self._sessions: dict[str, ChatSession] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info("Chat handler initialized: max_sessions=%d", self.MAX_CONCURRENT_SESSIONS)

    async def start_cleanup_loop(self) -> None:
        """Start background task to clean up stale sessions."""
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_sessions())

    async def handle_connection(
        self,
        websocket: WebSocket,
        user_id: str,
        department: Optional[str] = None,
        user_name: Optional[str] = None,
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
        )

        if len(self._sessions) >= self.MAX_CONCURRENT_SESSIONS:
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
            user_id,
            department,
        )

        try:
            await self._message_loop(session)
        except WebSocketDisconnect:
            logger.info("Client disconnected: session_id=%s", session_id)
        except Exception as e:
            logger.error("Session error: session_id=%s, error=%s", session_id, str(e))
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
            del self._sessions[session_id]
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
                    timeout=self.SESSION_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                await self._send_message(
                    session.websocket,
                    MessageType.SYSTEM,
                    {"message": "Session timed out due to inactivity."},
                )
                break

            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                await self._send_message(
                    session.websocket,
                    MessageType.ERROR,
                    {"error": "Invalid message format. Expected JSON."},
                )
                continue

            message_type = data.get("type", "user_message")
            content = data.get("content", "").strip()

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

        if len(content) > self.MAX_MESSAGE_LENGTH:
            await self._send_message(
                session.websocket,
                MessageType.ERROR,
                {"error": f"Message too long. Maximum {self.MAX_MESSAGE_LENGTH} characters."},
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
                "role": "staff",
                "user_id": session.user_id,
            }

            response = await self.ai_engine.process_message(
                message=content,
                session_id=session.session_id,
                user_context=user_context,
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
                response_data["warning"] = (
                    "PHI was detected and redacted from the conversation for HIPAA compliance."
                )

            await self._send_message(
                session.websocket,
                MessageType.AI_RESPONSE,
                response_data,
            )

            # Auto-escalation if confidence is too low
            if response.escalation_recommended:
                await self._send_message(
                    session.websocket,
                    MessageType.ESCALATION,
                    {
                        "message": "Based on the complexity of your issue, I recommend connecting you with a human support agent.",
                        "reason": "low_confidence",
                        "confidence": response.confidence,
                    },
                )

        except Exception as e:
            logger.error(
                "AI processing error: session=%s, error=%s",
                session.session_id,
                str(e),
            )
            await self._send_message(
                session.websocket,
                MessageType.TYPING_STOP,
                {},
            )
            await self._send_message(
                session.websocket,
                MessageType.ERROR,
                {"error": "I encountered an issue processing your request. Please try rephrasing or contact the help desk at ext. 4500."},
            )

    async def _handle_feedback(self, session: ChatSession, data: dict) -> None:
        """Handle user feedback on AI responses."""
        feedback_type = data.get("feedback_type", "")  # helpful, unhelpful, incorrect
        message_id = data.get("message_id", "")
        comment = data.get("comment", "")

        logger.info(
            "Feedback received: session=%s, type=%s, message_id=%s",
            session.session_id,
            feedback_type,
            message_id,
        )

        await self._send_message(
            session.websocket,
            MessageType.SYSTEM,
            {"message": "Thank you for your feedback. It helps improve our AI support system."},
        )

    async def _handle_escalation(self, session: ChatSession, reason: str) -> None:
        """Handle manual escalation request."""
        logger.info(
            "Manual escalation requested: session=%s, reason=%s",
            session.session_id,
            reason,
        )

        await self._send_message(
            session.websocket,
            MessageType.ESCALATION,
            {
                "message": "Your request has been escalated to a human support agent. Expected wait time: 5-10 minutes.",
                "ticket_created": True,
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
            logger.warning("Failed to send message: %s", str(e))

    async def _cleanup_stale_sessions(self) -> None:
        """Periodically clean up inactive sessions."""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes

            now = datetime.now(timezone.utc)
            stale_sessions = []

            for session_id, session in self._sessions.items():
                last_activity = datetime.fromisoformat(session.last_activity)
                idle_seconds = (now - last_activity).total_seconds()

                if idle_seconds > self.SESSION_TIMEOUT_SECONDS:
                    stale_sessions.append(session_id)

            for session_id in stale_sessions:
                session = self._sessions.get(session_id)
                if session:
                    try:
                        await self._send_message(
                            session.websocket,
                            MessageType.SESSION_END,
                            {"message": "Session closed due to inactivity."},
                        )
                        await session.websocket.close()
                    except Exception:
                        pass
                    session.is_active = False
                    del self._sessions[session_id]

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
