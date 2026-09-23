#!/usr/bin/env python3
"""
API Route Definitions
Project: AI-Powered Healthcare IT Support System
Timeline: April 2024 - June 2024

FastAPI route definitions for chat, ticket classification,
knowledge base management, and system administration endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket

from src.api.models import (
    ChatRequest,
    ChatResponse,
    ClassifyRequest,
    ClassifyResponse,
    IngestRequest,
    IngestResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)

logger = logging.getLogger("ai_support.api")

router = APIRouter()


def get_ai_engine():
    """Dependency injection for AI engine."""
    from src.main import app_state
    if not app_state.ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not initialized")
    return app_state.ai_engine


def get_knowledge_base():
    """Dependency injection for knowledge base."""
    from src.main import app_state
    if not app_state.knowledge_base:
        raise HTTPException(status_code=503, detail="Knowledge base not initialized")
    return app_state.knowledge_base


# -------------------------------------------------------------------
# Chat Endpoints
# -------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ai_engine=Depends(get_ai_engine),
):
    """
    Process a chat message and return AI response.
    Supports single-turn and multi-turn conversations via session_id.
    """
    logger.info(
        "Chat request: session=%s, message_length=%d",
        request.session_id,
        len(request.message),
    )

    user_context = None
    if request.department or request.user_role:
        user_context = {
            "department": request.department,
            "role": request.user_role or "staff",
        }

    response = await ai_engine.process_message(
        message=request.message,
        session_id=request.session_id,
        user_context=user_context,
    )

    return ChatResponse(
        message=response.message,
        session_id=request.session_id,
        confidence=response.confidence,
        source_documents=response.source_documents,
        escalation_recommended=response.escalation_recommended,
        response_time_ms=response.response_time_ms,
        phi_warning="PHI was detected and filtered" if response.phi_detected else None,
    )


@router.websocket("/chat/ws/{user_id}")
async def websocket_chat(
    websocket: WebSocket,
    user_id: str,
    department: Optional[str] = Query(None),
):
    """WebSocket endpoint for real-time chat sessions."""
    from src.chat_handler import ChatHandler
    from src.main import app_state

    handler = ChatHandler(ai_engine=app_state.ai_engine)
    await handler.handle_connection(
        websocket=websocket,
        user_id=user_id,
        department=department,
    )


# -------------------------------------------------------------------
# Ticket Classification Endpoints
# -------------------------------------------------------------------

@router.post("/classify", response_model=ClassifyResponse)
async def classify_ticket(request: ClassifyRequest):
    """
    Classify an IT support ticket into category, priority, and severity.
    Returns routing information and SLA targets.
    """
    from src.ticket_classifier import TicketClassifier

    classifier = TicketClassifier()

    result = await classifier.classify(
        title=request.title,
        description=request.description,
        reporter_department=request.department,
        reporter_role=request.reporter_role,
    )

    logger.info(
        "Ticket classified: category=%s, priority=%s, confidence=%.2f",
        result.category.value,
        result.priority.value,
        result.confidence,
    )

    return ClassifyResponse(
        category=result.category.value,
        priority=result.priority.value,
        severity=result.severity.value,
        confidence=result.confidence,
        assigned_team=result.assigned_team,
        sla_response_hours=result.sla_response_hours,
        sla_resolution_hours=result.sla_resolution_hours,
        suggested_tags=result.suggested_tags,
        reasoning=result.reasoning,
        requires_escalation=result.requires_escalation,
    )


@router.post("/classify/batch")
async def batch_classify_tickets(tickets: list[ClassifyRequest]):
    """Classify multiple tickets in a single request."""
    from src.ticket_classifier import TicketClassifier

    if len(tickets) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 tickets per batch")

    classifier = TicketClassifier()
    ticket_dicts = [
        {
            "title": t.title,
            "description": t.description,
            "department": t.department,
            "role": t.reporter_role,
        }
        for t in tickets
    ]

    results = await classifier.batch_classify(ticket_dicts)

    return {
        "total": len(results),
        "classifications": [r.to_dict() for r in results],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# -------------------------------------------------------------------
# Knowledge Base Endpoints
# -------------------------------------------------------------------

@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge_base(
    request: KnowledgeSearchRequest,
    kb=Depends(get_knowledge_base),
):
    """Search the IT knowledge base for relevant articles."""
    results = await kb.search(
        query=request.query,
        n_results=request.max_results,
        category=request.category,
        min_relevance=request.min_relevance,
    )

    return KnowledgeSearchResponse(
        results=results,
        total_results=len(results),
        query=request.query,
    )


@router.post("/knowledge/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    kb=Depends(get_knowledge_base),
):
    """Ingest a new document into the knowledge base."""
    from src.knowledge_base import Document

    doc = Document(
        content=request.content,
        source=request.source,
        category=request.category,
        title=request.title,
    )

    result = await kb.ingest_document(doc)

    return IngestResponse(
        doc_id=result["doc_id"],
        chunks_created=result["chunks_created"],
        total_documents=result["total_documents"],
    )


@router.get("/knowledge/stats")
async def knowledge_base_stats(kb=Depends(get_knowledge_base)):
    """Return knowledge base statistics."""
    return await kb.get_stats()


# -------------------------------------------------------------------
# System Endpoints
# -------------------------------------------------------------------

@router.get("/stats")
async def system_stats(ai_engine=Depends(get_ai_engine)):
    """Return comprehensive system statistics."""
    return {
        "engine_stats": ai_engine.get_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
