"""REST and WebSocket routes.

Services are resolved from ``request.app.state``, where the application
lifespan stores them. Every route that runs model or embedding work holds a
slot of the app's ``ConcurrencyLimiter`` (``work_slot``), and ingest requires
the operator ingest key (``require_ingest_key``).
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket

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
from src.phi_filter import redact
from src.security import bearer_token, keys_match, log_value, origin_allowed

logger = logging.getLogger("ai_support.api")

router = APIRouter()


MAX_BATCH_SIZE = 50
_USER_ID = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")


def _service(request: Request, name: str, label: str):
    service = getattr(request.app.state, name, None)
    if service is None:
        raise HTTPException(status_code=503, detail=f"{label} not initialized")
    return service


def get_ai_engine(request: Request):
    return _service(request, "ai_engine", "AI engine")


def get_knowledge_base(request: Request):
    return _service(request, "knowledge_base", "Knowledge base")


def get_classifier(request: Request):
    return _service(request, "classifier", "Ticket classifier")


def get_settings(request: Request):
    return request.app.state.settings


def client_key(connection) -> str:
    """The address a request or WebSocket came from, used for per-client quotas."""
    client = connection.client
    return client.host if client and client.host else "unknown"


async def work_slot(request: Request):
    """Hold one concurrency slot for the duration of the request, or answer 503."""
    limiter = request.app.state.work_limiter
    key = client_key(request)
    if not limiter.try_acquire(key):
        raise HTTPException(status_code=503, detail="Server busy, please retry shortly")
    try:
        yield
    finally:
        limiter.release(key)


async def require_ingest_key(request: Request) -> None:
    """Deny unless the configured ingest key is presented as a Bearer token."""
    configured = request.app.state.settings.security.ingest_api_key
    if configured is None:
        raise HTTPException(status_code=503, detail="Ingest is disabled: no ingest key is configured")
    failures = request.app.state.auth_failures
    key = client_key(request)
    if failures.blocked(key):
        raise HTTPException(status_code=429, detail="Too many failed attempts, retry later")
    supplied = bearer_token(request.headers.get("authorization"))
    if not keys_match(supplied, configured.get_secret_value()):
        failures.record_failure(key)
        logger.warning("Ingest key rejected: client=%s", log_value(key))
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid ingest key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# -------------------------------------------------------------------
# Chat Endpoints
# -------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(work_slot)])
async def chat(
    request: ChatRequest,
    http_request: Request,
    ai_engine=Depends(get_ai_engine),
):
    """
    Process a chat message and return AI response.
    Supports single-turn and multi-turn conversations via session_id. The
    history is kept per (client address, session_id), so a session id used
    from another address starts a separate conversation.
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
        session_id=f"{client_key(http_request)}|{request.session_id}",
        user_context=user_context,
        owner=client_key(http_request),
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
    role: Optional[str] = Query(None),
):
    """WebSocket endpoint for real-time chat sessions on the app's shared handler.

    Upgrades from another origin, malformed user ids and over-long query
    parameters are refused with close code 1008 before the socket is accepted.
    """
    settings = websocket.app.state.settings
    if not origin_allowed(
        websocket.headers.get("origin"), websocket.headers.get("host"), settings.security.allowed_origins
    ):
        await websocket.close(code=1008)
        return
    if (
        not _USER_ID.match(user_id)
        or len(department or "") > 100
        or len(role or "") > 50
    ):
        await websocket.close(code=1008)
        return
    handler = getattr(websocket.app.state, "chat_handler", None)
    if handler is None:
        await websocket.close(code=1013)
        return
    await handler.handle_connection(
        websocket=websocket,
        user_id=user_id,
        department=department,
        role=role,
        client=client_key(websocket),
    )


# -------------------------------------------------------------------
# Ticket Classification Endpoints
# -------------------------------------------------------------------

@router.post("/classify", response_model=ClassifyResponse, dependencies=[Depends(work_slot)])
async def classify_ticket(request: ClassifyRequest, classifier=Depends(get_classifier)):
    """
    Classify an IT support ticket into category, priority, and severity.
    Returns routing information and SLA targets.
    """
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


@router.post("/classify/batch", dependencies=[Depends(work_slot)])
async def batch_classify_tickets(
    tickets: list[ClassifyRequest],
    classifier=Depends(get_classifier),
):
    """Classify multiple tickets in a single request."""
    if len(tickets) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_BATCH_SIZE} tickets per batch"
        )

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

@router.post("/knowledge/search", response_model=KnowledgeSearchResponse, dependencies=[Depends(work_slot)])
async def search_knowledge_base(
    request: KnowledgeSearchRequest,
    kb=Depends(get_knowledge_base),
    settings=Depends(get_settings),
):
    """Search the IT knowledge base; min_relevance defaults to the active profile."""
    min_relevance = request.min_relevance
    if min_relevance is None:
        min_relevance = settings.retrieval.min_relevance_score
    results = await kb.search(
        query=request.query,
        n_results=request.max_results,
        category=request.category,
        min_relevance=min_relevance,
    )

    return KnowledgeSearchResponse(
        results=results,
        total_results=len(results),
        query=redact(request.query).text,
    )


@router.post(
    "/knowledge/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_ingest_key), Depends(work_slot)],
)
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
        phi_redacted=result.get("phi_redacted", []),
    )


@router.get("/knowledge/stats")
async def knowledge_base_stats(kb=Depends(get_knowledge_base)):
    """Return knowledge base statistics."""
    return await kb.get_stats()


# -------------------------------------------------------------------
# System Endpoints
# -------------------------------------------------------------------

@router.get("/stats")
async def system_stats(
    ai_engine=Depends(get_ai_engine),
    classifier=Depends(get_classifier),
):
    """Return engine and classifier counters."""
    return {
        "engine_stats": ai_engine.get_stats(),
        "classifier_stats": classifier.get_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
