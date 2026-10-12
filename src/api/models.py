"""Pydantic request and response models for the API endpoints.

Request models forbid unknown fields and bound every string, so oversized or
unexpected input is rejected with 422 before any service runs.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ingest import validate_source

SESSION_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"
CATEGORY_PATTERN = r"^[A-Za-z0-9_-]{1,50}$"
MAX_INGEST_CHARS = 100_000


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -------------------------------------------------------------------
# Chat Models
# -------------------------------------------------------------------

class ChatRequest(_Request):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User message to process",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=SESSION_ID_PATTERN,
        description="Unique session identifier for conversation continuity",
    )
    department: Optional[str] = Field(
        None,
        max_length=100,
        description="Reporter's department for context-aware responses",
    )
    user_role: Optional[str] = Field(
        None,
        max_length=50,
        description="Reporter's role (e.g., nurse, physician, admin)",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "message": "My VPN keeps disconnecting when I try to access the EHR system",
                    "session_id": "user-abc-123",
                    "department": "Radiology",
                    "user_role": "physician",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    message: str = Field(..., description="AI-generated response")
    session_id: str = Field(..., description="Session identifier")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Response confidence score",
    )
    source_documents: list[str] = Field(
        default_factory=list,
        description="Knowledge base sources referenced",
    )
    escalation_recommended: bool = Field(
        False,
        description="Whether human escalation is recommended",
    )
    response_time_ms: float = Field(
        ...,
        description="Response generation time in milliseconds",
    )
    phi_warning: Optional[str] = Field(
        None,
        description="Warning if PHI was detected and filtered",
    )


# -------------------------------------------------------------------
# Ticket Classification Models
# -------------------------------------------------------------------

class ClassifyRequest(_Request):
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Ticket title/subject",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Ticket description with issue details",
    )
    department: Optional[str] = Field(
        None,
        max_length=100,
        description="Reporter's department",
    )
    reporter_role: Optional[str] = Field(
        None,
        max_length=50,
        description="Reporter's organizational role",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Cannot access email on mobile device",
                    "description": "Outlook app on iPhone showing 'authentication failed' error since this morning. Restarted phone, cleared cache, still not working.",
                    "department": "Nursing",
                    "reporter_role": "nurse",
                }
            ]
        }
    }


class ClassifyResponse(BaseModel):
    category: str = Field(..., description="Ticket category")
    priority: str = Field(..., description="Ticket priority level")
    severity: str = Field(..., description="Ticket severity")
    confidence: float = Field(..., description="Classification confidence")
    assigned_team: str = Field(..., description="Assigned support team")
    sla_response_hours: float = Field(..., description="SLA response target in hours")
    sla_resolution_hours: float = Field(..., description="SLA resolution target in hours")
    suggested_tags: list[str] = Field(default_factory=list, description="Suggested tags")
    reasoning: str = Field("", description="Classification reasoning")
    requires_escalation: bool = Field(False, description="Whether escalation is needed")


# -------------------------------------------------------------------
# Knowledge Base Models
# -------------------------------------------------------------------

class KnowledgeSearchRequest(_Request):
    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Search query",
    )
    max_results: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of results",
    )
    category: Optional[str] = Field(
        None,
        pattern=CATEGORY_PATTERN,
        description="Filter by knowledge category",
    )
    min_relevance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score; None uses the active retrieval profile",
    )


class KnowledgeSearchResponse(BaseModel):
    results: list[dict] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total matching results")
    query: str = Field(..., description="Original search query")


class IngestRequest(_Request):
    content: str = Field(
        ...,
        min_length=10,
        max_length=MAX_INGEST_CHARS,
        description="Document content to ingest",
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Relative document name, e.g. vpn/disconnects.md (no '..', absolute paths or backslashes)",
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=CATEGORY_PATTERN,
        description="Knowledge category",
    )
    title: str = Field(
        "",
        max_length=256,
        description="Document title",
    )

    @field_validator("source")
    @classmethod
    def _safe_source(cls, value: str) -> str:
        return validate_source(value)


class IngestResponse(BaseModel):
    doc_id: str = Field(..., description="Generated document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    total_documents: int = Field(..., description="Total documents in knowledge base")
    phi_redacted: list[str] = Field(
        default_factory=list,
        description="Kinds of patient identifiers redacted from the article before storage",
    )
