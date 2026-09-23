#!/usr/bin/env python3
"""
Pydantic Request/Response Models
Project: AI-Powered Healthcare IT Support System
Timeline: April 2024 - June 2024

Data validation models for all API endpoints using Pydantic v2.
"""

from typing import Optional

from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Chat Models
# -------------------------------------------------------------------

class ChatRequest(BaseModel):
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

class ClassifyRequest(BaseModel):
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

class KnowledgeSearchRequest(BaseModel):
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
        description="Filter by knowledge category",
    )
    min_relevance: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold",
    )


class KnowledgeSearchResponse(BaseModel):
    results: list[dict] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., description="Total matching results")
    query: str = Field(..., description="Original search query")


class IngestRequest(BaseModel):
    content: str = Field(
        ...,
        min_length=10,
        description="Document content to ingest",
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Document source identifier",
    )
    category: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Knowledge category",
    )
    title: str = Field(
        "",
        max_length=256,
        description="Document title",
    )


class IngestResponse(BaseModel):
    doc_id: str = Field(..., description="Generated document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    total_documents: int = Field(..., description="Total documents in knowledge base")
