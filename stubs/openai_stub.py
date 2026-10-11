"""Deterministic OpenAI-compatible API for keyless provider testing.

Implements the subset of the OpenAI REST API that ``langchain-openai`` uses:

- ``POST /v1/chat/completions``: when the prompt carries the classification
  marker, returns classifier JSON built from the fallback keyword rules applied
  to the ticket text; otherwise returns the extractive answer used by the
  offline chat model.
- ``POST /v1/embeddings``: ``input`` as a string or a list of strings, embedded
  with ``HashingEmbeddings``. Token-ID arrays are rejected with 400, which is
  what a client built without ``check_embedding_ctx_length=False`` sends.
- ``GET /v1/models``: the model names from the settings.

Run with ``uvicorn stubs.openai_stub:app --port 8001``.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any, List, Optional, Union

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from src.config import load_settings
from src.fakes import HashingEmbeddings, extractive_answer
from src.ticket_classifier import FALLBACK_KEYWORDS, keyword_rules

CLASSIFICATION_MARKER = "### TICKET_CLASSIFICATION_JSON ###"
CLASSIFICATION_CONFIDENCE = 0.75

app = FastAPI(title="OpenAI-compatible stub", docs_url=None, redoc_url=None)
_embeddings = HashingEmbeddings()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Any]]] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: str = "float"


def _error(status: int, message: str, param: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": None,
            }
        },
    )


def _message_text(message: ChatMessage) -> str:
    content = message.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(parts)


def _ticket_text(prompt: str) -> str:
    """The ticket fields of a classification prompt.

    Only the text before the marker is used, and from the ``Title:`` line on
    when present, so the category names listed in the output contract and the
    system prompt never influence the result.
    """
    head = prompt.split(CLASSIFICATION_MARKER, 1)[0]
    start = head.rfind("Title:")
    return head[start:] if start != -1 else head


def classification_json(prompt: str) -> str:
    """Classifier JSON for the ticket in ``prompt``."""
    text = _ticket_text(prompt)
    category, priority, severity = keyword_rules(text)
    keywords = FALLBACK_KEYWORDS.get(category, [])
    lowered = text.lower()
    tags = [kw for kw in keywords if kw in lowered][:3] or [category.value]
    return json.dumps(
        {
            "category": category.value,
            "priority": priority.value,
            "severity": severity.value,
            "confidence": CLASSIFICATION_CONFIDENCE,
            "suggested_tags": tags,
            "reasoning": f"Keyword rules matched category '{category.value}'.",
            "requires_escalation": priority.value == "critical",
        }
    )


def _word_count(text: str) -> int:
    return len(text.split())


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        return _error(400, "Streaming is not supported by this stub.", "stream")

    prompt = "\n".join(_message_text(m) for m in request.messages)
    if CLASSIFICATION_MARKER in prompt:
        content = classification_json(prompt)
    else:
        content = extractive_answer(prompt)

    prompt_tokens = _word_count(prompt)
    completion_tokens = _word_count(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _encode(vector: List[float], encoding_format: str) -> Union[List[float], str]:
    if encoding_format == "base64":
        return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")
    return vector


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest):
    raw = request.input
    texts = [raw] if isinstance(raw, str) else raw
    if not texts or not all(isinstance(item, str) for item in texts):
        return _error(
            400,
            "Token-ID arrays are not supported; send raw strings "
            "(build OpenAIEmbeddings with check_embedding_ctx_length=False).",
            "input",
        )
    if request.encoding_format not in ("float", "base64"):
        return _error(400, f"Unsupported encoding_format '{request.encoding_format}'.",
                      "encoding_format")

    vectors = _embeddings.embed_documents(texts)
    total = sum(_word_count(text) for text in texts)
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": _encode(vector, request.encoding_format),
            }
            for i, vector in enumerate(vectors)
        ],
        "model": request.model,
        "usage": {"prompt_tokens": total, "total_tokens": total},
    }


@app.get("/v1/models")
def models():
    llm = load_settings().llm
    names = sorted({llm.chat_model, llm.classification_model, llm.embedding_model})
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "stub"}
            for name in names
        ],
    }
