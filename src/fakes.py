"""Deterministic offline stand-ins for the OpenAI chat model and embeddings.

``HashingEmbeddings`` turns text into a fixed-size bag-of-words vector using a
stable hash, so vectors are identical across processes. ``ExtractiveFakeChatModel``
answers by quoting lines from the knowledge base context block of the prompt.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage

EMBEDDING_DIM = 512

STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself
    yourselves
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

CONTEXT_HEADER = "Knowledge base context:"
CONTEXT_END_MARKERS = ("Conversation so far:", "Question:")
NO_CONTEXT_ANSWER = (
    "I could not find a relevant article in the knowledge base for this question. "
    "Please contact the service desk for further help."
)
MAX_STEPS = 5


def tokenize(text: str) -> List[str]:
    """Lowercase ``[a-z0-9]+`` tokens with stop words removed."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in STOP_WORDS]


def _bucket(token: str) -> tuple[int, float]:
    value = int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
    )
    sign = -1.0 if value >> 63 else 1.0
    return value % EMBEDDING_DIM, sign


def hashing_vector(text: str) -> List[float]:
    """Sublinear-tf hashed bag of words, L2 normalised (all zeros for empty input)."""
    vector = [0.0] * EMBEDDING_DIM
    for token, count in Counter(tokenize(text)).items():
        index, sign = _bucket(token)
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


class HashingEmbeddings(Embeddings):
    """Lexical embeddings that need no model download and no network."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [hashing_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return hashing_vector(text)


def extract_context(prompt: str) -> str:
    """Return the text between the context header and the next section header."""
    start = prompt.find(CONTEXT_HEADER)
    if start == -1:
        return ""
    body = prompt[start + len(CONTEXT_HEADER):]
    end = len(body)
    for marker in CONTEXT_END_MARKERS:
        pos = body.find(marker)
        if pos != -1:
            end = min(end, pos)
    return body[:end].strip()


def _context_lines(context: str) -> List[str]:
    lines = []
    for raw in context.splitlines():
        line = raw.strip()
        if not line or (line.startswith("[") and line.endswith("]")):
            continue
        line = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line)
        if line:
            lines.append(line)
    return lines


def extractive_answer(prompt: str) -> str:
    """Build a "Suggested steps" answer from the prompt's context block."""
    lines = _context_lines(extract_context(prompt))
    if not lines:
        return NO_CONTEXT_ANSWER
    steps = "\n".join(f"{i}. {line}" for i, line in enumerate(lines[:MAX_STEPS], start=1))
    return f"Suggested steps from the knowledge base:\n{steps}"


class ExtractiveFakeChatModel(SimpleChatModel):
    """Chat model that quotes the retrieved context instead of generating text."""

    @property
    def _llm_type(self) -> str:
        return "extractive-fake"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = "\n".join(
            m.content if isinstance(m.content, str) else str(m.content) for m in messages
        )
        return extractive_answer(prompt)
