"""Retrieval-augmented chat engine built as an LCEL chain.

For each message the engine redacts patient identifiers, retrieves knowledge
base chunks, picks a system prompt from the message content and the user's
role, renders the prompt with the session history and calls the chat model.
Confidence is the best retrieval relevance (0 when nothing is retrieved) and
escalation is recommended when it falls below the active profile threshold.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import Settings
from src.phi_filter import redact

logger = logging.getLogger("ai_support.ai_engine")

SECURITY_KEYWORDS = ("phishing", "malware", "ransomware", "breach", "suspicious", "virus", "hacked")
_SECURITY_PATTERN = re.compile(
    rf"\b(?:{'|'.join(SECURITY_KEYWORDS)})(?:s|es|ing|ed)?\b", re.IGNORECASE
)

ROLE_PROMPTS = {
    "physician": "clinical_staff",
    "nurse": "clinical_staff",
    "clinician": "clinical_staff",
    "clinical": "clinical_staff",
    "it": "technical_staff",
    "engineer": "technical_staff",
    "admin": "technical_staff",
    "technical": "technical_staff",
}

NO_HISTORY = "(none)"
USER, ASSISTANT = "user", "assistant"


@dataclass
class EngineResponse:
    message: str
    confidence: float
    source_documents: List[str]
    escalation_recommended: bool
    response_time_ms: float
    phi_detected: bool


@dataclass
class _Stats:
    total_messages: int = 0
    escalations_recommended: int = 0
    phi_detections: int = 0
    total_response_time_ms: float = 0.0
    prompts_used: Dict[str, int] = field(default_factory=dict)


class AIEngine:
    """Answers support questions from the knowledge base through a chat model."""

    def __init__(self, settings: Settings, prompts: Mapping[str, Any], knowledge_base, chat_model: BaseChatModel):
        self.settings = settings
        self.knowledge_base = knowledge_base
        self.system_prompts: Mapping[str, str] = prompts["system_prompts"]
        self.chain = (
            ChatPromptTemplate.from_messages(
                [("system", "{system_prompt}"), ("human", prompts["chat"]["user_template"])]
            )
            | chat_model
            | StrOutputParser()
        )
        self._history: Dict[str, Deque[Tuple[str, str]]] = {}
        self._stats = _Stats()

    @staticmethod
    def select_prompt_key(message: str, role: Optional[str]) -> str:
        """Security content first, regardless of role; then the role map; else default."""
        if _SECURITY_PATTERN.search(message):
            return "security_incident"
        return ROLE_PROMPTS.get((role or "").strip().lower(), "default")

    def _system_prompt(self, key: str) -> str:
        base = self.system_prompts["default"].strip()
        if key == "default":
            return base
        return f"{base}\n\n{self.system_prompts[key].strip()}"

    @staticmethod
    def _format_context(results: List[dict]) -> str:
        blocks = []
        for result in results:
            label = result.get("title") or result["source"]
            blocks.append(f"[{result['source']}: {label}]\n{result['content']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_history(history: Deque[Tuple[str, str]]) -> str:
        if not history:
            return NO_HISTORY
        names = {USER: "User", ASSISTANT: "Assistant"}
        return "\n".join(f"{names[role]}: {text}" for role, text in history)

    def _session(self, session_id: str) -> Deque[Tuple[str, str]]:
        if session_id not in self._history:
            self._history[session_id] = deque(maxlen=self.settings.chat.max_context_messages)
        return self._history[session_id]

    async def process_message(
        self,
        message: str,
        session_id: str,
        user_context: Optional[Mapping[str, Any]] = None,
    ) -> EngineResponse:
        started = time.perf_counter()
        user_context = user_context or {}

        phi_detected = False
        if self.settings.security.phi_detection_enabled:
            redaction = redact(message)
            message, phi_detected = redaction.text, bool(redaction.found)

        profile = self.settings.retrieval
        results = await self.knowledge_base.search(
            message,
            n_results=self.settings.knowledge_base.max_search_results,
            min_relevance=profile.min_relevance_score,
        )
        confidence = max((r["relevance_score"] for r in results), default=0.0)
        sources = list(dict.fromkeys(r["source"] for r in results))

        prompt_key = self.select_prompt_key(message, user_context.get("role"))
        history = self._session(session_id)
        answer = await self.chain.ainvoke(
            {
                "system_prompt": self._system_prompt(prompt_key),
                "context": self._format_context(results),
                "history": self._format_history(history),
                "question": message,
            }
        )
        history.append((USER, message))
        history.append((ASSISTANT, answer))

        escalate = confidence < profile.confidence_threshold
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        stats = self._stats
        stats.total_messages += 1
        stats.escalations_recommended += int(escalate)
        stats.phi_detections += int(phi_detected)
        stats.total_response_time_ms += elapsed_ms
        stats.prompts_used[prompt_key] = stats.prompts_used.get(prompt_key, 0) + 1

        logger.info(
            "Chat answered: session=%s prompt=%s sources=%d confidence=%.3f escalate=%s",
            session_id, prompt_key, len(sources), confidence, escalate,
        )
        return EngineResponse(
            message=answer,
            confidence=confidence,
            source_documents=sources,
            escalation_recommended=escalate,
            response_time_ms=elapsed_ms,
            phi_detected=phi_detected,
        )

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        return list(self._history.get(session_id, ()))

    def get_stats(self) -> dict:
        stats = self._stats
        average = stats.total_response_time_ms / stats.total_messages if stats.total_messages else 0.0
        return {
            "total_messages": stats.total_messages,
            "active_sessions": len(self._history),
            "escalations_recommended": stats.escalations_recommended,
            "phi_detections": stats.phi_detections,
            "average_response_time_ms": round(average, 2),
            "prompts_used": dict(stats.prompts_used),
            "retrieval_profile": self.settings.active_profile_name,
        }
