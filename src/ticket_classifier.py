#!/usr/bin/env python3
"""
IT support ticket classifier.

Classifies incoming IT support tickets into categories and priority levels
with a chat model returning JSON, and falls back to keyword rules when the
model call or the parse fails. Results carry the routed team and SLA targets.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import Settings, load_prompts, load_settings
from src.phi_filter import redact

logger = logging.getLogger("ai_support.classifier")


class TicketCategory(Enum):
    NETWORK = "network"
    EMAIL = "email"
    VPN = "vpn"
    EHR_SYSTEMS = "ehr_systems"
    ACTIVE_DIRECTORY = "active_directory"
    PRINTING = "printing"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    SECURITY = "security"
    CLOUD_SERVICES = "cloud_services"
    MOBILE_DEVICES = "mobile_devices"
    TELEPHONY = "telephony"


class TicketPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TicketSeverity(Enum):
    SEV1 = "sev1"  # Complete service outage
    SEV2 = "sev2"  # Major feature degradation
    SEV3 = "sev3"  # Minor feature issue
    SEV4 = "sev4"  # Cosmetic or informational


# SLA definitions in hours by priority
SLA_RESPONSE_HOURS = {
    TicketPriority.CRITICAL: 0.25,  # 15 minutes
    TicketPriority.HIGH: 1.0,
    TicketPriority.MEDIUM: 4.0,
    TicketPriority.LOW: 8.0,
}

SLA_RESOLUTION_HOURS = {
    TicketPriority.CRITICAL: 4.0,
    TicketPriority.HIGH: 8.0,
    TicketPriority.MEDIUM: 24.0,
    TicketPriority.LOW: 72.0,
}

# Routing rules per category
ROUTING_MAP = {
    TicketCategory.NETWORK: "network-ops-team",
    TicketCategory.EMAIL: "messaging-team",
    TicketCategory.VPN: "network-ops-team",
    TicketCategory.EHR_SYSTEMS: "clinical-it-team",
    TicketCategory.ACTIVE_DIRECTORY: "identity-team",
    TicketCategory.PRINTING: "desktop-support",
    TicketCategory.HARDWARE: "desktop-support",
    TicketCategory.SOFTWARE: "application-support",
    TicketCategory.SECURITY: "security-ops-team",
    TicketCategory.CLOUD_SERVICES: "cloud-ops-team",
    TicketCategory.MOBILE_DEVICES: "mobile-device-mgmt",
    TicketCategory.TELEPHONY: "telecom-team",
}


# Fallback keyword map. Order matters: the first category with a match wins,
# so security is checked before email (a phishing email is a security issue).
FALLBACK_KEYWORDS = {
    TicketCategory.SECURITY: ["virus", "malware", "ransomware", "phishing", "breach", "suspicious", "unauthorized", "hacked"],
    TicketCategory.NETWORK: ["network", "wifi", "internet", "connectivity", "dns", "dhcp"],
    TicketCategory.EMAIL: ["email", "outlook", "mailbox", "smtp", "exchange"],
    TicketCategory.VPN: ["vpn", "remote access", "tunnel", "connection dropped"],
    TicketCategory.EHR_SYSTEMS: ["ehr", "epic", "cerner", "clinical", "patient record", "emr"],
    TicketCategory.ACTIVE_DIRECTORY: ["password", "login", "locked out", "active directory", "ad account"],
    TicketCategory.PRINTING: ["printer", "print", "scanner", "fax"],
    TicketCategory.HARDWARE: ["laptop", "monitor", "keyboard", "mouse", "docking"],
    TicketCategory.CLOUD_SERVICES: ["aws", "azure", "cloud", "s3", "ec2"],
    TicketCategory.MOBILE_DEVICES: ["iphone", "android", "mobile", "tablet", "mdm"],
    TicketCategory.TELEPHONY: ["phone", "voip", "teams call", "headset"],
}

CRITICAL_KEYWORDS = ["ehr", "patient", "clinical", "emergency", "down", "outage"]
HIGH_KEYWORDS = ["security", "breach", "virus", "unauthorized"]


def _keyword_pattern(keywords: list[str]) -> re.Pattern:
    """Whole-word match for any keyword, allowing an s/es/ing/ed suffix."""
    alternatives = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(rf"\b(?:{alternatives})(?:s|es|ing|ed)?\b", re.IGNORECASE)


_CATEGORY_PATTERNS = {cat: _keyword_pattern(kws) for cat, kws in FALLBACK_KEYWORDS.items()}
_CRITICAL_PATTERN = _keyword_pattern(CRITICAL_KEYWORDS)
_HIGH_PATTERN = _keyword_pattern(HIGH_KEYWORDS)


@dataclass
class ClassificationResult:
    category: TicketCategory
    priority: TicketPriority
    severity: TicketSeverity
    confidence: float
    assigned_team: str
    sla_response_hours: float
    sla_resolution_hours: float
    suggested_tags: list[str] = field(default_factory=list)
    reasoning: str = ""
    requires_escalation: bool = False
    classification_time_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "priority": self.priority.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "assigned_team": self.assigned_team,
            "sla_response_hours": self.sla_response_hours,
            "sla_resolution_hours": self.sla_resolution_hours,
            "suggested_tags": self.suggested_tags,
            "reasoning": self.reasoning,
            "requires_escalation": self.requires_escalation,
            "classification_time_ms": self.classification_time_ms,
            "timestamp": self.timestamp,
        }


class TicketClassifier:
    """
    Chat-model ticket classifier with structured output parsing.

    The prompt comes from ``prompts.yml`` (``classification.system_prompt`` and
    ``classification.user_template``). Patient identifiers are redacted from the
    ticket before rendering. Categories listed in
    ``settings.escalation.auto_escalate_categories`` always require escalation.
    An OpenAI client is built only when no ``llm`` is injected.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        settings: Optional[Settings] = None,
        model_name: str = "gpt-4-0125-preview",
        temperature: float = 0.1,
        prompts: Optional[Mapping[str, Any]] = None,
    ):
        self.settings = settings if settings is not None else load_settings()
        prompts = prompts if prompts is not None else load_prompts()
        self.system_prompt: str = prompts["classification"]["system_prompt"]
        self.user_template: str = prompts["classification"]["user_template"]
        self.auto_escalate_categories = frozenset(
            self.settings.escalation.auto_escalate_categories
        )

        if llm is None:
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=500,
                request_timeout=15,
            )
            logger.info("Ticket classifier built OpenAI client: model=%s", model_name)
        self.llm = llm

        self._classification_count = 0
        self._total_time_ms = 0.0

    def build_messages(
        self,
        title: str,
        description: str,
        additional_context: str = "",
    ) -> list[BaseMessage]:
        """Render the system and user messages for an already-redacted ticket."""
        user_prompt = self.user_template.format(
            title=title,
            description=description,
            additional_context=additional_context or "none",
        )
        return [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]

    async def classify(
        self,
        title: str,
        description: str,
        reporter_department: Optional[str] = None,
        reporter_role: Optional[str] = None,
    ) -> ClassificationResult:
        """
        Classify a support ticket and return structured result
        with routing and SLA information.
        """
        start_time = time.perf_counter()

        context_parts = []
        if reporter_department:
            context_parts.append(f"Reporter Department: {reporter_department}")
        if reporter_role:
            context_parts.append(f"Reporter Role: {reporter_role}")

        title = redact(title).text
        description = redact(description).text
        additional_context = redact("\n".join(context_parts)).text

        messages = self.build_messages(title, description, additional_context)

        try:
            response = await self.llm.ainvoke(messages)
            raw_response = response.content.strip()

            # Parse JSON response
            classification_data = self._parse_response(raw_response)
            classification_time = (time.perf_counter() - start_time) * 1000

            category = TicketCategory(classification_data["category"])
            priority = TicketPriority(classification_data["priority"])
            severity = TicketSeverity(classification_data["severity"])

            requires_escalation = bool(classification_data.get("requires_escalation", False))
            if category.value in self.auto_escalate_categories:
                requires_escalation = True

            result = ClassificationResult(
                category=category,
                priority=priority,
                severity=severity,
                confidence=float(classification_data.get("confidence", 0.8)),
                assigned_team=ROUTING_MAP.get(category, "general-support"),
                sla_response_hours=SLA_RESPONSE_HOURS[priority],
                sla_resolution_hours=SLA_RESOLUTION_HOURS[priority],
                suggested_tags=classification_data.get("suggested_tags", []),
                reasoning=classification_data.get("reasoning", ""),
                requires_escalation=requires_escalation,
                classification_time_ms=classification_time,
            )

            self._classification_count += 1
            self._total_time_ms += classification_time

            logger.info(
                "Ticket classified: category=%s, priority=%s, confidence=%.2f, team=%s, time=%.0fms",
                category.value,
                priority.value,
                result.confidence,
                result.assigned_team,
                classification_time,
            )

            return result

        except Exception as e:
            logger.error("Classification failed: %s", str(e))
            return self._fallback_classification(title, description, start_time)

    async def batch_classify(
        self,
        tickets: list[dict],
    ) -> list[ClassificationResult]:
        """Classify multiple tickets in sequence."""
        results = []
        for ticket in tickets:
            result = await self.classify(
                title=ticket.get("title", ""),
                description=ticket.get("description", ""),
                reporter_department=ticket.get("department"),
                reporter_role=ticket.get("role"),
            )
            results.append(result)

        logger.info("Batch classification complete: %d tickets processed", len(results))
        return results

    def _parse_response(self, raw_response: str) -> dict:
        """Parse and validate the LLM JSON response."""
        # Handle markdown code blocks
        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0]
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].split("```")[0]

        try:
            data = json.loads(raw_response.strip())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse classifier response: %s", str(e))
            raise ValueError(f"Invalid JSON response from classifier: {str(e)}")

        # Validate required fields
        required_fields = ["category", "priority", "severity"]
        for field_name in required_fields:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")

        # Validate enum values
        valid_categories = [c.value for c in TicketCategory]
        if data["category"] not in valid_categories:
            raise ValueError(f"Invalid category: {data['category']}")

        valid_priorities = [p.value for p in TicketPriority]
        if data["priority"] not in valid_priorities:
            raise ValueError(f"Invalid priority: {data['priority']}")

        valid_severities = [s.value for s in TicketSeverity]
        if data["severity"] not in valid_severities:
            raise ValueError(f"Invalid severity: {data['severity']}")

        return data

    def _fallback_classification(
        self,
        title: str,
        description: str,
        start_time: Optional[float] = None,
    ) -> ClassificationResult:
        """Rule-based fallback when AI classification fails.

        ``start_time`` is a ``time.perf_counter()`` reading taken when the
        request began. Any value that is missing, zero or later than now is
        ignored and the fallback's own duration is reported instead.
        """
        method_start = time.perf_counter()
        combined = f"{title} {description}"

        category = TicketCategory.SOFTWARE
        priority = TicketPriority.MEDIUM
        severity = TicketSeverity.SEV3

        for cat, pattern in _CATEGORY_PATTERNS.items():
            if pattern.search(combined):
                category = cat
                break

        if _CRITICAL_PATTERN.search(combined):
            priority = TicketPriority.CRITICAL
            severity = TicketSeverity.SEV1
        elif category == TicketCategory.SECURITY or _HIGH_PATTERN.search(combined):
            priority = TicketPriority.HIGH
            severity = TicketSeverity.SEV2

        now = time.perf_counter()
        if start_time is not None and 0 < start_time <= now:
            classification_time = (now - start_time) * 1000
        else:
            classification_time = (now - method_start) * 1000

        return ClassificationResult(
            category=category,
            priority=priority,
            severity=severity,
            confidence=0.5,  # Lower confidence for fallback
            assigned_team=ROUTING_MAP.get(category, "general-support"),
            sla_response_hours=SLA_RESPONSE_HOURS[priority],
            sla_resolution_hours=SLA_RESOLUTION_HOURS[priority],
            suggested_tags=["fallback-classification"],
            reasoning="Classified using keyword-based fallback (AI classification unavailable)",
            requires_escalation=True,
            classification_time_ms=classification_time,
        )

    def get_stats(self) -> dict:
        """Return classifier statistics."""
        avg_time = (
            self._total_time_ms / self._classification_count
            if self._classification_count > 0
            else 0
        )
        return {
            "total_classifications": self._classification_count,
            "avg_classification_time_ms": round(avg_time, 2),
        }
