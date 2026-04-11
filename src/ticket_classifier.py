#!/usr/bin/env python3
"""
ML-Based IT Support Ticket Classifier
Project: AI-Powered Healthcare IT Support System
Author: Alexander Efrem - IT Operations Specialist, AEL Dubai
Timeline: April 2024 - June 2024

Classifies incoming IT support tickets into categories and priority levels
using GPT-4 with structured output. Supports automated routing and SLA
assignment for healthcare IT operations.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI

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
    GPT-4 based ticket classifier with structured output parsing.
    Classifies tickets by category, priority, and severity with
    automated routing and SLA assignment.
    """

    CLASSIFICATION_PROMPT = """You are an IT support ticket classifier for a healthcare organization.
Analyze the following support ticket and provide a structured classification.

IMPORTANT RULES:
- Healthcare/EHR system issues affecting patient care are always CRITICAL priority
- Security incidents are always HIGH priority or above
- Consider the healthcare environment when assessing impact
- If multiple categories apply, choose the primary one

Ticket Title: {title}
Ticket Description: {description}
{additional_context}

Respond in the following JSON format ONLY (no other text):
{{
    "category": "<one of: network, email, vpn, ehr_systems, active_directory, printing, hardware, software, security, cloud_services, mobile_devices, telephony>",
    "priority": "<one of: critical, high, medium, low>",
    "severity": "<one of: sev1, sev2, sev3, sev4>",
    "confidence": <float 0.0-1.0>,
    "suggested_tags": ["tag1", "tag2"],
    "reasoning": "<brief explanation of classification>",
    "requires_escalation": <true/false>
}}"""

    def __init__(
        self,
        model_name: str = "gpt-4-0125-preview",
        temperature: float = 0.1,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=500,
            request_timeout=15,
        )

        self._classification_count = 0
        self._total_time_ms = 0.0

        logger.info("Ticket classifier initialized: model=%s", model_name)

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
        start_time = time.time()

        additional_context = ""
        if reporter_department:
            additional_context += f"Reporter Department: {reporter_department}\n"
        if reporter_role:
            additional_context += f"Reporter Role: {reporter_role}\n"

        prompt = self.CLASSIFICATION_PROMPT.format(
            title=title,
            description=description,
            additional_context=additional_context,
        )

        try:
            response = await self.llm.ainvoke(prompt)
            raw_response = response.content.strip()

            # Parse JSON response
            classification_data = self._parse_response(raw_response)
            classification_time = (time.time() - start_time) * 1000

            category = TicketCategory(classification_data["category"])
            priority = TicketPriority(classification_data["priority"])
            severity = TicketSeverity(classification_data["severity"])

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
                requires_escalation=classification_data.get("requires_escalation", False),
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

        return data

    def _fallback_classification(
        self,
        title: str,
        description: str,
        start_time: float,
    ) -> ClassificationResult:
        """Rule-based fallback when AI classification fails."""
        combined = f"{title} {description}".lower()

        # Simple keyword-based classification
        category = TicketCategory.SOFTWARE
        priority = TicketPriority.MEDIUM
        severity = TicketSeverity.SEV3

        keyword_map = {
            TicketCategory.NETWORK: ["network", "wifi", "internet", "connectivity", "dns", "dhcp"],
            TicketCategory.EMAIL: ["email", "outlook", "mailbox", "smtp", "exchange"],
            TicketCategory.VPN: ["vpn", "remote access", "tunnel", "connection dropped"],
            TicketCategory.EHR_SYSTEMS: ["ehr", "epic", "cerner", "clinical", "patient record", "emr"],
            TicketCategory.ACTIVE_DIRECTORY: ["password", "login", "locked out", "active directory", "ad account"],
            TicketCategory.PRINTING: ["printer", "print", "scanner", "fax"],
            TicketCategory.HARDWARE: ["laptop", "monitor", "keyboard", "mouse", "docking"],
            TicketCategory.SECURITY: ["virus", "malware", "phishing", "breach", "suspicious", "unauthorized"],
            TicketCategory.CLOUD_SERVICES: ["aws", "azure", "cloud", "s3", "ec2"],
            TicketCategory.MOBILE_DEVICES: ["iphone", "android", "mobile", "tablet", "mdm"],
            TicketCategory.TELEPHONY: ["phone", "voip", "teams call", "headset"],
        }

        for cat, keywords in keyword_map.items():
            if any(kw in combined for kw in keywords):
                category = cat
                break

        # Priority escalation for healthcare-critical keywords
        if any(kw in combined for kw in ["ehr", "patient", "clinical", "emergency", "down", "outage"]):
            priority = TicketPriority.CRITICAL
            severity = TicketSeverity.SEV1

        elif any(kw in combined for kw in ["security", "breach", "virus", "unauthorized"]):
            priority = TicketPriority.HIGH
            severity = TicketSeverity.SEV2

        classification_time = (time.time() - start_time) * 1000

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
