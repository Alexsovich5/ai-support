#!/usr/bin/env python3
"""
Ticket Classifier Unit Tests
Project: AI-Powered Healthcare IT Support System
Timeline: April 2024 - June 2024

Unit tests for the ticket classification system including
AI-based and fallback classification paths.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.ticket_classifier import (
    ClassificationResult,
    TicketCategory,
    TicketClassifier,
    TicketPriority,
    TicketSeverity,
    ROUTING_MAP,
    SLA_RESPONSE_HOURS,
    SLA_RESOLUTION_HOURS,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def classifier():
    """Create a TicketClassifier instance with mocked LLM."""
    with patch("src.ticket_classifier.ChatOpenAI") as mock_llm:
        instance = TicketClassifier(model_name="gpt-4-test", temperature=0.1)
        instance.llm = mock_llm.return_value
        yield instance


@pytest.fixture
def sample_network_ticket():
    return {
        "title": "WiFi keeps disconnecting in Building A",
        "description": "Multiple users reporting intermittent WiFi drops on floor 3. "
        "Started around 10am today. Affects about 20 workstations.",
    }


@pytest.fixture
def sample_ehr_ticket():
    return {
        "title": "EHR system unresponsive",
        "description": "Epic application not loading for clinical staff. "
        "Error: Connection timeout. Multiple departments affected. "
        "Patient care impacted.",
    }


@pytest.fixture
def sample_security_ticket():
    return {
        "title": "Suspicious email with attachment",
        "description": "Received email appearing to be from hospital admin "
        "asking to click a link and enter credentials. "
        "Multiple staff members received the same email.",
    }


@pytest.fixture
def sample_password_ticket():
    return {
        "title": "Account locked out",
        "description": "Cannot log into my computer. Says my Active Directory "
        "account is locked. Need access urgently for morning rounds.",
    }


# -------------------------------------------------------------------
# Classification Result Tests
# -------------------------------------------------------------------

class TestClassificationResult:
    def test_to_dict(self):
        result = ClassificationResult(
            category=TicketCategory.NETWORK,
            priority=TicketPriority.HIGH,
            severity=TicketSeverity.SEV2,
            confidence=0.92,
            assigned_team="network-ops-team",
            sla_response_hours=1.0,
            sla_resolution_hours=8.0,
            suggested_tags=["wifi", "building-a"],
            reasoning="WiFi connectivity issue affecting multiple users",
        )

        d = result.to_dict()

        assert d["category"] == "network"
        assert d["priority"] == "high"
        assert d["severity"] == "sev2"
        assert d["confidence"] == 0.92
        assert d["assigned_team"] == "network-ops-team"
        assert "wifi" in d["suggested_tags"]
        assert d["requires_escalation"] is False

    def test_default_values(self):
        result = ClassificationResult(
            category=TicketCategory.EMAIL,
            priority=TicketPriority.MEDIUM,
            severity=TicketSeverity.SEV3,
            confidence=0.8,
            assigned_team="messaging-team",
            sla_response_hours=4.0,
            sla_resolution_hours=24.0,
        )

        assert result.suggested_tags == []
        assert result.reasoning == ""
        assert result.requires_escalation is False
        assert result.classification_time_ms == 0.0


# -------------------------------------------------------------------
# Routing Configuration Tests
# -------------------------------------------------------------------

class TestRoutingConfiguration:
    def test_all_categories_have_routing(self):
        """Every ticket category must have an assigned team."""
        for category in TicketCategory:
            assert category in ROUTING_MAP, f"Missing routing for {category.value}"

    def test_all_priorities_have_sla_response(self):
        """Every priority level must have an SLA response time."""
        for priority in TicketPriority:
            assert priority in SLA_RESPONSE_HOURS, f"Missing SLA response for {priority.value}"

    def test_all_priorities_have_sla_resolution(self):
        """Every priority level must have an SLA resolution time."""
        for priority in TicketPriority:
            assert priority in SLA_RESOLUTION_HOURS, f"Missing SLA resolution for {priority.value}"

    def test_critical_sla_is_fastest(self):
        """Critical priority must have the shortest SLA."""
        critical_response = SLA_RESPONSE_HOURS[TicketPriority.CRITICAL]
        for priority in TicketPriority:
            assert SLA_RESPONSE_HOURS[priority] >= critical_response

    def test_sla_response_less_than_resolution(self):
        """SLA response time must be less than resolution time."""
        for priority in TicketPriority:
            assert SLA_RESPONSE_HOURS[priority] < SLA_RESOLUTION_HOURS[priority]

    def test_ehr_routed_to_clinical_team(self):
        assert ROUTING_MAP[TicketCategory.EHR_SYSTEMS] == "clinical-it-team"

    def test_security_routed_to_security_ops(self):
        assert ROUTING_MAP[TicketCategory.SECURITY] == "security-ops-team"


# -------------------------------------------------------------------
# Fallback Classification Tests
# -------------------------------------------------------------------

class TestFallbackClassification:
    def test_network_keyword_detection(self, classifier, sample_network_ticket):
        result = classifier._fallback_classification(
            title=sample_network_ticket["title"],
            description=sample_network_ticket["description"],
            start_time=0,
        )
        assert result.category == TicketCategory.NETWORK
        assert result.confidence == 0.5  # Fallback confidence
        assert "fallback-classification" in result.suggested_tags

    def test_ehr_critical_priority(self, classifier, sample_ehr_ticket):
        result = classifier._fallback_classification(
            title=sample_ehr_ticket["title"],
            description=sample_ehr_ticket["description"],
            start_time=0,
        )
        assert result.category == TicketCategory.EHR_SYSTEMS
        assert result.priority == TicketPriority.CRITICAL
        assert result.severity == TicketSeverity.SEV1

    def test_security_high_priority(self, classifier, sample_security_ticket):
        result = classifier._fallback_classification(
            title=sample_security_ticket["title"],
            description=sample_security_ticket["description"],
            start_time=0,
        )
        assert result.category == TicketCategory.SECURITY
        assert result.priority == TicketPriority.HIGH

    def test_password_ad_classification(self, classifier, sample_password_ticket):
        result = classifier._fallback_classification(
            title=sample_password_ticket["title"],
            description=sample_password_ticket["description"],
            start_time=0,
        )
        assert result.category == TicketCategory.ACTIVE_DIRECTORY

    def test_fallback_always_recommends_escalation(self, classifier):
        result = classifier._fallback_classification(
            title="Generic issue",
            description="Something is not working properly",
            start_time=0,
        )
        assert result.requires_escalation is True

    def test_unknown_ticket_defaults_to_software(self, classifier):
        result = classifier._fallback_classification(
            title="Application behaving oddly",
            description="The system is doing something unexpected with the UI",
            start_time=0,
        )
        assert result.category == TicketCategory.SOFTWARE


# -------------------------------------------------------------------
# Response Parsing Tests
# -------------------------------------------------------------------

class TestResponseParsing:
    def test_valid_json_parsing(self, classifier):
        raw = '{"category": "network", "priority": "high", "severity": "sev2", "confidence": 0.9, "suggested_tags": ["wifi"], "reasoning": "test", "requires_escalation": false}'
        result = classifier._parse_response(raw)
        assert result["category"] == "network"
        assert result["priority"] == "high"
        assert result["confidence"] == 0.9

    def test_json_in_markdown_code_block(self, classifier):
        raw = '```json\n{"category": "email", "priority": "medium", "severity": "sev3"}\n```'
        result = classifier._parse_response(raw)
        assert result["category"] == "email"

    def test_invalid_json_raises_error(self, classifier):
        with pytest.raises(ValueError, match="Invalid JSON"):
            classifier._parse_response("not valid json at all")

    def test_missing_required_field_raises_error(self, classifier):
        raw = '{"category": "network", "priority": "high"}'  # missing severity
        with pytest.raises(ValueError, match="Missing required field"):
            classifier._parse_response(raw)

    def test_invalid_category_raises_error(self, classifier):
        raw = '{"category": "invalid_category", "priority": "high", "severity": "sev2"}'
        with pytest.raises(ValueError, match="Invalid category"):
            classifier._parse_response(raw)


# -------------------------------------------------------------------
# AI Classification Tests (Mocked)
# -------------------------------------------------------------------

class TestAIClassification:
    @pytest.mark.asyncio
    async def test_successful_classification(self, classifier):
        mock_response = MagicMock()
        mock_response.content = '{"category": "network", "priority": "high", "severity": "sev2", "confidence": 0.92, "suggested_tags": ["wifi", "connectivity"], "reasoning": "WiFi issue affecting multiple users", "requires_escalation": false}'
        classifier.llm.ainvoke = AsyncMock(return_value=mock_response)

        result = await classifier.classify(
            title="WiFi dropping",
            description="WiFi keeps disconnecting for users in Building A",
        )

        assert result.category == TicketCategory.NETWORK
        assert result.priority == TicketPriority.HIGH
        assert result.confidence == 0.92
        assert result.assigned_team == "network-ops-team"

    @pytest.mark.asyncio
    async def test_api_failure_triggers_fallback(self, classifier):
        classifier.llm.ainvoke = AsyncMock(side_effect=Exception("API timeout"))

        result = await classifier.classify(
            title="EHR system down",
            description="Epic is not responding, patient care affected",
        )

        # Should fall back to keyword-based classification
        assert result.confidence == 0.5
        assert result.requires_escalation is True

    @pytest.mark.asyncio
    async def test_stats_tracking(self, classifier):
        mock_response = MagicMock()
        mock_response.content = '{"category": "email", "priority": "low", "severity": "sev4", "confidence": 0.85}'
        classifier.llm.ainvoke = AsyncMock(return_value=mock_response)

        await classifier.classify(title="Test", description="Test ticket")

        stats = classifier.get_stats()
        assert stats["total_classifications"] == 1
        assert stats["avg_classification_time_ms"] > 0
