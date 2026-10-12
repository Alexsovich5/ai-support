"""
Ticket classifier unit tests.

Unit tests for the ticket classification system including
AI-based and fallback classification paths.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import load_settings
from src.fakes import ExtractiveFakeChatModel
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
from tests.helpers import RecordingFakeChatModel


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

    def test_phishing_email_is_security(self, classifier):
        result = classifier._fallback_classification(
            title="Phishing email received",
            description="An email asked me to confirm my mailbox password on an outside site.",
            start_time=0,
        )
        assert result.category == TicketCategory.SECURITY
        assert result.assigned_team == "security-ops-team"

    def test_security_fallback_priority_is_high(self, classifier, sample_security_ticket):
        result = classifier._fallback_classification(
            title=sample_security_ticket["title"],
            description=sample_security_ticket["description"],
            start_time=0,
        )
        assert result.category == TicketCategory.SECURITY
        assert result.priority == TicketPriority.HIGH
        assert result.severity == TicketSeverity.SEV2

    def test_security_with_critical_keyword_is_critical(self, classifier):
        result = classifier._fallback_classification(
            title="Malware outbreak",
            description="Ransom note on several workstations, file shares are down.",
            start_time=0,
        )
        assert result.category == TicketCategory.SECURITY
        assert result.priority == TicketPriority.CRITICAL
        assert result.severity == TicketSeverity.SEV1

    def test_download_does_not_trigger_critical(self, classifier):
        result = classifier._fallback_classification(
            title="Cannot download installer",
            description="The download of the PDF reader stops halfway.",
            start_time=0,
        )
        assert result.priority == TicketPriority.MEDIUM
        assert result.severity == TicketSeverity.SEV3

    def test_down_still_triggers_critical(self, classifier):
        result = classifier._fallback_classification(
            title="Shared drive down",
            description="Nobody can open the shared drive.",
            start_time=0,
        )
        assert result.priority == TicketPriority.CRITICAL

    def test_printing_matches_print(self, classifier):
        result = classifier._fallback_classification(
            title="Printing fails on 3rd floor",
            description="Jobs sit in the queue.",
            start_time=0,
        )
        assert result.category == TicketCategory.PRINTING

    def test_phones_match_telephony(self, classifier):
        result = classifier._fallback_classification(
            title="phones not ringing",
            description="Incoming calls go straight to voicemail.",
            start_time=0,
        )
        assert result.category == TicketCategory.TELEPHONY

    def test_keyword_inside_other_word_does_not_match(self, classifier):
        result = classifier._fallback_classification(
            title="Spreadsheet crashes on the sprint workbook",
            description="The dropdown in the expense sheet is empty.",
            start_time=0,
        )
        # "print" inside "sprint" and "down" inside "dropdown" must not match
        assert result.category == TicketCategory.SOFTWARE
        assert result.priority == TicketPriority.MEDIUM

    def test_fallback_time_is_small_when_start_is_zero(self, classifier):
        result = classifier._fallback_classification(
            title="Generic issue",
            description="Something is not working properly",
            start_time=0,
        )
        assert 0 <= result.classification_time_ms < 1000

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
    def test_invalid_severity_raises_error(self, classifier):
        raw = '{"category": "network", "priority": "high", "severity": "sev9"}'
        with pytest.raises(ValueError, match="Invalid severity"):
            classifier._parse_response(raw)

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


# -------------------------------------------------------------------
# Injected model, prompt contract, PHI redaction and auto-escalation
# -------------------------------------------------------------------

CONTRACT_KEYS = (
    "category",
    "priority",
    "severity",
    "confidence",
    "suggested_tags",
    "reasoning",
    "requires_escalation",
)


def _reply(**overrides) -> str:
    data = {
        "category": "network",
        "priority": "medium",
        "severity": "sev3",
        "confidence": 0.9,
        "suggested_tags": ["wifi"],
        "reasoning": "connectivity",
        "requires_escalation": False,
    }
    data.update(overrides)
    return json.dumps(data)


@pytest.fixture
def fake_settings():
    return load_settings(env={"LLM_PROVIDER": "fake"})


class TestInjectedModel:
    @pytest.mark.asyncio
    async def test_rendered_prompt_contains_contract_and_marker(self, fake_settings):
        model = RecordingFakeChatModel(responses=[_reply()])
        clf = TicketClassifier(llm=model, settings=fake_settings)

        await clf.classify(title="WiFi drops", description="Floor 3 loses WiFi")

        prompt = model.last_prompt_text()
        assert "### TICKET_CLASSIFICATION_JSON ###" in prompt
        for key in CONTRACT_KEYS:
            assert f'"{key}"' in prompt, key
        assert "WiFi drops" in prompt and "Floor 3 loses WiFi" in prompt
        assert "ticket classifier" in prompt  # system prompt from prompts.yml

    def test_in_code_prompt_constant_removed(self):
        assert not hasattr(TicketClassifier, "CLASSIFICATION_PROMPT")

    @pytest.mark.asyncio
    async def test_phi_is_redacted_before_the_model(self, fake_settings):
        model = RecordingFakeChatModel(responses=[_reply()])
        clf = TicketClassifier(llm=model, settings=fake_settings)

        await clf.classify(
            title="Printer jam for MRN 12345678",
            description="Cannot print labels for MRN 12345678, call 555-123-4567",
            reporter_department="Ward MRN 87654321",
        )

        prompt = model.last_prompt_text()
        assert "12345678" not in prompt
        assert "87654321" not in prompt
        assert "555-123-4567" not in prompt
        assert "[REDACTED:mrn]" in prompt
        assert "[REDACTED:phone]" in prompt

    @pytest.mark.asyncio
    async def test_auto_escalate_category_forces_escalation(self, fake_settings):
        assert "ehr_systems" in fake_settings.escalation.auto_escalate_categories
        model = FakeListChatModel(
            responses=[_reply(category="ehr_systems", priority="critical", severity="sev1",
                              requires_escalation=False)]
        )
        clf = TicketClassifier(llm=model, settings=fake_settings)

        result = await clf.classify(title="Epic slow", description="Charts load slowly")

        assert result.category == TicketCategory.EHR_SYSTEMS
        assert result.confidence == 0.9
        assert result.requires_escalation is True

    @pytest.mark.asyncio
    async def test_other_category_keeps_model_escalation_flag(self, fake_settings):
        model = FakeListChatModel(responses=[_reply(requires_escalation=False)])
        clf = TicketClassifier(llm=model, settings=fake_settings)

        result = await clf.classify(title="WiFi", description="Slow WiFi")

        assert result.category == TicketCategory.NETWORK
        assert result.requires_escalation is False

    @pytest.mark.asyncio
    async def test_fenced_json_reply_parses(self, fake_settings):
        fenced = "```json\n" + _reply(category="email", priority="low", severity="sev4",
                                      confidence=0.77) + "\n```"
        clf = TicketClassifier(llm=FakeListChatModel(responses=[fenced]), settings=fake_settings)

        result = await clf.classify(title="Outlook", description="Signature missing")

        assert result.category == TicketCategory.EMAIL
        assert result.priority == TicketPriority.LOW
        assert result.confidence == 0.77

    @pytest.mark.asyncio
    async def test_extractive_fake_model_falls_back(self, fake_settings):
        clf = TicketClassifier(llm=ExtractiveFakeChatModel(), settings=fake_settings)

        result = await clf.classify(title="Printer offline", description="Jobs stuck")

        assert result.confidence == 0.5
        assert result.category == TicketCategory.PRINTING
        assert "fallback-classification" in result.suggested_tags
        assert result.requires_escalation is True

    @pytest.mark.asyncio
    async def test_batch_classify_preserves_order(self, fake_settings):
        replies = [
            _reply(category="vpn"),
            _reply(category="printing"),
            _reply(category="telephony"),
        ]
        clf = TicketClassifier(llm=FakeListChatModel(responses=replies), settings=fake_settings)

        results = await clf.batch_classify(
            [{"title": f"t{i}", "description": f"d{i}"} for i in range(3)]
        )

        assert [r.category for r in results] == [
            TicketCategory.VPN,
            TicketCategory.PRINTING,
            TicketCategory.TELEPHONY,
        ]

    def test_injected_model_does_not_build_openai_client(self, fake_settings):
        model = FakeListChatModel(responses=[_reply()])
        with patch("src.ticket_classifier.ChatOpenAI") as mock_openai:
            clf = TicketClassifier(llm=model, settings=fake_settings)
        mock_openai.assert_not_called()
        assert clf.llm is model

    def test_existing_fixture_still_constructs(self, classifier):
        assert isinstance(classifier, TicketClassifier)
        assert classifier.llm is not None

    def test_default_construction_uses_keyword_arguments(self):
        with patch("src.ticket_classifier.ChatOpenAI") as mock_openai:
            TicketClassifier(model_name="gpt-4-test", temperature=0.2)
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["model"] == "gpt-4-test"
        assert kwargs["temperature"] == 0.2
