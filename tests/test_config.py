import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Settings, load_prompts, load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = REPO_ROOT / "config" / "settings.yml"

ENV_VARS = (
    "APP_CONFIG",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CHROMA_PERSIST_DIR",
    "RETRIEVAL_PROFILE",
    "KB_SEED_DIR",
    "LOG_LEVEL",
    "OPENAI_API_KEY_FILE",
    "INGEST_API_KEY_FILE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


class TestLoadSettings:
    def test_defaults_load_from_repo_config(self):
        settings = load_settings()
        assert isinstance(settings, Settings)
        assert settings.app.name == "AI IT Support"
        assert settings.app.log_level == "INFO"
        assert settings.llm.provider == "fake"
        assert settings.llm.chat_model == "gpt-4-0125-preview"
        assert settings.llm.classification_model == "gpt-4-0125-preview"
        assert settings.llm.temperature == 0.3
        assert settings.llm.classification_temperature == 0.1
        assert settings.llm.max_tokens == 1024
        assert settings.llm.request_timeout == 30
        assert settings.llm.embedding_model == "text-embedding-3-small"
        assert settings.llm.api_key is None
        assert settings.llm.base_url is None
        assert settings.knowledge_base.persist_directory == "./data/chroma"
        assert settings.knowledge_base.collection_name == "it_support_kb"
        assert settings.knowledge_base.max_search_results == 5
        assert settings.chat.max_message_length == 4096
        assert settings.chat.max_context_messages == 10
        assert settings.chat.session_timeout_seconds == 1800
        assert settings.chat.max_concurrent_sessions == 200
        assert settings.security.phi_detection_enabled is True
        assert list(settings.escalation.auto_escalate_categories) == ["security", "ehr_systems"]
        assert settings.kb_seed_dir is None

    def test_profiles_match_spec(self):
        settings = load_settings()
        openai = settings.retrieval_profiles["openai"]
        hashing = settings.retrieval_profiles["hashing"]
        assert (openai.chunk_size, openai.chunk_overlap) == (1000, 200)
        assert (openai.min_relevance_score, openai.confidence_threshold) == (0.3, 0.4)
        assert (hashing.chunk_size, hashing.chunk_overlap) == (400, 80)
        assert (hashing.min_relevance_score, hashing.confidence_threshold) == (0.15, 0.2)

    def test_settings_are_frozen(self):
        settings = load_settings()
        with pytest.raises(ValidationError):
            settings.chat = settings.chat
        with pytest.raises(ValidationError):
            settings.chat.max_message_length = 1

    def test_model_copy_updates_nested_value(self):
        settings = load_settings()
        changed = settings.model_copy(
            update={"chat": settings.chat.model_copy(update={"max_concurrent_sessions": 1})}
        )
        assert changed.chat.max_concurrent_sessions == 1
        assert settings.chat.max_concurrent_sessions == 200

    def test_env_overrides_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://stub:8001/v1")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("RETRIEVAL_PROFILE", "hashing")
        monkeypatch.setenv("KB_SEED_DIR", str(tmp_path / "kb"))
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = load_settings()

        assert settings.llm.provider == "openai"
        assert settings.llm.api_key.get_secret_value() == "sk-test"
        assert settings.llm.base_url == "http://stub:8001/v1"
        assert settings.knowledge_base.persist_directory == str(tmp_path / "chroma")
        assert settings.retrieval_profile == "hashing"
        assert settings.kb_seed_dir == str(tmp_path / "kb")
        assert settings.app.log_level == "DEBUG"

    def test_retrieval_is_hashing_for_fake_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "fake")
        settings = load_settings()
        assert settings.retrieval == settings.retrieval_profiles["hashing"]
        assert settings.retrieval.chunk_size == 400

    def test_retrieval_is_openai_for_openai_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        settings = load_settings()
        assert settings.retrieval == settings.retrieval_profiles["openai"]
        assert settings.retrieval.min_relevance_score == 0.3

    def test_retrieval_profile_env_overrides_provider_default(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("RETRIEVAL_PROFILE", "hashing")
        assert load_settings().retrieval == load_settings().retrieval_profiles["hashing"]

        monkeypatch.setenv("LLM_PROVIDER", "fake")
        monkeypatch.setenv("RETRIEVAL_PROFILE", "openai")
        assert load_settings().retrieval.confidence_threshold == 0.4

    def test_app_config_points_at_other_file(self, monkeypatch, tmp_path):
        text = SETTINGS_FILE.read_text().replace("it_support_kb", "other_kb")
        alt = tmp_path / "alt.yml"
        alt.write_text(text)
        monkeypatch.setenv("APP_CONFIG", str(alt))
        assert load_settings().knowledge_base.collection_name == "other_kb"

    def test_missing_config_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("APP_CONFIG", str(tmp_path / "missing.yml"))
        with pytest.raises(FileNotFoundError):
            load_settings()

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "no-such-provider")
        with pytest.raises(ValidationError):
            load_settings()

    def test_unknown_profile_raises(self, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_PROFILE", "semantic")
        with pytest.raises(ValidationError):
            load_settings()

    def test_unknown_key_in_file_raises(self, monkeypatch, tmp_path):
        alt = tmp_path / "alt.yml"
        alt.write_text(SETTINGS_FILE.read_text() + "\nredis:\n  url: redis://localhost:6379\n")
        monkeypatch.setenv("APP_CONFIG", str(alt))
        with pytest.raises(ValidationError):
            load_settings()


class TestSettingsFileContent:
    def test_removed_blocks_are_gone(self):
        text = SETTINGS_FILE.read_text().lower()
        for word in ("redis", "prometheus", "servicenow", "jira", "slack", "jwt", "allowed_hosts"):
            assert word not in text, word


class TestLoadPrompts:
    def test_role_prompts_present(self):
        prompts = load_prompts()
        for role in ("default", "clinical_staff", "technical_staff", "security_incident"):
            assert prompts["system_prompts"][role].strip(), role

    def test_chat_user_template_placeholders(self):
        template = load_prompts()["chat"]["user_template"]
        for placeholder in ("{context}", "{history}", "{question}"):
            assert placeholder in template

    def test_classification_user_template(self):
        template = load_prompts()["classification"]["user_template"]
        for placeholder in ("{title}", "{description}", "{additional_context}"):
            assert placeholder in template
        assert "### TICKET_CLASSIFICATION_JSON ###" in template

        rendered = template.format(title="t", description="d", additional_context="")
        assert "### TICKET_CLASSIFICATION_JSON ###" in rendered
        for key in (
            '"category"',
            '"priority"',
            '"severity"',
            '"confidence"',
            '"suggested_tags"',
            '"reasoning"',
            '"requires_escalation"',
        ):
            assert key in rendered, key
        assert '{"category"' in rendered

    def test_classification_system_prompt_rules_aligned(self):
        system = load_prompts()["classification"]["system_prompt"].lower()
        assert "critical" in system
        assert "security incidents are always high" in system
        assert "multiple departments" not in system
        assert "after-hours" not in system

    def test_unused_prompts_removed(self):
        prompts = load_prompts()
        assert "knowledge_search" not in prompts
        assert "ticket_summary" not in prompts["escalation"]
        assert "threshold_warning" in prompts["escalation"]
        assert "warning_message" in prompts["phi_detection"]

    def test_no_compliance_or_ticket_creation_claims(self):
        for text in _iter_strings(load_prompts()):
            assert not re.search(r"hipaa compliance", text, re.IGNORECASE), text
            assert not re.search(r"create a support ticket", text, re.IGNORECASE), text

    def test_default_prompt_and_warnings_reworded(self):
        prompts = load_prompts()
        assert "never repeat patient identifiers" in prompts["system_prompts"]["default"]
        assert prompts["phi_detection"]["warning_message"].startswith(
            "I've filtered what looked like patient identifiers from your message"
        )
        assert "service desk" in prompts["escalation"]["threshold_warning"]

    def test_missing_prompts_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompts(tmp_path / "missing.yml")
