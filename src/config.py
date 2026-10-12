"""Typed application settings and prompt templates loaded from config/."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yml"
DEFAULT_PROMPTS_PATH = CONFIG_DIR / "prompts.yml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AppConfig(_Frozen):
    name: str
    log_level: str = "INFO"


class LLMConfig(_Frozen):
    provider: Literal["fake", "openai"] = "fake"
    chat_model: str
    classification_model: str
    temperature: float
    classification_temperature: float
    max_tokens: int
    request_timeout: float
    embedding_model: str
    api_key: Optional[SecretStr] = None
    base_url: Optional[str] = None


class KnowledgeBaseConfig(_Frozen):
    persist_directory: str
    collection_name: str
    max_search_results: int


class RetrievalProfile(_Frozen):
    chunk_size: int
    chunk_overlap: int
    min_relevance_score: float
    confidence_threshold: float


class ChatConfig(_Frozen):
    max_message_length: int
    max_context_messages: int
    session_timeout_seconds: float
    max_concurrent_sessions: int
    max_sessions_per_client: int = Field(20, ge=1)
    max_tracked_sessions: int = Field(1000, ge=1)
    max_tracked_sessions_per_client: int = Field(50, ge=1)


class SecurityConfig(_Frozen):
    phi_detection_enabled: bool = True
    # Never read from the YAML file: only from INGEST_API_KEY_FILE.
    ingest_api_key: Optional[SecretStr] = None
    allowed_origins: tuple[str, ...] = ()
    auth_failure_limit: int = Field(10, ge=1)
    auth_failure_window_seconds: float = Field(60, gt=0)


class LimitsConfig(_Frozen):
    max_request_bytes: int = Field(1_048_576, ge=1024)
    max_concurrent_requests: int = Field(8, ge=1)
    max_concurrent_requests_per_client: int = Field(4, ge=1)


class EscalationConfig(_Frozen):
    auto_escalate_categories: tuple[str, ...] = ()


class Settings(_Frozen):
    app: AppConfig
    llm: LLMConfig
    knowledge_base: KnowledgeBaseConfig
    retrieval_profiles: dict[Literal["openai", "hashing"], RetrievalProfile]
    retrieval_profile: Optional[Literal["openai", "hashing"]] = None
    chat: ChatConfig
    security: SecurityConfig
    limits: LimitsConfig = LimitsConfig()
    escalation: EscalationConfig
    kb_seed_dir: Optional[str] = None

    @model_validator(mode="after")
    def _active_profile_exists(self) -> "Settings":
        if self.active_profile_name not in self.retrieval_profiles:
            raise ValueError(
                f"retrieval profile '{self.active_profile_name}' is not defined in retrieval_profiles"
            )
        return self

    @property
    def active_profile_name(self) -> str:
        if self.retrieval_profile:
            return self.retrieval_profile
        return "hashing" if self.llm.provider == "fake" else "openai"

    @property
    def retrieval(self) -> RetrievalProfile:
        return self.retrieval_profiles[self.active_profile_name]


def _read_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping at the top level")
    return data


def _read_key_file(var: str, path: str) -> str:
    """Return the stripped contents of a key file named by ``var``."""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"{var} names a file that cannot be read ({exc.strerror})") from None
    if not value:
        raise ValueError(f"{var} names an empty file")
    return value


def _apply_env(data: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    if "ingest_api_key" in (data.get("security") or {}):
        raise ValueError("security.ingest_api_key is read only from INGEST_API_KEY_FILE")
    overrides = {
        "LLM_PROVIDER": ("llm", "provider"),
        "OPENAI_API_KEY": ("llm", "api_key"),
        "OPENAI_BASE_URL": ("llm", "base_url"),
        "CHROMA_PERSIST_DIR": ("knowledge_base", "persist_directory"),
        "LOG_LEVEL": ("app", "log_level"),
    }
    for var, (section, key) in overrides.items():
        value = env.get(var)
        if value:
            data.setdefault(section, {})
            data[section][key] = value
    if env.get("RETRIEVAL_PROFILE"):
        data["retrieval_profile"] = env["RETRIEVAL_PROFILE"]
    if env.get("KB_SEED_DIR"):
        data["kb_seed_dir"] = env["KB_SEED_DIR"]
    key_files = {
        "OPENAI_API_KEY_FILE": ("llm", "api_key"),
        "INGEST_API_KEY_FILE": ("security", "ingest_api_key"),
    }
    for var, (section, key) in key_files.items():
        if env.get(var):
            data.setdefault(section, {})
            data[section][key] = _read_key_file(var, env[var])
    return data


def load_settings(
    path: Optional[Path | str] = None, env: Optional[Mapping[str, str]] = None
) -> Settings:
    """Load settings from YAML (APP_CONFIG or config/settings.yml) and apply env overrides."""
    env = os.environ if env is None else env
    config_path = Path(path or env.get("APP_CONFIG") or DEFAULT_SETTINGS_PATH)
    data = _apply_env(_read_yaml(config_path), env)
    return Settings.model_validate(data)


def load_prompts(path: Optional[Path | str] = None) -> dict[str, Any]:
    """Load prompt templates from config/prompts.yml."""
    return _read_yaml(Path(path or DEFAULT_PROMPTS_PATH))
