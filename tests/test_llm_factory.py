import pytest
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import load_settings
from src.fakes import ExtractiveFakeChatModel, HashingEmbeddings
from src.llm_factory import get_chat_model, get_embeddings

ENV_VARS = (
    "APP_CONFIG",
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CHROMA_PERSIST_DIR",
    "RETRIEVAL_PROFILE",
    "KB_SEED_DIR",
    "LOG_LEVEL",
)

BASE_URL = "http://openai-stub:8080/v1"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_settings():
    return load_settings(env={"LLM_PROVIDER": "fake"})


@pytest.fixture
def openai_settings():
    return load_settings(
        env={
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-dummy-key",
            "OPENAI_BASE_URL": BASE_URL,
        }
    )


def test_fake_provider_returns_fake_chat_model(fake_settings):
    assert isinstance(get_chat_model(fake_settings, "chat"), ExtractiveFakeChatModel)
    assert isinstance(get_chat_model(fake_settings, "classification"), ExtractiveFakeChatModel)


def test_fake_provider_returns_hashing_embeddings(fake_settings):
    assert isinstance(get_embeddings(fake_settings), HashingEmbeddings)


def test_openai_chat_model_uses_configured_model_and_base_url(openai_settings):
    model = get_chat_model(openai_settings, "chat")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == openai_settings.llm.chat_model
    assert model.temperature == openai_settings.llm.temperature
    assert model.openai_api_base == BASE_URL
    assert model.openai_api_key.get_secret_value() == "sk-dummy-key"


def test_openai_classification_model_uses_classification_settings(openai_settings):
    model = get_chat_model(openai_settings, "classification")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == openai_settings.llm.classification_model
    assert model.temperature == openai_settings.llm.classification_temperature


def test_openai_embeddings_skip_ctx_length_check(openai_settings):
    embeddings = get_embeddings(openai_settings)
    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.check_embedding_ctx_length is False
    assert embeddings.model == openai_settings.llm.embedding_model
    assert embeddings.openai_api_base == BASE_URL


def test_openai_clients_are_passed_through(openai_settings):
    import httpx

    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    try:
        model = get_chat_model(
            openai_settings, "chat", http_client=sync_client, http_async_client=async_client
        )
        embeddings = get_embeddings(
            openai_settings, http_client=sync_client, http_async_client=async_client
        )
        assert model.http_client is sync_client
        assert model.http_async_client is async_client
        assert embeddings.http_client is sync_client
        assert embeddings.http_async_client is async_client
    finally:
        sync_client.close()


def test_unknown_purpose_raises(fake_settings):
    with pytest.raises(ValueError):
        get_chat_model(fake_settings, "summary")
