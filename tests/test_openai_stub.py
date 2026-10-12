"""The real langchain-openai clients against the in-process OpenAI-compatible stub.

Sync calls go through ``TestClient(stub_app)`` and async calls through
``httpx.ASGITransport``, so no socket is opened. ``make test`` runs with
``--network none``, so any attempt to download a tiktoken encoding fails here.
"""

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import load_prompts, load_settings
from src.fakes import NO_CONTEXT_ANSWER, HashingEmbeddings
from src.llm_factory import get_chat_model, get_embeddings
from src.ticket_classifier import TicketCategory, TicketClassifier, TicketPriority
from stubs.openai_stub import app as stub_app

TEXTS = [
    "VPN tunnel keeps dropping every few minutes",
    "Printer on floor two shows offline",
]

CONTEXT_PROMPT = (
    "Knowledge base context:\n"
    "[vpn-disconnects.md]\n"
    "1. Reconnect to the corporate network.\n"
    "2. Update the VPN client.\n"
    "Question: my VPN drops"
)


@pytest.fixture
def stub_settings():
    return load_settings(
        env={
            "LLM_PROVIDER": "openai",
            "OPENAI_BASE_URL": "http://stub/v1",
            "OPENAI_API_KEY": "sk-test",
        }
    )


@pytest.fixture
def sync_client():
    with TestClient(stub_app, base_url="http://stub") as client:
        yield client


@pytest.fixture
def async_client():
    # ASGITransport holds no sockets, so the client needs no explicit close.
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=stub_app), base_url="http://stub"
    )


@pytest.fixture
def chat_model(stub_settings, sync_client, async_client):
    model = get_chat_model(
        stub_settings, "chat", http_client=sync_client, http_async_client=async_client
    )
    assert isinstance(model, ChatOpenAI)
    return model


@pytest.fixture
def classification_model(stub_settings, sync_client, async_client):
    return get_chat_model(
        stub_settings,
        "classification",
        http_client=sync_client,
        http_async_client=async_client,
    )


@pytest.fixture
def embeddings(stub_settings, sync_client, async_client):
    model = get_embeddings(
        stub_settings, http_client=sync_client, http_async_client=async_client
    )
    assert isinstance(model, OpenAIEmbeddings)
    return model


def _as_float32(vectors):
    return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]


class TestChatCompletions:
    def test_invoke_returns_extractive_answer(self, chat_model):
        reply = chat_model.invoke([HumanMessage(content=CONTEXT_PROMPT)])
        assert reply.content.startswith("Suggested steps from the knowledge base:")
        assert "1. Reconnect to the corporate network." in reply.content
        assert "2. Update the VPN client." in reply.content

    @pytest.mark.asyncio
    async def test_ainvoke_returns_extractive_answer(self, chat_model):
        reply = await chat_model.ainvoke(
            [SystemMessage(content="You help with IT issues."), HumanMessage(content=CONTEXT_PROMPT)]
        )
        assert "Update the VPN client." in reply.content

    def test_prompt_without_context_gets_no_context_answer(self, chat_model):
        reply = chat_model.invoke([HumanMessage(content="hello there")])
        assert reply.content == NO_CONTEXT_ANSWER

    def test_streaming_request_is_rejected(self, sync_client):
        response = sync_client.post(
            "/v1/chat/completions",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 400
        assert "error" in response.json()

    def test_response_follows_chat_completion_shape(self, sync_client):
        response = sync_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "gpt-test"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["finish_reason"] == "stop"


class TestEmbeddings:
    def test_embed_documents_match_hashing_embeddings(self, embeddings):
        vectors = embeddings.embed_documents(TEXTS)
        assert [len(v) for v in vectors] == [512, 512]
        assert _as_float32(vectors) == _as_float32(HashingEmbeddings().embed_documents(TEXTS))

    @pytest.mark.asyncio
    async def test_aembed_documents_match_hashing_embeddings(self, embeddings):
        vectors = await embeddings.aembed_documents(TEXTS)
        assert [len(v) for v in vectors] == [512, 512]
        assert _as_float32(vectors) == _as_float32(HashingEmbeddings().embed_documents(TEXTS))

    def test_embed_query_matches_hashing_embeddings(self, embeddings):
        vector = embeddings.embed_query(TEXTS[0])
        assert _as_float32([vector]) == _as_float32([HashingEmbeddings().embed_query(TEXTS[0])])

    def test_single_string_input_with_float_format(self, sync_client):
        response = sync_client.post(
            "/v1/embeddings",
            json={"model": "m", "input": TEXTS[0], "encoding_format": "float"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert len(data) == 1
        assert data[0]["embedding"] == HashingEmbeddings().embed_query(TEXTS[0])

    @pytest.mark.parametrize("token_input", [[15339, 1917], [[15339, 1917], [2028]]])
    def test_token_id_arrays_are_rejected(self, sync_client, token_input):
        response = sync_client.post(
            "/v1/embeddings", json={"model": "m", "input": token_input}
        )
        assert response.status_code == 400
        assert "check_embedding_ctx_length" in response.json()["error"]["message"]


def test_models_lists_configured_models(sync_client, stub_settings):
    response = sync_client.get("/v1/models")
    assert response.status_code == 200
    ids = {model["id"] for model in response.json()["data"]}
    assert stub_settings.llm.chat_model in ids
    assert stub_settings.llm.embedding_model in ids


class TestClassificationThroughStub:
    @pytest.mark.asyncio
    async def test_classifier_gets_parsed_non_fallback_result(
        self, classification_model, stub_settings
    ):
        classifier = TicketClassifier(
            llm=classification_model, settings=stub_settings, prompts=load_prompts()
        )
        result = await classifier.classify(
            title="Printer jam",
            description="The printer on floor two keeps jamming on every job",
        )
        assert result.confidence != 0.5
        assert "fallback-classification" not in result.suggested_tags
        assert result.category == TicketCategory.PRINTING
        assert result.priority == TicketPriority.MEDIUM

    @pytest.mark.asyncio
    async def test_classification_uses_ticket_text_not_template(
        self, classification_model, stub_settings
    ):
        # The output contract in the template lists every category name; only
        # the ticket fields may decide the category.
        classifier = TicketClassifier(
            llm=classification_model, settings=stub_settings, prompts=load_prompts()
        )
        result = await classifier.classify(
            title="Phishing message received",
            description="Got a suspicious email asking for my password",
        )
        assert result.confidence != 0.5
        assert result.category == TicketCategory.SECURITY
        assert result.priority == TicketPriority.HIGH
        assert result.requires_escalation is True


def test_embedding_batches_above_the_cap_are_rejected():
    with TestClient(stub_app, base_url="http://stub") as client:
        response = client.post(
            "/v1/embeddings", json={"model": "m", "input": ["vpn"] * 2049}
        )
    assert response.status_code == 400
