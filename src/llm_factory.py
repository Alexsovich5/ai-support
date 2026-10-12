"""Build the chat model and embeddings for the configured provider."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.fakes import ExtractiveFakeChatModel, HashingEmbeddings

PURPOSES = ("chat", "classification")


def _openai_kwargs(settings: Settings, http_client: Any, http_async_client: Any) -> dict:
    kwargs: dict = {}
    if settings.llm.api_key:
        kwargs["api_key"] = settings.llm.api_key.get_secret_value()
    if settings.llm.base_url:
        kwargs["base_url"] = settings.llm.base_url
    if http_client is not None:
        kwargs["http_client"] = http_client
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    return kwargs


def get_chat_model(
    settings: Settings,
    purpose: str,
    http_client: Optional[Any] = None,
    http_async_client: Optional[Any] = None,
) -> BaseChatModel:
    """Return the chat model for ``purpose`` ("chat" or "classification")."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown model purpose '{purpose}', expected one of {PURPOSES}")
    llm = settings.llm
    if llm.provider == "fake":
        return ExtractiveFakeChatModel()

    from langchain_openai import ChatOpenAI

    if purpose == "classification":
        model, temperature = llm.classification_model, llm.classification_temperature
    else:
        model, temperature = llm.chat_model, llm.temperature
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=llm.max_tokens,
        timeout=llm.request_timeout,
        **_openai_kwargs(settings, http_client, http_async_client),
    )


def get_embeddings(
    settings: Settings,
    http_client: Optional[Any] = None,
    http_async_client: Optional[Any] = None,
) -> Embeddings:
    """Return the embeddings for the configured provider."""
    if settings.llm.provider == "fake":
        return HashingEmbeddings()

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.llm.embedding_model,
        check_embedding_ctx_length=False,
        timeout=settings.llm.request_timeout,
        **_openai_kwargs(settings, http_client, http_async_client),
    )
