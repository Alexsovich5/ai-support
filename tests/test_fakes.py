import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.fakes import (
    EMBEDDING_DIM,
    NO_CONTEXT_ANSWER,
    ExtractiveFakeChatModel,
    HashingEmbeddings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SAMPLE = "VPN keeps disconnecting every few minutes when working from home"


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def _vector_in_subprocess(text, seed):
    code = (
        "import json, sys\n"
        "from src.fakes import HashingEmbeddings\n"
        "print(json.dumps(HashingEmbeddings().embed_query(sys.argv[1])))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run(
        [sys.executable, "-c", code, text],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def test_embeddings_are_deterministic_across_instances():
    first = HashingEmbeddings().embed_query(SAMPLE)
    second = HashingEmbeddings().embed_query(SAMPLE)
    assert first == second
    assert len(first) == EMBEDDING_DIM == 512


def test_embeddings_identical_across_hash_seeds():
    in_process = HashingEmbeddings().embed_query(SAMPLE)
    seed_one = _vector_in_subprocess(SAMPLE, 1)
    seed_two = _vector_in_subprocess(SAMPLE, 2)
    assert seed_one == seed_two
    assert seed_one == pytest.approx(in_process, abs=1e-12)


def test_embedding_norm_is_one():
    vector = HashingEmbeddings().embed_query(SAMPLE)
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0)


def test_embed_documents_matches_embed_query():
    emb = HashingEmbeddings()
    docs = emb.embed_documents([SAMPLE, "printer jam on the third floor"])
    assert docs[0] == emb.embed_query(SAMPLE)
    assert len(docs) == 2


def test_stop_words_only_text_gives_zero_vector():
    vector = HashingEmbeddings().embed_query("the and of to")
    assert all(v == 0.0 for v in vector)


def test_overlapping_text_is_more_similar_than_disjoint():
    emb = HashingEmbeddings()
    query = emb.embed_query("vpn keeps disconnecting")
    overlapping = emb.embed_query(
        "If the VPN client keeps disconnecting, reinstall the VPN profile and reconnect."
    )
    disjoint = emb.embed_query("How do I bake sourdough bread at home")
    assert _cosine(query, overlapping) > _cosine(query, disjoint)
    assert _cosine(query, overlapping) > 0.2


PROMPT_WITH_CONTEXT = """Use the knowledge base context below to answer the question.

Knowledge base context:
[VPN Troubleshooting]
Open the VPN client and sign out.
Restart the laptop and reconnect to the VPN.

Conversation so far:
(none)

Question:
my vpn drops
"""

PROMPT_WITHOUT_CONTEXT = """Use the knowledge base context below to answer the question.

Knowledge base context:


Conversation so far:
(none)

Question:
how do I bake bread
"""


@pytest.mark.asyncio
async def test_fake_model_ainvoke_returns_context_text():
    model = ExtractiveFakeChatModel()
    result = await model.ainvoke(
        [SystemMessage(content="You are helpful."), HumanMessage(content=PROMPT_WITH_CONTEXT)]
    )
    assert result.content.startswith("Suggested steps")
    assert "Restart the laptop and reconnect to the VPN." in result.content
    assert "Open the VPN client and sign out." in result.content
    assert "my vpn drops" not in result.content


@pytest.mark.asyncio
async def test_fake_model_without_context_says_no_article():
    model = ExtractiveFakeChatModel()
    result = await model.ainvoke([HumanMessage(content=PROMPT_WITHOUT_CONTEXT)])
    assert result.content == NO_CONTEXT_ANSWER
    assert "I could not find a relevant article" in result.content


def test_fake_model_is_deterministic():
    model = ExtractiveFakeChatModel()
    assert model.invoke(PROMPT_WITH_CONTEXT).content == model.invoke(PROMPT_WITH_CONTEXT).content


def test_fake_model_without_context_block_says_no_article():
    model = ExtractiveFakeChatModel()
    assert model.invoke("Classify this ticket: printer broken").content == NO_CONTEXT_ANSWER
