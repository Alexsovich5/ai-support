import uuid
from pathlib import Path

import chromadb
import pytest
import pytest_asyncio
from chromadb.config import Settings as ChromaSettings

from src.ai_engine import AIEngine, EngineResponse
from src.config import load_prompts, load_settings
from src.fakes import HashingEmbeddings
from src.ingest import ingest_directory
from src.knowledge_base import KnowledgeBaseManager
from tests.helpers import RecordingFakeChatModel

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"
CANNED = "Here is what to try."


@pytest.fixture
def settings():
    return load_settings(env={"LLM_PROVIDER": "fake"})


@pytest.fixture
def prompts():
    return load_prompts()


@pytest_asyncio.fixture
async def kb(tmp_path, settings):
    profile = settings.retrieval
    manager = KnowledgeBaseManager(
        embeddings=HashingEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name=f"kb_{uuid.uuid4().hex}",
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
        client=chromadb.PersistentClient(
            path=str(tmp_path / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        ),
    )
    await manager.initialize()
    summary = await ingest_directory(manager, SEED_DIR)
    assert summary["errors"] == [] and summary["total_files"] == 6
    return manager


@pytest.fixture
def model():
    return RecordingFakeChatModel(responses=[CANNED])


@pytest.fixture
def engine(settings, prompts, kb, model):
    return AIEngine(settings, prompts, kb, model)


def _system_text(model):
    return str(model.calls[-1][0].content)


@pytest.mark.asyncio
async def test_vpn_question_cites_vpn_article_with_top_relevance(engine, kb, settings, model):
    question = "My VPN connection keeps dropping"
    response = await engine.process_message(question, "s1")

    assert isinstance(response, EngineResponse)
    assert response.message == CANNED
    assert response.source_documents[0] == "vpn-disconnects.md"

    expected = await kb.search(
        question,
        n_results=settings.knowledge_base.max_search_results,
        min_relevance=settings.retrieval.min_relevance_score,
    )
    assert response.confidence == expected[0]["relevance_score"]
    assert response.confidence >= settings.retrieval.confidence_threshold
    assert response.escalation_recommended is False
    assert response.phi_detected is False
    assert response.response_time_ms >= 0
    # The retrieved article text reaches the model as context.
    assert "power saving on the network adapter" in model.last_prompt_text()


@pytest.mark.asyncio
async def test_unrelated_question_has_zero_confidence_and_escalates(engine):
    response = await engine.process_message("how do I bake sourdough bread", "s1")
    assert response.confidence == 0
    assert response.source_documents == []
    assert response.escalation_recommended is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role, key",
    [
        ("physician", "clinical_staff"),
        ("Nurse", "clinical_staff"),
        ("it", "technical_staff"),
        ("ENGINEER", "technical_staff"),
    ],
)
async def test_role_selects_matching_system_prompt(engine, prompts, model, role, key):
    await engine.process_message("printer shows offline", "s1", {"role": role})
    system = _system_text(model)
    assert prompts["system_prompts"][key].strip() in system
    others = {"clinical_staff", "technical_staff", "security_incident"} - {key}
    for other in others:
        assert prompts["system_prompts"][other].strip() not in system


@pytest.mark.asyncio
async def test_security_content_overrides_role(engine, prompts, model):
    await engine.process_message(
        "I clicked a link in a phishing email", "s1", {"role": "nurse"}
    )
    system = _system_text(model)
    assert prompts["system_prompts"]["security_incident"].strip() in system
    assert prompts["system_prompts"]["clinical_staff"].strip() not in system


@pytest.mark.asyncio
async def test_security_keywords_match_inflections(engine, prompts, model):
    await engine.process_message("I think my laptop was hacked by viruses", "s1")
    assert prompts["system_prompts"]["security_incident"].strip() in _system_text(model)


@pytest.mark.parametrize(
    "message, role, key",
    [
        ("printer is offline", None, "default"),
        ("printer is offline", "staff", "default"),
        ("printer is offline", "Clinical", "clinical_staff"),
        ("printer is offline", "admin", "technical_staff"),
        ("received suspicious mail", "it", "security_incident"),
        ("the virus scanner update", None, "security_incident"),
        ("viral video will not play", None, "default"),
    ],
)
def test_select_prompt_key(message, role, key):
    assert AIEngine.select_prompt_key(message, role) == key


@pytest.mark.asyncio
async def test_unknown_role_uses_default_prompt(engine, prompts, model):
    await engine.process_message("printer shows offline", "s1", {"role": "pharmacist"})
    system = _system_text(model)
    assert prompts["system_prompts"]["default"].strip() in system
    for other in ("clinical_staff", "technical_staff", "security_incident"):
        assert prompts["system_prompts"][other].strip() not in system


@pytest.mark.asyncio
async def test_history_is_capped_and_sent_to_model(settings, prompts, kb):
    capped = settings.model_copy(
        update={"chat": settings.chat.model_copy(update={"max_context_messages": 4})}
    )
    model = RecordingFakeChatModel(responses=[CANNED])
    engine = AIEngine(capped, prompts, kb, model)

    for i in range(5):
        await engine.process_message(f"question number q{i}x", "capped")

    history = engine.get_history("capped")
    assert len(history) == 4
    # Each turn adds the user message and the answer; only the latest 4 remain.
    assert history[0] == ("user", "question number q3x")
    assert history[-1] == ("assistant", CANNED)

    # The last prompt carried history from before the final turn, already capped.
    last = model.last_prompt_text()
    assert "q2x" in last and "q3x" in last
    assert "q0x" not in last and "q1x" not in last
    # Other sessions are independent.
    assert engine.get_history("other") == []


@pytest.mark.asyncio
async def test_phi_is_redacted_before_reaching_model(engine, model):
    response = await engine.process_message(
        "Printer offline while printing labels for MRN 12345678, call me at 555-123-4567",
        "s1",
    )
    prompt = model.last_prompt_text()
    assert "12345678" not in prompt
    assert "555-123-4567" not in prompt
    assert "[REDACTED:mrn]" in prompt
    assert response.phi_detected is True
    assert all("12345678" not in text for _, text in engine.get_history("s1"))


@pytest.mark.asyncio
async def test_stats_count_messages(engine):
    assert engine.get_stats()["total_messages"] == 0
    await engine.process_message("My VPN connection keeps dropping", "a")
    await engine.process_message("how do I bake sourdough bread", "b")
    await engine.process_message("MRN 12345678 printer offline", "b")
    stats = engine.get_stats()
    assert stats["total_messages"] == 3
    assert stats["active_sessions"] == 2
    assert stats["escalations_recommended"] >= 1
    assert stats["phi_detections"] == 1
    assert stats["average_response_time_ms"] >= 0
