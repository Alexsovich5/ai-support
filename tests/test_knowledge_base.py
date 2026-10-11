import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from src.config import load_settings
from src.fakes import HashingEmbeddings
from src.knowledge_base import Document, KnowledgeBaseManager

ARTICLES = {
    "vpn-disconnects.md": (
        "vpn",
        "VPN keeps disconnecting\n\n"
        "If the VPN client disconnects every few minutes, the tunnel is usually "
        "dropped by an unstable home connection or by power saving on the network "
        "adapter.\n\n"
        "1. Connect the laptop to the router with a cable, or move closer to the "
        "access point, and check whether the VPN still disconnects.\n"
        "2. Open the VPN client, sign out, quit it completely and reconnect.\n"
        "3. In Device Manager, open the network adapter properties and turn off "
        "\"Allow the computer to turn off this device to save power\".\n"
        "4. Update the VPN client to the version published on the software portal.\n"
        "5. If the VPN connection keeps dropping, note the time of each disconnect "
        "and send the VPN client log to the service desk.",
    ),
    "account-lockout.md": (
        "active_directory",
        "Account locked out\n\n"
        "An account locks after several failed sign-in attempts. Old passwords "
        "saved on a phone or a mapped drive often cause repeated lockouts.\n\n"
        "1. Wait fifteen minutes for the lockout to expire, or ask the service desk "
        "to unlock the account.\n"
        "2. Remove saved credentials for the domain from Credential Manager.\n"
        "3. Update the password on every phone and tablet that reads work mail.\n"
        "4. Sign out of any shared workstation where you are still signed in.\n"
        "5. If the account locks again within the hour, report it so the source "
        "of the failed attempts can be traced.",
    ),
    "printer-offline.md": (
        "printing",
        "Printer shows offline\n\n"
        "A printer listed as offline cannot receive print jobs from the workstation.\n\n"
        "1. Check that the printer is powered on and shows no error on its panel.\n"
        "2. Open the print queue, cancel stuck jobs and clear \"Use printer offline\".\n"
        "3. Remove the printer and add it again from the print server list.\n"
        "4. Restart the Print Spooler service.\n"
        "5. If other people on the floor cannot print either, report the printer "
        "name and location.",
    ),
    "outlook-mobile-auth.md": (
        "email",
        "Outlook on a phone keeps asking for a password\n\n"
        "After a password change, the mail app on a phone prompts for credentials "
        "again and again until the account is added again.\n\n"
        "1. Remove the work account from the Outlook mobile app.\n"
        "2. Install the authenticator app and complete multi-factor registration.\n"
        "3. Add the account again and approve the sign-in request.\n"
        "4. Check that the phone date and time are set automatically.\n"
        "5. If mail still does not sync, send the app diagnostics ID to the "
        "service desk.",
    ),
    "wifi-drops.md": (
        "network",
        "Wireless network drops\n\n"
        "Laptops that lose the wireless network in some rooms are often roaming "
        "between access points or using an old driver.\n\n"
        "1. Forget the wireless network and join it again.\n"
        "2. Update the wireless adapter driver from the software portal.\n"
        "3. Note the room and time whenever the signal is lost.\n"
        "4. Try a wired port to rule out a fault in the laptop.\n"
        "5. Report rooms with weak signal so the access points can be checked.",
    ),
    "report-phishing.md": (
        "security",
        "Reporting a phishing message\n\n"
        "A phishing message tries to trick you into opening an attachment, "
        "following a link or giving away a password.\n\n"
        "1. Do not click links or open attachments in the suspicious message.\n"
        "2. Use the Report Phishing button in the mail client.\n"
        "3. If you entered your password on a linked page, change it at once.\n"
        "4. Tell the service desk if you opened an attachment.\n"
        "5. Delete the message after reporting it.",
    ),
}

LONG_TEXT = " ".join(
    f"Sentence number {i} describes restarting the docking station and checking cables."
    for i in range(60)
)


@pytest.fixture
def chroma_client(tmp_path):
    return chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@pytest.fixture
def collection_name():
    return f"kb_{uuid.uuid4().hex}"


def _hashing_profile():
    settings = load_settings(env={})
    return settings.retrieval_profiles["hashing"]


async def _manager(tmp_path, collection_name, client=None, chunk_size=400, chunk_overlap=80):
    kb = KnowledgeBaseManager(
        embeddings=HashingEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        client=client,
    )
    await kb.initialize()
    return kb


async def _ingest_articles(kb):
    for source, (category, content) in ARTICLES.items():
        await kb.ingest_document(
            Document(content=content, source=source, category=category, title=source)
        )


def test_doc_id_is_derived_from_source_only():
    a = Document(content="first text", source="a.md", category="vpn")
    b = Document(content="completely different", source="a.md", category="vpn")
    c = Document(content="first text", source="b.md", category="vpn")
    assert a.doc_id == b.doc_id
    assert a.doc_id != c.doc_id
    assert len(a.doc_id) == 16


UNIQUE_TOKEN_TEXT = " ".join(f"w{i}" for i in range(400))


async def _first_two_chunks(kb, source):
    stored = kb._collection.get(where={"source": source}, include=["documents", "metadatas"])
    by_index = sorted(
        zip(stored["metadatas"], stored["documents"]), key=lambda pair: pair[0]["chunk_index"]
    )
    return by_index[0][1].split(), by_index[1][1].split()


@pytest.mark.asyncio
async def test_long_document_splits_into_overlapping_chunks(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client, chunk_size=200, chunk_overlap=60)
    result = await kb.ingest_document(
        Document(content=UNIQUE_TOKEN_TEXT, source="tokens.md", category="hardware")
    )
    assert result["chunks_created"] > 1

    first, second = await _first_two_chunks(kb, "tokens.md")
    shared = set(first) & set(second)
    assert shared
    assert second[0] in first[-15:]
    # The shared run sits at the end of chunk 0 and the start of chunk 1.
    run = len(shared)
    assert first[-run:] == second[:run]


@pytest.mark.asyncio
async def test_zero_overlap_setting_produces_disjoint_chunks(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client, chunk_size=200, chunk_overlap=0)
    result = await kb.ingest_document(
        Document(content=UNIQUE_TOKEN_TEXT, source="tokens.md", category="hardware")
    )
    assert result["chunks_created"] > 1

    first, second = await _first_two_chunks(kb, "tokens.md")
    assert set(first) & set(second) == set()


@pytest.mark.asyncio
async def test_reingest_same_document_keeps_count(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    doc = Document(content=LONG_TEXT, source="dock.md", category="hardware")
    await kb.ingest_document(doc)
    first_count = kb._collection.count()
    await kb.ingest_document(Document(content=LONG_TEXT, source="dock.md", category="hardware"))
    assert kb._collection.count() == first_count


@pytest.mark.asyncio
async def test_reingest_shortened_source_removes_stale_chunks(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    await kb.ingest_document(Document(content=LONG_TEXT, source="dock.md", category="hardware"))
    long_count = kb._collection.count()

    short = "Restart the docking station once and reseat the cable."
    result = await kb.ingest_document(Document(content=short, source="dock.md", category="hardware"))

    assert result["chunks_created"] == 1
    assert kb._collection.count() == 1 < long_count
    stored = kb._collection.get(where={"source": "dock.md"}, include=["documents", "metadatas"])
    assert stored["documents"] == [short]
    assert stored["metadatas"][0]["total_chunks"] == 1


@pytest.mark.asyncio
async def test_vpn_query_ranks_vpn_article_first_with_hashing_profile(
    tmp_path, chroma_client, collection_name
):
    profile = _hashing_profile()
    assert (profile.chunk_size, profile.chunk_overlap) == (400, 80)
    kb = await _manager(
        tmp_path,
        collection_name,
        client=chroma_client,
        chunk_size=profile.chunk_size,
        chunk_overlap=profile.chunk_overlap,
    )
    await _ingest_articles(kb)

    results = await kb.search(
        "vpn keeps disconnecting", min_relevance=profile.min_relevance_score
    )
    assert results
    assert results[0]["source"] == "vpn-disconnects.md"
    assert results[0]["relevance_score"] > profile.confidence_threshold


@pytest.mark.asyncio
async def test_unrelated_query_returns_nothing(tmp_path, chroma_client, collection_name):
    profile = _hashing_profile()
    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    await _ingest_articles(kb)
    results = await kb.search("sourdough bread recipe", min_relevance=profile.min_relevance_score)
    assert results == []


@pytest.mark.asyncio
async def test_category_filter_excludes_other_categories(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    await _ingest_articles(kb)
    results = await kb.search("password sign-in", category="email", min_relevance=0.0)
    assert results
    assert {r["category"] for r in results} == {"email"}


@pytest.mark.asyncio
async def test_high_min_relevance_returns_nothing(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    await _ingest_articles(kb)
    assert await kb.search("vpn keeps disconnecting", min_relevance=0.99) == []


@pytest.mark.asyncio
async def test_delete_removes_all_chunks_of_document(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client, chunk_size=200, chunk_overlap=40)
    await kb.ingest_document(Document(content="Printer note.", source="p.md", category="printing"))
    result = await kb.ingest_document(Document(content=LONG_TEXT, source="dock.md", category="hardware"))
    assert result["chunks_created"] > 1

    assert await kb.delete_document(result["doc_id"]) is True
    assert kb._collection.get(where={"source": "dock.md"})["ids"] == []
    assert kb._collection.count() == 1
    assert await kb.delete_document(result["doc_id"]) is False


@pytest.mark.asyncio
async def test_stats_count_chunks_per_category(tmp_path, chroma_client, collection_name):
    kb = await _manager(tmp_path, collection_name, client=chroma_client, chunk_size=200, chunk_overlap=40)
    a = await kb.ingest_document(Document(content=LONG_TEXT, source="dock.md", category="hardware"))
    await kb.ingest_document(Document(content="Printer note.", source="p.md", category="printing"))
    await kb.ingest_document(Document(content="Second printer note.", source="p2.md", category="printing"))

    stats = await kb.get_stats()
    assert stats["categories"] == {"hardware": a["chunks_created"], "printing": 2}
    assert stats["total_chunks"] == a["chunks_created"] + 2
    assert stats["collection_name"] == collection_name


@pytest.mark.asyncio
async def test_ingest_directory_reports_unreadable_files(tmp_path, chroma_client, collection_name):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text("Restart the router and wait two minutes.", encoding="utf-8")
    (docs / "bad.md").write_bytes(b"\xff\xfe\xfa not utf-8 \x80\x81")

    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    results = await kb.ingest_directory(str(docs), category="network")

    assert results["total_files"] == 1
    assert len(results["errors"]) == 1
    assert results["errors"][0]["file"].endswith("bad.md")
    assert kb._collection.count() == 1


@pytest.mark.asyncio
async def test_data_persists_across_managers(tmp_path, collection_name):
    first = await _manager(tmp_path, collection_name)
    await first.ingest_document(
        Document(content=ARTICLES["vpn-disconnects.md"][1], source="vpn-disconnects.md", category="vpn")
    )
    count = first._collection.count()
    assert count > 0

    second = await _manager(tmp_path, collection_name)
    assert second.document_count == count
    results = await second.search("vpn keeps disconnecting", min_relevance=0.15)
    assert results[0]["source"] == "vpn-disconnects.md"


@pytest.mark.asyncio
async def test_initialize_creates_cosine_collection_without_embedding_function(
    tmp_path, chroma_client, collection_name
):
    kb = await _manager(tmp_path, collection_name, client=chroma_client)
    assert kb._collection.metadata["hnsw:space"] == "cosine"
    assert kb._collection._embedding_function is None
