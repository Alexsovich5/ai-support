import json
import shutil
from pathlib import Path

import chromadb
import httpx
import pytest
from chromadb.config import Settings as ChromaSettings

from src import ingest
from src.config import load_settings

SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

EXPECTED_CATEGORIES = {
    "vpn-disconnects.md": "vpn",
    "account-lockout.md": "active_directory",
    "printer-offline.md": "printing",
    "outlook-mobile-auth.md": "email",
    "wifi-drops.md": "network",
    "report-phishing.md": "security",
}


@pytest.fixture
def kb_dir(tmp_path):
    target = tmp_path / "kb"
    shutil.copytree(SEED_DIR, target)
    return target


@pytest.fixture
def chroma_env(tmp_path, monkeypatch):
    persist = tmp_path / "chroma"
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(persist))
    monkeypatch.delenv("APP_CONFIG", raising=False)
    return persist, load_settings(env={}).knowledge_base.collection_name


def _collection(persist, name):
    client = chromadb.PersistentClient(
        path=str(persist), settings=ChromaSettings(anonymized_telemetry=False)
    )
    return client.get_collection(name)


def _summary(capsys):
    out = capsys.readouterr().out.strip().splitlines()
    return json.loads(out[-1])


def test_seed_directory_has_six_articles_with_front_matter():
    files = sorted(p.name for p in SEED_DIR.glob("*.md"))
    assert files == sorted(EXPECTED_CATEGORIES)
    for name, category in EXPECTED_CATEGORIES.items():
        first_line = (SEED_DIR / name).read_text(encoding="utf-8").splitlines()[0]
        assert first_line == f"category: {category}"


def test_parse_front_matter_strips_category_line():
    category, body = ingest.parse_front_matter("category: vpn\n\nVPN drops\n\nSteps.")
    assert category == "vpn"
    assert body == "VPN drops\n\nSteps."
    assert ingest.parse_front_matter("No front matter here.") == (None, "No front matter here.")


def test_local_main_ingests_seed_with_front_matter_categories(kb_dir, chroma_env, capsys):
    persist, collection = chroma_env
    code = ingest.main([str(kb_dir), "--category", "general"])
    assert code == 0

    summary = _summary(capsys)
    assert summary["total_files"] == 6
    assert summary["errors"] == []
    assert summary["total_chunks"] >= 6

    stored = _collection(persist, collection).get(include=["metadatas", "documents"])
    by_source = {m["source"]: m["category"] for m in stored["metadatas"]}
    assert by_source == EXPECTED_CATEGORIES
    assert summary["total_chunks"] == len(stored["ids"])
    assert not any(doc.startswith("category:") for doc in stored["documents"])


def test_category_option_applies_without_front_matter(tmp_path, chroma_env, capsys):
    persist, collection = chroma_env
    docs = tmp_path / "plain"
    docs.mkdir()
    (docs / "dock.txt").write_text("Reseat the docking station cable.", encoding="utf-8")
    (docs / "ignored.rst").write_text("Not ingested.", encoding="utf-8")

    assert ingest.main([str(docs), "--category", "hardware"]) == 0
    assert _summary(capsys)["total_files"] == 1
    stored = _collection(persist, collection).get(include=["metadatas"])
    assert [m["category"] for m in stored["metadatas"]] == ["hardware"]


def test_reset_empties_collection_first(kb_dir, tmp_path, chroma_env, capsys):
    persist, collection = chroma_env
    assert ingest.main([str(kb_dir)]) == 0
    capsys.readouterr()

    other = tmp_path / "other"
    other.mkdir()
    (other / "dock.md").write_text(
        "category: hardware\n\nReseat the docking station cable.", encoding="utf-8"
    )

    assert ingest.main([str(other)]) == 0
    capsys.readouterr()
    sources = {m["source"] for m in _collection(persist, collection).get(include=["metadatas"])["metadatas"]}
    assert sources == set(EXPECTED_CATEGORIES) | {"dock.md"}

    assert ingest.main([str(other), "--reset"]) == 0
    stored = _collection(persist, collection).get(include=["metadatas"])
    assert [m["source"] for m in stored["metadatas"]] == ["dock.md"]


def test_exit_code_is_one_when_a_file_fails_to_decode(kb_dir, chroma_env, capsys):
    (kb_dir / "broken.md").write_bytes(b"\xff\xfe\xfa not utf-8 \x80\x81")
    assert ingest.main([str(kb_dir)]) == 1
    summary = _summary(capsys)
    assert summary["total_files"] == 6
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["file"].endswith("broken.md")


def test_api_url_with_reset_is_a_usage_error(kb_dir, capsys):
    with pytest.raises(SystemExit) as excinfo:
        ingest.main([str(kb_dir), "--api-url", "http://api:8000", "--reset"])
    assert excinfo.value.code == 2
    assert "--reset" in capsys.readouterr().err


def test_api_url_mode_posts_one_request_per_file(kb_dir, capsys):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.method, str(request.url), body))
        return httpx.Response(
            200, json={"doc_id": "x", "chunks_created": 2, "total_documents": 2 * len(requests)}
        )

    code = ingest.main(
        [str(kb_dir), "--api-url", "http://api:8000/"], transport=httpx.MockTransport(handler)
    )
    assert code == 0
    assert len(requests) == 6
    assert {url for _, url, _ in requests} == {"http://api:8000/api/v1/knowledge/ingest"}
    assert {m for m, _, _ in requests} == {"POST"}
    posted = {body["source"]: body for _, _, body in requests}
    assert {s: b["category"] for s, b in posted.items()} == EXPECTED_CATEGORIES
    assert all(not b["content"].startswith("category:") and b["title"] for b in posted.values())

    summary = _summary(capsys)
    assert summary == {"total_files": 6, "total_chunks": 12, "errors": []}


def test_api_url_mode_reports_http_errors(kb_dir, capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["source"] == "wifi-drops.md":
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={"doc_id": "x", "chunks_created": 1, "total_documents": 1})

    code = ingest.main([str(kb_dir), "--api-url", "http://api:8000"], transport=httpx.MockTransport(handler))
    assert code == 1
    summary = _summary(capsys)
    assert summary["total_files"] == 5
    assert summary["errors"][0]["file"].endswith("wifi-drops.md")
