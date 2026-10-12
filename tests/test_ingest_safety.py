"""Directory ingest stays inside its root and only reads bounded regular files."""

import json
import os
import threading

import pytest

from src import ingest


class RecordingKB:
    """Stands in for KnowledgeBaseManager; keeps every ingested document."""

    def __init__(self):
        self.docs = []

    async def ingest_document(self, document):
        self.docs.append(document)
        return {"chunks_created": 1}


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "good.md").write_text("Restart the router and wait two minutes.", encoding="utf-8")
    (root / "sub" / "nested.txt").write_text("Reseat the docking station cable.", encoding="utf-8")
    return root


@pytest.fixture
def outside(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    (target / "secret.md").write_text("OUTSIDE-ROOT-SENTINEL contents", encoding="utf-8")
    return target


def _reasons(summary):
    return {entry["file"]: entry["error"] for entry in summary["errors"]}


@pytest.mark.asyncio
async def test_symlink_to_file_outside_root_is_not_ingested(docs, outside):
    os.symlink(outside / "secret.md", docs / "link.md")
    os.symlink("/etc/passwd", docs / "passwd.md")
    kb = RecordingKB()

    summary = await ingest.ingest_directory(kb, docs)

    assert sorted(d.source for d in kb.docs) == ["good.md", "sub/nested.txt"]
    assert not any("OUTSIDE-ROOT-SENTINEL" in d.content or "root:" in d.content for d in kb.docs)
    reasons = _reasons(summary)
    assert "symlink" in reasons["link.md"]
    assert "symlink" in reasons["passwd.md"]


@pytest.mark.asyncio
async def test_symlinked_directory_is_not_traversed(docs, outside):
    os.symlink(outside, docs / "linked", target_is_directory=True)
    kb = RecordingKB()

    summary = await ingest.ingest_directory(kb, docs)

    assert sorted(d.source for d in kb.docs) == ["good.md", "sub/nested.txt"]
    assert "symlink" in _reasons(summary)["linked"]


def test_fifo_does_not_hang_the_run(docs):
    os.mkfifo(docs / "pipe.md")
    kb = RecordingKB()
    outcome = {}

    def run():
        import asyncio

        outcome["summary"] = asyncio.run(ingest.ingest_directory(kb, docs))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), "ingest blocked on a FIFO"
    assert "not a regular file" in _reasons(outcome["summary"])["pipe.md"]
    assert len(kb.docs) == 2


@pytest.mark.asyncio
async def test_oversized_file_is_skipped_and_counted(docs):
    (docs / "huge.md").write_text("x" * 5000, encoding="utf-8")
    kb = RecordingKB()

    summary = await ingest.ingest_directory(kb, docs, limits=ingest.IngestLimits(max_file_bytes=1000))

    assert "huge.md" not in {d.source for d in kb.docs}
    assert "too large" in _reasons(summary)["huge.md"]
    assert summary["total_files"] == 2


@pytest.mark.asyncio
async def test_file_count_and_total_byte_caps_stop_the_run(docs):
    for i in range(5):
        (docs / f"extra{i}.md").write_text(f"Article {i} body text.", encoding="utf-8")
    kb = RecordingKB()
    summary = await ingest.ingest_directory(kb, docs, limits=ingest.IngestLimits(max_files=3))
    assert len(kb.docs) == 3
    assert any("file limit" in e["error"] for e in summary["errors"])

    kb = RecordingKB()
    summary = await ingest.ingest_directory(kb, docs, limits=ingest.IngestLimits(max_total_bytes=60))
    assert sum(len(d.content) for d in kb.docs) <= 60
    assert any("byte limit" in e["error"] for e in summary["errors"])


@pytest.mark.asyncio
async def test_only_allow_listed_extensions_are_read(docs):
    (docs / "notes.md.sh").write_text("echo hi", encoding="utf-8")
    (docs / "page.html").write_text("<p>hi</p>", encoding="utf-8")
    kb = RecordingKB()
    await ingest.ingest_directory(kb, docs)
    assert sorted(d.source for d in kb.docs) == ["good.md", "sub/nested.txt"]


@pytest.mark.asyncio
async def test_summary_and_metadata_hold_only_relative_paths(docs, tmp_path):
    (docs / "broken.md").write_bytes(b"\xff\xfe\xfa not utf-8")
    os.symlink("/etc/hostname", docs / "host.md")
    kb = RecordingKB()

    summary = await ingest.ingest_directory(kb, docs)

    assert str(tmp_path) not in json.dumps(summary)
    assert {e["file"] for e in summary["errors"]} == {"broken.md", "host.md"}
    assert all(not d.source.startswith("/") for d in kb.docs)
    assert all(str(tmp_path) not in json.dumps(d.metadata) for d in kb.docs)


def test_regular_file_opener_refuses_a_symlink_swapped_in(docs, outside):
    os.symlink(outside / "secret.md", docs / "swapped.md")
    with pytest.raises(ingest.UnsafePathError):
        ingest.read_regular_file(docs, "swapped.md", max_bytes=10_000)


@pytest.mark.parametrize(
    "source",
    ["../etc/passwd", "a/../../b.md", "/etc/passwd", "a\\b.md", "C:\\x.md", "a\x00b.md", "", "a/\nb.md"],
)
def test_validate_source_rejects_unsafe_names(source):
    with pytest.raises(ValueError):
        ingest.validate_source(source)


@pytest.mark.parametrize("source", ["vpn-disconnects.md", "sub/dir/article.txt", "badge-reader-1a2b.md"])
def test_validate_source_accepts_relative_names(source):
    assert ingest.validate_source(source) == source
