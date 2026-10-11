"""Knowledge-base ingest command.

Local mode opens the Chroma store directly and must only be used on a
directory that no running API process is serving:

    python -m src.ingest <directory> [--category NAME] [--reset]

HTTP mode posts each file to a running API, which owns the store:

    python -m src.ingest <directory> [--category NAME] --api-url URL

Both modes ingest ``*.md`` and ``*.txt`` files. A first line of the form
``category: <name>`` overrides ``--category`` and is removed from the text.
A JSON summary ``{"total_files", "total_chunks", "errors"}`` is printed and
the exit code is 1 when any file failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence

import httpx

FILE_EXTENSIONS = (".md", ".txt")
DEFAULT_CATEGORY = "general"
INGEST_PATH = "/api/v1/knowledge/ingest"

_FRONT_MATTER = re.compile(r"^category:\s*([A-Za-z0-9_\-]+)\s*$")


def parse_front_matter(text: str) -> tuple[Optional[str], str]:
    """Return (category, body). The category line must be the first line."""
    lines = text.splitlines()
    if lines:
        match = _FRONT_MATTER.match(lines[0].strip())
        if match:
            return match.group(1), "\n".join(lines[1:]).strip()
    return None, text


def _title_for(path: Path, body: str) -> str:
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return path.stem.replace("_", " ").replace("-", " ").title()


def iter_files(directory: Path) -> Iterator[Path]:
    """Yield ingestible files under ``directory`` in a stable order."""
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in FILE_EXTENSIONS]
    yield from sorted(files)


def read_article(path: Path, root: Path, default_category: str) -> dict:
    """Read one file into the fields of an ingest request.

    Raises UnicodeDecodeError or OSError when the file cannot be read.
    """
    text = path.read_text(encoding="utf-8")
    category, body = parse_front_matter(text)
    return {
        "content": body,
        "source": path.relative_to(root).as_posix(),
        "category": category or default_category,
        "title": _title_for(path, body),
    }


def _new_summary() -> dict:
    return {"total_files": 0, "total_chunks": 0, "errors": []}


def _check_directory(directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"directory not found: {directory}")


async def ingest_directory(kb, path, category: str = DEFAULT_CATEGORY) -> dict:
    """Ingest every article under ``path`` into an initialised KnowledgeBaseManager."""
    from src.knowledge_base import Document

    root = Path(path)
    _check_directory(root)
    summary = _new_summary()
    for file_path in iter_files(root):
        try:
            fields = read_article(file_path, root, category)
            result = await kb.ingest_document(Document(**fields))
        except Exception as exc:  # reported per file, the rest still ingest
            summary["errors"].append({"file": str(file_path), "error": str(exc)})
            continue
        summary["total_files"] += 1
        summary["total_chunks"] += result["chunks_created"]
    return summary


def ingest_directory_http(
    client: httpx.Client, api_url: str, path, category: str = DEFAULT_CATEGORY
) -> dict:
    """POST every article under ``path`` to ``<api_url>/api/v1/knowledge/ingest``."""
    root = Path(path)
    _check_directory(root)
    url = api_url.rstrip("/") + INGEST_PATH
    summary = _new_summary()
    for file_path in iter_files(root):
        try:
            fields = read_article(file_path, root, category)
            response = client.post(url, json=fields)
            response.raise_for_status()
            chunks = int(response.json()["chunks_created"])
        except Exception as exc:
            summary["errors"].append({"file": str(file_path), "error": str(exc)})
            continue
        summary["total_files"] += 1
        summary["total_chunks"] += chunks
    return summary


async def _run_local(directory: str, category: str, reset: bool) -> dict:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from src.config import load_settings
    from src.knowledge_base import KnowledgeBaseManager
    from src.llm_factory import get_embeddings

    settings = load_settings()
    persist = settings.knowledge_base.persist_directory
    collection = settings.knowledge_base.collection_name
    Path(persist).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist, settings=ChromaSettings(anonymized_telemetry=False)
    )
    if reset and collection in {c.name for c in client.list_collections()}:
        client.delete_collection(collection)

    kb = KnowledgeBaseManager(
        embeddings=get_embeddings(settings),
        persist_directory=persist,
        collection_name=collection,
        chunk_size=settings.retrieval.chunk_size,
        chunk_overlap=settings.retrieval.chunk_overlap,
        client=client,
    )
    await kb.initialize()
    return await ingest_directory(kb, directory, category)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingest",
        description="Ingest *.md and *.txt articles into the knowledge base.",
    )
    parser.add_argument("directory", help="directory holding the articles")
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="category for files without a 'category:' first line",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="local mode only: empty the collection before ingesting",
    )
    parser.add_argument(
        "--api-url",
        help="post each file to a running API instead of opening Chroma directly",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None, transport: Optional[httpx.BaseTransport] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.api_url and args.reset:
        parser.error("--reset cannot be combined with --api-url; the API owns the store")

    try:
        if args.api_url:
            with httpx.Client(transport=transport, timeout=60.0) as client:
                summary = ingest_directory_http(client, args.api_url, args.directory, args.category)
        else:
            summary = asyncio.run(_run_local(args.directory, args.category, args.reset))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
