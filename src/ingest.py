"""Knowledge-base ingest command.

Local mode opens the Chroma store directly and must only be used on a
directory that no running API process is serving:

    python -m src.ingest <directory> [--category NAME] [--reset]

HTTP mode posts each file to a running API, which owns the store:

    python -m src.ingest <directory> [--category NAME] --api-url URL

Both modes ingest ``*.md`` and ``*.txt`` files. A first line of the form
``category: <name>`` overrides ``--category`` and is removed from the text.
A JSON summary ``{"total_files", "total_chunks", "errors"}`` is printed and
the exit code is 1 when any file failed or was skipped.

The directory walk never follows symlinks and reads only regular files that
resolve inside the directory, opened once with O_NOFOLLOW. File size, file
count and total bytes per run are capped (``IngestLimits``). Summaries and
stored metadata carry paths relative to the directory only. HTTP mode sends
the ingest key from ``--api-key-file`` or ``$INGEST_API_KEY_FILE``.
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import httpx

FILE_EXTENSIONS = (".md", ".txt")
DEFAULT_CATEGORY = "general"
INGEST_PATH = "/api/v1/knowledge/ingest"
MAX_SOURCE_LENGTH = 256

_FRONT_MATTER = re.compile(r"^category:\s*([A-Za-z0-9_\-]+)\s*$")
_SOURCE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ +\-]*$")


class UnsafePathError(ValueError):
    """A path that is not a plain regular file inside the ingest root."""


@dataclass(frozen=True)
class IngestLimits:
    """Caps applied to one directory ingest run."""

    max_file_bytes: int = 1_048_576
    max_files: int = 1_000
    max_total_bytes: int = 20 * 1_048_576
    max_depth: int = 16


def validate_source(source: str) -> str:
    """Return ``source`` if it is a safe relative name, else raise ValueError.

    A source is a ``/``-separated relative path: no empty, ``.`` or ``..``
    parts, no absolute path, drive letter, backslash, NUL or other control
    character, at most 256 characters.
    """
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty relative path")
    if len(source) > MAX_SOURCE_LENGTH:
        raise ValueError(f"source is longer than {MAX_SOURCE_LENGTH} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in source) or "\\" in source:
        raise ValueError("source contains a control character or backslash")
    if source.startswith("/") or re.match(r"^[A-Za-z]:", source):
        raise ValueError("source must be relative")
    for part in source.split("/"):
        if part in ("", ".", "..") or not _SOURCE_PART.match(part):
            raise ValueError("source contains an empty, '.', '..' or unsupported path part")
    return source


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
            return line[:256]
    return path.stem.replace("_", " ").replace("-", " ").title()[:256]


@dataclass(frozen=True)
class Candidate:
    """One directory entry found by ``scan``; ``problem`` is set when it is skipped."""

    rel: str
    problem: Optional[str] = None
    size: int = 0
    identity: tuple[int, int] = (0, 0)


def _resolved_root(root: Path) -> str:
    return os.path.realpath(root)


def _inside(root_real: str, path: str) -> bool:
    resolved = os.path.realpath(path)
    return os.path.commonpath([root_real, resolved]) == root_real


def scan(root: Path, limits: IngestLimits = IngestLimits()) -> Iterator[Candidate]:
    """Yield allow-listed files under ``root`` in a stable order, without following links.

    Symlinks (to files or directories), FIFOs, devices and sockets are reported
    as skipped candidates; other files with an extension outside
    ``FILE_EXTENSIONS`` are ignored.
    """
    root_real = _resolved_root(root)

    def _walk(directory: str, prefix: str, depth: int) -> Iterator[Candidate]:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
        for entry in entries:
            rel = f"{prefix}{entry.name}"
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                yield Candidate(rel, "skipped: cannot stat entry")
                continue
            if stat.S_ISLNK(info.st_mode):
                yield Candidate(rel, "skipped: symlink")
                continue
            if stat.S_ISDIR(info.st_mode):
                if depth >= limits.max_depth:
                    yield Candidate(rel, "skipped: directory nesting limit reached")
                    continue
                yield from _walk(entry.path, f"{rel}/", depth + 1)
                continue
            if Path(entry.name).suffix not in FILE_EXTENSIONS:
                continue
            if not stat.S_ISREG(info.st_mode):
                yield Candidate(rel, "skipped: not a regular file")
                continue
            if not _inside(root_real, entry.path):
                yield Candidate(rel, "skipped: outside the ingest root")
                continue
            yield Candidate(rel, None, info.st_size, (info.st_dev, info.st_ino))

    yield from _walk(root_real, "", 0)


def read_regular_file(
    root: Path, rel: str, max_bytes: int, identity: Optional[tuple[int, int]] = None
) -> str:
    """Read ``root/rel`` as UTF-8 text, refusing anything but a bounded regular file.

    The file is opened once with O_NOFOLLOW and O_NONBLOCK (a FIFO cannot block
    the run), and the open descriptor must be a regular file inside ``root``
    no larger than ``max_bytes``; with ``identity`` it must also be the same
    (device, inode) that the scan saw.
    """
    validate_source(rel)
    root_real = _resolved_root(root)
    path = os.path.join(root_real, *rel.split("/"))
    if not _inside(root_real, path):
        raise UnsafePathError("outside the ingest root")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP or os.path.islink(path):
            raise UnsafePathError("symlink") from None
        raise
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise UnsafePathError("not a regular file")
        if identity is not None and (info.st_dev, info.st_ino) != identity:
            raise UnsafePathError("file changed while ingesting")
        if info.st_size > max_bytes:
            raise UnsafePathError(f"too large (over {max_bytes} bytes)")
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UnsafePathError(f"too large (over {max_bytes} bytes)")
    return data.decode("utf-8")


def read_article(root: Path, candidate: Candidate, default_category: str, max_bytes: int) -> dict:
    """Read one scanned file into the fields of an ingest request."""
    text = read_regular_file(root, candidate.rel, max_bytes, candidate.identity)
    category, body = parse_front_matter(text)
    return {
        "content": body,
        "source": candidate.rel,
        "category": category or default_category,
        "title": _title_for(Path(candidate.rel), body),
    }


def safe_error(exc: BaseException) -> str:
    """An error message that never carries a host path or upstream response body."""
    if isinstance(exc, UnsafePathError):
        return f"skipped: {exc}"
    if isinstance(exc, UnicodeDecodeError):
        return "not valid UTF-8 text"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"API answered HTTP {exc.response.status_code}"
    if isinstance(exc, OSError):
        return f"cannot read file ({exc.strerror or type(exc).__name__})"
    if isinstance(exc, ValueError):
        return str(exc)[:200]
    return f"ingest failed ({type(exc).__name__})"


def _new_summary() -> dict:
    return {"total_files": 0, "total_chunks": 0, "errors": []}


def _check_directory(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError("ingest directory not found or not a plain directory")


def _articles(root: Path, category: str, limits: IngestLimits, summary: dict) -> Iterator[dict]:
    """Yield article fields within the limits, recording every skipped file."""
    taken = 0
    total_bytes = 0
    for candidate in scan(root, limits):
        if candidate.problem:
            summary["errors"].append({"file": candidate.rel, "error": candidate.problem})
            continue
        if candidate.size > limits.max_file_bytes:
            summary["errors"].append(
                {"file": candidate.rel, "error": f"skipped: too large (over {limits.max_file_bytes} bytes)"}
            )
            continue
        if taken >= limits.max_files:
            summary["errors"].append(
                {"file": candidate.rel, "error": f"stopped: file limit reached ({limits.max_files} files)"}
            )
            return
        if total_bytes + candidate.size > limits.max_total_bytes:
            summary["errors"].append(
                {"file": candidate.rel, "error": f"stopped: total byte limit reached ({limits.max_total_bytes} bytes)"}
            )
            return
        try:
            fields = read_article(root, candidate, category, limits.max_file_bytes)
        except Exception as exc:
            summary["errors"].append({"file": candidate.rel, "error": safe_error(exc)})
            continue
        taken += 1
        total_bytes += candidate.size
        yield fields


async def ingest_directory(
    kb, path, category: str = DEFAULT_CATEGORY, limits: Optional[IngestLimits] = None
) -> dict:
    """Ingest every article under ``path`` into an initialised KnowledgeBaseManager."""
    from src.knowledge_base import Document

    root = Path(path)
    _check_directory(root)
    limits = limits or IngestLimits()
    summary = _new_summary()
    for fields in _articles(root, category, limits, summary):
        try:
            result = await kb.ingest_document(Document(**fields))
        except Exception as exc:  # reported per file, the rest still ingest
            summary["errors"].append({"file": fields["source"], "error": safe_error(exc)})
            continue
        summary["total_files"] += 1
        summary["total_chunks"] += result["chunks_created"]
    return summary


def ingest_directory_http(
    client: httpx.Client,
    api_url: str,
    path,
    category: str = DEFAULT_CATEGORY,
    limits: Optional[IngestLimits] = None,
    api_key: Optional[str] = None,
) -> dict:
    """POST every article under ``path`` to ``<api_url>/api/v1/knowledge/ingest``."""
    root = Path(path)
    _check_directory(root)
    limits = limits or IngestLimits()
    url = api_url.rstrip("/") + INGEST_PATH
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    summary = _new_summary()
    for fields in _articles(root, category, limits, summary):
        try:
            response = client.post(url, json=fields, headers=headers)
            response.raise_for_status()
            chunks = int(response.json()["chunks_created"])
        except Exception as exc:
            summary["errors"].append({"file": fields["source"], "error": safe_error(exc)})
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
    parser.add_argument(
        "--api-key-file",
        default=os.environ.get("INGEST_API_KEY_FILE"),
        help="--api-url mode: file holding the ingest key (default: $INGEST_API_KEY_FILE)",
    )
    return parser


def _read_api_key(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8").strip() or None
    except OSError as exc:
        raise FileNotFoundError(f"cannot read the ingest key file ({exc.strerror})") from None


def main(argv: Optional[Sequence[str]] = None, transport: Optional[httpx.BaseTransport] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.api_url and args.reset:
        parser.error("--reset cannot be combined with --api-url; the API owns the store")

    try:
        if args.api_url:
            api_key = _read_api_key(args.api_key_file)
            with httpx.Client(transport=transport, timeout=60.0) as client:
                summary = ingest_directory_http(
                    client, args.api_url, args.directory, args.category, api_key=api_key
                )
        else:
            summary = asyncio.run(_run_local(args.directory, args.category, args.reset))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
