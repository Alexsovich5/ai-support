"""Create the stack's key files on first start.

    python -m src.keygen <directory>

Writes, only when absent, ``ingest_api_key`` (a random 256-bit URL-safe
token) and ``openai_api_key`` (a placeholder for the local stub; replace the
file's contents to use a real key). Files are created with O_EXCL and mode
0400, never through a symlink, and their values are never printed.
"""

from __future__ import annotations

import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

STUB_OPENAI_KEY = "stub-openai-key-not-a-real-key"

KEY_FILES: Dict[str, Callable[[], str]] = {
    "ingest_api_key": lambda: secrets.token_urlsafe(32),
    "openai_api_key": lambda: STUB_OPENAI_KEY,
}


def ensure_keys(directory: Path | str) -> List[str]:
    """Create missing key files in ``directory``; return the names created."""
    directory = Path(directory)
    info = os.lstat(directory)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("key directory must be a real directory, not a symlink")
    created = []
    for name, make in KEY_FILES.items():
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(directory / name, flags, 0o400)
        except FileExistsError:
            existing = os.lstat(directory / name)
            if not stat.S_ISREG(existing.st_mode):
                raise ValueError(f"{name} exists but is not a regular file") from None
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(make() + "\n")
        created.append(name)
    return created


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m src.keygen <directory>", file=sys.stderr)
        return 2
    try:
        created = ensure_keys(args[0])
    except (OSError, ValueError) as exc:
        print(f"key generation failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print("created: " + (", ".join(created) if created else "nothing, all key files exist"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
