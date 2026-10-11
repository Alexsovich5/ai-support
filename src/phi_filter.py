"""Regex-based redaction of patient identifiers in free text.

Best-effort only: free-text identifiers such as patient names are not
detected. Matches are replaced with ``[REDACTED:<kind>]``.
"""

import re
from typing import List, NamedTuple

# Order matters: earlier alternatives win when patterns could overlap
# (an email's digits must not be read as a phone number, for example).
_PATTERNS = [
    ("email", r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    ("mrn", r"\bMRN\s*[:#]?\s*\d{6,10}\b"),
    ("dob", r"\b(?:DOB|date\s+of\s+birth)\s*:?\s*\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})\b"),
    ("ssn", r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])"),
    (
        "phone",
        r"(?<![\w.+-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]\d{4}(?![\w-])",
    ),
]

_COMBINED = re.compile(
    "|".join(f"(?P<{kind}>{pattern})" for kind, pattern in _PATTERNS),
    re.IGNORECASE,
)


class RedactionResult(NamedTuple):
    text: str
    found: List[str]


def redact(text: str) -> RedactionResult:
    """Replace identifiers in ``text`` and list the kinds found, in order of first appearance."""
    found: List[str] = []

    def _replace(match: "re.Match[str]") -> str:
        kind = match.lastgroup
        if kind not in found:
            found.append(kind)
        return f"[REDACTED:{kind}]"

    return RedactionResult(_COMBINED.sub(_replace, text), found)
