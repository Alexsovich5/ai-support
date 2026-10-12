"""Regex-based redaction of patient identifiers in free text and structured data.

Best-effort only, not a compliance control. The text is normalised first
(HTML entities and percent-encoding decoded, Unicode NFKC, dashes unified,
zero-width/control/bidi characters removed, runs of spaces collapsed), so
full-width digits, zero-width-split numbers and encoded fragments are matched
like plain ones. Matches are replaced with ``[REDACTED:<kind>]``.

Fail closed: if the input is larger than ``MAX_REDACT_CHARS`` or the filter
raises, the whole text is replaced by ``WITHHELD_TEXT`` and the kind
``withheld`` is reported. Every pattern is linear in the input length: each
one either starts only at a token boundary or has bounded repetition.
"""

from __future__ import annotations

import base64
import binascii
import html
import logging
import re
import unicodedata
from typing import Any, List, NamedTuple, Tuple
from urllib.parse import unquote

logger = logging.getLogger("ai_support.phi_filter")

MAX_REDACT_CHARS = 200_000
WITHHELD_TEXT = "[WITHHELD: text could not be checked for patient identifiers]"
WITHHELD_KIND = "withheld"

_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_DATE_FORMS = (
    r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}",
    r"\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})",
    rf"{_MONTH} {_DAY},? \d{{4}}",
    rf"{_DAY} {_MONTH},? \d{{4}}",
)
_ANY_DATE = "(?:" + "|".join(_DATE_FORMS) + ")"
# Unlabelled dates must carry a four-digit year, so version strings and
# addresses such as 1.2.24 or 10.0.0.1 are left alone.
_FULL_DATE = "(?:" + "|".join((_DATE_FORMS[0], r"\d{1,2}[/.-]\d{1,2}[/.-]\d{4}", *_DATE_FORMS[2:])) + ")"

_SENSITIVE_JSON_KEYS = (
    r"ssn|social_?security(?:_?number)?|mrn|medical_?record(?:_?number)?|dob|date_?of_?birth"
    r"|birth_?date|patient_?name|first_?name|last_?name|full_?name|name|phone(?:_?number)?"
    r"|email|address|street"
)
_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way"
    r"|Place|Pl|Terrace|Circle|Cir)"
)
_CAPITALISED_WORD = r"[A-Z][a-z'\-]+"

# Order matters: at any position the first alternative that matches wins.
_PATTERNS = [
    ("userinfo", r"(?<=://)[^\s/@:\[\]]{1,256}(?::[^\s/@\[\]]{0,256})?@"),
    ("field", rf"(?P<field_keep>\"(?:{_SENSITIVE_JSON_KEYS})\"[ ]?:[ ]?)\"[^\"\\\n]{{0,256}}\""),
    ("email", r"(?<![\w.+-])[\w.+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})++"),
    (
        "mrn",
        r"\b(?:MRN|medical[ ]record[ ](?:number|no\.?|\#))[ ]?[:#]?[ ]?"
        r"[A-Z]{0,3}\d(?:[ ]?\d){5,11}\b",
    ),
    (
        "dob",
        rf"\b(?:DOB|D\.O\.B\.?|date[ ]of[ ]birth|birth[ ]?date|born(?:[ ]on)?)[ ]?[:#-]?[ ]?{_ANY_DATE}\b",
    ),
    (
        "name",
        rf"(?P<name_keep>\b(?:patient(?:'s)?(?:[ ]name)?(?:[ ]?:)?|(?:full[ ])?name[ ]?:)[ ])"
        rf"(?-i:{_CAPITALISED_WORD}(?:[ ]{_CAPITALISED_WORD}){{1,2}})",
    ),
    ("address", rf"\b\d{{1,5}}[ ](?-i:(?:[A-Z][a-z]{{1,20}}[ ]){{1,3}}{_STREET_SUFFIX})\b\.?"),
    ("date", rf"\b{_FULL_DATE}\b"),
    ("ssn", r"(?<![\w.-])\d{3}[ -]?\d{2}[ -]?\d{4}(?![\w-]|\.\d)"),
    (
        "phone",
        r"(?<![\w.+-])(?:\+?1[ .-]?)?(?:\(\d{3}\)[ ]?|\d{3}[ .-]?)\d{3}[ .-]?\d{4}(?![\w-]|\.\d)",
    ),
    # Digits typed one by one with spaces (7 or more): read as an identifier.
    ("spaced_digits", r"(?<![\w.])(?<!\d[ ])\d(?:[ ]\d){6,}+(?![\w.])"),
]

_COMBINED = re.compile(
    "|".join(f"(?P<{kind}>{pattern})" for kind, pattern in _PATTERNS),
    re.IGNORECASE,
)

_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}+={0,2}(?![A-Za-z0-9+/=])")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_HORIZONTAL_SPACE = re.compile(r"[ \t]+")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣"), "-")
_REMOVED_CATEGORIES = {"Cc", "Cf"}
_KEPT_CONTROLS = {"\n", "\t"}


class RedactionResult(NamedTuple):
    text: str
    found: List[str]


def _strip_invisible(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if ch in _KEPT_CONTROLS or unicodedata.category(ch) not in _REMOVED_CATEGORIES
    )


def normalise(text: str) -> str:
    """Return the form of ``text`` that the patterns are matched against."""
    for _ in range(2):  # double encoding is unwrapped once more
        decoded = html.unescape(text)
        if _PERCENT_ESCAPE.search(decoded):
            decoded = unquote(decoded, errors="replace")
        if decoded == text:
            break
        text = decoded
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _strip_invisible(text.translate(_DASHES))
    return _HORIZONTAL_SPACE.sub(" ", text)


def _kind_for_digits(match_text: str) -> str:
    count = sum(ch.isdigit() for ch in match_text)
    if count == 9:
        return "ssn"
    if count in (10, 11):
        return "phone"
    return "id_number"


def _redact_plain(text: str, found: List[str]) -> str:
    def _replace(match: "re.Match[str]") -> str:
        kind = match.lastgroup
        keep = ""
        if kind == "spaced_digits":
            kind = _kind_for_digits(match.group(0))
        elif kind in ("field", "name"):
            # The label (JSON key or "patient") stays readable.
            keep = match.group(f"{kind}_keep") or ""
        if kind not in found:
            found.append(kind)
        if kind == "field":
            return f'{keep}"[REDACTED:{kind}]"'
        if kind == "userinfo":
            return f"[REDACTED:{kind}]@"
        return f"{keep}[REDACTED:{kind}]"

    return _COMBINED.sub(_replace, text)


def _decoded_fragment(token: str) -> str | None:
    if len(token) % 4:
        return None
    try:
        raw = base64.b64decode(token, validate=True)
        decoded = raw.decode("utf-8")
    except (binascii.Error, ValueError):
        return None
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in decoded)
    if not decoded or printable < 0.9 * len(decoded):
        return None
    return decoded


def _redact_encoded(text: str, found: List[str]) -> str:
    """Replace base64 fragments whose decoded text holds an identifier."""

    def _replace(match: "re.Match[str]") -> str:
        decoded = _decoded_fragment(match.group(0))
        if decoded is None:
            return match.group(0)
        inner: List[str] = []
        _redact_plain(normalise(decoded), inner)
        if not inner:
            return match.group(0)
        for kind in inner + ["encoded"]:
            if kind not in found:
                found.append(kind)
        return "[REDACTED:encoded]"

    return _BASE64_TOKEN.sub(_replace, text)


def _redact_normalised(text: str) -> RedactionResult:
    found: List[str] = []
    text = _redact_plain(text, found)
    text = _redact_encoded(text, found)
    return RedactionResult(text, found)


def _withheld(reason: str) -> RedactionResult:
    logger.warning("PHI filter withheld text: %s", reason)
    return RedactionResult(WITHHELD_TEXT, [WITHHELD_KIND])


def redact(text: str) -> RedactionResult:
    """Replace identifiers in ``text`` and list the kinds found, in order of first appearance.

    Never returns the input unfiltered: oversized input or any internal error
    yields ``WITHHELD_TEXT``.
    """
    if not text:
        return RedactionResult("", [])
    try:
        if not isinstance(text, str):
            return _withheld("input is not text")
        if len(text) > MAX_REDACT_CHARS:
            return _withheld("input exceeds the size cap")
        normalised = normalise(text)
        if len(normalised) > MAX_REDACT_CHARS:
            return _withheld("normalised input exceeds the size cap")
        return _redact_normalised(normalised)
    except Exception as exc:  # fail closed, never pass the text through
        return _withheld(type(exc).__name__)


_SENSITIVE_KEY = re.compile(rf"^(?:{_SENSITIVE_JSON_KEYS}|patient)$", re.IGNORECASE)


def _key_is_sensitive(key: Any) -> bool:
    return isinstance(key, str) and bool(_SENSITIVE_KEY.match(key.strip().replace("-", "_").replace(" ", "_")))


def redact_structure(data: Any, _depth: int = 0) -> Tuple[Any, List[str]]:
    """Redact nested dicts/lists: values under identifying keys and every string.

    Scalars stored under keys such as ``ssn``, ``mrn``, ``dob`` or ``name`` are
    masked whatever they look like; string keys and values are passed through
    ``redact``. Nesting deeper than 32 levels is withheld.
    """
    found: List[str] = []

    def _note(kinds: List[str]) -> None:
        for kind in kinds:
            if kind not in found:
                found.append(kind)

    def _walk(value: Any, depth: int, sensitive: bool) -> Any:
        if depth > 32:
            _note([WITHHELD_KIND])
            return WITHHELD_TEXT
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                new_key = key
                if isinstance(key, str):
                    result = redact(key)
                    new_key = result.text
                    _note(result.found)
                cleaned[new_key] = _walk(item, depth + 1, sensitive or _key_is_sensitive(key))
            return cleaned
        if isinstance(value, (list, tuple)):
            return type(value)(_walk(item, depth + 1, sensitive) for item in value)
        if sensitive and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            _note(["field"])
            return "[REDACTED:field]"
        if isinstance(value, str):
            result = redact(value)
            _note(result.found)
            return result.text
        return value

    return _walk(data, _depth, False), found
