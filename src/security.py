"""Request guards shared by the HTTP and WebSocket routes.

- ``RequestGuardMiddleware``: caps request bodies on raw bytes before any
  parsing (413) and, for state-changing methods, refuses cross-origin requests
  (403) and anything that is not ``application/json`` (415), which is what a
  cross-site HTML form or a "simple" cross-origin fetch would send. Unhandled
  errors become a generic 500 and are logged by exception type only.
- ``origin_allowed``: the same-origin rule, also used for WebSocket upgrades.
- ``AuthFailureLimiter``: per-client throttle for failed ingest-key checks.
- ``ConcurrencyLimiter``: the single cap on concurrent model/embedding work,
  with a per-client quota so one client cannot take every slot.
- ``log_value``: makes request-supplied text safe to put in a log line.
"""

from __future__ import annotations

import hmac
import re
import time
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import AsyncIterator, Iterable, Optional
from urllib.parse import urlsplit

from starlette.datastructures import Headers

logger = logging.getLogger("ai_support.security")

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_LOG_UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f  ‪-‮⁦-⁩]")


def log_value(value: object, limit: int = 64) -> str:
    """Return ``value`` as one printable, truncated token for a log line."""
    text = _LOG_UNSAFE.sub("?", str(value))
    return text if len(text) <= limit else text[:limit] + "..."


def keys_match(supplied: Optional[str], expected: str) -> bool:
    """Constant-time comparison of a presented key with the configured one."""
    if not isinstance(supplied, str) or not supplied:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


def origin_allowed(origin: Optional[str], host: Optional[str], allowed: Iterable[str]) -> bool:
    """True when there is no Origin, it is listed exactly, or it names this server's Host."""
    if origin is None:
        return True
    if origin in set(allowed):
        return True
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.netloc or not host:
        return False
    return parts.netloc.lower() == host.lower()


class RequestGuardMiddleware:
    """ASGI middleware: body size cap, JSON-only and same-origin unsafe requests."""

    def __init__(self, app, max_body_bytes: int, allowed_origins: Iterable[str] = ()):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.allowed_origins = tuple(allowed_origins)

    @staticmethod
    async def _reject(send, status: int, detail: str) -> None:
        body = ('{"detail":"%s"}' % detail).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        if scope["method"] in UNSAFE_METHODS:
            if not origin_allowed(headers.get("origin"), headers.get("host"), self.allowed_origins):
                await self._reject(send, 403, "cross-origin request refused")
                return
            media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                await self._reject(send, 415, "request body must be application/json")
                return

        declared = headers.get("content-length")
        if declared is not None:
            if not declared.isdigit():
                await self._reject(send, 400, "invalid content-length")
                return
            if int(declared) > self.max_body_bytes:
                await self._reject(send, 413, "request body too large")
                return

        received = 0
        started = False
        rejected = False

        async def guarded_receive():
            nonlocal received, started, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Answer 413 here: the framework turns read errors into 400s.
                    rejected = True
                    if not started:
                        started = True
                        await self._reject(send, 413, "request body too large")
                    return {"type": "http.disconnect"}
            return message

        async def tracking_send(message):
            nonlocal started
            if rejected:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, tracking_send)
        except Exception as exc:
            if rejected:
                return
            # The exception text may carry request content or upstream error
            # bodies; only its type is logged and the client gets a generic 500.
            logger.error(
                "Unhandled error: %s %s -> %s",
                scope["method"],
                log_value(scope.get("path", ""), 128),
                type(exc).__name__,
            )
            if not started:
                await self._reject(send, 500, "internal error")


class AuthFailureLimiter:
    """Counts failed key checks per client address in a bounded LRU table.

    A client with ``limit`` failures inside ``window_seconds`` is refused until
    the window has passed; the count then starts again. Other clients are not
    affected, and the table never holds more than ``max_clients`` entries.
    """

    def __init__(self, limit: int, window_seconds: float, max_clients: int = 4096, clock=time.monotonic):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._clock = clock
        self._failures: "OrderedDict[str, tuple[int, float]]" = OrderedDict()

    def _current(self, client: str) -> int:
        entry = self._failures.get(client)
        if entry is None:
            return 0
        count, first = entry
        if self._clock() - first >= self.window_seconds:
            del self._failures[client]
            return 0
        return count

    def blocked(self, client: str) -> bool:
        return self._current(client) >= self.limit

    def record_failure(self, client: str) -> None:
        count = self._current(client)
        first = self._failures[client][1] if count else self._clock()
        self._failures[client] = (count + 1, first)
        self._failures.move_to_end(client)
        while len(self._failures) > self.max_clients:
            self._failures.popitem(last=False)

    def __len__(self) -> int:
        return len(self._failures)


class LimiterBusy(Exception):
    """No work slot is free for this client."""


class ConcurrencyLimiter:
    """Caps concurrent model/embedding work overall and per client.

    ``try_acquire`` never waits: a full limiter refuses at once, so a slow
    upstream cannot queue unbounded work. Every acquire is paired with exactly
    one ``release`` through ``slot``.
    """

    def __init__(self, max_total: int, max_per_client: int):
        self.max_total = max_total
        self.max_per_client = min(max_per_client, max_total)
        self._total = 0
        self._per_client: dict[str, int] = {}

    @property
    def in_use(self) -> int:
        return self._total

    def try_acquire(self, client: str) -> bool:
        held = self._per_client.get(client, 0)
        if self._total >= self.max_total or held >= self.max_per_client:
            return False
        self._total += 1
        self._per_client[client] = held + 1
        return True

    def release(self, client: str) -> None:
        held = self._per_client.get(client, 0)
        if held <= 0 or self._total <= 0:
            raise RuntimeError("release without a matching acquire")
        self._total -= 1
        if held == 1:
            del self._per_client[client]
        else:
            self._per_client[client] = held - 1

    @asynccontextmanager
    async def slot(self, client: str) -> AsyncIterator[None]:
        if not self.try_acquire(client):
            raise LimiterBusy()
        try:
            yield
        finally:
            self.release(client)
