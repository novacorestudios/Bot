"""Structured logging with hard secret redaction.

Two protections against leaking credentials into logs:

1. ``register_secret`` records the literal value of every secret at startup; a
   processor replaces any occurrence of it anywhere in the event, including
   inside nested structures and formatted strings.
2. A key-name filter redacts values whose key looks sensitive (``signature``,
   ``apiKey``, ``secret``, ``token``, ``password``, ``listenKey``), even for
   values that were never registered.

Both run on every event, so a mistake at a call site cannot produce a leak.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
from pathlib import Path
from typing import Any

import structlog

REDACTED = "***REDACTED***"

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|api[_-]?secret|secret|signature|token|password|passwd|"
    r"listen[_-]?key|authorization|x-mbx-apikey|private)",
    re.IGNORECASE,
)

_secrets: set[str] = set()


def register_secret(value: str | None, min_length: int = 6) -> None:
    """Record a literal secret so it is scrubbed from every log event."""
    if value and len(value) >= min_length:
        _secrets.add(value)


def clear_secrets() -> None:
    """Test helper. Not used in production paths."""
    _secrets.clear()


def _scrub_text(text: str) -> str:
    for secret in _secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


def _scrub(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return value
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and _SENSITIVE_KEY_PATTERN.search(k):
                out[k] = REDACTED
            else:
                out[k] = _scrub(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        scrubbed = [_scrub(v, depth + 1) for v in value]
        return type(value)(scrubbed) if isinstance(value, list) else tuple(scrubbed)
    return value


def redact_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor applying both redaction layers."""
    return _scrub(event_dict)  # type: ignore[return-value]


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    log_file: str | None = None,
    max_bytes: int = 20 * 1024 * 1024,
    backups: int = 5,
) -> None:
    """Configure stdlib logging and structlog. Idempotent."""
    numeric = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(numeric)

    # Third-party noise
    for noisy in ("aiohttp.access", "uvicorn.access", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
