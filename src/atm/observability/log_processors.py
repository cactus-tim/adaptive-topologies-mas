"""Log processors for structlog. filter_secrets redacts sensitive keys."""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import Any, Union

_SECRET_RE = re.compile(r"(?i).*(api_key|token|password|secret).*")
_REDACTED = "<redacted>"

# Recursive type alias — structlog event dicts contain arbitrary values
_WalkResult = Union[
    "dict[str, Any]",
    "list[Any]",
    "tuple[Any, ...]",
    str,
    int,
    float,
    bool,
    None,
]


def _walk(obj: Any, seen: set[int]) -> Any:
    if id(obj) in seen:
        return obj
    if isinstance(obj, dict):
        seen.add(id(obj))
        return {
            k: (_REDACTED if isinstance(k, str) and _SECRET_RE.match(k) else _walk(v, seen))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        seen.add(id(obj))
        return [_walk(item, seen) for item in obj]
    if isinstance(obj, tuple):
        seen.add(id(obj))
        return tuple(_walk(item, seen) for item in obj)
    return obj


def filter_secrets(
    _logger: Any,
    _method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that redacts values whose keys match secret patterns.

    Returns a NEW dict; does not mutate input. Cycle-safe via id()-based seen-set.
    """
    seen: set[int] = set()
    result: Any = _walk(dict(event_dict), seen)
    return result  # type: ignore[no-any-return]


def configure_structlog() -> None:
    """Idempotent structlog configuration with filter_secrets enabled."""
    import structlog

    structlog.configure(
        processors=[
            filter_secrets,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )
