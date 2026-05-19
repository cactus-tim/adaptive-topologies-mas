"""Signal emission helpers (emit_signal, increment_signal) and signal key constants."""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

STUCK: Literal["stuck"] = "stuck"
REJECTED_COUNT: Literal["rejected_count"] = "rejected_count"
NEEDS_DEBATE: Literal["needs_debate"] = "needs_debate"
READY_FOR_EXECUTION: Literal["ready_for_execution"] = "ready_for_execution"
READY_FOR_VERIFICATION: Literal["ready_for_verification"] = "ready_for_verification"
CRITIC_APPROVED: Literal["critic_approved"] = "critic_approved"

SignalKey = Literal[
    "stuck",
    "rejected_count",
    "needs_debate",
    "ready_for_execution",
    "ready_for_verification",
    "critic_approved",
]


def emit_signal(shared: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Return a shallow copy of *shared* with ``signals[key] = value``; logs DEBUG.

    The caller must reassign: ``shared = emit_signal(shared, STUCK, True)``.

    TODO(M11): When ``langgraph.types.dispatch_custom_event`` becomes
    available in the installed langgraph version, add a call here:
        dispatch_custom_event(
            "signal_emit",
            {"key": key, "value": value, "at_iter": at_iter},
        )
    This will feed the analytics event stream consumed by M11.
    """
    existing_signals: dict[str, Any] = dict(shared.get("signals") or {})
    existing_signals[key] = value

    at_iter: int = int(shared.get("iter_total") or 0)

    logger.debug(
        "signal_emit",
        extra={"key": key, "value": value, "at_iter": at_iter},
    )

    updated = dict(shared)
    updated["signals"] = existing_signals
    return updated


def increment_signal(shared: dict[str, Any], key: str) -> tuple[dict[str, Any], int]:
    """Increment integer signal *key* and return ``(new_shared, new_value)``.

    Defaults to 0 if the key is absent or non-integer.
    """
    current: int
    existing_signals: dict[str, Any] = dict(shared.get("signals") or {})
    try:
        current = int(existing_signals.get(key) or 0)
    except (TypeError, ValueError):
        current = 0

    new_value = current + 1
    return emit_signal(shared, key, new_value), new_value
