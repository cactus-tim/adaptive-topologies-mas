"""Signal emission helpers for ATM adaptive topology layer (M8.5).

Defines signal key constants, the ``emit_signal`` helper, and
``increment_signal`` for counter signals (e.g. rejected_count).

Signal keys:
    STUCK                  — bool, set by Executor after N failed code_run calls
    REJECTED_COUNT         — int, incremented by Critic on each reject decision
    NEEDS_DEBATE           — bool, set by Critic when rejected_count >= 3
    READY_FOR_EXECUTION    — bool, set by Planner upon plan finalisation
    READY_FOR_VERIFICATION — bool, set by Executor after first successful code_run
    CRITIC_APPROVED        — bool, set by Critic on approve decision

Usage::

    from atm.phases.signals import (
        emit_signal,
        increment_signal,
        STUCK,
        REJECTED_COUNT,
        NEEDS_DEBATE,
        READY_FOR_EXECUTION,
        READY_FOR_VERIFICATION,
        CRITIC_APPROVED,
    )

    # In an agent step, after building the updated SharedState dict:
    new_shared = emit_signal(shared, READY_FOR_EXECUTION, True)

Note on dispatch_custom_event:
    The LangGraph ``dispatch_custom_event`` API (``langgraph.types``) is not
    available in the installed version of langgraph (>=0.3,<1).  When it
    becomes available, replace the TODO block below with:

        from langgraph.types import dispatch_custom_event
        dispatch_custom_event("signal_emit", {"key": key, "value": value, "at_iter": at_iter})

    Until then, signal emission is a pure state update with a DEBUG log.
    Downstream consumers (TopologyRouter, analytics) read signals directly
    from SharedState.signals.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal key constants
# ---------------------------------------------------------------------------

STUCK: Literal["stuck"] = "stuck"
REJECTED_COUNT: Literal["rejected_count"] = "rejected_count"
NEEDS_DEBATE: Literal["needs_debate"] = "needs_debate"
READY_FOR_EXECUTION: Literal["ready_for_execution"] = "ready_for_execution"
READY_FOR_VERIFICATION: Literal["ready_for_verification"] = "ready_for_verification"
CRITIC_APPROVED: Literal["critic_approved"] = "critic_approved"

# Union type for all known signal keys (for use in type annotations)
SignalKey = Literal[
    "stuck",
    "rejected_count",
    "needs_debate",
    "ready_for_execution",
    "ready_for_verification",
    "critic_approved",
]

# ---------------------------------------------------------------------------
# emit_signal — pure function; returns new SharedState dict
# ---------------------------------------------------------------------------


def emit_signal(shared: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Emit a signal by updating the shared state's signals dict.

    Returns a *new* dict (immutable-style update) — the caller's reference
    is not mutated.  The caller must reassign the result, e.g.::

        shared = emit_signal(shared, STUCK, True)

    Args:
        shared: The current SharedState dict (or any dict with an optional
                ``signals`` and ``iter_total`` key).
        key:    Signal key (use the module-level constants for safety).
        value:  New signal value (bool for flags, int for counters).

    Returns:
        Updated shallow copy of *shared* with ``signals[key] = value``.

    Side effects:
        Logs a DEBUG message with key/value/iter_total.

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

    # Build updated shared dict (shallow copy — only signals is replaced)
    updated = dict(shared)
    updated["signals"] = existing_signals
    return updated


# ---------------------------------------------------------------------------
# increment_signal — convenience helper for integer counter signals
# ---------------------------------------------------------------------------


def increment_signal(shared: dict[str, Any], key: str) -> tuple[dict[str, Any], int]:
    """Increment an integer signal counter and return (new_shared, new_value).

    The counter defaults to 0 if the key is absent or has a non-integer value.

    Args:
        shared: Current SharedState dict.
        key:    Signal key to increment (typically ``REJECTED_COUNT``).

    Returns:
        Tuple of (updated SharedState dict, new counter value).

    Example::

        shared, count = increment_signal(shared, REJECTED_COUNT)
        if count >= 3:
            shared = emit_signal(shared, NEEDS_DEBATE, True)
    """
    current: int
    existing_signals: dict[str, Any] = dict(shared.get("signals") or {})
    try:
        current = int(existing_signals.get(key) or 0)
    except (TypeError, ValueError):
        current = 0

    new_value = current + 1
    return emit_signal(shared, key, new_value), new_value
