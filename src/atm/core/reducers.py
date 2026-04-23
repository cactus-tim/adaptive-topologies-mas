"""LangGraph-compatible reducers for atm.core.

Public API:
  - merge_agent_states(left, right) -> dict
  - dedup_by_id_reducer(key, *, sort_by=None) -> Callable[[list, list], list]
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(item: Any, field: str) -> Any:
    """Access a field from a dict or object attribute.

    Supports both dict-style access and attribute access (e.g. namedtuple, dataclass).
    """
    if isinstance(item, dict):
        return item[field]
    return getattr(item, field)


def _merge_one(left_agent: dict[str, Any], right_agent: dict[str, Any]) -> dict[str, Any]:
    """Merge a single agent state dict (per-agent rules)."""
    merged: dict[str, Any] = {}

    # Immutable identity fields — left wins
    for key in ("agent_id", "role"):
        merged[key] = left_agent.get(key) or right_agent.get(key)

    # Append-only list fields — concat left + right
    for key in ("inbox", "outbox", "scratchpad", "tool_calls", "tool_results"):
        merged[key] = list(left_agent.get(key, [])) + list(right_agent.get(key, []))

    # Summary — right wins if non-empty, else left
    merged["summary_before_window"] = right_agent.get(
        "summary_before_window"
    ) or left_agent.get("summary_before_window", "")

    # Monotonically increasing numeric fields — max wins
    for key in ("step_count", "tokens_spent"):
        merged[key] = max(left_agent.get(key, 0), right_agent.get(key, 0))
    merged["cost_spent_usd"] = max(
        left_agent.get("cost_spent_usd", 0.0), right_agent.get("cost_spent_usd", 0.0)
    )

    return merged


# ---------------------------------------------------------------------------
# Public reducers
# ---------------------------------------------------------------------------


def merge_agent_states(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge two dicts of agent_id -> AgentState.

    Invariants (from arch.md §3.3):
      1. Idempotency (per numeric/identity fields): max and left-wins are stable.
      2. Associativity: holds for numeric max and list concat.
      3. Empty/None neutral element: merge({}/None, x) == x; merge(x, {}/None) == x.
      4. Input immutability: does not mutate left or right.

    Per-key rules:
      inbox, outbox, scratchpad, tool_calls, tool_results  — list concat (left + right)
      summary_before_window                                 — right wins if non-empty else left
      step_count, tokens_spent, cost_spent_usd             — max(left, right)
      agent_id, role                                       — left wins (immutable)
    """
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)

    out: dict[str, Any] = {}
    for aid in set(left) | set(right):
        la = left.get(aid, {})
        ra = right.get(aid, {})
        out[aid] = _merge_one(la, ra)
    return out


def dedup_by_id_reducer(
    key: str = "id",
    *,
    sort_by: str | None = None,
) -> Callable[[list[Any] | None, list[Any] | None], list[Any]]:
    """Factory that returns a reducer closure deduplicating a list by a unique id field.

    Invariants (from arch.md §3.3bis):
      1. Idempotency: reducer(x, x) == x  (by id set).
      2. Associativity: ((a⊕b)⊕c) id-set == (a⊕(b⊕c)) id-set.
      3. Left-wins on id collision: the element from `left` is kept in full.
      4. Empty/None neutral element: reducer([], x) == x; reducer(x, []) == x.
      5. Optional stable sort: if sort_by is given, result sorted ASC by that field.

    The returned closure is named `_reduce` (checked by Step 4 test MC-3).
    """

    def _reduce(
        left: list[Any] | None,
        right: list[Any] | None,
    ) -> list[Any]:
        if not left:
            return list(right or [])
        if not right:
            return list(left)

        # Build seen set from left (left-wins: skip right items whose id already present)
        seen: set[Hashable] = {_get(x, key) for x in left}
        merged = list(left) + [x for x in right if _get(x, key) not in seen]

        if sort_by is not None:
            merged.sort(key=lambda x: _get(x, sort_by))

        return merged

    return _reduce
