"""LangGraph-compatible reducers for atm.core.

Public API:
  - merge_agent_states(left, right) -> dict
  - dedup_by_id_reducer(key, *, sort_by=None) -> Callable[[list, list], list]
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, cast

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get(item: Any, field: str) -> Hashable:
    """Access a field from a dict or object attribute.

    Returns a Hashable value (as required by arch.md §3.3bis lines 565-566).
    The caller is responsible for ensuring the accessed field value is Hashable.
    Supports both dict-style access and attribute access (e.g. namedtuple, dataclass).
    """
    if isinstance(item, dict):
        return cast(Hashable, item[field])
    return cast(Hashable, getattr(item, field))


def _dedup_keep_left(items: list[Any], *, key_fn: Callable[[Any], Hashable]) -> list[Any]:
    """Stable left-wins dedup by ``key_fn`` applied to each element."""
    seen: set[Hashable] = set()
    out: list[Any] = []
    for item in items:
        k = key_fn(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def _msg_id(m: Any) -> Hashable:
    return cast(Hashable, getattr(m, "id", None) or id(m))


def _toolcall_id(t: Any) -> Hashable:
    return cast(Hashable, getattr(t, "id", None) or id(t))


def _toolresult_id(t: Any) -> Hashable:
    return cast(Hashable, getattr(t, "call_id", None) or id(t))


def _merge_one(left_agent: dict[str, Any], right_agent: dict[str, Any]) -> dict[str, Any]:
    """Merge a single agent state dict (per-agent rules)."""
    merged: dict[str, Any] = {}

    # Immutable identity fields — left wins
    for key in ("agent_id", "role"):
        merged[key] = left_agent.get(key) or right_agent.get(key)

    # Append-only list fields with unique id — dedup-by-id (left wins on id
    # collision).  Plain list-concat here causes a geometric explosion in
    # adaptive's meta-graph: each sub-graph dispatch reads the parent's
    # accumulated agent state, appends in its own internal reducer, then
    # returns the cumulative result as a "delta" to the parent.  The parent
    # reducer concatenates left + right → duplicates everything from left.
    # After N iterations the list length is 2^N - 1.  At N=12 a single agent's
    # outbox carried 4096 Message objects and the serialized blob hit 830 MB,
    # crashing the PG checkpointer with "invalid message length".
    for key, key_fn in (
        ("inbox", _msg_id),
        ("outbox", _msg_id),
        ("tool_calls", _toolcall_id),
        ("tool_results", _toolresult_id),
    ):
        combined = list(left_agent.get(key, [])) + list(right_agent.get(key, []))
        merged[key] = _dedup_keep_left(combined, key_fn=key_fn)

    # scratchpad — list of plain dicts without an inherent id.  Same
    # geometric-blowup hazard, but no key to dedup on.  Sub-graphs always
    # return a superset of the parent's scratchpad (the parent's events
    # are read as input then more are appended), so "longer side wins" is
    # the correct deltification: it drops the parent's prefix that is
    # already contained in the sub-graph's return.
    left_sp = list(left_agent.get("scratchpad", []))
    right_sp = list(right_agent.get("scratchpad", []))
    merged["scratchpad"] = right_sp if len(right_sp) >= len(left_sp) else left_sp

    # Summary — right wins if non-empty, else left
    merged["summary_before_window"] = right_agent.get("summary_before_window") or left_agent.get(
        "summary_before_window", ""
    )

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
      inbox, outbox, tool_calls, tool_results              — dedup-by-id (left + right, left wins on collision)
      scratchpad                                            — longer-list wins (no inherent id; sub-graph delta is always a superset)
      summary_before_window                                 — right wins if non-empty else left
      step_count, tokens_spent, cost_spent_usd             — max(left, right)
      agent_id, role                                       — left wins (immutable)

    Pre-fix history: ``inbox``/``outbox``/``scratchpad``/``tool_calls``/
    ``tool_results`` were merged with plain list-concat ``left + right``.
    This produced a geometric blowup in adaptive's meta-graph (a single
    agent's outbox reached 2^N - 1 entries after N meta-ticks because each
    sub-graph dispatch returned its cumulative agent state — including the
    parent's items — and the parent reducer concatenated them back in).
    After 12 iterations the agents-channel blob exceeded 830 MB and crashed
    the PG checkpointer with "invalid message length".
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
            merged.sort(key=lambda x: _get(x, sort_by))  # type: ignore[arg-type, return-value]  # _get returns Hashable; caller ensures field is also orderable (e.g. datetime, UUID)

        return merged

    return _reduce
