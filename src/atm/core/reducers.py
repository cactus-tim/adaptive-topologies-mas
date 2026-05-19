"""LangGraph-compatible reducers for atm.core."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, cast


def _get(item: Any, field: str) -> Hashable:
    """Access a field from a dict or object attribute (dict-style or getattr)."""
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

    for key in ("agent_id", "role"):
        merged[key] = left_agent.get(key) or right_agent.get(key)

    for key, key_fn in (
        ("inbox", _msg_id),
        ("outbox", _msg_id),
        ("tool_calls", _toolcall_id),
        ("tool_results", _toolresult_id),
    ):
        combined = list(left_agent.get(key, [])) + list(right_agent.get(key, []))
        merged[key] = _dedup_keep_left(combined, key_fn=key_fn)

    left_sp = list(left_agent.get("scratchpad", []))
    right_sp = list(right_agent.get("scratchpad", []))
    merged["scratchpad"] = right_sp if len(right_sp) >= len(left_sp) else left_sp

    merged["summary_before_window"] = right_agent.get("summary_before_window") or left_agent.get(
        "summary_before_window", ""
    )

    for key in ("step_count", "tokens_spent"):
        merged[key] = max(left_agent.get(key, 0), right_agent.get(key, 0))
    merged["cost_spent_usd"] = max(
        left_agent.get("cost_spent_usd", 0.0), right_agent.get("cost_spent_usd", 0.0)
    )

    return merged


def merge_agent_states(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge two dicts of agent_id -> AgentState.

    Per-key rules: inbox/outbox/tool_calls/tool_results — dedup-by-id (left wins on collision);
    scratchpad — longer-list wins; summary_before_window — right wins if non-empty else left;
    step_count/tokens_spent/cost_spent_usd — max; agent_id/role — left wins.
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
    """Factory returning a reducer that deduplicates a list by ``key`` (left wins on collision)."""

    def _reduce(
        left: list[Any] | None,
        right: list[Any] | None,
    ) -> list[Any]:
        if not left:
            return list(right or [])
        if not right:
            return list(left)

        seen: set[Hashable] = {_get(x, key) for x in left}
        merged = list(left) + [x for x in right if _get(x, key) not in seen]

        if sort_by is not None:
            merged.sort(key=lambda x: _get(x, sort_by))  # type: ignore[arg-type, return-value]  # _get returns Hashable; caller ensures field is also orderable (e.g. datetime, UUID)

        return merged

    return _reduce
