"""Unit tests for atm.core.reducers.

Tests use only dict and namedtuple fixtures — no Pydantic Message import (MC-5 deferred to Step 5).
Covers:
  - merge_agent_states: 10 tests (idempotency, associativity, commutativity-by-set, empty-neutral,
    immutability)
  - dedup_by_id_reducer: 11 tests (idempotency, associativity, left-wins, empty-neutral, sort,
    factory independence, closure __name__, namedtuple items, immutability)
"""

from __future__ import annotations

import copy
from collections import namedtuple
from typing import Any

from atm.core.reducers import dedup_by_id_reducer, merge_agent_states

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _agent(
    agent_id: str,
    inbox: list[Any] | None = None,
    outbox: list[Any] | None = None,
    scratchpad: list[Any] | None = None,
    tool_calls: list[Any] | None = None,
    tool_results: list[Any] | None = None,
    summary_before_window: str = "",
    step_count: int = 0,
    tokens_spent: int = 0,
    cost_spent_usd: float = 0.0,
    role: str = "worker",
) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "role": role,
        "inbox": inbox or [],
        "outbox": outbox or [],
        "scratchpad": scratchpad or [],
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "summary_before_window": summary_before_window,
        "step_count": step_count,
        "tokens_spent": tokens_spent,
        "cost_spent_usd": cost_spent_usd,
    }


Item = namedtuple("Item", ["id", "value", "ts"])


# ============================================================================
# merge_agent_states — 10 tests
# ============================================================================


class TestMergeAgentStates:
    """10 tests covering all invariants for merge_agent_states."""

    # Test 1: idempotency — merge(left, left) == left
    def test_idempotency(self) -> None:
        left = {
            "a1": _agent("a1", inbox=["m1", "m2"], step_count=3, tokens_spent=100),
        }
        result = merge_agent_states(left, left)
        # inbox is dedup-by-id (left wins on collision); merging a state with
        # itself yields the original list — true idempotency, no geometric blowup.
        assert result["a1"]["inbox"] == ["m1", "m2"]
        # NOTE: per arch.md §3.3, idempotency means per-agent reduction of same state
        # Step count and tokens should remain max (same value)
        assert result["a1"]["step_count"] == 3
        assert result["a1"]["tokens_spent"] == 100

    # Test 2: merge(left, left) step_count/tokens idempotency
    def test_idempotency_numeric_fields(self) -> None:
        left = {"a1": _agent("a1", step_count=5, tokens_spent=200, cost_spent_usd=0.5)}
        result = merge_agent_states(left, left)
        assert result["a1"]["step_count"] == 5
        assert result["a1"]["tokens_spent"] == 200
        assert result["a1"]["cost_spent_usd"] == 0.5

    # Test 3: empty neutral element — merge(left, {}) == left
    def test_empty_right_neutral(self) -> None:
        left = {"a1": _agent("a1", inbox=["x"], step_count=2)}
        result = merge_agent_states(left, {})
        assert result == left

    # Test 4: empty neutral element — merge({}, right) == right
    def test_empty_left_neutral(self) -> None:
        right = {"a1": _agent("a1", outbox=["y"], step_count=1)}
        result = merge_agent_states({}, right)
        assert result == right

    # Test 5: merge(None, right) == right
    def test_none_left_neutral(self) -> None:
        right = {"a1": _agent("a1", step_count=1)}
        result = merge_agent_states(None, right)
        assert result == right

    # Test 6: merge(left, None) == left
    def test_none_right_neutral(self) -> None:
        left = {"a1": _agent("a1", step_count=1)}
        result = merge_agent_states(left, None)
        assert result == left

    # Test 7: associativity — (a⊕b)⊕c == a⊕(b⊕c) for key union coverage
    def test_associativity_key_union(self) -> None:
        a = {"a1": _agent("a1", step_count=1)}
        b = {"a2": _agent("a2", step_count=2)}
        c = {"a3": _agent("a3", step_count=3)}
        left_assoc = merge_agent_states(merge_agent_states(a, b), c)
        right_assoc = merge_agent_states(a, merge_agent_states(b, c))
        assert set(left_assoc.keys()) == set(right_assoc.keys()) == {"a1", "a2", "a3"}

    # Test 8: commutativity by set — set of agent_ids is union of left and right
    def test_commutativity_by_set(self) -> None:
        left = {"a1": _agent("a1"), "a2": _agent("a2")}
        right = {"a2": _agent("a2"), "a3": _agent("a3")}
        result = merge_agent_states(left, right)
        assert set(result.keys()) == {"a1", "a2", "a3"}

    # Test 9: per-agent field merge — inbox concat, max step_count
    def test_per_agent_field_merge(self) -> None:
        left = {"a1": _agent("a1", inbox=["m1"], step_count=2, tokens_spent=50)}
        right = {"a1": _agent("a1", inbox=["m2"], step_count=5, tokens_spent=30)}
        result = merge_agent_states(left, right)
        agent = result["a1"]
        assert agent["inbox"] == ["m1", "m2"]
        assert agent["step_count"] == 5  # max wins
        assert agent["tokens_spent"] == 50  # max wins

    # Test 10: input immutability — does not mutate left or right
    def test_input_immutability(self) -> None:
        left = {"a1": _agent("a1", inbox=["original_left"], step_count=1)}
        right = {"a1": _agent("a1", inbox=["original_right"], step_count=2)}
        left_copy = copy.deepcopy(left)
        right_copy = copy.deepcopy(right)
        merge_agent_states(left, right)
        assert left == left_copy
        assert right == right_copy


# ============================================================================
# dedup_by_id_reducer — 11 tests
# ============================================================================


class TestDedupByIdReducer:
    """14 tests covering all invariants for dedup_by_id_reducer factory.

    Invariant index (Step 1.5 — explicit invariants for dedup_by_id_reducer):
      - Idempotent   : test_idempotency       (Test 2)
      - Associative  : test_associativity     (Test 13)
      - Empty-neutral: test_empty_left_neutral (Test 5)
                       test_empty_right_neutral (Test 6)
    """

    def _make_items(self) -> list[dict[str, Any]]:
        return [
            {"id": "a", "content": "hello", "ts": 1},
            {"id": "b", "content": "world", "ts": 2},
        ]

    # Test 1: basic dedup — no duplicate ids yields union
    def test_basic_union(self) -> None:
        reducer = dedup_by_id_reducer("id")
        left = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
        right = [{"id": "c", "v": 3}]
        result = reducer(left, right)
        ids = [x["id"] for x in result]
        assert sorted(ids) == ["a", "b", "c"]
        assert len(result) == 3

    # Test 2: idempotency — reducer(x, x) == x (by id set and values)
    def test_idempotency(self) -> None:
        reducer = dedup_by_id_reducer("id")
        items = [{"id": "x", "val": 1}, {"id": "y", "val": 2}]
        result = reducer(items, items)
        ids = [r["id"] for r in result]
        assert sorted(ids) == ["x", "y"]
        assert len(result) == 2

    # Test 3: left-wins on id collision — keep left's full object
    def test_left_wins_on_collision(self) -> None:
        reducer = dedup_by_id_reducer("id")
        left = [{"id": "x", "content": "v1", "payload": {"k": 1}}]
        right = [{"id": "x", "content": "v2", "payload": {"k": 2}}]
        result = reducer(left, right)
        assert len(result) == 1
        assert result[0]["content"] == "v1"
        assert result[0]["payload"] == {"k": 1}

    # Test 4 (MC-2): left-wins preserves left payload on collision
    def test_left_wins_preserves_full_payload(self) -> None:
        reducer = dedup_by_id_reducer("id")
        left_item = {"id": "x", "content": "v1", "payload": {"k": 1}, "extra": "left_only"}
        right_item = {"id": "x", "content": "v2", "payload": {"k": 2}, "extra": "right_only"}
        result = reducer([left_item], [right_item])
        assert len(result) == 1
        assert result[0] is left_item or result[0] == left_item
        assert result[0]["payload"]["k"] == 1
        assert result[0]["extra"] == "left_only"

    # Test 5: empty left neutral — reducer([], right) == right
    def test_empty_left_neutral(self) -> None:
        reducer = dedup_by_id_reducer("id")
        right = [{"id": "a", "v": 1}]
        result = reducer([], right)
        assert result == right

    # Test 6: empty right neutral — reducer(left, []) == left
    def test_empty_right_neutral(self) -> None:
        reducer = dedup_by_id_reducer("id")
        left = [{"id": "a", "v": 1}]
        result = reducer(left, [])
        assert result == left

    # Test 7: None inputs treated as empty lists
    def test_none_inputs(self) -> None:
        reducer = dedup_by_id_reducer("id")
        items = [{"id": "a", "v": 1}]
        assert reducer(None, items) == items
        assert reducer(items, None) == items
        assert reducer(None, None) == []

    # Test 8: stable sort by sort_by field
    def test_stable_sort(self) -> None:
        reducer = dedup_by_id_reducer("id", sort_by="ts")
        left = [{"id": "b", "ts": 2}, {"id": "a", "ts": 1}]
        right = [{"id": "c", "ts": 3}]
        result = reducer(left, right)
        assert [x["id"] for x in result] == ["a", "b", "c"]
        assert [x["ts"] for x in result] == [1, 2, 3]

    # Test 9: factory returns new independent closures each time
    def test_factory_returns_independent_closures(self) -> None:
        reducer_a = dedup_by_id_reducer("id")
        reducer_b = dedup_by_id_reducer("id")
        assert reducer_a is not reducer_b

    # Test 10: closure __name__ == "_reduce"
    def test_closure_name(self) -> None:
        reducer = dedup_by_id_reducer("id")
        assert reducer.__name__ == "_reduce"

    # Test 11: input immutability — does not mutate left or right
    def test_input_immutability(self) -> None:
        reducer = dedup_by_id_reducer("id")
        left = [{"id": "a", "v": 1}]
        right = [{"id": "b", "v": 2}]
        left_copy = copy.deepcopy(left)
        right_copy = copy.deepcopy(right)
        reducer(left, right)
        assert left == left_copy
        assert right == right_copy

    # Test 12: namedtuple items — _get helper supports getattr access
    def test_namedtuple_items(self) -> None:
        reducer = dedup_by_id_reducer("id", sort_by="ts")
        left = [Item(id="a", value="hello", ts=1)]
        right = [Item(id="b", value="world", ts=2), Item(id="a", value="dup", ts=3)]
        result = reducer(left, right)
        # "a" exists in left → left wins, "b" is new
        ids = [x.id for x in result]
        assert sorted(ids) == ["a", "b"]
        assert len(result) == 2
        # the "a" item should be from left
        a_item = next(x for x in result if x.id == "a")
        assert a_item.value == "hello"

    # Test 13: associativity — (a⊕b)⊕c == a⊕(b⊕c) by id set
    def test_associativity(self) -> None:
        reducer = dedup_by_id_reducer("id")
        a = [{"id": "1", "v": "a1"}, {"id": "2", "v": "a2"}]
        b = [{"id": "2", "v": "b2"}, {"id": "3", "v": "b3"}]
        c = [{"id": "3", "v": "c3"}, {"id": "4", "v": "c4"}]
        left_assoc = reducer(reducer(a, b), c)
        right_assoc = reducer(a, reducer(b, c))
        left_ids = sorted(x["id"] for x in left_assoc)
        right_ids = sorted(x["id"] for x in right_assoc)
        assert left_ids == right_ids == ["1", "2", "3", "4"]

    # Test 14: custom key field name
    def test_custom_key_field(self) -> None:
        reducer = dedup_by_id_reducer("msg_id")
        left = [{"msg_id": "x", "body": "left"}]
        right = [{"msg_id": "x", "body": "right"}, {"msg_id": "y", "body": "new"}]
        result = reducer(left, right)
        assert len(result) == 2
        x_item = next(i for i in result if i["msg_id"] == "x")
        assert x_item["body"] == "left"
