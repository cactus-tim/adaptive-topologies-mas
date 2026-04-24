"""Unit tests for src/atm/core/state.py.

Coverage:
  - AgentState TypedDict structure (fields, total=False)
  - SharedState TypedDict structure (all L2 fields)
  - GraphState TypedDict structure with Annotated reducers
  - MC-3: closure name check for dedup_by_id_reducer-produced reducers
  - MC-4: that Annotated metadata contains callable reducers
  - get_type_hints with include_extras=True returns Annotated
  - M2 step 1.3: llm_calls reducer sorts by started_at
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, get_args, get_origin, get_type_hints
from uuid import uuid4

from atm.core.state import AgentState, GraphState, SharedState

# ---------------------------------------------------------------------------
# 1. AgentState structure
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_is_typeddict(self) -> None:
        assert issubclass(AgentState, dict)

    def test_has_agent_id_field(self) -> None:
        hints = get_type_hints(AgentState)
        assert "agent_id" in hints

    def test_has_inbox_outbox_scratchpad(self) -> None:
        hints = get_type_hints(AgentState)
        assert "inbox" in hints
        assert "outbox" in hints
        assert "scratchpad" in hints

    def test_has_numeric_fields(self) -> None:
        hints = get_type_hints(AgentState)
        assert "step_count" in hints
        assert "tokens_spent" in hints
        assert "cost_spent_usd" in hints

    def test_total_false_allows_empty_creation(self) -> None:
        # AgentState is total=False — all keys optional
        state: AgentState = {}  # type: ignore[typeddict-item]
        assert isinstance(state, dict)

    def test_can_instantiate_with_values(self) -> None:
        state: AgentState = {  # type: ignore[typeddict-item]
            "agent_id": "agent_1",
            "role": "planner",
            "step_count": 5,
        }
        assert state["agent_id"] == "agent_1"
        assert state["step_count"] == 5


# ---------------------------------------------------------------------------
# 2. SharedState structure
# ---------------------------------------------------------------------------


class TestSharedState:
    def test_is_typeddict(self) -> None:
        assert issubclass(SharedState, dict)

    def test_has_phase_field(self) -> None:
        hints = get_type_hints(SharedState)
        assert "phase" in hints

    def test_has_topology_fields(self) -> None:
        hints = get_type_hints(SharedState)
        # Required L2 topology fields — field name matches arch.md §3.2 line 419
        assert "active_topology" in hints
        assert "topology_switch_count" in hints
        assert "topology_history" in hints

    def test_has_iter_fields(self) -> None:
        hints = get_type_hints(SharedState)
        assert "iter_total" in hints
        assert "phase_started_at_iter" in hints
        assert "topology_started_at_iter" in hints

    def test_has_signals_field(self) -> None:
        hints = get_type_hints(SharedState)
        assert "signals" in hints

    def test_total_false_allows_empty_creation(self) -> None:
        state: SharedState = {}  # type: ignore[typeddict-item]
        assert isinstance(state, dict)


# ---------------------------------------------------------------------------
# 3. GraphState structure
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_is_typeddict(self) -> None:
        assert issubclass(GraphState, dict)

    def test_has_required_top_level_keys(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        assert "agents" in hints
        assert "messages" in hints
        assert "llm_calls" in hints
        assert "budget_events" in hints
        assert "topology_transitions" in hints

    def test_agents_annotated_with_merge_reducer(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        agents_hint = hints["agents"]
        assert get_origin(agents_hint) is Annotated
        args = get_args(agents_hint)
        # args[0] is the type, args[1] is the reducer callable
        assert callable(args[1])
        assert args[1].__name__ == "merge_agent_states"

    def test_messages_annotated_with_dedup_reducer(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        msg_hint = hints["messages"]
        assert get_origin(msg_hint) is Annotated
        args = get_args(msg_hint)
        assert callable(args[1])

    def test_llm_calls_annotated_with_dedup_reducer(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        hint = hints["llm_calls"]
        assert get_origin(hint) is Annotated
        args = get_args(hint)
        assert callable(args[1])

    def test_budget_events_annotated(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        hint = hints["budget_events"]
        assert get_origin(hint) is Annotated
        args = get_args(hint)
        assert callable(args[1])

    def test_topology_transitions_annotated(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        hint = hints["topology_transitions"]
        assert get_origin(hint) is Annotated
        args = get_args(hint)
        assert callable(args[1])


# ---------------------------------------------------------------------------
# 4. MC-3: closure name check
# ---------------------------------------------------------------------------


class TestMC3ClosureName:
    """MC-3: factory-produced reducers must be named `_reduce`;
    the bare merge_agent_states must be named `merge_agent_states`.
    """

    def test_dedup_reducer_closure_name_is_reduce(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        # All dedup-based reducers (messages, llm_calls, budget_events, topology_transitions)
        for field in ("messages", "llm_calls", "budget_events", "topology_transitions"):
            hint = hints[field]
            reducer = get_args(hint)[1]
            assert reducer.__name__ == "_reduce", (
                f"Field '{field}': expected reducer name '_reduce', got '{reducer.__name__}'"
            )

    def test_merge_agent_states_reducer_name(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        agents_hint = hints["agents"]
        reducer = get_args(agents_hint)[1]
        assert reducer.__name__ == "merge_agent_states"


# ---------------------------------------------------------------------------
# 5. MC-4: reducers are callable (runtime check)
# ---------------------------------------------------------------------------


class TestMC4ReducersCallable:
    """MC-4: every Annotated reducer in GraphState must be callable at runtime."""

    def test_all_reducers_are_callable(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        reducer_fields = (
            "agents",
            "messages",
            "llm_calls",
            "budget_events",
            "topology_transitions",
        )
        for field in reducer_fields:
            hint = hints[field]
            assert get_origin(hint) is Annotated, f"Field '{field}' is not Annotated"
            reducer = get_args(hint)[1]
            assert callable(reducer), f"Field '{field}' reducer is not callable"

    def test_dedup_reducers_work_with_empty_lists(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        for field in ("messages", "llm_calls", "budget_events", "topology_transitions"):
            reducer = get_args(hints[field])[1]
            result = reducer([], [])
            assert result == []

    def test_merge_agent_states_reducer_works_with_empty_dicts(self) -> None:
        hints = get_type_hints(GraphState, include_extras=True)
        reducer = get_args(hints["agents"])[1]
        result = reducer({}, {})
        assert result == {}


# ---------------------------------------------------------------------------
# 6. M2 step 1.3: llm_calls reducer sorts by started_at
# ---------------------------------------------------------------------------


class TestLlmCallsReducerSortByStartedAt:
    """M2 step 1.3: GraphState.llm_calls reducer must sort by started_at ASC."""

    def _make_llm_response(self, started_at: datetime) -> dict:
        """Create a minimal dict that mimics LLMResponse for reducer testing."""
        return {"id": uuid4(), "started_at": started_at}

    def test_llm_calls_reducer_sorts_ascending(self) -> None:
        """Reducer must return items in started_at ascending order."""
        hints = get_type_hints(GraphState, include_extras=True)
        reducer = get_args(hints["llm_calls"])[1]

        now = datetime.now(UTC)
        early = self._make_llm_response(started_at=now - timedelta(seconds=10))
        late = self._make_llm_response(started_at=now)

        # Pass late first in left, early in right — result must be sorted ASC
        result = reducer([late], [early])
        assert len(result) == 2
        assert result[0]["started_at"] < result[1]["started_at"]
        assert result[0]["started_at"] == early["started_at"]
        assert result[1]["started_at"] == late["started_at"]

    def test_llm_calls_reducer_sorts_with_multiple_items(self) -> None:
        """Reducer must sort correctly across many items with different timestamps."""
        hints = get_type_hints(GraphState, include_extras=True)
        reducer = get_args(hints["llm_calls"])[1]

        now = datetime.now(UTC)
        items = [
            self._make_llm_response(started_at=now + timedelta(seconds=i))
            for i in [3, 1, 4, 1, 5, 9, 2, 6]
        ]
        # Shuffle: put some in left, some in right
        # Dedup by id means all unique IDs so all 8 kept
        result = reducer(items[:4], items[4:])
        started_ats = [r["started_at"] for r in result]
        assert started_ats == sorted(started_ats)

    def test_llm_calls_reducer_dedup_by_id(self) -> None:
        """Reducer must still deduplicate by id (left-wins) even with sort_by."""
        hints = get_type_hints(GraphState, include_extras=True)
        reducer = get_args(hints["llm_calls"])[1]

        shared_id = uuid4()
        now = datetime.now(UTC)
        left_item = {"id": shared_id, "started_at": now - timedelta(seconds=5)}
        right_item = {"id": shared_id, "started_at": now}  # same id, left wins

        result = reducer([left_item], [right_item])
        assert len(result) == 1
        assert result[0]["started_at"] == left_item["started_at"]  # left wins
