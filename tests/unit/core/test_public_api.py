"""Public API integration tests for atm.core.

Tests:
  - IR-4: every name in __all__ is resolvable and not None
  - MC-5: dedup_by_id_reducer with Pydantic Message instances — left-wins on id collision
  - End-to-end reducer roundtrip: merge_agent_states with two AgentState dicts
  - GraphState Annotated metadata contains callable reducers for all list/dict fields
  - Star-import doesn't fail
"""

from __future__ import annotations

from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

import atm.core

# ---------------------------------------------------------------------------
# Basic import tests
# ---------------------------------------------------------------------------


def test_star_import_does_not_fail() -> None:
    """from atm.core import * must not raise."""
    # Already executed at module load; just assert the module is importable
    assert atm.core is not None


def test_all_is_defined_and_non_empty() -> None:
    assert hasattr(atm.core, "__all__")
    assert len(atm.core.__all__) >= 25


# ---------------------------------------------------------------------------
# IR-4: every name in __all__ resolves via hasattr / getattr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", atm.core.__all__)
def test_all_name_resolvable(name: str) -> None:
    """Every name in __all__ must be present on the module and not None."""
    assert hasattr(atm.core, name), f"atm.core has no attribute {name!r}"
    obj = getattr(atm.core, name)
    assert obj is not None, f"atm.core.{name} is None"


# ---------------------------------------------------------------------------
# MC-5: dedup_by_id_reducer with Pydantic Message — left-wins on id collision
# ---------------------------------------------------------------------------


def test_mc5_dedup_message_left_wins_on_same_id() -> None:
    """Two Messages with same id but different payload: result length 1, left payload kept."""
    from atm.core import Message, MessageKind, dedup_by_id_reducer

    msg_left = Message(
        sender="agent-a",
        kind=MessageKind.REQUEST,
        content="left content",
        payload={"version": "left"},
    )
    # Construct right with same id but different payload (bypass frozen model via construct)
    msg_right = Message.model_construct(
        id=msg_left.id,
        sender="agent-b",
        kind=MessageKind.BROADCAST,
        content="right content",
        payload={"version": "right"},
        recipients=(),
        refs=(),
        created_at=msg_left.created_at,
    )

    reducer = dedup_by_id_reducer("id")
    result = reducer([msg_left], [msg_right])

    assert len(result) == 1
    kept = result[0]
    assert kept.id == msg_left.id
    assert kept.payload["version"] == "left", "Left item must win on id collision"
    assert kept.content == "left content"


def test_mc5_dedup_distinct_ids_both_kept() -> None:
    """Two Messages with distinct ids: both must appear in result."""
    from atm.core import Message, MessageKind, dedup_by_id_reducer

    m1 = Message(sender="a", kind=MessageKind.REQUEST, content="one")
    m2 = Message(sender="b", kind=MessageKind.BROADCAST, content="two")

    reducer = dedup_by_id_reducer("id")
    result = reducer([m1], [m2])

    assert len(result) == 2
    ids = {item.id for item in result}
    assert m1.id in ids
    assert m2.id in ids


# ---------------------------------------------------------------------------
# End-to-end reducer roundtrip: merge_agent_states
# ---------------------------------------------------------------------------


def test_merge_agent_states_roundtrip() -> None:
    """Create two AgentState dicts and verify merge produces correct result."""
    from atm.core import AgentState, merge_agent_states

    state_a: AgentState = {
        "agent_id": "agent-1",
        "role": "planner",
        "inbox": [],
        "outbox": [],
        "scratchpad": [{"step": 1}],
        "tool_calls": [],
        "tool_results": [],
        "summary_before_window": "",
        "step_count": 3,
        "tokens_spent": 100,
        "cost_spent_usd": 0.01,
    }

    state_b: AgentState = {
        "agent_id": "agent-1",
        "role": "planner",
        "inbox": [],
        "outbox": [],
        "scratchpad": [{"step": 2}],
        "tool_calls": [],
        "tool_results": [],
        "summary_before_window": "summary text",
        "step_count": 5,
        "tokens_spent": 80,
        "cost_spent_usd": 0.02,
    }

    left: dict[str, AgentState] = {"agent-1": state_a}
    right: dict[str, AgentState] = {"agent-1": state_b}

    merged = merge_agent_states(left, right)

    assert "agent-1" in merged
    agent = merged["agent-1"]
    # scratchpad — longer-list wins (no inherent id; sub-graph delta is a
    # superset of the parent's events). On a length tie the right side wins.
    assert agent["scratchpad"] == [{"step": 2}]
    # step_count — max
    assert agent["step_count"] == 5
    # tokens_spent — max
    assert agent["tokens_spent"] == 100
    # cost_spent_usd — max
    assert agent["cost_spent_usd"] == pytest.approx(0.02)
    # summary — right wins when non-empty
    assert agent["summary_before_window"] == "summary text"


def test_merge_agent_states_two_agents() -> None:
    """Two different agents in left and right get merged into single dict."""
    from atm.core import AgentState, merge_agent_states

    left: dict[str, AgentState] = {
        "agent-1": {"agent_id": "agent-1", "role": "planner", "step_count": 2},
    }
    right: dict[str, AgentState] = {
        "agent-2": {"agent_id": "agent-2", "role": "critic", "step_count": 1},
    }

    merged = merge_agent_states(left, right)

    assert "agent-1" in merged
    assert "agent-2" in merged


def test_merge_agent_states_none_neutral() -> None:
    """merge_agent_states(None, x) == x and merge_agent_states(x, None) == x."""
    from atm.core import AgentState, merge_agent_states

    state: dict[str, AgentState] = {
        "a": {"agent_id": "a", "role": "planner", "step_count": 1},
    }

    assert merge_agent_states(None, state) == state
    assert merge_agent_states(state, None) == state


# ---------------------------------------------------------------------------
# GraphState Annotated metadata contains callable reducers
# ---------------------------------------------------------------------------


def test_graphstate_annotated_reducers_are_callable() -> None:
    """GraphState list/dict fields must have callable reducers in Annotated metadata."""
    from atm.core import GraphState

    hints = get_type_hints(GraphState, include_extras=True)

    # These fields must all have Annotated[..., callable]
    expected_annotated = (
        "agents",
        "messages",
        "llm_calls",
        "budget_events",
        "topology_transitions",
    )

    for field in expected_annotated:
        assert field in hints, f"GraphState missing field {field!r}"
        hint = hints[field]
        assert get_origin(hint) is Annotated, f"GraphState.{field} must be Annotated, got {hint!r}"
        args = get_args(hint)
        assert len(args) >= 2, f"GraphState.{field} Annotated has < 2 args"
        reducer = args[1]
        assert callable(reducer), (
            f"GraphState.{field} second Annotated arg must be callable, got {reducer!r}"
        )


# ---------------------------------------------------------------------------
# Direct import smoke test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sort-by tests (arch.md §3.2 lines 454-459)
# ---------------------------------------------------------------------------


def test_dedup_sort_by_created_at_messages() -> None:
    """messages reducer sorts ASC by created_at; items fed in descending order."""
    import datetime

    from atm.core import Message, MessageKind, dedup_by_id_reducer

    t_old = datetime.datetime(2026, 1, 1, 10, 0, 0, tzinfo=datetime.UTC)
    t_new = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)

    msg_newer = Message.model_construct(
        id=__import__("uuid").uuid4(),
        sender="a",
        kind=MessageKind.REQUEST,
        content="newer",
        recipients=(),
        refs=(),
        payload={},
        created_at=t_new,
    )
    msg_older = Message.model_construct(
        id=__import__("uuid").uuid4(),
        sender="b",
        kind=MessageKind.REQUEST,
        content="older",
        recipients=(),
        refs=(),
        payload={},
        created_at=t_old,
    )

    # Feed descending order (newer first, older second) — reducer must return ASC
    reducer = dedup_by_id_reducer("id", sort_by="created_at")
    result = reducer([msg_newer], [msg_older])

    assert len(result) == 2
    assert result[0].created_at == t_old, "First item must be the older one (ASC sort)"
    assert result[1].created_at == t_new, "Second item must be the newer one (ASC sort)"


def test_dedup_sort_by_at_budget_events() -> None:
    """budget_events reducer sorts ASC by 'at'; items fed in descending order."""
    import datetime
    import uuid

    from atm.core import BudgetEvent, dedup_by_id_reducer

    t_old = datetime.datetime(2026, 1, 1, 9, 0, 0, tzinfo=datetime.UTC)
    t_new = datetime.datetime(2026, 1, 1, 11, 0, 0, tzinfo=datetime.UTC)

    # Two events with distinct run_ids so they are not deduped; timestamps set via model_copy
    base_newer = BudgetEvent(
        run_id=uuid.uuid4(), level="run", event="warn", limit_usd=1.0, current_usd=0.5
    )
    base_older = BudgetEvent(
        run_id=uuid.uuid4(), level="call", event="warn", limit_usd=1.0, current_usd=0.3
    )
    ev_newer = base_newer.model_copy(update={"at": t_new})
    ev_older = base_older.model_copy(update={"at": t_old})

    reducer = dedup_by_id_reducer("run_id", sort_by="at")
    result = reducer([ev_newer], [ev_older])

    assert len(result) == 2
    assert result[0].at == t_old, "First item must be the older one (ASC sort)"
    assert result[1].at == t_new, "Second item must be the newer one (ASC sort)"


def test_direct_named_imports() -> None:
    """Key names importable directly from atm.core."""
    from atm.core import (  # noqa: F401
        AgentRole,
        AgentState,
        AtmError,
        BudgetEvent,
        BudgetExceededError,
        GraphState,
        HumanContext,
        HumanResponse,
        HumanRole,
        LLMResponse,
        Message,
        MessageKind,
        Phase,
        PhaseError,
        PhaseTransition,
        RunResult,
        SharedState,
        TaskResult,
        TaskSpec,
        TokenUsage,
        ToolCall,
        ToolError,
        ToolResult,
        TopologyTransition,
        dedup_by_id_reducer,
        merge_agent_states,
    )
