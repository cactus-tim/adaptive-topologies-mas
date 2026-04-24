"""Unit tests for atm.observability.serializers — 6 pure row-conversion functions.

For each of the 6 serializers the test suite checks:
1. test_row_keys_match_schema — set(row.keys()) == set(SCHEMA.names)
2. test_pa_table_from_row — pa.Table.from_pylist([row], schema=SCHEMA) must not raise
3. test_tz_aware_preserved — timestamp field tzinfo is not None after conversion
4. test_json_fields_deterministic — same input → same *_json values on two calls

Additional _dumps tests:
- test_dumps_sort_keys
- test_dumps_unicode
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pyarrow as pa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _make_run_id() -> UUID:
    return uuid4()


# ---------------------------------------------------------------------------
# _dumps tests
# ---------------------------------------------------------------------------


def test_dumps_sort_keys() -> None:
    """_dumps must produce sorted keys in alphabetical order."""
    from atm.observability.serializers import _dumps

    result = _dumps({"b": 1, "a": 2})
    assert result == '{"a":2,"b":1}', f"Unexpected result: {result}"


def test_dumps_unicode() -> None:
    """_dumps must preserve Cyrillic/Unicode literals (not \\uXXXX escapes)."""
    from atm.observability.serializers import _dumps

    result = _dumps({"ru": "привет"})
    assert "привет" in result, f"Unicode was escaped: {result}"


def test_dumps_compact_no_spaces() -> None:
    """_dumps must use compact separators (no spaces after , or :)."""
    from atm.observability.serializers import _dumps

    result = _dumps({"x": 1, "y": 2})
    assert " " not in result, f"Unexpected spaces in: {result}"


# ---------------------------------------------------------------------------
# Fixtures / sample builders
# ---------------------------------------------------------------------------


def _make_llm_response() -> object:
    from atm.core.types import LLMResponse, TokenUsage

    return LLMResponse(
        model="openai:gpt-4o",
        text="hello",
        usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            cached_input_tokens=2,
            total_tokens=15,
        ),
        cost_usd=0.001,
        latency_ms=123,
        finish_reason="stop",
        started_at=_utcnow(),
    )


def _make_message() -> object:
    from atm.core.types import Message, MessageKind

    return Message(
        sender="agent-1",
        recipients=("agent-2",),
        kind=MessageKind.REQUEST,
        content="Hello",
        payload={"key": "value"},
        created_at=_utcnow(),
    )


def _make_tool_call_and_result() -> tuple[object, object]:
    from atm.core.types import ToolCall, ToolResult

    call = ToolCall(
        tool_name="search",
        args={"query": "test"},
        issued_by="agent-1",
        issued_at=_utcnow(),
    )
    result = ToolResult(
        call_id=call.id,
        ok=True,
        output={"hits": 3},
        error=None,
        latency_ms=50,
        finished_at=_utcnow(),
    )
    return call, result


def _make_phase_transition() -> object:
    from atm.core.types import Phase, PhaseTransition

    return PhaseTransition(
        run_id=uuid4(),
        from_phase=Phase.PLANNING,
        to_phase=Phase.EXECUTION,
        entry_reason="all agents agree",
        iter_total=5,
        decided_by="rule",
        at=_utcnow(),
    )


def _make_topology_transition() -> object:
    from atm.core.types import Phase, TopologyTransition

    return TopologyTransition(
        run_id=uuid4(),
        from_topology="star",
        to_topology="mesh",
        phase_at_decision=Phase.EXECUTION,
        iter_within_phase=2,
        iter_within_topology=0,
        decided_by="llm_router",
        reason="mesh better for creative tasks",
        considered_alternatives=("chain", "star"),
        guards_applied=("budget_guard",),
        signals_snapshot={"quality": 0.7},
        router_cost_usd=0.0005,
        at=_utcnow(),
    )


def _make_scratchpad_entry() -> object:
    from atm.observability.serializers import ScratchpadEntry

    return ScratchpadEntry(
        at=_utcnow(),
        role="assistant",
        content="Let me think about this...",
        tool_calls=[{"name": "search", "args": {}}],
    )


# ---------------------------------------------------------------------------
# llm_response_to_row
# ---------------------------------------------------------------------------


def test_llm_response_row_keys_match_schema() -> None:
    """set(row.keys()) == set(LLM_CALL_SCHEMA.names)."""
    from atm.observability.serializers import llm_response_to_row
    from atm.storage.schemas import LLM_CALL_SCHEMA

    row = llm_response_to_row(
        run_id=_make_run_id(),
        agent_id="agent-1",
        model="openai:gpt-4o",
        at=_utcnow(),
        response=_make_llm_response(),  # type: ignore[arg-type]
    )
    assert set(row.keys()) == set(LLM_CALL_SCHEMA.names), (
        f"Key mismatch: row={set(row.keys())}, schema={set(LLM_CALL_SCHEMA.names)}"
    )


def test_llm_response_pa_table_from_row() -> None:
    """pa.Table.from_pylist([row], schema=LLM_CALL_SCHEMA) must not raise."""
    from atm.observability.serializers import llm_response_to_row
    from atm.storage.schemas import LLM_CALL_SCHEMA

    row = llm_response_to_row(
        run_id=_make_run_id(),
        agent_id="agent-1",
        model="openai:gpt-4o",
        at=_utcnow(),
        response=_make_llm_response(),  # type: ignore[arg-type]
    )
    pa.Table.from_pylist([row], schema=LLM_CALL_SCHEMA)


def test_llm_response_tz_aware_preserved() -> None:
    """row['at'].tzinfo must not be None."""
    from atm.observability.serializers import llm_response_to_row

    at = _utcnow()
    row = llm_response_to_row(
        run_id=_make_run_id(),
        agent_id="agent-1",
        model="openai:gpt-4o",
        at=at,
        response=_make_llm_response(),  # type: ignore[arg-type]
    )
    assert row["at"].tzinfo is not None, "at.tzinfo must not be None"


def test_llm_response_json_fields_deterministic() -> None:
    """Two identical calls must produce the same result (pure function)."""
    from atm.observability.serializers import llm_response_to_row

    run_id = _make_run_id()
    at = _utcnow()
    resp = _make_llm_response()
    row1 = llm_response_to_row(run_id=run_id, agent_id="a", model="m", at=at, response=resp)  # type: ignore[arg-type]
    row2 = llm_response_to_row(run_id=run_id, agent_id="a", model="m", at=at, response=resp)  # type: ignore[arg-type]
    assert row1["fingerprint"] == row2["fingerprint"]
    assert row1["cache_scope"] == row2["cache_scope"]


# ---------------------------------------------------------------------------
# message_to_row
# ---------------------------------------------------------------------------


def test_message_row_keys_match_schema() -> None:
    """set(row.keys()) == set(MESSAGE_SCHEMA.names)."""
    from atm.observability.serializers import message_to_row
    from atm.storage.schemas import MESSAGE_SCHEMA

    row = message_to_row(run_id=_make_run_id(), msg=_make_message())  # type: ignore[arg-type]
    assert set(row.keys()) == set(MESSAGE_SCHEMA.names)


def test_message_pa_table_from_row() -> None:
    """pa.Table.from_pylist([row], schema=MESSAGE_SCHEMA) must not raise."""
    from atm.observability.serializers import message_to_row
    from atm.storage.schemas import MESSAGE_SCHEMA

    row = message_to_row(run_id=_make_run_id(), msg=_make_message())  # type: ignore[arg-type]
    pa.Table.from_pylist([row], schema=MESSAGE_SCHEMA)


def test_message_tz_aware_preserved() -> None:
    """row['at'].tzinfo must not be None."""
    from atm.observability.serializers import message_to_row

    row = message_to_row(run_id=_make_run_id(), msg=_make_message())  # type: ignore[arg-type]
    assert row["at"].tzinfo is not None


def test_message_json_fields_deterministic() -> None:
    """payload_json must be equal across two calls with the same msg."""
    from atm.observability.serializers import message_to_row

    run_id = _make_run_id()
    msg = _make_message()
    row1 = message_to_row(run_id=run_id, msg=msg)  # type: ignore[arg-type]
    row2 = message_to_row(run_id=run_id, msg=msg)  # type: ignore[arg-type]
    assert row1["payload_json"] == row2["payload_json"]


# ---------------------------------------------------------------------------
# tool_call_to_row
# ---------------------------------------------------------------------------


def test_tool_call_row_keys_match_schema() -> None:
    """set(row.keys()) == set(TOOL_CALL_SCHEMA.names)."""
    from atm.observability.serializers import tool_call_to_row
    from atm.storage.schemas import TOOL_CALL_SCHEMA

    call, result = _make_tool_call_and_result()
    row = tool_call_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        agent_id="agent-1",
        call=call,  # type: ignore[arg-type]
        result=result,  # type: ignore[arg-type]
    )
    assert set(row.keys()) == set(TOOL_CALL_SCHEMA.names)


def test_tool_call_pa_table_from_row() -> None:
    """pa.Table.from_pylist([row], schema=TOOL_CALL_SCHEMA) must not raise."""
    from atm.observability.serializers import tool_call_to_row
    from atm.storage.schemas import TOOL_CALL_SCHEMA

    call, result = _make_tool_call_and_result()
    row = tool_call_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        agent_id="agent-1",
        call=call,  # type: ignore[arg-type]
        result=result,  # type: ignore[arg-type]
    )
    pa.Table.from_pylist([row], schema=TOOL_CALL_SCHEMA)


def test_tool_call_tz_aware_preserved() -> None:
    """row['at'].tzinfo must not be None."""
    from atm.observability.serializers import tool_call_to_row

    call, result = _make_tool_call_and_result()
    row = tool_call_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        agent_id="agent-1",
        call=call,  # type: ignore[arg-type]
        result=result,  # type: ignore[arg-type]
    )
    assert row["at"].tzinfo is not None


def test_tool_call_json_fields_deterministic() -> None:
    """args_json and result_json must be equal across two calls with the same inputs."""
    from atm.observability.serializers import tool_call_to_row

    run_id = _make_run_id()
    call, result = _make_tool_call_and_result()
    row1 = tool_call_to_row(run_id=run_id, agent_id="a", call=call, result=result)  # type: ignore[arg-type]
    row2 = tool_call_to_row(run_id=run_id, agent_id="a", call=call, result=result)  # type: ignore[arg-type]
    assert row1["args_json"] == row2["args_json"]
    assert row1["result_json"] == row2["result_json"]


# ---------------------------------------------------------------------------
# phase_transition_to_row
# ---------------------------------------------------------------------------


def test_phase_transition_row_keys_match_schema() -> None:
    """set(row.keys()) == set(PHASE_SCHEMA.names)."""
    from atm.observability.serializers import phase_transition_to_row
    from atm.storage.schemas import PHASE_SCHEMA

    row = phase_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_phase_transition(),  # type: ignore[arg-type]
        topology_used="star",
        ended_at=_utcnow(),
    )
    assert set(row.keys()) == set(PHASE_SCHEMA.names)


def test_phase_transition_pa_table_from_row() -> None:
    """pa.Table.from_pylist([row], schema=PHASE_SCHEMA) must not raise."""
    from atm.observability.serializers import phase_transition_to_row
    from atm.storage.schemas import PHASE_SCHEMA

    row = phase_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_phase_transition(),  # type: ignore[arg-type]
        topology_used="star",
        ended_at=_utcnow(),
    )
    pa.Table.from_pylist([row], schema=PHASE_SCHEMA)


def test_phase_transition_tz_aware_preserved() -> None:
    """row['started_at'].tzinfo must not be None."""
    from atm.observability.serializers import phase_transition_to_row

    row = phase_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_phase_transition(),  # type: ignore[arg-type]
        topology_used="star",
        ended_at=_utcnow(),
    )
    assert row["started_at"].tzinfo is not None


def test_phase_transition_ended_at_none() -> None:
    """ended_at=None must produce row['ended_at'] == None and still build a pa.Table."""
    from atm.observability.serializers import phase_transition_to_row
    from atm.storage.schemas import PHASE_SCHEMA

    row = phase_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_phase_transition(),  # type: ignore[arg-type]
        topology_used="star",
        ended_at=None,
    )
    assert row["ended_at"] is None
    pa.Table.from_pylist([row], schema=PHASE_SCHEMA)


def test_phase_transition_json_fields_deterministic() -> None:
    """Two identical calls must produce identical rows."""
    from atm.observability.serializers import phase_transition_to_row

    run_id = _make_run_id()
    transition = _make_phase_transition()
    ended = _utcnow()
    row1 = phase_transition_to_row(
        run_id=run_id, transition=transition, topology_used="star", ended_at=ended
    )  # type: ignore[arg-type]
    row2 = phase_transition_to_row(
        run_id=run_id, transition=transition, topology_used="star", ended_at=ended
    )  # type: ignore[arg-type]
    assert row1["entry_reason"] == row2["entry_reason"]
    assert row1["phase_name"] == row2["phase_name"]


# ---------------------------------------------------------------------------
# topology_transition_to_row
# ---------------------------------------------------------------------------


def test_topology_transition_row_keys_match_schema() -> None:
    """set(row.keys()) == set(TOPOLOGY_TRANSITION_SCHEMA.names)."""
    from atm.observability.serializers import topology_transition_to_row
    from atm.storage.schemas import TOPOLOGY_TRANSITION_SCHEMA

    row = topology_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_topology_transition(),  # type: ignore[arg-type]
    )
    assert set(row.keys()) == set(TOPOLOGY_TRANSITION_SCHEMA.names)


def test_topology_transition_pa_table_from_row() -> None:
    """pa.Table.from_pylist([row], schema=TOPOLOGY_TRANSITION_SCHEMA) must not raise."""
    from atm.observability.serializers import topology_transition_to_row
    from atm.storage.schemas import TOPOLOGY_TRANSITION_SCHEMA

    row = topology_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_topology_transition(),  # type: ignore[arg-type]
    )
    pa.Table.from_pylist([row], schema=TOPOLOGY_TRANSITION_SCHEMA)


def test_topology_transition_tz_aware_preserved() -> None:
    """row['at'].tzinfo must not be None."""
    from atm.observability.serializers import topology_transition_to_row

    row = topology_transition_to_row(  # type: ignore[arg-type]
        run_id=_make_run_id(),
        transition=_make_topology_transition(),  # type: ignore[arg-type]
    )
    assert row["at"].tzinfo is not None


def test_topology_transition_json_fields_deterministic() -> None:
    """considered_alternatives_json and guards_applied_json must be stable."""
    from atm.observability.serializers import topology_transition_to_row

    run_id = _make_run_id()
    transition = _make_topology_transition()
    row1 = topology_transition_to_row(run_id=run_id, transition=transition)  # type: ignore[arg-type]
    row2 = topology_transition_to_row(run_id=run_id, transition=transition)  # type: ignore[arg-type]
    assert row1["considered_alternatives_json"] == row2["considered_alternatives_json"]
    assert row1["guards_applied_json"] == row2["guards_applied_json"]
    assert row1["signals_snapshot_json"] == row2["signals_snapshot_json"]


# ---------------------------------------------------------------------------
# scratchpad_entry_to_row
# ---------------------------------------------------------------------------


def test_scratchpad_entry_row_keys_match_schema() -> None:
    """set(row.keys()) == set(SCRATCHPAD_SCHEMA.names)."""
    from atm.observability.serializers import scratchpad_entry_to_row
    from atm.storage.schemas import SCRATCHPAD_SCHEMA

    row = scratchpad_entry_to_row(
        run_id=_make_run_id(),
        agent_id="agent-1",
        step_idx=3,
        entry=_make_scratchpad_entry(),  # type: ignore[arg-type]
    )
    assert set(row.keys()) == set(SCRATCHPAD_SCHEMA.names)


def test_scratchpad_entry_pa_table_from_row() -> None:
    """pa.Table.from_pylist([row], schema=SCRATCHPAD_SCHEMA) must not raise."""
    from atm.observability.serializers import scratchpad_entry_to_row
    from atm.storage.schemas import SCRATCHPAD_SCHEMA

    row = scratchpad_entry_to_row(
        run_id=_make_run_id(),
        agent_id="agent-1",
        step_idx=3,
        entry=_make_scratchpad_entry(),  # type: ignore[arg-type]
    )
    pa.Table.from_pylist([row], schema=SCRATCHPAD_SCHEMA)


def test_scratchpad_entry_tz_aware_preserved() -> None:
    """row['at'].tzinfo must not be None."""
    from atm.observability.serializers import scratchpad_entry_to_row

    row = scratchpad_entry_to_row(
        run_id=_make_run_id(),
        agent_id="agent-1",
        step_idx=0,
        entry=_make_scratchpad_entry(),  # type: ignore[arg-type]
    )
    assert row["at"].tzinfo is not None


def test_scratchpad_entry_json_fields_deterministic() -> None:
    """tool_calls_json must be equal for identical inputs."""
    from atm.observability.serializers import scratchpad_entry_to_row

    run_id = _make_run_id()
    entry = _make_scratchpad_entry()
    row1 = scratchpad_entry_to_row(run_id=run_id, agent_id="a", step_idx=0, entry=entry)  # type: ignore[arg-type]
    row2 = scratchpad_entry_to_row(run_id=run_id, agent_id="a", step_idx=0, entry=entry)  # type: ignore[arg-type]
    assert row1["tool_calls_json"] == row2["tool_calls_json"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_broadcast_message_to_agent_empty() -> None:
    """Broadcast message (no recipients) → row['to_agent'] == ''."""
    from atm.core.types import Message, MessageKind
    from atm.observability.serializers import message_to_row

    msg = Message(
        sender="agent-1",
        recipients=(),
        kind=MessageKind.BROADCAST,
        content="broadcast",
        created_at=_utcnow(),
    )
    row = message_to_row(run_id=_make_run_id(), msg=msg)
    assert row["to_agent"] == ""


def test_tool_result_error_none_to_empty_string() -> None:
    """ToolResult.error=None → row['error'] == ''."""
    from atm.core.types import ToolCall, ToolResult
    from atm.observability.serializers import tool_call_to_row

    call = ToolCall(tool_name="noop", args={}, issued_by="a", issued_at=_utcnow())
    result = ToolResult(
        call_id=call.id, ok=True, output=None, error=None, latency_ms=1, finished_at=_utcnow()
    )
    row = tool_call_to_row(run_id=_make_run_id(), agent_id="a", call=call, result=result)
    assert row["error"] == ""


def test_topology_transition_initial_from_none() -> None:
    """from_topology=None (initial decision) → row['from_topology'] == ''."""
    from atm.core.types import Phase, TopologyTransition
    from atm.observability.serializers import topology_transition_to_row

    t = TopologyTransition(
        run_id=uuid4(),
        from_topology=None,
        to_topology="star",
        phase_at_decision=Phase.PLANNING,
        iter_within_phase=0,
        iter_within_topology=0,
        decided_by="initial",
        reason="bootstrap",
        at=_utcnow(),
    )
    row = topology_transition_to_row(run_id=_make_run_id(), transition=t)
    assert row["from_topology"] == ""


def test_phase_transition_initial_from_none() -> None:
    """from_phase=None (initial) → row['from_phase'] == ''."""
    from atm.core.types import Phase, PhaseTransition
    from atm.observability.serializers import phase_transition_to_row

    t = PhaseTransition(
        run_id=uuid4(),
        from_phase=None,
        to_phase=Phase.PLANNING,
        entry_reason="start",
        iter_total=0,
        decided_by="initial",
        at=_utcnow(),
    )
    row = phase_transition_to_row(run_id=_make_run_id(), transition=t)
    assert row["from_phase"] == ""
