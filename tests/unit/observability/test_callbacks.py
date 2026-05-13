"""Unit tests for atm.observability.callbacks — ExperimentCallbackHandler.

Tests cover:
1. Root run detection (metadata flag + fallback)
2. on_chain_end / on_chain_error triggering parquet_writer.close()
3. on_llm_end writes parquet and updates PG budget atomically
4. budget warn/exceed branching (no duplicate events)
5. phase_transition ordering invariant (flush BEFORE pg insert)
6. phase_transition prev ended_at update
7. topology_transition ordering invariant
8. Exception swallowing — handler must not propagate errors
9. on_tool_end latency computation
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_llm_result(
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost_usd: float = 0.01,
    agent_id: str = "agent-1",
    model: str = "gpt-4o",
    latency_ms: int = 200,
    cache_hit_tokens: int = 0,
    cache_scope: str = "none",
    fingerprint: str = "fp-abc123",
) -> MagicMock:
    """Build a mock LLMResult whose llm_output dict matches the expected contract."""
    llm_output = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "agent_id": agent_id,
        "model": model,
        "latency_ms": latency_ms,
        "cache_hit_tokens": cache_hit_tokens,
        "cache_scope": cache_scope,
        "fingerprint": fingerprint,
    }
    result = MagicMock()
    result.llm_output = llm_output
    result.generations = [[MagicMock(generation_info={})]]
    return result


def _make_handler(
    *,
    budget_warn_threshold: Decimal | None = None,
    budget_exceed_threshold: Decimal | None = None,
    session_factory: AsyncMock | None = None,
    parquet_writer: AsyncMock | None = None,
) -> object:
    """Create an ExperimentCallbackHandler with mock dependencies."""
    from atm.observability.callbacks import ExperimentCallbackHandler

    run_id = uuid4()
    exp_id = uuid4()

    if session_factory is None:
        session_factory = _build_session_factory()
    if parquet_writer is None:
        parquet_writer = AsyncMock()
        parquet_writer.write_llm_call = AsyncMock()
        parquet_writer.write_message = AsyncMock()
        parquet_writer.write_tool_call = AsyncMock()
        parquet_writer.write_phase = AsyncMock()
        parquet_writer.write_topology_transition = AsyncMock()
        parquet_writer.flush = AsyncMock()
        parquet_writer.close = AsyncMock()

    handler = ExperimentCallbackHandler(
        run_id=run_id,
        exp_id=exp_id,
        session_factory=session_factory,
        parquet_writer=parquet_writer,
        budget_warn_threshold=budget_warn_threshold,
        budget_exceed_threshold=budget_exceed_threshold,
    )
    return handler


def _build_session_factory(scalar_values: list[Decimal] | None = None) -> AsyncMock:
    """Build an async session_factory mock.

    The scalar_one() side_effect cycles through ``scalar_values`` in order.
    Default: returns Decimal("0.0") for all calls.
    """
    if scalar_values is None:
        scalar_values = [Decimal("0.0"), Decimal("0.0")]

    mock_result = MagicMock()
    mock_result.scalar_one = MagicMock(side_effect=scalar_values)

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    return factory


# ---------------------------------------------------------------------------
# 1. Root detection — metadata flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_detection_metadata_flag() -> None:
    """First on_chain_start with parent_run_id=None and metadata.is_root_run=True sets _root_run_id;
    subsequent starts do NOT overwrite it."""
    from atm.observability.callbacks import ExperimentCallbackHandler

    handler = _make_handler()
    assert isinstance(handler, ExperimentCallbackHandler)
    assert handler._root_run_id is None

    root_id = uuid4()
    await handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=root_id,
        parent_run_id=None,
        metadata={"is_root_run": True},
    )
    assert handler._root_run_id == root_id

    # Second call with another run_id should NOT overwrite
    other_id = uuid4()
    await handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=other_id,
        parent_run_id=None,
        metadata={"is_root_run": True},
    )
    assert handler._root_run_id == root_id  # unchanged


# ---------------------------------------------------------------------------
# 2. Root detection — fallback (no metadata)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_detection_fallback() -> None:
    """If metadata is absent, first on_chain_start with parent_run_id=None still sets root."""

    handler = _make_handler()
    root_id = uuid4()
    await handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=root_id,
        parent_run_id=None,
    )
    assert handler._root_run_id == root_id


@pytest.mark.asyncio
async def test_root_not_set_for_child_chain() -> None:
    """on_chain_start with parent_run_id != None should not set root."""

    handler = _make_handler()
    child_id = uuid4()
    parent_id = uuid4()
    await handler.on_chain_start(
        serialized={},
        inputs={},
        run_id=child_id,
        parent_run_id=parent_id,
    )
    assert handler._root_run_id is None


# ---------------------------------------------------------------------------
# 3. on_chain_end of root triggers parquet_writer.close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_chain_end_root_triggers_close() -> None:
    """on_chain_end(run_id=root_id) must await parquet_writer.close() exactly once."""
    parquet_writer = AsyncMock()
    parquet_writer.close = AsyncMock()
    parquet_writer.flush = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)
    root_id = uuid4()
    handler._root_run_id = root_id  # type: ignore[union-attr]

    await handler.on_chain_end(outputs={}, run_id=root_id)  # type: ignore[union-attr]

    parquet_writer.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_chain_end_non_root_does_not_close() -> None:
    """on_chain_end for a non-root run_id must NOT call parquet_writer.close()."""
    parquet_writer = AsyncMock()
    parquet_writer.close = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)
    root_id = uuid4()
    handler._root_run_id = root_id  # type: ignore[union-attr]

    await handler.on_chain_end(outputs={}, run_id=uuid4())  # type: ignore[union-attr]

    parquet_writer.close.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. on_chain_error of root triggers parquet_writer.close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_chain_error_root_triggers_close() -> None:
    """on_chain_error(run_id=root_id) must await parquet_writer.close() — invariant (c)."""
    parquet_writer = AsyncMock()
    parquet_writer.close = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)
    root_id = uuid4()
    handler._root_run_id = root_id  # type: ignore[union-attr]

    await handler.on_chain_error(  # type: ignore[union-attr]
        error=RuntimeError("boom"), run_id=root_id
    )

    parquet_writer.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. on_llm_end writes parquet and updates PG budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_llm_end_writes_parquet_and_updates_budget() -> None:
    """on_llm_end must call parquet_writer.write_llm_call and execute two UPDATEs."""
    parquet_writer = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock()

    scalar_values = [Decimal("0.01"), Decimal("0.01")]
    session_factory = _build_session_factory(scalar_values)

    handler = _make_handler(
        parquet_writer=parquet_writer,
        session_factory=session_factory,
    )

    llm_result = _make_llm_result(cost_usd=0.01)
    await handler.on_llm_end(response=llm_result, run_id=uuid4())  # type: ignore[union-attr]

    parquet_writer.write_llm_call.assert_awaited_once()

    # Check that session.execute was called at least twice (two UPDATE ... RETURNING)
    session = session_factory.return_value
    assert session.execute.await_count >= 2


# ---------------------------------------------------------------------------
# 6. Budget warn branching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_warn_branching() -> None:
    """With budget_warn_threshold=1.0 and run total=1.5 → BudgetEvent(event='warn') inserted once.
    Second call with total=1.8 → no second warn (already warned)."""
    # First call: run total = 1.5 > threshold 1.0 → warn inserted
    # Second call: run total = 1.8 > threshold 1.0 → no second warn
    scalar_values_call1 = [Decimal("1.5"), Decimal("1.5")]
    scalar_values_call2 = [Decimal("1.8"), Decimal("1.8")]

    parquet_writer = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock()

    # We need a session factory that we can inspect for session.add calls
    mock_result1 = MagicMock()
    mock_result1.scalar_one = MagicMock(side_effect=scalar_values_call1)

    mock_result2 = MagicMock()
    mock_result2.scalar_one = MagicMock(side_effect=scalar_values_call2)

    session = AsyncMock()
    session.execute = AsyncMock(
        side_effect=[mock_result1, mock_result1, mock_result2, mock_result2]
    )
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    handler = _make_handler(
        parquet_writer=parquet_writer,
        session_factory=factory,
        budget_warn_threshold=Decimal("1.0"),
    )

    from atm.storage.models import BudgetEvent as BudgetEventModel

    llm_result = _make_llm_result(cost_usd=0.5)

    # First call → warn should fire
    await handler.on_llm_end(response=llm_result, run_id=uuid4())  # type: ignore[union-attr]

    add_calls = session.add.call_args_list
    warn_inserts = [
        c
        for c in add_calls
        if isinstance(c.args[0], BudgetEventModel) and c.args[0].event == "warn"
    ]
    assert len(warn_inserts) == 1, f"Expected 1 warn insert, got {len(warn_inserts)}"

    # Second call — total is still > threshold, but flag already set → no new warn
    session.add.reset_mock()
    await handler.on_llm_end(response=llm_result, run_id=uuid4())  # type: ignore[union-attr]

    add_calls2 = session.add.call_args_list
    warn_inserts2 = [
        c
        for c in add_calls2
        if isinstance(c.args[0], BudgetEventModel) and c.args[0].event == "warn"
    ]
    assert len(warn_inserts2) == 0, "Duplicate warn should not be inserted"


# ---------------------------------------------------------------------------
# 7. Budget exceed branching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceed_branching() -> None:
    """With budget_exceed_threshold=10.0 and run total=10.5 → BudgetEvent(event='exceed') inserted once."""
    mock_result = MagicMock()
    mock_result.scalar_one = MagicMock(side_effect=[Decimal("10.5"), Decimal("10.5")])

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    parquet_writer = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock()

    handler = _make_handler(
        parquet_writer=parquet_writer,
        session_factory=factory,
        budget_exceed_threshold=Decimal("10.0"),
    )

    from atm.storage.models import BudgetEvent as BudgetEventModel

    llm_result = _make_llm_result(cost_usd=0.5)
    await handler.on_llm_end(response=llm_result, run_id=uuid4())  # type: ignore[union-attr]

    add_calls = session.add.call_args_list
    exceed_inserts = [
        c
        for c in add_calls
        if isinstance(c.args[0], BudgetEventModel) and c.args[0].event == "exceed"
    ]
    assert len(exceed_inserts) == 1, f"Expected 1 exceed insert, got {len(exceed_inserts)}"


# ---------------------------------------------------------------------------
# 8. phase_transition ordering invariant — flush BEFORE pg insert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_transition_ordering() -> None:
    """For 'phase_transition' event: parquet_writer.flush() must be called BEFORE session.execute.

    Uses a parent mock to track call ordering across flush and execute.
    """
    from atm.core.types import Phase, PhaseTransition

    mock_result = MagicMock()
    mock_result.scalar_one = MagicMock(return_value=None)  # no previous phase

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    parquet_writer = AsyncMock()
    parquet_writer.flush = AsyncMock()
    parquet_writer.write_phase = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer, session_factory=factory)
    run_id = uuid4()
    handler._run_id = run_id  # type: ignore[union-attr]

    transition = PhaseTransition(
        run_id=run_id,
        from_phase=None,
        to_phase=Phase.PLANNING,
        entry_reason="initial",
        iter_total=0,
        decided_by="initial",
    )

    # Track order of flush vs execute calls using a shared call log
    call_log: list[str] = []

    async def tracked_flush(*a: object, **kw: object) -> None:
        call_log.append("flush")

    async def tracked_execute(*a: object, **kw: object) -> MagicMock:
        call_log.append("execute")
        return mock_result

    parquet_writer.flush.side_effect = tracked_flush
    session.execute.side_effect = tracked_execute

    await handler.on_custom_event(  # type: ignore[union-attr]
        name="phase_transition",
        data=transition,
        run_id=uuid4(),
    )

    assert "flush" in call_log, "flush was never called"
    assert "execute" in call_log, "execute was never called"
    flush_idx = call_log.index("flush")
    first_execute_idx = call_log.index("execute")
    assert flush_idx < first_execute_idx, (
        f"flush (idx={flush_idx}) must precede execute (idx={first_execute_idx})"
    )


# ---------------------------------------------------------------------------
# 9. phase_transition prev ended_at update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_transition_prev_ended_at_update() -> None:
    """When SELECT returns a previous phase id, an UPDATE to set ended_at must be executed."""
    from atm.core.types import Phase, PhaseTransition

    prev_phase_id = uuid4()

    # First execute call: SELECT → returns prev_phase_id
    # Subsequent execute calls: UPDATEs
    # SELECT scalar_one returns prev_phase_id
    select_result = MagicMock()
    select_result.scalar_one = MagicMock(return_value=prev_phase_id)
    # UPDATE results
    update_result = MagicMock()
    update_result.scalar_one = MagicMock(return_value=None)

    call_count = 0

    session = AsyncMock()

    async def mock_execute(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return select_result  # SELECT previous phase
        return update_result

    session.execute = AsyncMock(side_effect=mock_execute)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    parquet_writer = AsyncMock()
    parquet_writer.flush = AsyncMock()
    parquet_writer.write_phase = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer, session_factory=factory)
    run_id = uuid4()

    transition = PhaseTransition(
        run_id=run_id,
        from_phase=Phase.PLANNING,
        to_phase=Phase.EXECUTION,
        entry_reason="task done",
        iter_total=5,
        decided_by="rule",
    )

    await handler.on_custom_event(  # type: ignore[union-attr]
        name="phase_transition",
        data=transition,
        run_id=uuid4(),
    )

    # Should have been called at least twice: SELECT + UPDATE ended_at + INSERT
    assert call_count >= 2, f"Expected >=2 execute calls, got {call_count}"


# ---------------------------------------------------------------------------
# 10. topology_transition ordering invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topology_transition_ordering() -> None:
    """For 'topology_transition' event: flush() must be called BEFORE PG insert (invariant b).

    Uses a shared call log to verify that parquet_writer.flush() is called BEFORE
    session.add() which inserts the topology_transition row.
    """
    from atm.core.types import Phase, TopologyTransition

    mock_result = MagicMock()
    mock_result.scalar_one = MagicMock(return_value=None)

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock()
    factory.return_value = session
    factory.return_value.__aenter__ = AsyncMock(return_value=session)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    parquet_writer = AsyncMock()
    parquet_writer.flush = AsyncMock()
    parquet_writer.write_topology_transition = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer, session_factory=factory)
    run_id = uuid4()

    transition = TopologyTransition(
        run_id=run_id,
        from_topology=None,
        to_topology="star",
        phase_at_decision=Phase.PLANNING,
        iter_within_phase=0,
        iter_within_topology=0,
        decided_by="initial",
        reason="first",
    )

    call_log: list[str] = []

    async def tracked_flush(*a: object, **kw: object) -> None:
        call_log.append("flush")

    def tracked_add(*a: object, **kw: object) -> None:
        call_log.append("add")

    parquet_writer.flush.side_effect = tracked_flush
    session.add = MagicMock(side_effect=tracked_add)

    await handler.on_custom_event(  # type: ignore[union-attr]
        name="topology_transition",
        data=transition,
        run_id=uuid4(),
    )

    assert "flush" in call_log, "flush was never called"
    assert "add" in call_log, "session.add (INSERT) was never called"
    flush_idx = call_log.index("flush")
    first_add_idx = call_log.index("add")
    assert flush_idx < first_add_idx, (
        f"flush (idx={flush_idx}) must precede session.add (idx={first_add_idx})"
    )


# ---------------------------------------------------------------------------
# 11. Exception in hook is swallowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exception_in_hook_is_swallowed() -> None:
    """If parquet_writer.write_llm_call raises, on_llm_end must NOT propagate the exception."""
    parquet_writer = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock(side_effect=RuntimeError("parquet error"))

    session_factory = _build_session_factory()
    handler = _make_handler(parquet_writer=parquet_writer, session_factory=session_factory)

    llm_result = _make_llm_result()

    # Must not raise
    await handler.on_llm_end(response=llm_result, run_id=uuid4())  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 12. on_tool_end latency computed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_end_latency_computed() -> None:
    """on_tool_start followed by on_tool_end produces a row with latency_ms > 0 and < 1000,
    tool_name and agent_id propagated from on_tool_start, result_json is valid JSON."""
    import json

    from atm.observability.callbacks import ExperimentCallbackHandler

    parquet_writer = AsyncMock()
    parquet_writer.write_tool_call = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)
    assert isinstance(handler, ExperimentCallbackHandler)

    tool_run_id = uuid4()

    await handler.on_tool_start(  # type: ignore[union-attr]
        serialized={"name": "search"},
        input_str="q",
        run_id=tool_run_id,
        metadata={"agent_id": "agent-1"},
    )

    # Small sleep to ensure measurable latency
    await asyncio.sleep(0.001)

    await handler.on_tool_end(  # type: ignore[union-attr]
        output="result",
        run_id=tool_run_id,
        parent_run_id=uuid4(),
    )

    parquet_writer.write_tool_call.assert_awaited_once()
    written_row = parquet_writer.write_tool_call.call_args.args[0]
    assert written_row["latency_ms"] > 0, "latency_ms should be > 0"
    assert written_row["latency_ms"] < 1000, "latency_ms should be < 1000 for a trivial tool call"
    assert written_row["ok"] is True
    assert written_row["error"] == ""
    assert written_row["tool_name"] == "search", (
        f"Expected 'search', got {written_row['tool_name']!r}"
    )
    assert written_row["agent_id"] == "agent-1", (
        f"Expected 'agent-1', got {written_row['agent_id']!r}"
    )
    assert json.loads(written_row["result_json"]) == "result", (
        "result_json should be valid JSON encoding the output string"
    )


# ---------------------------------------------------------------------------
# 13. on_tool_error writes row with ok=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_error_writes_row() -> None:
    """on_tool_start + on_tool_error produces a parquet row with ok=False."""
    parquet_writer = AsyncMock()
    parquet_writer.write_tool_call = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)

    tool_run_id = uuid4()
    await handler.on_tool_start(  # type: ignore[union-attr]
        serialized={"name": "bad_tool"},
        input_str="arg",
        run_id=tool_run_id,
        parent_run_id=uuid4(),
    )

    await handler.on_tool_error(  # type: ignore[union-attr]
        error=ValueError("tool failed"),
        run_id=tool_run_id,
        parent_run_id=uuid4(),
    )

    parquet_writer.write_tool_call.assert_awaited_once()
    written_row = parquet_writer.write_tool_call.call_args.args[0]
    assert written_row["ok"] is False
    assert "tool failed" in written_row["error"]


# ---------------------------------------------------------------------------
# 14. on_tool_error propagates tool_name, agent_id from on_tool_start
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_error_records_error() -> None:
    """on_tool_start + on_tool_error: row has ok=False, error string, and tool_name/agent_id from start."""
    parquet_writer = AsyncMock()
    parquet_writer.write_tool_call = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)

    tool_run_id = uuid4()
    await handler.on_tool_start(  # type: ignore[union-attr]
        serialized={"name": "failing_tool"},
        input_str="some_arg",
        run_id=tool_run_id,
        metadata={"agent_id": "agent-error"},
    )

    await handler.on_tool_error(  # type: ignore[union-attr]
        error=RuntimeError("network timeout"),
        run_id=tool_run_id,
        parent_run_id=uuid4(),
    )

    parquet_writer.write_tool_call.assert_awaited_once()
    written_row = parquet_writer.write_tool_call.call_args.args[0]
    assert written_row["ok"] is False
    assert "network timeout" in written_row["error"]
    assert written_row["tool_name"] == "failing_tool", (
        f"Expected 'failing_tool', got {written_row['tool_name']!r}"
    )
    assert written_row["agent_id"] == "agent-error", (
        f"Expected 'agent-error', got {written_row['agent_id']!r}"
    )


# ---------------------------------------------------------------------------
# 16. message_emit custom event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_emit_custom_event() -> None:
    """on_custom_event('message_emit', message) → parquet_writer.write_message awaited."""
    from atm.core.types import Message, MessageKind

    parquet_writer = AsyncMock()
    parquet_writer.write_message = AsyncMock()

    handler = _make_handler(parquet_writer=parquet_writer)
    run_id = uuid4()
    handler._run_id = run_id  # type: ignore[union-attr]

    message = Message(
        sender="agent-1",
        kind=MessageKind.BROADCAST,
        content="hello world",
    )

    await handler.on_custom_event(  # type: ignore[union-attr]
        name="message_emit",
        data=message,
        run_id=uuid4(),
    )

    parquet_writer.write_message.assert_awaited_once()
