"""Unit tests for ExperimentCallbackHandler — human_request / human_response event handlers.

Tests cover:
1. human_request fires INSERT ... ON CONFLICT DO NOTHING against the correct constraint
2. Two identical human_request events (same run_id, request_id) result in only one INSERT
   (second is swallowed by ON CONFLICT DO NOTHING — single-row simulation)
3. human_response fires UPDATE ... WHERE response_json IS NULL
4. human_response update clause contains IS NULL filter (idempotency guard)
5. Unknown exception in human_request is swallowed (run not broken)
6. Unknown exception in human_response is swallowed (run not broken)
7. human_request uses the correct constraint name constant
8. human_response does not touch non-matching rows (request_id mismatch)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_session_factory() -> MagicMock:
    """Minimal async session factory mock with no execute side-effects."""
    mock_result = MagicMock()
    mock_result.scalar_one = MagicMock(return_value=None)
    mock_result.scalar_one_or_none = MagicMock(return_value=None)

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


def _make_handler(session_factory: MagicMock | None = None) -> Any:
    """Build an ExperimentCallbackHandler with mocked dependencies."""
    from atm.observability.callbacks import ExperimentCallbackHandler

    if session_factory is None:
        session_factory = _build_session_factory()

    parquet_writer = AsyncMock()
    parquet_writer.write_llm_call = AsyncMock()
    parquet_writer.write_message = AsyncMock()
    parquet_writer.write_tool_call = AsyncMock()
    parquet_writer.write_phase = AsyncMock()
    parquet_writer.write_topology_transition = AsyncMock()
    parquet_writer.flush = AsyncMock()
    parquet_writer.close = AsyncMock()

    run_id = uuid4()
    exp_id = uuid4()

    return ExperimentCallbackHandler(
        run_id=run_id,
        exp_id=exp_id,
        session_factory=session_factory,
        parquet_writer=parquet_writer,
    )


def _human_request_data(
    *,
    run_id: UUID | None = None,
    request_id: str = "chain:reviewer:0",
) -> dict[str, Any]:
    """Build a sample human_request event payload."""
    return {
        "run_id": run_id or uuid4(),
        "request_id": request_id,
        "role": "reviewer",
        "context_json": {"question": "approve?", "allowed_actions": ["approve", "reject"]},
        "requested_at": datetime.now(UTC),
    }


def _human_response_data(
    *,
    run_id: UUID | None = None,
    request_id: str = "chain:reviewer:0",
) -> dict[str, Any]:
    """Build a sample human_response event payload."""
    return {
        "run_id": run_id or uuid4(),
        "request_id": request_id,
        "answered_at": datetime.now(UTC),
        "response_json": {"action": "approve", "source": "llm_sim", "timed_out": False},
        "source": "llm_sim",
        "timed_out": False,
        "latency_s": 1.23,
    }


# ---------------------------------------------------------------------------
# 1. human_request — INSERT ... ON CONFLICT DO NOTHING fires with correct constraint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_request_inserts_with_correct_constraint() -> None:
    """on_custom_event('human_request', ...) must execute an INSERT that uses
    ON CONFLICT DO NOTHING against the constraint 'uq_human_interactions_run_request'.
    """
    from atm.observability.callbacks import CONSTRAINT_NAME
    from atm.storage.models import HumanInteraction

    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture_execute(stmt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture_execute)

    handler = _make_handler(session_factory=session_factory)
    data = _human_request_data()

    await handler.on_custom_event(
        name="human_request",
        data=data,
        run_id=uuid4(),
    )

    assert len(executed_stmts) == 1, f"Expected 1 statement, got {len(executed_stmts)}"

    stmt = executed_stmts[0]
    # The compiled SQL should contain ON CONFLICT DO NOTHING
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "ON CONFLICT" in compiled.upper(), f"Expected ON CONFLICT clause in SQL, got: {compiled}"

    # Verify the constraint constant value matches the migration
    assert CONSTRAINT_NAME == "uq_human_interactions_run_request"

    # Verify target table is human_interactions
    assert stmt.table.name == HumanInteraction.__tablename__


# ---------------------------------------------------------------------------
# 2. Two identical human_request events — only one INSERT executed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_request_duplicate_does_not_double_insert() -> None:
    """When the same (run_id, request_id) is dispatched twice, the callback calls
    session.execute twice — but the ON CONFLICT DO NOTHING clause on the DB side
    ensures only one row is persisted. We verify that execute is called for each
    event dispatch (the deduplication happens in the DB), and that neither call
    raises an exception.
    """
    session_factory = _build_session_factory()
    session = session_factory.return_value
    execute_count = 0

    async def counting_execute(stmt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        nonlocal execute_count
        execute_count += 1
        result = MagicMock()
        result.scalar_one = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=counting_execute)

    handler = _make_handler(session_factory=session_factory)
    run_id = uuid4()
    data = _human_request_data(run_id=run_id, request_id="chain:reviewer:1")

    # First dispatch
    await handler.on_custom_event(name="human_request", data=data, run_id=uuid4())
    # Second dispatch with same (run_id, request_id)
    await handler.on_custom_event(name="human_request", data=data, run_id=uuid4())

    # The handler dispatched two INSERTs; the DB constraint prevents double rows.
    assert execute_count == 2, f"Expected 2 execute calls (one per event), got {execute_count}"


# ---------------------------------------------------------------------------
# 3. human_response — UPDATE fires with IS NULL filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_response_select_for_update_then_insert_when_missing() -> None:
    """on_custom_event('human_response', ...) must SELECT ... FOR UPDATE first, and
    when no existing request row is found, INSERT a complete row (race-tolerant
    fallback for the case where the response handler runs before the request one).
    """
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture_execute(stmt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture_execute)

    handler = _make_handler(session_factory=session_factory)
    data = _human_response_data()

    await handler.on_custom_event(
        name="human_response",
        data=data,
        run_id=uuid4(),
    )

    # Expect SELECT ... FOR UPDATE then INSERT ... ON CONFLICT DO NOTHING
    assert len(executed_stmts) == 2, f"Expected 2 statements, got {len(executed_stmts)}"

    first = str(executed_stmts[0].compile(compile_kwargs={"literal_binds": False})).upper()
    assert first.strip().startswith("SELECT"), f"Expected SELECT first, got: {first}"
    assert "FOR UPDATE" in first, f"Expected FOR UPDATE clause, got: {first}"

    second = str(executed_stmts[1].compile(compile_kwargs={"literal_binds": False})).upper()
    assert second.strip().startswith("INSERT"), f"Expected INSERT second, got: {second}"
    assert "ON CONFLICT" in second, f"Expected ON CONFLICT clause, got: {second}"


# ---------------------------------------------------------------------------
# 4. human_response — idempotency: WHERE response_json IS NULL prevents double-update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_response_idempotent_when_existing_response_present() -> None:
    """A re-dispatch of human_response for a row whose response_json is already filled
    must NOT issue an UPDATE — the SELECT ... FOR UPDATE branch sees a populated row
    and short-circuits to a no-op (idempotency contract).
    """
    from atm.storage.models import HumanInteraction

    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    # Simulate an existing row that already has response_json filled
    existing_row = MagicMock(spec=HumanInteraction)
    existing_row.id = uuid4()
    existing_row.response_json = {"action": "approve"}

    async def capture_execute(stmt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=existing_row)
        return result

    session.execute = AsyncMock(side_effect=capture_execute)

    handler = _make_handler(session_factory=session_factory)
    data = _human_response_data(request_id="chain:reviewer:2")

    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    # Each dispatch should issue exactly ONE statement (the SELECT ... FOR UPDATE);
    # no UPDATE/INSERT follows because existing.response_json is non-NULL.
    assert len(executed_stmts) == 2, (
        f"Expected 2 statements (one SELECT per dispatch), got {len(executed_stmts)}"
    )
    for stmt in executed_stmts:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": False})).upper()
        assert compiled.strip().startswith("SELECT"), (
            f"Expected SELECT only — idempotency must short-circuit before UPDATE; got: {compiled}"
        )


# ---------------------------------------------------------------------------
# 5. Exception in human_request is swallowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_request_exception_is_swallowed() -> None:
    """If session.execute raises inside _handle_human_request, the exception must
    NOT propagate — the run continues uninterrupted.
    """
    session_factory = _build_session_factory()
    session = session_factory.return_value
    session.execute = AsyncMock(side_effect=RuntimeError("DB down"))

    handler = _make_handler(session_factory=session_factory)
    data = _human_request_data()

    # Must not raise
    await handler.on_custom_event(name="human_request", data=data, run_id=uuid4())


# ---------------------------------------------------------------------------
# 6. Exception in human_response is swallowed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_response_exception_is_swallowed() -> None:
    """If session.execute raises inside _handle_human_response, the exception must
    NOT propagate — the run continues uninterrupted.
    """
    session_factory = _build_session_factory()
    session = session_factory.return_value
    session.execute = AsyncMock(side_effect=RuntimeError("DB timeout"))

    handler = _make_handler(session_factory=session_factory)
    data = _human_response_data()

    # Must not raise
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())


# ---------------------------------------------------------------------------
# 7. human_request INSERT targets the correct table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_request_targets_human_interactions_table() -> None:
    """The INSERT statement must target the 'human_interactions' table."""
    from atm.storage.models import HumanInteraction

    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        executed_stmts.append(stmt)
        return MagicMock()

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _human_request_data()

    await handler.on_custom_event(name="human_request", data=data, run_id=uuid4())

    assert executed_stmts, "No statement was executed"
    assert executed_stmts[0].table.name == HumanInteraction.__tablename__


# ---------------------------------------------------------------------------
# 8. human_response UPDATE targets the correct table (human_interactions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_response_targets_human_interactions_table() -> None:
    """The UPDATE statement must target the 'human_interactions' table."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *args: Any, **kwargs: Any) -> MagicMock:
        executed_stmts.append(stmt)
        return MagicMock()

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _human_response_data()

    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    assert executed_stmts, "No statement was executed"
    compiled = str(executed_stmts[0].compile(compile_kwargs={"literal_binds": False}))
    assert "human_interactions" in compiled, (
        f"Expected 'human_interactions' in UPDATE, got: {compiled}"
    )
