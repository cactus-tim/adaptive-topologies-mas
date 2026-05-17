"""Unit tests for _handle_human_response — TLX fields and study_session_id extension (M14 §3.2).

Tests cover BOTH branches of _handle_human_response:
  - INSERT path  (existing is None)
  - UPDATE path  (existing.response_json is None)

For each branch the tests verify:
  1. tlx_scores JSONB written when valid TLX dict present
  2. raw_tlx_score FLOAT computed correctly via NasaTLX.raw_score
  3. study_session_id UUID extracted from response_json["payload"]["study_session_id"]
  4. tlx_scores = None when response_json has no tlx_scores key
  5. raw_tlx_score = None when tlx_scores is None or invalid
  6. study_session_id = None when key missing from payload
  7. study_session_id = None when UUID string is malformed
  8. NasaTLX raises ValidationError (e.g. value out-of-range) → raw_tlx_score = None, no exception
  9. Third branch (response already filled) — no UPDATE/INSERT after the initial SELECT
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers shared between all tests
# ---------------------------------------------------------------------------

_VALID_TLX: dict[str, int] = {
    "mental_demand": 60,
    "physical_demand": 20,
    "temporal_demand": 40,
    "performance": 50,
    "effort": 70,
    "frustration": 30,
}

# raw_score = (60 + 20 + 40 + (100-50) + 70 + 30) / 6 = 270/6 = 45.0
_VALID_TLX_RAW_SCORE: float = 45.0

_SAMPLE_SESSION_ID: str = "12345678-1234-5678-1234-567812345678"
_SAMPLE_SESSION_UUID: UUID = UUID(_SAMPLE_SESSION_ID)


def _build_session_factory() -> MagicMock:
    """Minimal async session factory mock; scalar_one_or_none returns None by default."""
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

    return ExperimentCallbackHandler(
        run_id=uuid4(),
        exp_id=uuid4(),
        session_factory=session_factory,
        parquet_writer=parquet_writer,
    )


def _response_data(
    *,
    run_id: UUID | None = None,
    request_id: str = "chain:reviewer:0",
    tlx_scores: dict[str, int] | None = _VALID_TLX,
    study_session_id: str | None = _SAMPLE_SESSION_ID,
) -> dict[str, Any]:
    """Build a human_response event payload with optional TLX and study_session_id."""
    payload: dict[str, Any] = {}
    if study_session_id is not None:
        payload["study_session_id"] = study_session_id

    resp: dict[str, Any] = {
        "action": "approve",
        "source": "human",
        "timed_out": False,
        "payload": payload,
    }
    if tlx_scores is not None:
        resp["tlx_scores"] = tlx_scores

    return {
        "run_id": run_id or uuid4(),
        "request_id": request_id,
        "answered_at": datetime.now(UTC),
        "response_json": resp,
    }


def _extract_insert_values(stmt: Any) -> dict[str, Any]:
    """Extract the VALUES dict from a SQLAlchemy INSERT statement, keyed by column name."""
    # stmt._values is an immutabledict keyed by Column objects.
    # Normalize to string-keyed dict for easy test assertions.
    return {col.key: bind for col, bind in stmt._values.items()}  # type: ignore[attr-defined]


def _extract_update_values(stmt: Any) -> dict[str, Any]:
    """Extract the VALUES dict from a SQLAlchemy UPDATE statement, keyed by column name."""
    return {col.key: bind for col, bind in stmt._values.items()}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# INSERT path (existing is None) — TLX fields present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_path_writes_tlx_scores() -> None:
    """INSERT branch must include tlx_scores in its VALUES."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    # 2 statements: SELECT then INSERT
    assert len(executed_stmts) == 2
    insert_stmt = executed_stmts[1]
    compiled = str(insert_stmt.compile(compile_kwargs={"literal_binds": False})).upper()
    assert "INSERT" in compiled
    # tlx_scores must appear in the INSERT column list
    compiled_cols = str(insert_stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "tlx_scores" in compiled_cols


@pytest.mark.asyncio
async def test_insert_path_writes_raw_tlx_score_correct_value() -> None:
    """INSERT branch must compute raw_tlx_score from NasaTLX and include it."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    raw = values["raw_tlx_score"].value  # BindParameter wraps the actual value
    assert raw == pytest.approx(_VALID_TLX_RAW_SCORE, abs=1e-9)


@pytest.mark.asyncio
async def test_insert_path_writes_study_session_id() -> None:
    """INSERT branch must extract study_session_id UUID from response_json payload."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    session_id = values["study_session_id"].value
    assert session_id == _SAMPLE_SESSION_UUID


# ---------------------------------------------------------------------------
# INSERT path — TLX missing / invalid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_path_tlx_none_stores_null_fields() -> None:
    """INSERT branch with tlx_scores=None must write None for both TLX fields."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(tlx_scores=None, study_session_id=None)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    assert values["tlx_scores"].value is None
    assert values["raw_tlx_score"].value is None


@pytest.mark.asyncio
async def test_insert_path_invalid_tlx_dict_stores_null_raw_score() -> None:
    """INSERT branch with out-of-range TLX scores must store None for raw_tlx_score."""
    bad_tlx = {
        "mental_demand": 200,  # out of range — NasaTLX validation fails
        "physical_demand": 20,
        "temporal_demand": 40,
        "performance": 50,
        "effort": 70,
        "frustration": 30,
    }
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(tlx_scores=bad_tlx)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    assert values["raw_tlx_score"].value is None


@pytest.mark.asyncio
async def test_insert_path_missing_tlx_keys_stores_null_raw_score() -> None:
    """INSERT branch with incomplete TLX dict (missing keys) must store None for raw_tlx_score."""
    incomplete_tlx = {"mental_demand": 50}  # 5 keys missing
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(tlx_scores=incomplete_tlx)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    assert values["raw_tlx_score"].value is None


# ---------------------------------------------------------------------------
# INSERT path — study_session_id edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_path_missing_study_session_id_stores_null() -> None:
    """INSERT branch with no study_session_id in payload must store None."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(study_session_id=None)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    assert values["study_session_id"].value is None


@pytest.mark.asyncio
async def test_insert_path_invalid_uuid_study_session_id_stores_null() -> None:
    """INSERT branch with a non-UUID string for study_session_id must store None."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(study_session_id="not-a-valid-uuid")
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    assert values["study_session_id"].value is None


# ---------------------------------------------------------------------------
# UPDATE path (existing.response_json is None) — TLX fields present
# ---------------------------------------------------------------------------


def _make_existing_row(response_json: dict[str, Any] | None = None) -> MagicMock:
    """Build a mock HumanInteraction row for the UPDATE-branch scenario."""
    from atm.storage.models import HumanInteraction

    row = MagicMock(spec=HumanInteraction)
    row.id = uuid4()
    row.response_json = response_json
    return row


@pytest.mark.asyncio
async def test_update_path_writes_tlx_scores() -> None:
    """UPDATE branch must include tlx_scores in its VALUES."""
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        # First call is SELECT — return existing row
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    # 2 statements: SELECT then UPDATE
    assert len(executed_stmts) == 2
    update_stmt = executed_stmts[1]
    compiled = str(update_stmt.compile(compile_kwargs={"literal_binds": False})).upper()
    assert "UPDATE" in compiled
    compiled_full = str(update_stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "tlx_scores" in compiled_full


@pytest.mark.asyncio
async def test_update_path_writes_raw_tlx_score_correct_value() -> None:
    """UPDATE branch must compute raw_tlx_score from NasaTLX and include it."""
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    update_stmt = executed_stmts[1]
    values = _extract_update_values(update_stmt)
    raw = values["raw_tlx_score"].value
    assert raw == pytest.approx(_VALID_TLX_RAW_SCORE, abs=1e-9)


@pytest.mark.asyncio
async def test_update_path_writes_study_session_id() -> None:
    """UPDATE branch must extract study_session_id UUID from response_json payload."""
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    update_stmt = executed_stmts[1]
    values = _extract_update_values(update_stmt)
    session_id = values["study_session_id"].value
    assert session_id == _SAMPLE_SESSION_UUID


# ---------------------------------------------------------------------------
# UPDATE path — TLX missing / invalid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_path_tlx_none_stores_null_fields() -> None:
    """UPDATE branch with tlx_scores=None must write None for both TLX fields."""
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(tlx_scores=None, study_session_id=None)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    update_stmt = executed_stmts[1]
    values = _extract_update_values(update_stmt)
    assert values["tlx_scores"].value is None
    assert values["raw_tlx_score"].value is None


@pytest.mark.asyncio
async def test_update_path_invalid_tlx_dict_stores_null_raw_score() -> None:
    """UPDATE branch with out-of-range TLX values must store None for raw_tlx_score."""
    bad_tlx = {
        "mental_demand": 999,  # out of range
        "physical_demand": 20,
        "temporal_demand": 40,
        "performance": 50,
        "effort": 70,
        "frustration": 30,
    }
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(tlx_scores=bad_tlx)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    update_stmt = executed_stmts[1]
    values = _extract_update_values(update_stmt)
    assert values["raw_tlx_score"].value is None


# ---------------------------------------------------------------------------
# UPDATE path — study_session_id edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_path_missing_study_session_id_stores_null() -> None:
    """UPDATE branch with no study_session_id in payload must store None."""
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(study_session_id=None)
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    update_stmt = executed_stmts[1]
    values = _extract_update_values(update_stmt)
    assert values["study_session_id"].value is None


@pytest.mark.asyncio
async def test_update_path_invalid_uuid_stores_null() -> None:
    """UPDATE branch with malformed study_session_id string must store None."""
    existing_row = _make_existing_row(response_json=None)
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    call_count = 0

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        nonlocal call_count
        executed_stmts.append(stmt)
        result = MagicMock()
        if call_count == 0:
            result.scalar_one_or_none = MagicMock(return_value=existing_row)
        else:
            result.scalar_one_or_none = MagicMock(return_value=None)
        call_count += 1
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data(study_session_id="NOT-A-UUID")
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    update_stmt = executed_stmts[1]
    values = _extract_update_values(update_stmt)
    assert values["study_session_id"].value is None


# ---------------------------------------------------------------------------
# Third branch — response already filled (no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_third_branch_no_op_when_response_already_filled() -> None:
    """When existing.response_json is already filled, no UPDATE/INSERT is issued."""
    existing_row = _make_existing_row(response_json={"action": "approve", "source": "human"})
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=existing_row)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    data = _response_data()
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    # Only the SELECT should have been executed
    assert len(executed_stmts) == 1
    compiled = str(executed_stmts[0].compile(compile_kwargs={"literal_binds": False})).upper()
    assert compiled.strip().startswith("SELECT")


# ---------------------------------------------------------------------------
# Regression: existing tests still pass (no TLX in old-style payloads)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_path_no_payload_no_exception() -> None:
    """INSERT branch with a legacy response_json (no payload/tlx_scores) must not raise."""
    executed_stmts: list[Any] = []
    session_factory = _build_session_factory()
    session = session_factory.return_value

    async def capture(stmt: Any, *a: Any, **kw: Any) -> MagicMock:
        executed_stmts.append(stmt)
        result = MagicMock()
        result.scalar_one_or_none = MagicMock(return_value=None)
        return result

    session.execute = AsyncMock(side_effect=capture)

    handler = _make_handler(session_factory=session_factory)
    # Old-style payload — no tlx_scores, no payload.study_session_id
    data = {
        "run_id": uuid4(),
        "request_id": "chain:reviewer:legacy",
        "answered_at": datetime.now(UTC),
        "response_json": {"action": "approve", "source": "llm_sim", "timed_out": False},
    }
    await handler.on_custom_event(name="human_response", data=data, run_id=uuid4())

    assert len(executed_stmts) == 2
    insert_stmt = executed_stmts[1]
    values = _extract_insert_values(insert_stmt)
    assert values["tlx_scores"].value is None
    assert values["raw_tlx_score"].value is None
    assert values["study_session_id"].value is None
