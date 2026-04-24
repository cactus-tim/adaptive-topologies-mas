"""Unit tests for atm.storage.parquet_writer — ParquetWriter buffered async writer.

Tests cover:
1. tz-aware round-trip: write row, flush, read back, tzinfo is not None
2. 3 writes + flush → pa.parquet.read_table reports num_rows == 3
3. auto-flush on buffer_rows overflow: buffer_rows=2, 3 writes → file has 2 rows before explicit flush
4. close idempotent: double-close raises no exception
5. schema mismatch raises pa.lib.ArrowInvalid
6. scratchpad per-agent file: two different agent_ids → two separate files
7. multi-stream concurrency: concurrent writes to different streams → no corruption
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _sample_llm_row(run_id: str) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "agent_id": "agent-1",
        "model": "gpt-4o",
        "at": _utcnow(),
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_hit_tokens": 0,
        "cost_usd": 0.001,
        "latency_ms": 123.4,
        "cache_scope": "none",
        "fingerprint": "abc123",
    }


def _sample_message_row(run_id: str) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "message_id": str(uuid4()),
        "at": _utcnow(),
        "from_agent": "agent-1",
        "to_agent": "agent-2",
        "mtype": "text",
        "payload_json": "{}",
    }


def _sample_tool_call_row(run_id: str) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "agent_id": "agent-1",
        "tool_name": "search",
        "at": _utcnow(),
        "latency_ms": 200.0,
        "ok": True,
        "args_json": "{}",
        "result_json": "{}",
        "error": "",
    }


def _sample_phase_row(run_id: str) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "phase_name": "explore",
        "from_phase": "init",
        "started_at": _utcnow(),
        "ended_at": _utcnow(),
        "entry_reason": "start",
        "topology_used": "star",
        "decided_by": "coordinator",
    }


def _sample_topology_transition_row(run_id: str) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "from_topology": "star",
        "to_topology": "mesh",
        "phase_at_decision": "planning",
        "iter_within_phase": 1,
        "iter_within_topology": 0,
        "decided_by": "rule",
        "reason": "test",
        "considered_alternatives_json": "[]",
        "guards_applied_json": "[]",
        "signals_snapshot_json": "{}",
        "router_cost_usd": 0.001,
        "at": _utcnow(),
    }


def _sample_scratchpad_row(run_id: str) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "agent_id": "agent-1",
        "step_idx": 0,
        "at": _utcnow(),
        "role": "assistant",
        "content": "thinking...",
        "tool_calls_json": "[]",
    }


# ---------------------------------------------------------------------------
# Test 1: tz-aware round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tz_aware_roundtrip(tmp_path: Path) -> None:
    """Write tz-aware row, flush, read back; column 'at' tzinfo must not be None."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=100)

    row = _sample_llm_row(str(run_id))
    await writer.write_llm_call(row)
    await writer.flush()
    await writer.close()

    expected_path = (
        tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id) / "llm_calls.parquet"
    )
    assert expected_path.exists()

    table = pq.read_table(expected_path)
    at_value = table.column("at")[0].as_py()
    assert at_value is not None
    assert at_value.tzinfo is not None, "tzinfo should not be None after round-trip"


# ---------------------------------------------------------------------------
# Test 2: 3 writes + flush → 3 rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_3_flush_read_3(tmp_path: Path) -> None:
    """3 write_llm_call + flush → parquet file has num_rows == 3."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=100)

    for _ in range(3):
        await writer.write_llm_call(_sample_llm_row(str(run_id)))

    await writer.flush()
    await writer.close()

    path = tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id) / "llm_calls.parquet"
    table = pq.read_table(path)
    assert table.num_rows == 3


# ---------------------------------------------------------------------------
# Test 3: auto-flush on buffer_rows overflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_flush_on_buffer_rows(tmp_path: Path) -> None:
    """buffer_rows=2, after 3 writes, first 2 rows already on disk before explicit flush."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=2)

    # Write 3 rows — after 2nd write, auto-flush should have occurred
    await writer.write_llm_call(_sample_llm_row(str(run_id)))
    await writer.write_llm_call(_sample_llm_row(str(run_id)))
    await writer.write_llm_call(_sample_llm_row(str(run_id)))

    path = tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id) / "llm_calls.parquet"
    # File must exist and have at least 2 rows (the auto-flushed batch)
    assert path.exists(), "File should exist after auto-flush"
    table = pq.read_table(path)
    assert table.num_rows >= 2, f"Expected >= 2 rows after auto-flush, got {table.num_rows}"

    await writer.close()


# ---------------------------------------------------------------------------
# Test 4: close idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_idempotent(tmp_path: Path) -> None:
    """Double-close must not raise any exception."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id)

    await writer.write_llm_call(_sample_llm_row(str(run_id)))
    await writer.close()
    await writer.close()  # must not raise


# ---------------------------------------------------------------------------
# Test 5: schema mismatch raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_mismatch_raises(tmp_path: Path) -> None:
    """Writing a row with wrong field type should raise ArrowInvalid or similar."""
    import pyarrow as pa

    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=100)

    bad_row = _sample_llm_row(str(run_id))
    # input_tokens should be int32; provide a string that can't be cast
    bad_row["input_tokens"] = "not-an-integer"

    with pytest.raises((pa.lib.ArrowInvalid, pa.lib.ArrowTypeError, TypeError, ValueError)):
        await writer.write_llm_call(bad_row)
        await writer.flush()

    # cleanup — ignore errors during close since writer may be in bad state
    import contextlib

    with contextlib.suppress(Exception):
        await writer.close()


# ---------------------------------------------------------------------------
# Test 6: scratchpad per-agent file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scratchpad_per_agent_file(tmp_path: Path) -> None:
    """Two different agent_ids → two separate scratchpad files."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=100)

    row_a = _sample_scratchpad_row(str(run_id))
    row_a["agent_id"] = "agent-A"

    row_b = _sample_scratchpad_row(str(run_id))
    row_b["agent_id"] = "agent-B"

    await writer.write_scratchpad("agent-A", row_a)
    await writer.write_scratchpad("agent-B", row_b)
    await writer.flush()
    await writer.close()

    base = tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id) / "scratchpads"
    assert (base / "agent-A.parquet").exists(), "agent-A.parquet not found"
    assert (base / "agent-B.parquet").exists(), "agent-B.parquet not found"

    table_a = pq.read_table(base / "agent-A.parquet")
    table_b = pq.read_table(base / "agent-B.parquet")
    assert table_a.num_rows == 1
    assert table_b.num_rows == 1


# ---------------------------------------------------------------------------
# Test 7: multi-stream concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_stream_concurrency(tmp_path: Path) -> None:
    """Concurrent writes to different streams via asyncio.gather land correctly."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=100)

    n = 10
    llm_rows = [_sample_llm_row(str(run_id)) for _ in range(n)]
    msg_rows = [_sample_message_row(str(run_id)) for _ in range(n)]
    tool_rows = [_sample_tool_call_row(str(run_id)) for _ in range(n)]

    await asyncio.gather(
        *[writer.write_llm_call(r) for r in llm_rows],
        *[writer.write_message(r) for r in msg_rows],
        *[writer.write_tool_call(r) for r in tool_rows],
    )
    await writer.flush()
    await writer.close()

    base = tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id)
    assert pq.read_table(base / "llm_calls.parquet").num_rows == n
    assert pq.read_table(base / "messages.parquet").num_rows == n
    assert pq.read_table(base / "tool_calls.parquet").num_rows == n


# ---------------------------------------------------------------------------
# Test 8: write_scratchpad rejects path traversal / unsafe agent_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_scratchpad_rejects_path_traversal(tmp_path: Path) -> None:
    """write_scratchpad must raise ValueError for agent_id values that fail safety regex."""
    from atm.storage.parquet_writer import ParquetWriter

    writer = ParquetWriter(tmp_path, uuid4(), uuid4())
    for bad in ["../evil", "agent/../..", "", "agent\0", "ab/cd", "." * 100]:
        with pytest.raises(ValueError):
            await writer.write_scratchpad(bad, _sample_scratchpad_row(str(uuid4())))
    await writer.close()


# ---------------------------------------------------------------------------
# Test 9: all stream write methods + flush work without error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_streams_write_and_flush(tmp_path: Path) -> None:
    """write_* for all 6 streams + flush + close without error."""
    from atm.storage.parquet_writer import ParquetWriter

    run_id = uuid4()
    exp_id = uuid4()
    writer = ParquetWriter(tmp_path, run_id, exp_id, buffer_rows=100)

    await writer.write_llm_call(_sample_llm_row(str(run_id)))
    await writer.write_message(_sample_message_row(str(run_id)))
    await writer.write_tool_call(_sample_tool_call_row(str(run_id)))
    await writer.write_phase(_sample_phase_row(str(run_id)))
    await writer.write_topology_transition(_sample_topology_transition_row(str(run_id)))
    await writer.write_scratchpad("agent-1", _sample_scratchpad_row(str(run_id)))

    await writer.flush()
    await writer.close()

    base = tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id)
    assert (base / "llm_calls.parquet").exists()
    assert (base / "messages.parquet").exists()
    assert (base / "tool_calls.parquet").exists()
    assert (base / "phases.parquet").exists()
    assert (base / "topology_transitions.parquet").exists()
    assert (base / "scratchpads" / "agent-1.parquet").exists()
