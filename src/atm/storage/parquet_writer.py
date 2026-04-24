"""Buffered async Parquet writer for ATM storage streams.

Implements per-stream buffering with two auto-flush triggers:
  (a) buffer reaches buffer_rows rows  — invariant (d) from arch.md §10.3
  (b) time since first buffered row >= buffer_seconds

File layout:
  root/experiments/{exp_id}/runs/{run_id}/{stream}.parquet
  root/experiments/{exp_id}/runs/{run_id}/scratchpads/{agent_id}.parquet

Design note: pyarrow.parquet.ParquetWriter does not support append mode (reopening
the same path creates a new file, losing prior data). To guarantee the file is always
in a valid readable state after every flush AND to preserve all rows across multiple
flushes, each stream keeps a running ``flushed_rows`` list. On every flush, the complete
list (prior flushes + current buffer) is written atomically via a fresh
pq.ParquetWriter that is immediately closed. The cost is O(total_rows) per flush, which
is acceptable for the buffer sizes used in experiments (buffer_rows defaults to 100).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from atm.storage.schemas import (
    LLM_CALL_SCHEMA,
    MESSAGE_SCHEMA,
    PHASE_SCHEMA,
    SCRATCHPAD_SCHEMA,
    TOOL_CALL_SCHEMA,
    TOPOLOGY_TRANSITION_SCHEMA,
)


@dataclass
class _StreamState:
    """Internal state for a single parquet stream."""

    path: Path
    schema: pa.Schema
    # Rows that have been flushed at least once (complete history on disk)
    flushed_rows: list[dict] = field(default_factory=list)  # type: ignore[type-arg]
    # Rows buffered since last flush
    buffer: list[dict] = field(default_factory=list)  # type: ignore[type-arg]
    first_buffered_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def flush_to_disk(self, compression: str) -> None:
        """Write all rows (flushed + buffered) to disk as a single parquet file.

        After writing, moves buffer into flushed_rows and clears buffer.
        The file is always complete and readable after this call.
        """
        if not self.buffer:
            return
        all_rows = self.flushed_rows + self.buffer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(all_rows, schema=self.schema)
        writer = pq.ParquetWriter(self.path, self.schema, compression=compression)  # type: ignore[no-untyped-call]
        writer.write_table(table)  # type: ignore[no-untyped-call]
        writer.close()  # type: ignore[no-untyped-call]
        self.flushed_rows = all_rows
        self.buffer.clear()
        self.first_buffered_at = None


class ParquetWriter:
    """Buffered append-only async Parquet writer.

    One file per stream; auto-flushes when buffer reaches buffer_rows rows OR when
    time since first buffered row exceeds buffer_seconds.

    Args:
        root: Root directory for data storage.
        run_id: UUID of the current run.
        exp_id: UUID of the current experiment.
        buffer_rows: Number of rows to buffer before auto-flush (default 100).
        buffer_seconds: Seconds since first buffered row before auto-flush (default 5.0).
        compression: Parquet compression codec (default "snappy").
    """

    def __init__(
        self,
        root: Path,
        run_id: UUID,
        exp_id: UUID,
        *,
        buffer_rows: int = 100,
        buffer_seconds: float = 5.0,
        compression: str = "snappy",
    ) -> None:
        self._root = root
        self._run_id = run_id
        self._exp_id = exp_id
        self._buffer_rows = buffer_rows
        self._buffer_seconds = buffer_seconds
        self._compression = compression
        self._closed = False

        run_dir = root / "experiments" / str(exp_id) / "runs" / str(run_id)

        # Pre-defined streams (non-scratchpad)
        self._streams: dict[str, _StreamState] = {
            "llm_calls": _StreamState(run_dir / "llm_calls.parquet", LLM_CALL_SCHEMA),
            "messages": _StreamState(run_dir / "messages.parquet", MESSAGE_SCHEMA),
            "tool_calls": _StreamState(run_dir / "tool_calls.parquet", TOOL_CALL_SCHEMA),
            "phases": _StreamState(run_dir / "phases.parquet", PHASE_SCHEMA),
            "topology_transitions": _StreamState(
                run_dir / "topology_transitions.parquet", TOPOLOGY_TRANSITION_SCHEMA
            ),
        }

        # Scratchpad streams are created on demand, keyed by agent_id
        self._scratchpad_dir = run_dir / "scratchpads"
        self._scratchpads: dict[str, _StreamState] = {}
        self._scratchpad_registry_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_auto_flush(self, state: _StreamState) -> bool:
        if len(state.buffer) >= self._buffer_rows:
            return True
        return (
            state.first_buffered_at is not None
            and time.monotonic() - state.first_buffered_at >= self._buffer_seconds
        )

    async def _append(self, state: _StreamState, row: dict) -> None:  # type: ignore[type-arg]
        """Append a row to the stream's buffer and auto-flush if needed."""
        async with state.lock:
            if state.first_buffered_at is None:
                state.first_buffered_at = time.monotonic()
            state.buffer.append(row)
            if self._should_auto_flush(state):
                state.flush_to_disk(self._compression)

    async def _get_scratchpad_stream(self, agent_id: str) -> _StreamState:
        """Return (or lazily create) the scratchpad stream for agent_id."""
        async with self._scratchpad_registry_lock:
            if agent_id not in self._scratchpads:
                path = self._scratchpad_dir / f"{agent_id}.parquet"
                self._scratchpads[agent_id] = _StreamState(path, SCRATCHPAD_SCHEMA)
            return self._scratchpads[agent_id]

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    async def write_llm_call(self, row: dict) -> None:  # type: ignore[type-arg]
        """Buffer a row for the llm_calls stream."""
        await self._append(self._streams["llm_calls"], row)

    async def write_message(self, row: dict) -> None:  # type: ignore[type-arg]
        """Buffer a row for the messages stream."""
        await self._append(self._streams["messages"], row)

    async def write_tool_call(self, row: dict) -> None:  # type: ignore[type-arg]
        """Buffer a row for the tool_calls stream."""
        await self._append(self._streams["tool_calls"], row)

    async def write_phase(self, row: dict) -> None:  # type: ignore[type-arg]
        """Buffer a row for the phases stream."""
        await self._append(self._streams["phases"], row)

    async def write_topology_transition(self, row: dict) -> None:  # type: ignore[type-arg]
        """Buffer a row for the topology_transitions stream."""
        await self._append(self._streams["topology_transitions"], row)

    async def write_scratchpad(self, agent_id: str, row: dict) -> None:  # type: ignore[type-arg]
        """Buffer a row for the per-agent scratchpad stream."""
        state = await self._get_scratchpad_stream(agent_id)
        await self._append(state, row)

    # ------------------------------------------------------------------
    # Flush and close
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Flush ALL buffered rows to disk across all streams."""
        all_streams = list(self._streams.values()) + list(self._scratchpads.values())
        for state in all_streams:
            async with state.lock:
                state.flush_to_disk(self._compression)

    async def close(self) -> None:
        """Flush all streams and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        await self.flush()
