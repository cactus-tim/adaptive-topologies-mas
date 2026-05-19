"""Unit tests for `build_llm` replay mode wiring (m12-resume-replay).

Verifies that ``build_llm(model_id="fake:replay", replay_source=...)`` loads
the supplied Parquet file and returns an :class:`LLMWrapper` backed by a
:class:`FakeLLM` in replay mode that emits rows from the table.
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atm.llm.budget import BudgetTracker
from atm.llm.factory import build_llm
from atm.llm.fake import REPLAY_SCHEMA, FakeLLM
from atm.llm.pricing import Pricing


def _make_replay_table(call_id: str, content: str) -> pa.Table:
    """Build a single-row replay-schema table."""
    now_str = datetime.datetime.now(datetime.UTC).isoformat()
    data = {
        "call_id": pa.array([call_id], type=pa.string()),
        "model": pa.array(["fake:replay"], type=pa.string()),
        "content": pa.array([content], type=pa.string()),
        "usage_input": pa.array([10], type=pa.int64()),
        "usage_output": pa.array([5], type=pa.int64()),
        "usage_total": pa.array([15], type=pa.int64()),
        "usage_cached": pa.array([0], type=pa.int64()),
        "cost_usd": pa.array([0.0], type=pa.float64()),
        "latency_ms": pa.array([0], type=pa.int64()),
        "finish_reason": pa.array(["stop"], type=pa.string()),
        "started_at": pa.array([now_str], type=pa.string()),
        "tool_calls_json": pa.array(["[]"], type=pa.string()),
    }
    return pa.table(data, schema=REPLAY_SCHEMA)


def _budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0)


def _pricing() -> Pricing:
    return Pricing(version=1, models={})


def test_build_llm_replay_loads_parquet(tmp_path: Path) -> None:
    """build_llm(fake:replay, replay_source=...) loads the parquet table."""
    parquet_path = tmp_path / "llm_calls.parquet"
    table = _make_replay_table(call_id=str(uuid.uuid4()), content="hello replay")
    pq.write_table(table, parquet_path)

    wrapper = build_llm(
        model_id="fake:replay",
        pricing=_pricing(),
        budget=_budget(),
        replay_source=parquet_path,
    )

    assert wrapper.model_id == "fake:replay"
    inner = wrapper._llm  # type: ignore[attr-defined]
    assert isinstance(inner, FakeLLM)
    assert inner._replay_table is not None  # type: ignore[attr-defined]
    assert inner._replay_table.num_rows == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_build_llm_replay_emits_row_content(tmp_path: Path) -> None:
    """The FakeLLM created by build_llm returns content from the replay table."""
    parquet_path = tmp_path / "calls.parquet"
    call_uuid = str(uuid.uuid4())
    table = _make_replay_table(call_id=call_uuid, content="the replayed answer")
    pq.write_table(table, parquet_path)

    wrapper = build_llm(
        model_id="fake:replay",
        pricing=_pricing(),
        budget=_budget(),
        replay_source=parquet_path,
    )

    response = await wrapper._llm.ainvoke(messages=[], agent_id="any")  # type: ignore[attr-defined]
    assert response.text == "the replayed answer"
    assert str(response.id) == call_uuid


def test_build_llm_replay_missing_source_raises() -> None:
    """fake:replay without replay_source raises ValueError."""
    with pytest.raises(ValueError, match="replay_source"):
        build_llm(
            model_id="fake:replay",
            pricing=_pricing(),
            budget=_budget(),
        )


def test_build_llm_replay_missing_file_raises(tmp_path: Path) -> None:
    """fake:replay with a non-existent path raises FileNotFoundError."""
    missing = tmp_path / "no-such-file.parquet"
    with pytest.raises(FileNotFoundError):
        build_llm(
            model_id="fake:replay",
            pricing=_pricing(),
            budget=_budget(),
            replay_source=missing,
        )


def test_build_llm_replay_source_accepts_string(tmp_path: Path) -> None:
    """replay_source may be a string path, not just Path."""
    parquet_path = tmp_path / "calls.parquet"
    pq.write_table(_make_replay_table(str(uuid.uuid4()), "y"), parquet_path)

    wrapper = build_llm(
        model_id="fake:replay",
        pricing=_pricing(),
        budget=_budget(),
        replay_source=str(parquet_path),
    )

    inner = wrapper._llm  # type: ignore[attr-defined]
    assert isinstance(inner, FakeLLM)
