"""Unit tests for human_sim_cognitive_load_proxy (m9.2 step 2.2).

Tests:
  1. empty  — no rows for run_id → returns 0.0
  2. k=3    — three interactions with known context sizes and latencies
              → expected value computed using DEFAULT_WEIGHTS
  3. missing answered_at — latency contributes 0.0 for that row
  4. _load_weights       — returns DEFAULT_WEIGHTS when file is missing
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from atm.evaluation.metrics import DEFAULT_WEIGHTS, _load_weights, human_sim_cognitive_load_proxy


def _make_session(rows: list[tuple[Any, ...]]) -> AsyncMock:
    """Build a mock AsyncSession whose execute() returns *rows*."""
    mock_result = MagicMock()
    mock_result.fetchall.return_value = rows

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


def _ctx_bytes(ctx: dict[str, Any]) -> int:
    """Return the byte-length of json.dumps(ctx, default=str)."""
    return len(json.dumps(ctx, default=str))


@pytest.mark.asyncio
async def test_empty_returns_zero() -> None:
    """No human_interactions for run_id → cognitive_load_proxy == 0.0."""
    session = _make_session([])
    run_id = uuid.uuid4()

    result = await human_sim_cognitive_load_proxy(session, run_id)

    assert result == 0.0


@pytest.mark.asyncio
async def test_k3_interactions_expected_value() -> None:
    """Three interactions with known data → formula gives the expected float."""
    run_id = uuid.uuid4()
    now = datetime.now(UTC)

    ctx_a = {"question": "approve?", "messages": ["hello"]}
    ctx_b = {"question": "reject?", "messages": ["world", "foo"]}
    ctx_c = {"question": "review?", "messages": ["bar", "baz", "qux"]}

    lat_a = 5.0
    lat_b = 10.0
    lat_c = 15.0

    rows = [
        (
            uuid.uuid4(),
            ctx_a,
            now,
            now + timedelta(seconds=lat_a),
        ),
        (
            uuid.uuid4(),
            ctx_b,
            now,
            now + timedelta(seconds=lat_b),
        ),
        (
            uuid.uuid4(),
            ctx_c,
            now,
            now + timedelta(seconds=lat_c),
        ),
    ]

    session = _make_session(rows)

    result = await human_sim_cognitive_load_proxy(session, run_id)

    alpha = DEFAULT_WEIGHTS["alpha"]
    beta = DEFAULT_WEIGHTS["beta"]
    gamma = DEFAULT_WEIGHTS["gamma"]

    count = 3
    mean_ctx = (_ctx_bytes(ctx_a) + _ctx_bytes(ctx_b) + _ctx_bytes(ctx_c)) / count
    mean_lat = (lat_a + lat_b + lat_c) / count

    expected = alpha * count + beta * mean_ctx + gamma * mean_lat

    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"


@pytest.mark.asyncio
async def test_missing_answered_at_contributes_zero_latency() -> None:
    """Rows with answered_at=None contribute 0.0 to mean_latency."""
    run_id = uuid.uuid4()
    now = datetime.now(UTC)

    ctx_answered = {"q": "answered"}
    ctx_unanswered = {"q": "unanswered"}

    lat_answered = 8.0

    rows = [
        (
            uuid.uuid4(),
            ctx_answered,
            now,
            now + timedelta(seconds=lat_answered),
        ),
        (
            uuid.uuid4(),
            ctx_unanswered,
            now,
            None,
        ),
    ]

    session = _make_session(rows)

    result = await human_sim_cognitive_load_proxy(session, run_id)

    alpha = DEFAULT_WEIGHTS["alpha"]
    beta = DEFAULT_WEIGHTS["beta"]
    gamma = DEFAULT_WEIGHTS["gamma"]

    count = 2
    mean_ctx = (_ctx_bytes(ctx_answered) + _ctx_bytes(ctx_unanswered)) / count
    mean_lat = (lat_answered + 0.0) / count

    expected = alpha * count + beta * mean_ctx + gamma * mean_lat

    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"


def test_load_weights_missing_file_returns_defaults() -> None:
    """_load_weights falls back to DEFAULT_WEIGHTS when the file does not exist."""
    missing = Path("/tmp/nonexistent_cognitive_load_weights_m9_2.yaml")
    assert not missing.exists(), "Precondition: file must not exist"

    weights = _load_weights(path=missing)

    assert weights == DEFAULT_WEIGHTS
