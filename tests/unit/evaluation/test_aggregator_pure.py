"""Unit tests for atm.evaluation.aggregator — pure in-memory tests (no PG).

4 tests:
  1. compute_quality success path — returns (score, details) when evaluator succeeds.
  2. compute_quality error path — returns (None, {"error": ...}) on exception.
  3. persist_quality basic mock — executes a SQL UPDATE via the session.
  4. aggregate_run shape — returns a float on success, calls persist.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from atm.evaluation.aggregator import aggregate_run, compute_quality, persist_quality
from atm.tasks.base import EvalResult, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(**kwargs: Any) -> TaskSpec:
    defaults: dict[str, Any] = {
        "id": "mmlu/test/0",
        "type": "qa",
        "input": "What is 2+2?",
        "expected": "4",
        "evaluator_key": "mmlu_exact_match",
    }
    defaults.update(kwargs)
    return TaskSpec(**defaults)


# ---------------------------------------------------------------------------
# 1. compute_quality success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_quality_success() -> None:
    """compute_quality returns (score, details) when score_ground_truth succeeds."""
    spec = _make_spec()
    expected_result = EvalResult(score=0.9, passed=True, details={"correct": True})

    with patch(
        "atm.evaluation.aggregator.score_ground_truth",
        new_callable=AsyncMock,
        return_value=expected_result,
    ):
        score, details = await compute_quality(spec, "4", run_seed=42)

    assert score == pytest.approx(0.9)
    assert details == {"correct": True}


# ---------------------------------------------------------------------------
# 2. compute_quality error path — returns (None, {"error": ...})
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_quality_error_path_returns_none() -> None:
    """compute_quality catches exceptions and returns (None, {"error": ...})."""
    spec = _make_spec()

    with patch(
        "atm.evaluation.aggregator.score_ground_truth",
        new_callable=AsyncMock,
        side_effect=ValueError("evaluator blew up"),
    ):
        score, details = await compute_quality(spec, "some answer")

    assert score is None
    assert "error" in details
    assert "evaluator blew up" in details["error"]


# ---------------------------------------------------------------------------
# 3. persist_quality basic mock — issues SQL UPDATE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_quality_executes_update() -> None:
    """persist_quality calls session.execute with a SQL statement."""
    mock_session = AsyncMock()
    run_id = uuid.uuid4()

    await persist_quality(mock_session, run_id, 0.75)

    mock_session.execute.assert_called_once()
    # Verify the argument is a SQL construct (not None)
    call_args = mock_session.execute.call_args
    assert call_args is not None
    stmt = call_args[0][0]
    # The statement should be a SQLAlchemy Update object
    assert stmt is not None


# ---------------------------------------------------------------------------
# 4. aggregate_run shape — returns float on success, calls persist
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_run_shape_on_success() -> None:
    """aggregate_run returns a float and calls persist_quality once."""
    spec = _make_spec()
    mock_session = AsyncMock()
    run_id = uuid.uuid4()

    expected_result = EvalResult(score=1.0, passed=True, details={})

    with patch(
        "atm.evaluation.aggregator.score_ground_truth",
        new_callable=AsyncMock,
        return_value=expected_result,
    ):
        result = await aggregate_run(mock_session, run_id, spec, "correct answer")

    assert result == pytest.approx(1.0)
    # persist_quality should have been called (executes SQL UPDATE)
    mock_session.execute.assert_called_once()
