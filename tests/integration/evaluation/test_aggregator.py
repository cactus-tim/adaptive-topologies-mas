"""Integration tests for atm.evaluation.aggregator — requires PostgreSQL.

Skipped unless ATM_INTEGRATION_PG=1 is set in the environment.

4 tests:
  1. persist_quality writes quality_score column.
  2. persist_quality is idempotent (second call overwrites value).
  3. persist_quality on unknown run_id is a no-op (does not raise).
  4. compute_quality + persist_quality round-trip: MMLU correct answer → 1.0.

Uses fixtures from tests/integration/conftest.py:
  - pg_engine_fast, session_factory_fast (via pg_dsn).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.evaluation.aggregator import aggregate_run, persist_quality
from atm.storage.models import Experiment, Run
from atm.storage.session import create_session_factory, session_scope
from atm.tasks.base import EvalResult, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mmlu_spec() -> TaskSpec:
    return TaskSpec(
        id="gsm8k/test/0",
        type="reasoning",
        input="What is 6 * 7?",
        expected="42",
        evaluator_key="gsm8k_numeric",
        metadata={},
    )


async def _insert_exp_and_run(
    factory,  # type: ignore[no-untyped-def]
    *,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
) -> None:
    """Insert a minimal Experiment + Run into the DB."""
    async with session_scope(factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"eval-test-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="chain",
                task_id="gsm8k/test/0",
                agent_set="canonical_4",
                seed=42,
                model="fake:echo",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
            )
        )


# ---------------------------------------------------------------------------
# 1. persist_quality writes quality_score column
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_persist_quality_writes_column(
    pg_engine_fast: AsyncEngine,
) -> None:
    """persist_quality updates runs.quality_score to the given value."""
    factory = create_session_factory(pg_engine_fast)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    await _insert_exp_and_run(factory, exp_id=exp_id, run_id=run_id)

    async with session_scope(factory) as session:
        await persist_quality(session, run_id, 0.75)

    # Verify the column was updated
    async with session_scope(factory) as session:
        result = await session.execute(select(Run).where(Run.id == run_id))
        run_row = result.scalar_one()
        assert run_row.quality_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 2. persist_quality is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_persist_quality_idempotent(
    pg_engine_fast: AsyncEngine,
) -> None:
    """persist_quality can be called twice; the last value wins."""
    factory = create_session_factory(pg_engine_fast)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    await _insert_exp_and_run(factory, exp_id=exp_id, run_id=run_id)

    # First call
    async with session_scope(factory) as session:
        await persist_quality(session, run_id, 0.5)

    # Second call — overwrites
    async with session_scope(factory) as session:
        await persist_quality(session, run_id, 1.0)

    async with session_scope(factory) as session:
        result = await session.execute(select(Run).where(Run.id == run_id))
        run_row = result.scalar_one()
        assert run_row.quality_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. persist_quality on unknown run_id — no-op (does not raise)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_persist_quality_unknown_run_id_noop(
    pg_engine_fast: AsyncEngine,
) -> None:
    """persist_quality on an unknown run_id does not raise."""
    factory = create_session_factory(pg_engine_fast)
    ghost_run_id = uuid.uuid4()  # never inserted

    # Should not raise even though the run does not exist
    async with session_scope(factory) as session:
        await persist_quality(session, ghost_run_id, 0.9)  # no-op


# ---------------------------------------------------------------------------
# 4. compute_quality + persist_quality round-trip (MMLU correct answer → 1.0)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_compute_and_persist_round_trip(
    pg_engine_fast: AsyncEngine,
) -> None:
    """Full round-trip: compute_quality (mocked MMLU) + persist_quality → 1.0."""
    factory = create_session_factory(pg_engine_fast)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    await _insert_exp_and_run(factory, exp_id=exp_id, run_id=run_id)

    spec = _mmlu_spec()
    answer = "42"  # correct answer for the gsm8k spec returned by helper

    # Mock score_ground_truth to return a passing EvalResult (correct MMLU answer)
    mock_result = EvalResult(score=1.0, passed=True, details={"match": True})

    with patch(
        "atm.evaluation.aggregator.score_ground_truth",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        async with session_scope(factory) as session:
            score = await aggregate_run(session, run_id, spec, answer)

    assert score == pytest.approx(1.0)

    # Verify persisted
    async with session_scope(factory) as session:
        result = await session.execute(select(Run).where(Run.id == run_id))
        run_row = result.scalar_one()
        assert run_row.quality_score == pytest.approx(1.0)
