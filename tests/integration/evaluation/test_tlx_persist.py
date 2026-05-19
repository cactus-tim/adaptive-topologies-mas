"""Integration tests for atm.evaluation.tlx.persist_tlx — requires PostgreSQL.

Skipped unless ATM_ENABLE_PG_TESTS=1 is set in the environment.

3 tests:
  1. persist_tlx writes raw_tlx_score to human_interactions row.
  2. persist_tlx is idempotent (second call overwrites with new value).
  3. persist_tlx on unknown interaction_id is a no-op (does not raise).

Uses fixtures from tests/integration/conftest.py:
  - pg_engine_fast (function scope) — creates schema via Base.metadata.create_all.
  - session_factory_fast (function scope) — async_sessionmaker bound to pg_engine_fast.

FK chain: Experiment → Run → HumanInteraction (mirrors test_aggregator.py patterns).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.evaluation.tlx import persist_tlx
from atm.storage.models import Experiment, HumanInteraction, Run
from atm.storage.session import create_session_factory, session_scope


async def _insert_exp_run_interaction(
    factory,  # type: ignore[no-untyped-def]
    *,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    interaction_id: uuid.UUID,
) -> None:
    """Insert Experiment → Run → HumanInteraction with raw_tlx_score=None."""
    async with session_scope(factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"tlx-test-{exp_id}",
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
                task_id="mmlu/test/0",
                agent_set="canonical_4",
                seed=0,
                model="fake:echo",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
            )
        )
    async with session_scope(factory) as session:
        session.add(
            HumanInteraction(
                id=interaction_id,
                run_id=run_id,
                role="user",
                context_json={},
                raw_tlx_score=None,
            )
        )


@pytest.mark.integration
async def test_persist_tlx_writes_column(
    pg_engine_fast: AsyncEngine,
) -> None:
    """persist_tlx updates human_interactions.raw_tlx_score to given value."""
    factory = create_session_factory(pg_engine_fast)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    interaction_id = uuid.uuid4()
    await _insert_exp_run_interaction(
        factory, exp_id=exp_id, run_id=run_id, interaction_id=interaction_id
    )

    async with session_scope(factory) as session:
        await persist_tlx(session, interaction_id, raw_score=42.5)

    async with session_scope(factory) as session:
        result = await session.execute(
            select(HumanInteraction).where(HumanInteraction.id == interaction_id)
        )
        row = result.scalar_one()
        assert row.raw_tlx_score == pytest.approx(42.5)


@pytest.mark.integration
async def test_persist_tlx_idempotent_overwrite(
    pg_engine_fast: AsyncEngine,
) -> None:
    """persist_tlx can be called twice; the second value wins."""
    factory = create_session_factory(pg_engine_fast)
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    interaction_id = uuid.uuid4()
    await _insert_exp_run_interaction(
        factory, exp_id=exp_id, run_id=run_id, interaction_id=interaction_id
    )

    async with session_scope(factory) as session:
        await persist_tlx(session, interaction_id, raw_score=42.5)

    async with session_scope(factory) as session:
        await persist_tlx(session, interaction_id, raw_score=88.0)

    async with session_scope(factory) as session:
        result = await session.execute(
            select(HumanInteraction).where(HumanInteraction.id == interaction_id)
        )
        row = result.scalar_one()
        assert row.raw_tlx_score == pytest.approx(88.0)


@pytest.mark.integration
async def test_persist_tlx_unknown_id_noop(
    pg_engine_fast: AsyncEngine,
) -> None:
    """persist_tlx on an unknown interaction_id does not raise."""
    factory = create_session_factory(pg_engine_fast)
    ghost_id = uuid.uuid4()

    async with session_scope(factory) as session:
        await persist_tlx(session, ghost_id, raw_score=99.0)  # no-op
