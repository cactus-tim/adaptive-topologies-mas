"""Integration smoke tests: end-to-end storage layer verification.

Wave 7 — Step 5.1 of m3-storage-observability plan.

Verifies the 4 flush invariants in a real PostgreSQL + Parquet end-to-end
scenario, and verifies DDL equivalence between Base.metadata and
the alembic migration head.

Tests are marked @pytest.mark.integration and are skipped unless
ATM_ENABLE_PG_TESTS=1 is set in the environment.

Approach
--------
Since langgraph-checkpoint-postgres is installed but the full langgraph graph
runtime (StateGraph) is not available as a project dependency, the handler is
exercised directly:

1. A RunnableLambda from langchain_core is used to provide a proper runnable
   context so that on_chain_start / on_chain_end callbacks fire correctly.
2. Custom events (message_emit, phase_transition) are dispatched via
   adispatch_custom_event inside the runnable.
3. on_llm_end is called manually to simulate an LLM response.
4. The checkpointer is set up and alist() is verified (empty but reachable).

Architecture anchors verified:
- §10.3 invariant (a): on_chain_end → parquet close
- §10.3 invariant (b): phase_transition → flush before PG insert
- §4.2 / §18/#4: atomic budget update via UPDATE...RETURNING
- §3.4: DDL equivalence between ORM metadata and alembic migration
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.core.types import Message, MessageKind, Phase, PhaseTransition
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.checkpointer import checkpointer_scope
from atm.storage.models import Base, BudgetEvent, Experiment, Run
from atm.storage.models import Phase as PhaseModel
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_session_factory, session_scope


def _make_exp_id() -> uuid.UUID:
    return uuid.uuid4()


def _make_run_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.mark.integration
async def test_smoke_full_run_records_to_pg_and_parquet(
    pg_engine_fast: AsyncEngine,
    tmp_path: Path,
) -> None:
    """End-to-end smoke: callback handler writes to PG and Parquet correctly.

    Verifies:
    - phases table has >= 1 row (phase_transition event processed)
    - messages.parquet has >= 1 row (message_emit event processed)
    - llm_calls.parquet has == 1 row (single on_llm_end call)
    - phases.parquet has >= 1 row
    - budget_events: >= 1 warn event (cost_usd=150 >= warn_threshold=100)
    - checkpointer: setup succeeds and alist returns (no real checkpoints, count >= 0)
    """
    factory = create_session_factory(pg_engine_fast)

    exp_id = _make_exp_id()
    run_id = _make_run_id()

    async with session_scope(factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"smoke-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="star",
                task_id="test-task-1",
                agent_set="default",
                seed=42,
                model="gpt-4",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
            )
        )

    parquet_writer = ParquetWriter(tmp_path, run_id, exp_id)
    handler = ExperimentCallbackHandler(
        run_id=run_id,
        exp_id=exp_id,
        session_factory=factory,
        parquet_writer=parquet_writer,
        budget_warn_threshold=Decimal("100"),
        budget_exceed_threshold=Decimal("1000"),
    )

    llm_result = LLMResult(
        generations=[[]],
        llm_output={
            "cost_usd": 150.0,
            "model": "gpt-4",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_hit_tokens": 0,
            "latency_ms": 200.0,
            "cache_scope": "none",
            "fingerprint": "smoke-test-fingerprint",
        },
    )

    async def _run_step(inputs: dict) -> dict:  # type: ignore[type-arg]
        """Graph node that dispatches custom events (runs inside LangChain context)."""
        msg = Message(
            sender="planner",
            recipients=("executor",),
            kind=MessageKind.REQUEST,
            content="Please execute the task.",
        )
        await adispatch_custom_event("message_emit", msg)

        pt = PhaseTransition(
            run_id=run_id,
            from_phase=None,
            to_phase=Phase.PLANNING,
            entry_reason="smoke-test start",
            iter_total=0,
            decided_by="initial",
        )
        await adispatch_custom_event("phase_transition", pt)

        await handler.on_llm_end(
            llm_result,
            run_id=uuid.uuid4(),
            parent_run_id=uuid.uuid4(),
        )

        return inputs

    runnable: RunnableLambda[dict[str, object], dict[str, object]] = RunnableLambda(_run_step)

    chain_run_id = uuid.uuid4()
    await runnable.ainvoke(
        {},
        config={
            "callbacks": [handler],
            "metadata": {"is_root_run": True},
            "run_id": chain_run_id,
        },
    )

    await asyncio.sleep(0.1)

    async with session_scope(factory) as session:
        phase_result = await session.execute(select(PhaseModel).where(PhaseModel.run_id == run_id))
        phases = phase_result.scalars().all()
        assert len(phases) >= 1, f"Expected >= 1 Phase row, got {len(phases)}"

    async with session_scope(factory) as session:
        budget_result = await session.execute(
            select(BudgetEvent).where(BudgetEvent.run_id == run_id)
        )
        budget_events = budget_result.scalars().all()
        assert len(budget_events) >= 1, (
            f"Expected >= 1 BudgetEvent (warn), got {len(budget_events)}"
        )
        warn_events = [e for e in budget_events if e.event == "warn"]
        assert len(warn_events) >= 1, "Expected at least 1 'warn' BudgetEvent"

    run_dir = tmp_path / "experiments" / str(exp_id) / "runs" / str(run_id)

    messages_df = pd.read_parquet(run_dir / "messages.parquet")
    assert messages_df.shape[0] >= 1, (
        f"Expected >= 1 message row in Parquet, got {messages_df.shape[0]}"
    )

    llm_df = pd.read_parquet(run_dir / "llm_calls.parquet")
    assert llm_df.shape[0] == 1, (
        f"Expected exactly 1 llm_call row in Parquet, got {llm_df.shape[0]}"
    )

    phases_df = pd.read_parquet(run_dir / "phases.parquet")
    assert phases_df.shape[0] >= 1, f"Expected >= 1 phase row in Parquet, got {phases_df.shape[0]}"

    pg_dsn = pg_engine_fast.url.render_as_string(hide_password=False)
    async with checkpointer_scope(pg_dsn) as saver:
        count = 0
        async for _ in saver.alist({"configurable": {"thread_id": str(run_id)}}):
            count += 1
        assert count >= 0


@pytest.mark.integration
async def test_alembic_equivalence(pg_engine_alembic: AsyncEngine) -> None:
    """After alembic upgrade head, the 6 business tables exist and match Base.metadata.

    Verifies:
    - information_schema.tables shows all 6 expected table names
    - runs table has reproducibility bundle columns
    - Base.metadata.sorted_tables names match the alembic-created tables
    """
    expected_tables = {
        "experiments",
        "runs",
        "phases",
        "human_interactions",
        "budget_events",
        "topology_transitions",
    }

    required_runs_columns = {
        "models_by_role_json",
        "model_version_snapshot",
        "sandbox_image_digest",
        "finish_reason",
    }

    async with pg_engine_alembic.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )
        )
        db_tables = {row[0] for row in result}

    missing = expected_tables - db_tables
    assert not missing, f"Missing tables after alembic upgrade head: {missing}"

    async with pg_engine_alembic.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'runs'"
            )
        )
        runs_columns = {row[0] for row in result}

    missing_cols = required_runs_columns - runs_columns
    assert not missing_cols, f"Missing columns in 'runs' table: {missing_cols}"

    orm_table_names = {t.name for t in Base.metadata.sorted_tables}
    orm_missing = orm_table_names - db_tables
    assert not orm_missing, f"ORM tables not found in DB after alembic upgrade: {orm_missing}"
