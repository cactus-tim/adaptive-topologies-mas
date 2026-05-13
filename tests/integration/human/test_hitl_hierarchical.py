"""Integration tests for HierarchicalTopology HITL integration (M9.1 Step 5).

Two scenarios:
  Scenario A — scope="top": 1 human interaction row written at top level.
  Scenario B — scope="sub_team": 2 rows written (one per team), proving
    callback propagation through compiled subgraphs (LangGraph 0.3+ contract).

Both scenarios use FakeLLM scripted fixtures and assert:
  - human_interactions count is correct (1 or 2)
  - request_ids contain "team_a" or "team_b" (for sub_team scenario)
  - runs.human_role = 'reviewer'

PG required: skip when ATM_ENABLE_PG_TESTS is not set.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from atm.core.types import HumanRole, Message, MessageKind
from atm.experiment.config import HumanCfg
from atm.human.llm_simulated import LLMSimulatedGateway
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope
from atm.topology.base import TopologyConfig
from atm.topology.hierarchical import HierarchicalTopology

# ---------------------------------------------------------------------------
# Skip / DSN setup
# ---------------------------------------------------------------------------

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


async def _insert_experiment_and_run(
    session_factory: Any,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    human_role: str | None = "reviewer",
) -> None:
    """Insert minimal Experiment + Run rows for FK constraints."""
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"hitl-hier-test-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="hierarchical",
                task_id="hitl-hier-test-task",
                agent_set="default",
                seed=42,
                model="fake:scripted",
                human_role=human_role,
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
            )
        )


def _build_handler(
    run_id: uuid.UUID,
    exp_id: uuid.UUID,
    session_factory: Any,
    tmp_path: Path,
) -> ExperimentCallbackHandler:
    parquet_writer = ParquetWriter(tmp_path, run_id, exp_id)
    return ExperimentCallbackHandler(
        run_id=run_id,
        exp_id=exp_id,
        session_factory=session_factory,
        parquet_writer=parquet_writer,
        budget_warn_threshold=Decimal("100"),
        budget_exceed_threshold=Decimal("1000"),
    )


def _make_mock_agent(agent_id: str) -> Any:
    """Create an agent that returns a DRAFT message."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict[str, Any]) -> dict[str, Any]:
        msg = Message(
            sender=agent_id,
            kind=MessageKind.DRAFT,
            content=f"draft from {agent_id}",
        )
        return {
            "agents": {agent_id: {"agent_id": agent_id, "outbox": [msg]}},
            "messages": [],
        }

    agent.step = fake_step
    return agent


def _make_agents() -> dict[str, Any]:
    return {
        "executor_a1": _make_mock_agent("executor_a1"),
        "executor_a2": _make_mock_agent("executor_a2"),
        "executor_b1": _make_mock_agent("executor_b1"),
        "executor_b2": _make_mock_agent("executor_b2"),
    }


# ---------------------------------------------------------------------------
# Scenario A — scope="top": human reviewer fires once at top level
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_hierarchical_scope_top_writes_one_row(ephemeral_pg_dsn: str, tmp_path: Path) -> None:
    """Scenario A: scope='top' — exactly 1 human_interactions row is written.

    Verifies:
    - human_top_reviewer fires exactly once (after both teams produce drafts).
    - The callback handler writes 1 HumanInteraction row with request_id
      matching 'hierarchical:top:' prefix.
    - runs.human_role = 'reviewer' (inserted manually via _insert_experiment_and_run).
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="reviewer")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        # Build LLM wrapper for LLMSimulatedGateway (reviewer fixture)
        reviewer_wrapper = _make_llm_wrapper("m91_hierarchical_reviewer_approve.yaml")
        gateway = LLMSimulatedGateway(reviewer_wrapper)

        # Build topology with scope="top"
        human_cfg = HumanCfg(
            enabled=True,
            gateway="llm_simulated",
            role=HumanRole.REVIEWER,
            timeout_s=None,
            extra={"scope": "top"},
        )
        topology = HierarchicalTopology()
        cfg = TopologyConfig(
            name="hierarchical",
            max_iterations=20,
            extra={
                "max_rounds": 4,
                "final_answer_strategy": "json_concat",
                "finalize_signal": "top_coord_finalize",
            },
        )
        agents = _make_agents()

        from unittest.mock import patch

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = gateway
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=reviewer_wrapper
            )

        initial_state = {
            "shared": {
                "task_input": "integration test task",
                "iter_total": 0,
                "iteration": 0,
                "signals": {},
                "final_answer": None,
                "run_id": run_id,
            },
            "agents": {},
            "messages": [],
        }

        await compiled.ainvoke(
            initial_state,
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )

        # Allow async DB operations to complete
        await asyncio.sleep(0.2)

        # Assert: exactly 1 row in human_interactions
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count())
                .select_from(HumanInteraction)
                .where(HumanInteraction.run_id == run_id)
            )
            count = count_result.scalar_one()

        assert count == 1, (
            f"Scenario A (scope=top): Expected exactly 1 human_interactions row, got {count}"
        )

        # Assert: request_id has 'hierarchical:top:' prefix
        async with session_scope(factory) as session:
            rows_result = await session.execute(
                select(HumanInteraction).where(HumanInteraction.run_id == run_id)
            )
            rows = rows_result.scalars().all()

        assert len(rows) == 1
        row = rows[0]
        assert "hierarchical:top:" in row.request_id, (
            f"Expected 'hierarchical:top:' in request_id, got {row.request_id!r}"
        )
        assert row.role == "reviewer", f"Expected role='reviewer', got {row.role!r}"

        # Assert: runs.human_role = 'reviewer'
        async with session_scope(factory) as session:
            run_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_result.scalar_one_or_none()

        assert run_row is not None
        assert run_row.human_role == "reviewer", (
            f"Expected runs.human_role='reviewer', got {run_row.human_role!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Scenario B — scope="sub_team": 2 rows from subgraph callbacks (one per team)
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_hierarchical_scope_sub_team_writes_two_rows_with_team_ids(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Scenario B: scope='sub_team' — exactly 2 rows written from subgraph contexts.

    Verifies:
    - Each subgraph (team_a and team_b) fires the human reviewer independently.
    - 2 HumanInteraction rows are written via callback propagation from subgraph.
    - request_ids contain 'team_a' and 'team_b' respectively.
    - This proves LangGraph 0.3+ propagates RunnableConfig (including callbacks)
      into compiled subgraph ainvoke.
    - runs.human_role = 'reviewer'.
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="reviewer")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        # Build LLM wrapper for LLMSimulatedGateway (reviewer fixture with 2+ entries)
        reviewer_wrapper = _make_llm_wrapper("m91_hierarchical_reviewer_approve.yaml")
        # Create 2 gateway instances (one per team — shared LLM fixture has enough entries)
        gateway = LLMSimulatedGateway(reviewer_wrapper)

        # Build topology with scope="sub_team"
        human_cfg = HumanCfg(
            enabled=True,
            gateway="llm_simulated",
            role=HumanRole.REVIEWER,
            timeout_s=None,
            extra={"scope": "sub_team"},
        )
        topology = HierarchicalTopology()
        cfg = TopologyConfig(
            name="hierarchical",
            max_iterations=20,
            extra={
                "max_rounds": 4,
                "final_answer_strategy": "json_concat",
                "finalize_signal": "top_coord_finalize",
            },
        )
        agents = _make_agents()

        from unittest.mock import patch

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = gateway
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=reviewer_wrapper
            )

        initial_state = {
            "shared": {
                "task_input": "integration sub_team test task",
                "iter_total": 0,
                "iteration": 0,
                "signals": {},
                "final_answer": None,
                "run_id": run_id,
            },
            "agents": {},
            "messages": [],
        }

        await compiled.ainvoke(
            initial_state,
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )

        # Allow async DB operations to complete
        await asyncio.sleep(0.3)

        # Assert: exactly 2 rows (one per team) in human_interactions
        async with session_scope(factory) as session:
            rows_result = await session.execute(
                select(HumanInteraction).where(HumanInteraction.run_id == run_id)
            )
            rows = rows_result.scalars().all()

        assert len(rows) == 2, (
            f"Scenario B (scope=sub_team): Expected exactly 2 human_interactions rows "
            f"(one per subgraph team), got {len(rows)}. "
            f"Request IDs: {[r.request_id for r in rows]}. "
            "This proves callback propagation from inside compiled subgraphs."
        )

        # Assert: request_ids contain "team_a" and "team_b"
        request_ids = {r.request_id for r in rows}
        team_a_rows = [r for r in rows if "team_a" in r.request_id]
        team_b_rows = [r for r in rows if "team_b" in r.request_id]

        assert len(team_a_rows) >= 1, (
            f"Expected at least 1 row with 'team_a' in request_id; got request_ids: {request_ids}"
        )
        assert len(team_b_rows) >= 1, (
            f"Expected at least 1 row with 'team_b' in request_id; got request_ids: {request_ids}"
        )

        # Assert: roles are correct
        for row in rows:
            assert row.role == "reviewer", (
                f"Expected role='reviewer', got {row.role!r} for request_id={row.request_id!r}"
            )

        # Assert: runs.human_role = 'reviewer'
        async with session_scope(factory) as session:
            run_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_result.scalar_one_or_none()

        assert run_row is not None
        assert run_row.human_role == "reviewer", (
            f"Expected runs.human_role='reviewer', got {run_row.human_role!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
