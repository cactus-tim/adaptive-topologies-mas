"""StarTopology HITL integration test (M9.1 Step 2.3).

Acceptance scenario:
  - Build StarTopology with human_cfg.enabled=True (reviewer mode).
  - Run the graph end-to-end with scripted FakeLLM fixtures.
  - Assert: human_interactions count >= 1 in PostgreSQL.
  - Assert: runs.human_role == 'reviewer'.

Pattern: copied from tests/integration/human/test_m9_hitl.py (Chain reference).

Skipping rules:
  - Requires ATM_ENABLE_PG_TESTS=1 to run (marked @pytest.mark.integration).
  - Skips gracefully if fixture files are missing.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

import atm.topology.star  # noqa: F401 — register StarTopology
from atm.core.types import HumanRole, Message, MessageKind
from atm.experiment.config import HumanCfg
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope
from atm.topology.base import TopologyConfig, TopologyRegistry

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    """Build a scripted FakeLLM wrapper from fixture file."""
    fixture_path = FIXTURES_DIR / fixture_name
    fake = FakeLLM(mode="scripted", fixture=fixture_path)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_human_cfg(role: HumanRole = HumanRole.REVIEWER) -> HumanCfg:
    return HumanCfg(
        enabled=True,
        gateway="llm_simulated",
        role=role,
        timeout_s=None,
        timeout_policy="skip",  # type: ignore[arg-type]
    )


def _make_topology_cfg() -> TopologyConfig:
    return TopologyConfig(
        name="star",
        max_iterations=20,
        extra={
            "planning_max_iter": 1,
            "exec_max_iter": 1,
            "verify_max_iter": 2,
        },
    )


async def _insert_experiment_and_run(
    session_factory: Any,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    human_role: str | None = "reviewer",
) -> None:
    """Insert minimal Experiment + Run rows needed for FK constraints."""
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"hitl-star-test-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="star",
                task_id="hitl-star-test-task",
                agent_set="default",
                seed=42,
                model="fake:scripted",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
                human_role=human_role,
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
    """Create a simple async mock agent."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agents": {agent_id: {"agent_id": agent_id, "outbox": []}},
            "messages": [],
        }

    agent.step = fake_step
    return agent


def _make_critic_agent_with_approval() -> Any:
    """Critic agent that always approves after one step."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.agent_id = "critic"

    async def fake_step(state: dict[str, Any]) -> dict[str, Any]:
        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True, "comment": "Looks good"},
        )
        return {
            "agents": {"critic": {"agent_id": "critic", "outbox": [decision_msg]}},
            "messages": [decision_msg],
        }

    agent.step = fake_step
    return agent


@pytest.mark.integration
async def test_star_hitl_reviewer_e2e_writes_human_interaction(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Star HITL end-to-end: run completes, human_interactions >= 1, human_role='reviewer'.

    Verifies:
    - StarTopology with HITL enabled dispatches human_request + human_response events.
    - ExperimentCallbackHandler writes at least 1 HumanInteraction row to PG.
    - runs.human_role == 'reviewer' (set by _insert_run in the test setup).
    - The response action is 'approve' (from scripted fixture m91_star_human_reviewer.yaml).

    This test uses LLMSimulatedGateway with a scripted FakeLLM fixture so that the
    human reviewer node fires deterministically in the verification phase.
    """
    reviewer_fixture = FIXTURES_DIR / "m91_star_human_reviewer.yaml"
    if not reviewer_fixture.exists():
        pytest.skip(f"Fixture not found: {reviewer_fixture}")

    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="reviewer")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        from atm.human.llm_simulated import LLMSimulatedGateway

        gateway_llm_wrapper = _make_llm_wrapper("m91_star_human_reviewer.yaml")
        gateway = LLMSimulatedGateway(gateway_llm_wrapper)

        human_cfg = _make_human_cfg()

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_critic_agent_with_approval(),
        }

        topo_cfg = _make_topology_cfg()

        star_cls = TopologyRegistry.get("star")
        star = star_cls()

        from langgraph.checkpoint.memory import MemorySaver

        with patch("atm.topology.star.LLMSimulatedGateway", return_value=gateway):
            graph = star.build(
                agents,
                topo_cfg,
                checkpointer=MemorySaver(),
                human_cfg=human_cfg,
                human_gateway_llm=gateway_llm_wrapper,
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "run_id": run_id,
                "task_input": "integration test task",
                "phase": "planning",
                "iter_total": 0,
                "iteration": 0,
                "phase_started_at_iter": 0,
                "phase_history": [],
                "active_topology": "star",
                "topology_started_at_iter": 0,
                "topology_switch_count": 0,
                "topology_history": [],
                "final_answer": None,
                "signals": {},
                "broadcast_bus": [],
                "human_requests": [],
                "human_responses": [],
            },
            "agents": {},
            "messages": [],
            "llm_calls": [],
            "budget_events": [],
            "topology_transitions": [],
        }

        result = await graph.ainvoke(
            initial_state,
            config={
                "callbacks": [handler],
                "configurable": {"thread_id": str(run_id)},
                "recursion_limit": 100,
            },
        )

        await asyncio.sleep(0.2)

        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count())
                .select_from(HumanInteraction)
                .where(HumanInteraction.run_id == run_id)
            )
            count = count_result.scalar_one()

        assert count >= 1, (
            f"Expected at least 1 row in human_interactions for run_id={run_id}, "
            f"got {count}. The star HITL node must dispatch human_request + human_response."
        )

        async with session_scope(factory) as session:
            run_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_result.scalar_one_or_none()

        assert run_row is not None, f"No run row found for run_id={run_id}"
        assert run_row.human_role == "reviewer", (
            f"Expected runs.human_role='reviewer', got {run_row.human_role!r}"
        )

        assert result is not None, "Graph should have completed and returned a final state"

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.mark.integration
async def test_star_hitl_human_approved_state_is_true(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Star HITL: LLMSimulatedGateway approve response → shared['human_approved']=True.

    Verifies end-to-end: the approve response from the scripted fixture
    propagates through the human_reviewer node and sets human_approved=True
    in the final shared state.
    """
    reviewer_fixture = FIXTURES_DIR / "m91_star_human_reviewer.yaml"
    if not reviewer_fixture.exists():
        pytest.skip(f"Fixture not found: {reviewer_fixture}")

    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="reviewer")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        from atm.human.llm_simulated import LLMSimulatedGateway

        gateway_llm_wrapper = _make_llm_wrapper("m91_star_human_reviewer.yaml")
        gateway = LLMSimulatedGateway(gateway_llm_wrapper)

        human_cfg = _make_human_cfg()

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_critic_agent_with_approval(),
        }

        topo_cfg = _make_topology_cfg()
        star_cls = TopologyRegistry.get("star")
        star = star_cls()

        from langgraph.checkpoint.memory import MemorySaver

        with patch("atm.topology.star.LLMSimulatedGateway", return_value=gateway):
            graph = star.build(
                agents,
                topo_cfg,
                checkpointer=MemorySaver(),
                human_cfg=human_cfg,
                human_gateway_llm=gateway_llm_wrapper,
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "run_id": run_id,
                "task_input": "integration test task 2",
                "phase": "planning",
                "iter_total": 0,
                "iteration": 0,
                "phase_started_at_iter": 0,
                "phase_history": [],
                "active_topology": "star",
                "topology_started_at_iter": 0,
                "topology_switch_count": 0,
                "topology_history": [],
                "final_answer": None,
                "signals": {},
                "broadcast_bus": [],
                "human_requests": [],
                "human_responses": [],
            },
            "agents": {},
            "messages": [],
            "llm_calls": [],
            "budget_events": [],
            "topology_transitions": [],
        }

        result = await graph.ainvoke(
            initial_state,
            config={
                "callbacks": [handler],
                "configurable": {"thread_id": str(run_id)},
                "recursion_limit": 100,
            },
        )

        await asyncio.sleep(0.1)

        assert result is not None
        shared_final = result.get("shared", {})
        assert shared_final.get("human_approved") is True, (
            f"Expected shared['human_approved']=True after approve response; "
            f"got {shared_final.get('human_approved')!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
