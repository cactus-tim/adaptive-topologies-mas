"""Integration tests for AdaptiveTopology HITL human_advisor node (M9.1 Step 2.16).

Two scenarios (both require ATM_ENABLE_PG_TESTS=1):

  Scenario A — Advisory mode:
    Build adaptive graph with human_cfg.enabled=True but
    human_can_override_router=False (advisory mode).
    Drive ExperimentCallbackHandler to capture human_request/response events.
    Assert:
      - At least 1 row in human_interactions for this run_id.
      - topology_transitions rows do NOT contain decided_by='human_override'.
      - runs.human_role is set correctly.

  Scenario B — Override mode:
    Build adaptive graph with human_can_override_router=True.
    LLMSimulatedGateway returns switch_topology action.
    Assert:
      - topology_transitions contains >= 1 row with decided_by='human_override'.
      - human_interactions count >= 1.
      - runs.human_role is set correctly.

Skipping rules
--------------
All scenarios skip gracefully when ATM_ENABLE_PG_TESTS is not set.

Integration strategy
-------------------
These tests use RunnableLambda + adispatch_custom_event (same pattern as
test_m9_hitl.py) to verify the full callback → PG write path without needing
to run the full adaptive graph.  The override path is tested by directly
dispatching the topology_transition event with decided_by='human_override'.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableLambda
from sqlalchemy import func, select

from atm.core.types import (
    HumanContext,
    HumanRole,
    Message,
    MessageKind,
    Phase,
    TopologyTransition,
)
from atm.human.llm_simulated import LLMSimulatedGateway
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


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


def _make_context(run_id: uuid.UUID, role: HumanRole = HumanRole.REVIEWER) -> HumanContext:
    msg = Message(
        sender="topology_router",
        kind=MessageKind.DRAFT,
        content="Router chose 'linear'. Advisory mode active.",
    )
    return HumanContext(
        run_id=run_id,
        role=role,
        question="Should topology be changed?",
        recent_messages=(msg,),
        allowed_actions=("advise", "switch_topology", "abstain"),
    )


async def _insert_experiment_and_run(
    session_factory: Any,
    exp_id: uuid.UUID,
    run_id: uuid.UUID,
    human_role: str | None = None,
) -> None:
    """Insert minimal Experiment + Run rows for FK constraints."""
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"hitl-adaptive-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="adaptive",
                task_id="hitl-adaptive-test",
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


@pytest.mark.integration
async def test_advisory_mode_no_human_override_transitions(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Scenario A: Advisory mode — topology_transitions.decided_by != 'human_override'.

    Verifies:
    - human_request + human_response events write a row in human_interactions.
    - topology_transition event with decided_by='rule' is written correctly.
    - No topology_transition row has decided_by='human_override'.
    - runs.human_role = 'reviewer' when human_cfg.role = HumanRole.REVIEWER.
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = f"adaptive:{run_id}:0:advisor"

    fixture_path = FIXTURES_DIR / "m91_adaptive_advisor_approve.yaml"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="reviewer")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)
        ctx = _make_context(run_id)

        llm_wrapper = _make_llm_wrapper("m91_adaptive_advisor_approve.yaml")
        gateway = LLMSimulatedGateway(llm_wrapper)
        response = await gateway.request(ctx, request_id=request_id)
        assert response.action == "advise"

        topo_transition = TopologyTransition(
            run_id=run_id,
            from_topology="linear",
            to_topology="linear",
            phase_at_decision=Phase.PLANNING,
            iter_within_phase=0,
            iter_within_topology=0,
            decided_by="rule",
            reason="no-change: rule-based stay",
        )

        async def _dispatch_advisory_scenario(inputs: dict[str, Any]) -> dict[str, Any]:
            await adispatch_custom_event(
                "human_request",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "role": str(ctx.role),
                    "context_json": ctx.model_dump(mode="json"),
                    "requested_at": datetime.now(UTC),
                },
            )
            # human_response
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": response.model_dump(mode="json"),
                    "source": response.source,
                    "timed_out": response.timed_out,
                    "latency_s": 0.05,
                },
            )
            await adispatch_custom_event("topology_transition", topo_transition)
            return inputs

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _dispatch_advisory_scenario
        )
        await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
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
            f"Expected >= 1 row in human_interactions for run_id={run_id}, got {count}"
        )

        from atm.storage.models import TopologyTransition as TopologyTransitionRow

        async with session_scope(factory) as session:
            override_result = await session.execute(
                select(func.count())
                .select_from(TopologyTransitionRow)
                .where(
                    TopologyTransitionRow.run_id == run_id,
                    TopologyTransitionRow.decided_by == "human_override",
                )
            )
            override_count = override_result.scalar_one()

        assert override_count == 0, (
            f"Advisory mode should not write 'human_override' transitions; got {override_count}"
        )

        async with session_scope(factory) as session:
            run_row_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_row_result.scalar_one_or_none()

        assert run_row is not None
        assert run_row.human_role == "reviewer", (
            f"Expected runs.human_role='reviewer', got {run_row.human_role!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest.mark.integration
async def test_override_mode_writes_human_override_transition(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Scenario B: Override mode — topology_transitions contains >= 1 human_override row.

    Verifies:
    - human_request + human_response events write a row in human_interactions.
    - topology_transition event with decided_by='human_override' is written correctly.
    - SELECT COUNT(*) WHERE decided_by='human_override' >= 1.
    - runs.human_role = 'reviewer'.
    """
    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()
    request_id = f"adaptive:{run_id}:0:advisor"

    fixture_path = FIXTURES_DIR / "m91_adaptive_override.yaml"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="reviewer")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)
        ctx = _make_context(run_id)

        llm_wrapper = _make_llm_wrapper("m91_adaptive_override.yaml")
        gateway = LLMSimulatedGateway(llm_wrapper)
        response = await gateway.request(ctx, request_id=request_id)
        assert response.action == "switch_topology"

        override_transition = TopologyTransition(
            run_id=run_id,
            from_topology="linear",
            to_topology="mesh",
            phase_at_decision=Phase.PLANNING,
            iter_within_phase=0,
            iter_within_topology=0,
            decided_by="human_override",
            reason="human_override: switch to mesh",
            considered_alternatives=("linear",),
        )

        async def _dispatch_override_scenario(inputs: dict[str, Any]) -> dict[str, Any]:
            await adispatch_custom_event(
                "human_request",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "role": str(ctx.role),
                    "context_json": ctx.model_dump(mode="json"),
                    "requested_at": datetime.now(UTC),
                },
            )
            # human_response
            await adispatch_custom_event(
                "human_response",
                {
                    "run_id": run_id,
                    "request_id": request_id,
                    "answered_at": datetime.now(UTC),
                    "response_json": response.model_dump(mode="json"),
                    "source": response.source,
                    "timed_out": response.timed_out,
                    "latency_s": 0.05,
                },
            )
            await adispatch_custom_event("topology_transition", override_transition)
            return inputs

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(
            _dispatch_override_scenario
        )
        await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
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
            f"Expected >= 1 row in human_interactions for run_id={run_id}, got {count}"
        )

        from atm.storage.models import TopologyTransition as TopologyTransitionRow

        async with session_scope(factory) as session:
            override_result = await session.execute(
                select(func.count())
                .select_from(TopologyTransitionRow)
                .where(
                    TopologyTransitionRow.run_id == run_id,
                    TopologyTransitionRow.decided_by == "human_override",
                )
            )
            override_count = override_result.scalar_one()

        assert override_count >= 1, (
            f"Expected >= 1 topology_transition with decided_by='human_override', "
            f"got {override_count}. Override path did not write the transition."
        )

        async with session_scope(factory) as session:
            from atm.storage.models import TopologyTransition as TopologyTransitionRow

            row_result = await session.execute(
                select(TopologyTransitionRow).where(
                    TopologyTransitionRow.run_id == run_id,
                    TopologyTransitionRow.decided_by == "human_override",
                )
            )
            row = row_result.scalar_one_or_none()

        assert row is not None
        assert row.to_topology == "mesh", f"Expected to_topology='mesh', got {row.to_topology!r}"
        assert row.from_topology == "linear", (
            f"Expected from_topology='linear', got {row.from_topology!r}"
        )

        async with session_scope(factory) as session:
            run_row_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_row_result.scalar_one_or_none()

        assert run_row is not None
        assert run_row.human_role == "reviewer", (
            f"Expected runs.human_role='reviewer', got {run_row.human_role!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
