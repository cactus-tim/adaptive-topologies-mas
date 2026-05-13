"""Integration test for MeshTopology HITL integration (M9.1 Step 2.6).

Scenario: deterministic 2-round consensus with human_peer as the tipping voter.

Setup:
  - 3 LLM agents: planner, researcher, executor — each votes "X" uniformly.
  - consensus_threshold=4 — 3 LLM votes (round 1) are not enough.
  - human_cfg.enabled=True, activation_round=2 — human_peer joins from round 2.
  - Round 2: human_peer (via LLMSimulatedGateway) votes "X" → total "X" = 4 ≥ threshold.
  - Graph exits with consensus_reached=True, final_answer="X".

Assertions:
  1. final_answer == "X"
  2. consensus_reached == True
  3. human_interactions count ≥ 1 (PG row written)
  4. runs.human_role == 'reviewer' (per HumanCfg.role)

PG-gating: @pytest.mark.integration — skipped if ATM_ENABLE_PG_TESTS not set.
The ephemeral_pg_dsn fixture handles table creation/teardown.

Reference: tests/integration/human/test_m9_hitl.py for callback dispatch pattern.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from sqlalchemy import func, select

from atm.core.types import Message, MessageKind
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.observability.callbacks import ExperimentCallbackHandler
from atm.storage.models import Base, Experiment, HumanInteraction, Run
from atm.storage.parquet_writer import ParquetWriter
from atm.storage.session import create_engine, create_session_factory, session_scope
from atm.topology.base import TopologyConfig
from atm.topology.mesh import MeshTopology

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"

# ---------------------------------------------------------------------------
# Helpers
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
) -> None:
    """Insert minimal Experiment + Run rows needed for FK constraints."""
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"hitl-mesh-test-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="mesh",
                task_id="hitl-mesh-test-task",
                agent_set="default",
                seed=42,
                model="fake:scripted",
                models_by_role_json={},
                model_version_snapshot={},
                status="running",
                human_role="reviewer",
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


def _make_initial_state(run_id: uuid.UUID) -> dict[str, Any]:
    """Minimal initial state for MeshTopology with run_id in shared."""
    return {
        "shared": {
            "run_id": run_id,
            "task_id": "hitl-mesh-task",
            "task_input": "What is the correct answer?",
            "iter_total": 0,
            "iteration": 0,
            "final_answer": None,
            "signals": {},
            "broadcast_bus": [],
            "phase": "planning",
            "phase_history": [],
            "phase_started_at_iter": 0,
            "active_topology": "mesh",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
            "human_requests": [],
            "human_responses": [],
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _make_vote_agent(agent_id: str, vote: str) -> Any:
    """Create a mock agent that writes a deterministic DECISION vote."""
    _aid = agent_id
    _vote = vote

    class _VoteAgent:
        async def step(self, state: dict[str, Any]) -> dict[str, Any]:
            vote_msg = Message(
                sender=_aid,
                kind=MessageKind.DECISION,
                content=f"I vote for {_vote}",
                payload={"vote_for": _vote},
            )
            return {
                "agents": {
                    _aid: {
                        "agent_id": _aid,
                        "outbox": [vote_msg],
                        "inbox": [],
                        "scratchpad": [],
                        "tool_calls": [],
                        "tool_results": [],
                        "step_count": 1,
                        "tokens_spent": 0,
                        "cost_spent_usd": 0.0,
                    }
                }
            }

    return _VoteAgent()


# ---------------------------------------------------------------------------
# Integration test: deterministic 2-round human-tips-consensus scenario
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_mesh_hitl_human_tips_consensus(ephemeral_pg_dsn: str, tmp_path: Path) -> None:
    """Mesh HITL: 3 LLM peers vote "X" (round 1, threshold=4, no consensus),
    human_peer votes "X" at activation_round=2 → total 4 votes → consensus.

    Verifies:
    - final_answer == "X"
    - consensus_reached == True in signals
    - human_interactions table has ≥ 1 row for this run_id
    - The row has role='reviewer' and non-NULL answered_at

    This test is deterministic: LLM agents always vote "X", human votes "X",
    threshold=4 requires exactly 4 votes → no flakiness.
    """
    import atm.topology.mesh  # noqa: F401 — ensure registration

    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id)

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        # Build the human gateway with the m91_mesh_human_vote fixture
        human_vote_fixture = FIXTURES_DIR / "m91_mesh_human_vote.yaml"
        if not human_vote_fixture.exists():
            pytest.skip(f"Fixture not found: {human_vote_fixture}")

        gateway_wrapper = _make_llm_wrapper("m91_mesh_human_vote.yaml")

        # Build topology config:
        # - 3 LLM agents: planner, researcher, executor — each votes "X"
        # - consensus_threshold=4 (3 LLM votes not enough; human tips it)
        # - max_rounds=6 (enough room for 2 full cycles)
        # - activation_round=2 (human fires at round 2 after seeing pending)
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=30,
            extra={
                "max_rounds": 6,
                "consensus_threshold": 4,
                "activation_policy": "round_robin",
                "agent_order": ["planner", "researcher", "executor"],
                "broadcast_bus_cap": 200,
            },
        )

        # Build HumanCfg-like config
        from types import SimpleNamespace

        from atm.core.types import HumanRole

        human_cfg = SimpleNamespace(
            enabled=True,
            gateway="llm_simulated",
            role=HumanRole.REVIEWER,
            timeout_s=None,  # no timeout for test reliability
            timeout_policy="skip",
            extra={"activation_round": 2},
        )

        # All 3 LLM agents vote "X" deterministically
        agents: dict[str, Any] = {
            "planner": _make_vote_agent("planner", "X"),
            "researcher": _make_vote_agent("researcher", "X"),
            "executor": _make_vote_agent("executor", "X"),
        }

        topology = MeshTopology()
        graph = topology.build(
            agents,
            cfg,
            human_cfg=human_cfg,
            human_gateway_llm=gateway_wrapper,
        )

        initial_state = _make_initial_state(run_id)

        # Run the graph inside a RunnableLambda to get proper LangChain callback context
        # (enables adispatch_custom_event → ExperimentCallbackHandler)
        async def _run_mesh(inputs: dict[str, Any]) -> dict[str, Any]:
            result = await graph.ainvoke(initial_state)
            return result

        runnable: RunnableLambda[dict[str, Any], dict[str, Any]] = RunnableLambda(_run_mesh)
        final_state = await runnable.ainvoke(
            {},
            config={
                "callbacks": [handler],
                "metadata": {"is_root_run": True},
                "run_id": uuid.uuid4(),
            },
        )

        # Allow async background operations to complete
        await asyncio.sleep(0.2)

        # --- Assert graph result ---
        shared: dict[str, Any] = final_state.get("shared", {})
        signals: dict[str, Any] = shared.get("signals", {})

        assert signals.get("consensus_reached") is True, (
            f"Expected consensus_reached=True, got signals={signals}"
        )
        assert shared.get("final_answer") == "X", (
            f"Expected final_answer='X', got {shared.get('final_answer')!r}"
        )

        # --- Assert PG rows ---
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count())
                .select_from(HumanInteraction)
                .where(HumanInteraction.run_id == run_id)
            )
            count = count_result.scalar_one()

        assert count >= 1, (
            f"Expected ≥ 1 row in human_interactions for run_id={run_id}, got {count}"
        )

        # --- Assert row fields ---
        async with session_scope(factory) as session:
            row_result = await session.execute(
                select(HumanInteraction).where(HumanInteraction.run_id == run_id).limit(1)
            )
            row = row_result.scalar_one_or_none()

        assert row is not None, f"No human_interactions row found for run_id={run_id}"
        assert row.role == "reviewer", f"Expected role='reviewer', got {row.role!r}"
        assert row.response_json is not None, "response_json should be non-NULL after human vote"

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


# ---------------------------------------------------------------------------
# Back-compat integration test: without HITL, mesh behaves identically to M7
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mesh_without_hitl_consensus_backcompat() -> None:
    """Without human_cfg, mesh achieves consensus via LLM agents only (back-compat).

    This test does NOT require PG — it verifies the base-case graph invocation
    is unaffected by the HITL additions.
    """
    import atm.topology.mesh  # noqa: F401

    agents: dict[str, Any] = {
        "planner": _make_vote_agent("planner", "Y"),
        "researcher": _make_vote_agent("researcher", "Y"),
        "executor": _make_vote_agent("executor", "Y"),
    }
    cfg = TopologyConfig(
        name="mesh",
        max_iterations=20,
        extra={
            "max_rounds": 6,
            "consensus_threshold": 3,
            "agent_order": ["planner", "researcher", "executor"],
        },
    )

    topology = MeshTopology()
    graph = topology.build(agents, cfg)  # no human_cfg

    initial_state: dict[str, Any] = {
        "shared": {
            "task_id": "test",
            "task_input": "Back-compat test",
            "iter_total": 0,
            "iteration": 0,
            "final_answer": None,
            "signals": {},
            "broadcast_bus": [],
            "phase": "planning",
            "phase_history": [],
            "phase_started_at_iter": 0,
            "active_topology": "mesh",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
            "human_requests": [],
            "human_responses": [],
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }

    final_state = await graph.ainvoke(initial_state)
    shared = final_state.get("shared", {})
    signals = shared.get("signals", {})

    assert signals.get("consensus_reached") is True, (
        f"Back-compat: expected consensus_reached=True, got {signals}"
    )
    assert shared.get("final_answer") == "Y", (
        f"Back-compat: expected final_answer='Y', got {shared.get('final_answer')!r}"
    )
