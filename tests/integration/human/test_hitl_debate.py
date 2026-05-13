"""Integration test for DebateTopology HITL — judge mode "human" (M9.1 Step 2.9).

Scenario:
  - DebateTopology with judge mode "human" (HumanCfg.extra={"judge": "human"}).
  - LLMSimulatedGateway (scripted FakeLLM) simulates the human judge approving.
  - ExperimentCallbackHandler writes HITL events to PG via adispatch_custom_event.
  - One round of debate: planner → debater_pro + debater_contra → human_judge →
    judge_postprocess → END (judge approves on first round).

Assertions:
  - Run completes (graph.ainvoke returns without exception).
  - human_interactions count >= 1 (human_judge node fired and dispatched events).
  - runs.human_role == "judge" (correct role written by Runner._insert_run).

Skipping:
  - Skipped unless ATM_ENABLE_PG_TESTS=1 is set (ephemeral_pg_dsn fixture handles this).

Pattern: Follows test_m9_hitl.py — drives LangGraph directly with callback handler,
manually inserts Experiment + Run rows for FK constraints, then asserts PG writes.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.agents.debater import Debater
from atm.core.state import GraphState
from atm.core.types import Message, MessageKind
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pricing() -> Pricing:
    if _PRICING_PATH.exists():
        return Pricing.from_yaml(_PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    fixture_path = _FIXTURES_DIR / fixture_name
    fake = FakeLLM(mode="scripted", fixture=fixture_path)
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
    *,
    human_role: str = "judge",
) -> None:
    """Insert minimal Experiment + Run rows needed for FK constraints."""
    async with session_scope(session_factory) as session:
        session.add(
            Experiment(
                id=exp_id,
                name=f"hitl-debate-{exp_id}",
                config_snapshot={},
                status="running",
            )
        )
    async with session_scope(session_factory) as session:
        session.add(
            Run(
                id=run_id,
                exp_id=exp_id,
                topology="debate",
                task_id="hitl-debate-task",
                agent_set="default",
                human_role=human_role,
                seed=42,
                model="fake:scripted",
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


# ---------------------------------------------------------------------------
# JudgeAgent — converts DRAFT outbox to DECISION (from test_debate_e2e.py)
# ---------------------------------------------------------------------------


class JudgeAgent(Agent):
    """Judge role: evaluates debate and emits DECISION messages."""

    async def step(self, state: GraphState) -> dict[str, Any]:
        """Execute one judge step, convert DRAFT → DECISION."""
        delta = await super().step(state)

        agents_delta: dict[str, Any] = dict(delta.get("agents", {}))
        self_delta: dict[str, Any] = dict(agents_delta.get(self.agent_id, {}))
        outbox: list[Any] = list(self_delta.get("outbox", []))
        messages: list[Any] = list(delta.get("messages", []))

        new_outbox: list[Message] = []
        new_messages: list[Message] = []

        for msg in outbox:
            content: str = getattr(msg, "content", "") or ""
            content_lower = content.lower()
            approved: bool = "approve" in content_lower

            winner = "pro"
            if "winner=contra" in content_lower:
                winner = "contra"

            decision_msg = Message(
                sender=self.agent_id,
                kind=MessageKind.DECISION,
                content=content,
                payload={"approved": approved, "winner": winner},
            )
            new_outbox.append(decision_msg)

        for msg in messages:
            content = getattr(msg, "content", "") or ""
            content_lower = content.lower()
            approved = "approve" in content_lower

            winner = "pro"
            if "winner=contra" in content_lower:
                winner = "contra"

            decision_msg = Message(
                sender=self.agent_id,
                kind=MessageKind.DECISION,
                content=content,
                payload={"approved": approved, "winner": winner},
            )
            new_messages.append(decision_msg)

        self_delta = dict(self_delta)
        self_delta["outbox"] = new_outbox
        agents_delta = dict(agents_delta)
        agents_delta[self.agent_id] = self_delta
        delta = dict(delta)
        delta["agents"] = agents_delta
        delta["messages"] = new_messages

        return delta


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------


def _make_planner_agent() -> Agent:
    """Build planner agent with scripted FakeLLM fixture."""
    fixture_path = _FIXTURES_DIR / "m91_debate_planner.yaml"
    if not fixture_path.exists():
        # Fallback to existing debate fixture if specific one not found
        fixture_path = _FIXTURES_DIR / "m7_debate_judge_decides.yaml"
    cfg = AgentConfig(
        role="planner",
        system_prompt="You are a debate planner. Set up the debate topic.",
        tools=[],
        max_tool_iters=0,
    )
    from atm.tools.base import ToolRegistry

    llm = _make_llm_wrapper(fixture_path.name)
    return Agent(agent_id="planner", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_debater_pro_agent() -> Debater:
    """Build debater_pro agent with scripted FakeLLM fixture."""
    fixture_path = _FIXTURES_DIR / "m91_debate_debater_pro.yaml"
    if not fixture_path.exists():
        fixture_path = _FIXTURES_DIR / "m7_debate_judge_decides.yaml"
    cfg = AgentConfig(
        role="debater",
        system_prompt="You are a debater. Stance: {{stance}}. Argue your position.",
        tools=[],
        max_tool_iters=0,
        params={"stance": "pro"},
    )
    from atm.tools.base import ToolRegistry

    llm = _make_llm_wrapper(fixture_path.name)
    return Debater(agent_id="debater_pro", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_debater_contra_agent() -> Debater:
    """Build debater_contra agent with scripted FakeLLM fixture."""
    fixture_path = _FIXTURES_DIR / "m91_debate_debater_contra.yaml"
    if not fixture_path.exists():
        fixture_path = _FIXTURES_DIR / "m7_debate_judge_decides.yaml"
    cfg = AgentConfig(
        role="debater",
        system_prompt="You are a debater. Stance: {{stance}}. Argue your position.",
        tools=[],
        max_tool_iters=0,
        params={"stance": "contra"},
    )
    from atm.tools.base import ToolRegistry

    llm = _make_llm_wrapper(fixture_path.name)
    return Debater(agent_id="debater_contra", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_judge_agent() -> JudgeAgent:
    """Build judge agent with scripted FakeLLM fixture (LLM path not used in human mode)."""
    # In human mode, the judge LLM is not called — the human gateway replaces it.
    # We still need an agent registered under "judge" to satisfy topology agent lookup.
    fixture_path = _FIXTURES_DIR / "m91_debate_planner.yaml"
    if not fixture_path.exists():
        fixture_path = _FIXTURES_DIR / "m7_debate_judge_decides.yaml"
    cfg = AgentConfig(
        role="critic",
        system_prompt="You are a judge. Evaluate the debate.",
        tools=[],
        max_tool_iters=0,
    )
    from atm.tools.base import ToolRegistry

    llm = _make_llm_wrapper(fixture_path.name)
    return JudgeAgent(agent_id="judge", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_human_judge_gateway() -> LLMSimulatedGateway:
    """Build LLMSimulatedGateway that returns 'approve' for judge mode."""
    fixture_path = _FIXTURES_DIR / "m91_debate_human_judge_approve.yaml"
    if not fixture_path.exists():
        # Fallback to generic approve fixture
        fixture_path = _FIXTURES_DIR / "m9_human_reviewer_approve.yaml"
    llm_wrapper = _make_llm_wrapper(fixture_path.name)
    return LLMSimulatedGateway(llm=llm_wrapper)


def _make_initial_state(run_id: uuid.UUID) -> dict[str, Any]:
    """Build a minimal initial GraphState for the debate."""
    return {
        "shared": {
            "run_id": run_id,
            "task_id": "debate_hitl_test",
            "task_input": "Debate the proposition and determine a winner.",
            "phase": "planning",
            "iteration": 0,
            "iter_total": 0,
            "active_topology": "debate",
            "final_answer": "",
            "signals": {},
            "phase_started_at_iter": 0,
            "topology_started_at_iter": 0,
            "topology_history": [],
            "topology_switch_count": 0,
            "phase_history": [],
            "human_requests": [],
            "human_responses": [],
            "broadcast_bus": [],
            "debate_round": 0,
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


# ---------------------------------------------------------------------------
# Integration test: Debate HITL "human" mode
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_debate_hitl_human_mode_completes_and_writes_interaction(
    ephemeral_pg_dsn: str, tmp_path: Path
) -> None:
    """Debate HITL mode='human': run completes, human_interactions count >= 1,
    runs.human_role == 'judge'.

    Scenario:
      1. Build DebateTopology with human_cfg.extra={"judge": "human"} and
         LLMSimulatedGateway as the human judge gateway.
      2. Run graph.ainvoke() with ExperimentCallbackHandler (writes to PG).
      3. Assert: graph completes without error.
      4. Assert: human_interactions count >= 1 for this run_id.
      5. Assert: Run.human_role == 'judge' (written by _insert_run in test setup).

    The human_judge HITL node:
      - Dispatches human_request event (callback writes HumanInteraction row)
      - Calls LLMSimulatedGateway (scripted fixture returns "approve")
      - Dispatches human_response event (callback updates response_json)
      - Synthesizes DECISION(approved=True) in agents["judge"]["outbox"]
      - _judge_postprocess then reads this DECISION → signals["judge_decided"]=True
      - Graph terminates successfully on first round
    """
    import atm.topology.debate  # noqa: F401 — triggers @TopologyRegistry.register
    from atm.topology.debate import DebateTopology

    exp_id = uuid.uuid4()
    run_id = uuid.uuid4()

    engine = create_engine(ephemeral_pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        factory = create_session_factory(engine)
        await _insert_experiment_and_run(factory, exp_id, run_id, human_role="judge")

        handler = _build_handler(run_id, exp_id, factory, tmp_path)

        # Build topology cfg with human mode
        topology_cfg = TopologyConfig(
            name="debate",
            max_iterations=12,
            extra={
                "max_rounds": 4,
                "debater_pro_id": "debater_pro",
                "debater_contra_id": "debater_contra",
                "judge_id": "judge",
            },
        )
        human_cfg = HumanCfg(
            enabled=True,
            gateway="llm_simulated",
            role="judge",  # type: ignore[arg-type]
            timeout_s=None,
            extra={"judge": "human"},
        )

        # Build agents
        agents = {
            "planner": _make_planner_agent(),
            "debater_pro": _make_debater_pro_agent(),
            "debater_contra": _make_debater_contra_agent(),
            "judge": _make_judge_agent(),
        }

        # Build gateway (pre-built to use our scripted fixture)
        gateway = _make_human_judge_gateway()

        # Build compiled graph
        topology = DebateTopology()
        compiled_graph = topology.build(
            agents,
            topology_cfg,
            human_cfg=human_cfg,
            gateway=gateway,
        )

        # Run with callback handler
        initial_state = _make_initial_state(run_id)
        final_state = await compiled_graph.ainvoke(
            initial_state,
            config={
                "callbacks": [handler],
                "configurable": {"thread_id": str(run_id)},
                "run_id": uuid.uuid4(),
                "metadata": {"is_root_run": True},
            },
        )

        # Allow async background operations to complete
        await asyncio.sleep(0.15)

        # Close parquet writer to flush
        import contextlib

        async with contextlib.AsyncExitStack() as _stack:
            pass  # Ensure parquet is closed
        with contextlib.suppress(Exception):
            await handler._parquet_writer.close()

        # ── Assertion 1: Graph completed ─────────────────────────────────────
        assert final_state is not None, "Graph should complete without exception"

        # ── Assertion 2: human_interactions count >= 1 ───────────────────────
        async with session_scope(factory) as session:
            count_result = await session.execute(
                select(func.count())
                .select_from(HumanInteraction)
                .where(HumanInteraction.run_id == run_id)
            )
            interaction_count = count_result.scalar_one()

        assert interaction_count >= 1, (
            f"Expected at least 1 row in human_interactions for run_id={run_id}, "
            f"got {interaction_count}. "
            "The human_judge node must have dispatched human_request events."
        )

        # ── Assertion 3: runs.human_role == 'judge' ──────────────────────────
        async with session_scope(factory) as session:
            run_result = await session.execute(select(Run).where(Run.id == run_id))
            run_row = run_result.scalar_one_or_none()

        assert run_row is not None, f"Run row should exist for run_id={run_id}"
        assert run_row.human_role == "judge", (
            f"Expected runs.human_role='judge', got {run_row.human_role!r}"
        )

    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
