"""Integration E2E tests for DebateTopology.

Two test scenarios:
  1. test_debate_judge_decides_after_loop:
       - Round 1: judge rejects, loop continues
       - Round 2: judge approves (winner="pro")
       - Assert: signals["judge_decided"] == True
       - Assert: final_answer set from winning debater's DRAFT
       - Assert: all Message.id unique after fan-in (MC-3)
       - Assert: iter_total > 0

  2. test_debate_max_rounds:
       - All rounds: judge rejects
       - max_rounds=2 to keep fixture short
       - Assert: topology terminates (graph completes)
       - Assert: iter_total > 0
       - Assert: signals["judge_decided"] not set

Both tests use scripted FakeLLM fixtures (no real LLM calls) and invoke the
compiled LangGraph directly (no PG, no ParquetWriter).

The Judge agent is wrapped as a JudgeAgent subclass (mirroring Critic)
that converts DRAFT outbox messages to DECISION messages by scanning
for "APPROVE" and "winner=" keywords.

MC-3 invariant: all Message.id values must be unique after fan-in from
parallel debaters — guaranteed by uuid4 default_factory on Message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.agents.debater import Debater
from atm.core.state import GraphState
from atm.core.types import Message, MessageKind
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry
from atm.topology.base import TopologyConfig

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_JUDGE_DECIDES_FIXTURE = _FIXTURES_DIR / "m7_debate_judge_decides.yaml"
_MAX_ROUNDS_FIXTURE = _FIXTURES_DIR / "m7_debate_max_rounds.yaml"


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_pricing() -> Pricing:
    pricing_path = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"
    if pricing_path.exists():
        return Pricing.from_yaml(pricing_path)
    return Pricing(version=1, models={})


def _make_llm(fixture_path: Path, agent_id: str | None = None) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=fixture_path)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


class JudgeAgent(Agent):
    """Judge role: evaluates debate and emits DECISION messages.

    Parses LLM response for:
      - "APPROVE" → approved=True
      - "winner=pro" or "winner=contra" in the response → sets winner
      - Default winner: "pro" if approved

    Mirrors the Critic subclass pattern for DECISION emission.
    """

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
            elif "winner=pro" in content_lower:
                winner = "pro"

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
            elif "winner=pro" in content_lower:
                winner = "pro"

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


def _make_planner(fixture_path: Path) -> Agent:
    cfg = AgentConfig(
        role="planner",
        system_prompt="You are a debate planner. Set up the debate topic.",
        tools=[],
        max_tool_iters=0,
    )
    llm = _make_llm(fixture_path)
    return Agent(agent_id="planner", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_debater_pro(fixture_path: Path) -> Debater:
    cfg = AgentConfig(
        role="debater",
        system_prompt="You are a debater. Stance: {{stance}}. Argue your position.",
        tools=[],
        max_tool_iters=0,
        params={"stance": "pro"},
    )
    llm = _make_llm(fixture_path)
    return Debater(agent_id="debater_pro", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_debater_contra(fixture_path: Path) -> Debater:
    cfg = AgentConfig(
        role="debater",
        system_prompt="You are a debater. Stance: {{stance}}. Argue your position.",
        tools=[],
        max_tool_iters=0,
        params={"stance": "contra"},
    )
    llm = _make_llm(fixture_path)
    return Debater(agent_id="debater_contra", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_judge(fixture_path: Path) -> JudgeAgent:
    cfg = AgentConfig(
        role="critic",
        system_prompt="You are a judge evaluating a debate. Decide which side wins.",
        tools=[],
        max_tool_iters=0,
    )
    llm = _make_llm(fixture_path)
    return JudgeAgent(agent_id="judge", cfg=cfg, llm=llm, tools=ToolRegistry())


def _make_initial_state() -> dict[str, Any]:
    """Build a minimal initial GraphState for the debate."""
    return {
        "shared": {
            "task_id": "debate_test",
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


def _make_debate_cfg(*, max_rounds: int = 4, max_iterations: int = 12) -> TopologyConfig:
    return TopologyConfig(
        name="debate",
        max_iterations=max_iterations,
        extra={
            "max_rounds": max_rounds,
            "debater_pro_id": "debater_pro",
            "debater_contra_id": "debater_contra",
            "judge_id": "judge",
        },
    )


@pytest.mark.asyncio
async def test_debate_judge_decides_after_loop() -> None:
    """E2E: DebateTopology runs 2 rounds, judge approves in round 2.

    Assertions:
      - Graph terminates (ainvoke completes without exception)
      - signals["judge_decided"] == True in final state
      - final_answer is non-empty (from winning debater)
      - iter_total > 0 (at least one planner step executed)
      - All Message.id values are unique (MC-3 fan-in dedup)
    """
    if not _JUDGE_DECIDES_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {_JUDGE_DECIDES_FIXTURE}")

    fixture_path = _JUDGE_DECIDES_FIXTURE
    agents = {
        "planner": _make_planner(fixture_path),
        "debater_pro": _make_debater_pro(fixture_path),
        "debater_contra": _make_debater_contra(fixture_path),
        "judge": _make_judge(fixture_path),
    }

    cfg = _make_debate_cfg(max_rounds=4, max_iterations=20)

    import atm.topology.debate  # noqa: F401
    from atm.topology.debate import DebateTopology

    topology = DebateTopology()
    graph = topology.build(agents, cfg)

    initial_state = _make_initial_state()
    final_state = await graph.ainvoke(initial_state)

    shared = final_state.get("shared", {})
    signals = shared.get("signals", {})
    assert signals.get("judge_decided") is True, (
        f"Expected signals['judge_decided']=True, got: {signals}"
    )

    final_answer = shared.get("final_answer", "")
    assert final_answer and final_answer != "<incomplete>", (
        f"Expected non-empty final_answer, got: {final_answer!r}"
    )

    iter_total = shared.get("iter_total", 0)
    assert iter_total > 0, f"Expected iter_total > 0, got {iter_total}"

    messages: list[Any] = final_state.get("messages", [])
    if messages:
        msg_ids = [getattr(m, "id", None) for m in messages]
        msg_ids_set = {mid for mid in msg_ids if mid is not None}
        non_none_ids = [mid for mid in msg_ids if mid is not None]
        assert len(msg_ids_set) == len(non_none_ids), (
            f"Duplicate Message.id detected after fan-in! "
            f"Total: {len(non_none_ids)}, Unique: {len(msg_ids_set)}"
        )


@pytest.mark.asyncio
async def test_debate_max_rounds() -> None:
    """E2E: DebateTopology terminates due to max_rounds exceeded.

    Uses max_rounds=2 with a fixture where judge never approves.

    Assertions:
      - Graph terminates (ainvoke completes without exception)
      - signals["judge_decided"] is NOT True (judge never approved)
      - iter_total > 0 (debate ran for at least one iteration)
      - All Message.id values are unique (MC-3 fan-in dedup)
      - debate_round >= max_rounds (rounds were exhausted)
    """
    if not _MAX_ROUNDS_FIXTURE.exists():
        pytest.skip(f"Fixture not found: {_MAX_ROUNDS_FIXTURE}")

    fixture_path = _MAX_ROUNDS_FIXTURE
    agents = {
        "planner": _make_planner(fixture_path),
        "debater_pro": _make_debater_pro(fixture_path),
        "debater_contra": _make_debater_contra(fixture_path),
        "judge": _make_judge(fixture_path),
    }

    cfg = _make_debate_cfg(max_rounds=2, max_iterations=20)

    import atm.topology.debate  # noqa: F401
    from atm.topology.debate import DebateTopology

    topology = DebateTopology()
    graph = topology.build(agents, cfg)

    initial_state = _make_initial_state()
    final_state = await graph.ainvoke(initial_state)

    shared = final_state.get("shared", {})
    signals = shared.get("signals", {})
    assert signals.get("judge_decided") is not True, (
        f"Expected signals['judge_decided'] NOT True (max_rounds path), got: {signals}"
    )

    iter_total = shared.get("iter_total", 0)
    assert iter_total > 0, f"Expected iter_total > 0, got {iter_total}"

    messages: list[Any] = final_state.get("messages", [])
    if messages:
        msg_ids = [getattr(m, "id", None) for m in messages]
        msg_ids_set = {mid for mid in msg_ids if mid is not None}
        non_none_ids = [mid for mid in msg_ids if mid is not None]
        assert len(msg_ids_set) == len(non_none_ids), (
            f"Duplicate Message.id detected after fan-in! "
            f"Total: {len(non_none_ids)}, Unique: {len(msg_ids_set)}"
        )

    debate_round = shared.get("debate_round", 0)
    assert debate_round >= 2, (
        f"Expected debate_round >= 2 (max_rounds=2), got debate_round={debate_round}"
    )
