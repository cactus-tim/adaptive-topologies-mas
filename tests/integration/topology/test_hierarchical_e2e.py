"""Integration (e2e) tests for HierarchicalTopology.

Tests:
  - 4 worker agents (executor_a1, executor_a2, executor_b1, executor_b2)
  - rule-based coordinators (top_coord, sub_coord_team_a, sub_coord_team_b)
  - FakeLLM scripted fixture: m7_hierarchical_finalize.yaml
  - Assertions:
    - json.loads(state["shared"].final_answer) == {"team_a": ..., "team_b": ...}
    - signals["top_coord_finalize"] == True
    - iter_total > 0

The tests do NOT require PostgreSQL (no ephemeral_pg_dsn fixture).
They run the topology graph directly using ainvoke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import atm.topology.hierarchical  # noqa: F401
from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry
from atm.topology.base import TopologyConfig
from atm.topology.hierarchical import HierarchicalTopology

_FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_pricing() -> Pricing:
    if _PRICING_PATH.exists():
        return Pricing.from_yaml(_PRICING_PATH)
    return Pricing(entries=[])


def _make_llm_for_agent(agent_id: str, fixture_path: Path) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=fixture_path)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_worker_agent(agent_id: str, fixture_path: Path) -> Agent:
    llm = _make_llm_for_agent(agent_id, fixture_path)
    cfg = AgentConfig(
        role="executor",
        system_prompt=f"You are executor agent {agent_id}. Produce a draft.",
        tools=[],
        max_tool_iters=0,
    )
    registry = ToolRegistry()
    return Agent(agent_id=agent_id, cfg=cfg, llm=llm, tools=registry)


def _make_initial_state(
    task_input: str = "concat hello and world, then summarize",
) -> dict[str, Any]:
    return {
        "shared": {
            "task_input": task_input,
            "iter_total": 0,
            "iteration": 0,
            "signals": {},
            "final_answer": None,
            "phase": "execution",
            "phase_started_at_iter": 0,
            "phase_history": [],
            "active_topology": "hierarchical",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
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


@pytest.mark.asyncio
async def test_hierarchical_e2e_finalize_path() -> None:
    """E2E test: HierarchicalTopology with 4 workers, rule-based coordinators.

    Scenario:
      - Task: "concat hello and world, then summarize"
      - executor_a1: scripted → returns "hello" draft
      - executor_a2: scripted → returns "hello team_a result" draft
      - executor_b1: scripted → returns "world" draft
      - executor_b2: scripted → returns "world team_b result" draft
      - top_coord (rule-based): after both teams produce drafts → sets finalize signal

    Assertions:
      1. json.loads(final_answer) == {"team_a": ..., "team_b": ...}
      2. signals["top_coord_finalize"] == True
      3. iter_total > 0
    """
    fixture_path = _FIXTURES_DIR / "m7_hierarchical_finalize.yaml"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    agents: dict[str, Any] = {}
    for agent_id in ["executor_a1", "executor_a2", "executor_b1", "executor_b2"]:
        agents[agent_id] = _make_worker_agent(agent_id, fixture_path)

    cfg = TopologyConfig(
        name="hierarchical",
        max_iterations=20,
        extra={
            "max_rounds": 4,
            "final_answer_strategy": "json_concat",
            "finalize_signal": "top_coord_finalize",
            "sub_teams": [
                {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
            ],
        },
    )

    topology = HierarchicalTopology()
    compiled = topology.build(agents, cfg)

    initial_state = _make_initial_state()
    final_state = await compiled.ainvoke(initial_state)

    shared = final_state.get("shared", {})
    signals = shared.get("signals", {})
    final_answer = shared.get("final_answer")
    iter_total = shared.get("iter_total", 0)

    assert iter_total > 0, f"iter_total should be > 0, got {iter_total}"

    assert signals.get("top_coord_finalize") is True, (
        f"Expected signals['top_coord_finalize'] == True, got: {signals}"
    )

    assert final_answer is not None, "final_answer should not be None"
    parsed = json.loads(final_answer)
    assert isinstance(parsed, dict), (
        f"final_answer should parse to dict, got {type(parsed)}: {final_answer!r}"
    )
    assert "team_a" in parsed, f"Expected 'team_a' key in final_answer, got: {list(parsed.keys())}"
    assert "team_b" in parsed, f"Expected 'team_b' key in final_answer, got: {list(parsed.keys())}"

    assert parsed["team_a"] and parsed["team_a"] != "<incomplete>", (
        f"team_a draft should not be empty, got: {parsed['team_a']!r}"
    )
    assert parsed["team_b"] and parsed["team_b"] != "<incomplete>", (
        f"team_b draft should not be empty, got: {parsed['team_b']!r}"
    )


@pytest.mark.asyncio
async def test_hierarchical_e2e_max_iterations_stops_graph() -> None:
    """E2E test: graph terminates when max_iterations is reached.

    With max_iterations=3 and no finalize signal, the graph should stop.
    """
    from unittest.mock import MagicMock

    no_draft_agents: dict[str, Any] = {}
    for agent_id in ["executor_a1", "executor_a2", "executor_b1", "executor_b2"]:
        agent = MagicMock()
        agent.agent_id = agent_id

        async def noop_step(state: dict, _id: str = agent_id) -> dict:
            return {"agents": {_id: {"agent_id": _id, "outbox": []}}}

        agent.step = noop_step
        no_draft_agents[agent_id] = agent

    cfg = TopologyConfig(
        name="hierarchical",
        max_iterations=3,
        extra={
            "max_rounds": 10,
            "final_answer_strategy": "json_concat",
            "finalize_signal": "top_coord_finalize",
            "sub_teams": [
                {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
            ],
        },
    )

    topology = HierarchicalTopology()
    compiled = topology.build(no_draft_agents, cfg)

    initial_state = _make_initial_state()
    final_state = await compiled.ainvoke(initial_state)

    shared = final_state.get("shared", {})
    iter_total = shared.get("iter_total", 0)

    assert iter_total >= 1, "Should have run at least once"
    assert iter_total <= cfg.max_iterations + 10, (
        f"iter_total {iter_total} exceeded max_iterations {cfg.max_iterations} by too much"
    )
