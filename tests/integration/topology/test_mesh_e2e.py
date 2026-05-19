"""Integration (e2e) tests for MeshTopology — 2 scenarios.

Tests:
  1. test_mesh_consensus_path — 3 matching votes reach consensus_threshold=3;
     shared.final_answer == "4" after graph completes.
  2. test_mesh_max_rounds_path — no consensus reached; graph exits via
     max_rounds (topology_max) with no consensus_reached signal.

Both tests use mock agents that deterministically write DECISION messages
to their outbox with specific vote_for payloads — no real LLM calls.

The graph is invoked via ainvoke with a minimal initial state.
"""

from __future__ import annotations

from typing import Any

import pytest

import atm.topology.mesh  # noqa: F401
from atm.core.types import Message, MessageKind
from atm.topology.base import TopologyConfig
from atm.topology.mesh import MeshTopology


def _make_cfg(
    *,
    max_iterations: int = 20,
    max_rounds: int = 6,
    consensus_threshold: int = 3,
    activation_policy: str = "round_robin",
    broadcast_bus_cap: int = 200,
) -> TopologyConfig:
    return TopologyConfig(
        name="mesh",
        max_iterations=max_iterations,
        extra={
            "max_rounds": max_rounds,
            "consensus_threshold": consensus_threshold,
            "activation_policy": activation_policy,
            "agent_order": ["planner", "researcher", "executor", "critic"],
            "broadcast_bus_cap": broadcast_bus_cap,
        },
    )


def _make_initial_state(*, task_input: str = "What is 2+2?") -> dict[str, Any]:
    return {
        "shared": {
            "task_id": "test_task",
            "task_input": task_input,
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


def _make_vote_agent(vote_for: str, agent_id: str) -> Any:
    _aid = agent_id
    _vote = vote_for

    class _MockAgent:
        def __init__(self) -> None:
            self.agent_id = _aid

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

    return _MockAgent()


@pytest.mark.asyncio
async def test_mesh_consensus_path() -> None:
    """Consensus path: 3 agents vote for '4', threshold=3, final_answer=='4'.

    Scenario:
      - 4 agents: planner, researcher, executor, critic
      - Each agent emits a DECISION vote with vote_for="4"
      - consensus_threshold=3, max_rounds=6
      - After 3 rounds (planner+researcher+executor each voted), consensus is reached
      - Graph exits with shared.final_answer == "4"
    """
    cfg = _make_cfg(consensus_threshold=3, max_rounds=6)

    agents = {
        "planner": _make_vote_agent("4", "planner"),
        "researcher": _make_vote_agent("4", "researcher"),
        "executor": _make_vote_agent("4", "executor"),
        "critic": _make_vote_agent("4", "critic"),
    }

    topology = MeshTopology()
    graph = topology.build(agents, cfg)

    initial_state = _make_initial_state()
    final_state = await graph.ainvoke(initial_state)

    shared: dict[str, Any] = final_state.get("shared", {})
    signals: dict[str, Any] = shared.get("signals", {})

    assert signals.get("consensus_reached") is True, (
        f"Expected consensus_reached=True, got signals={signals}"
    )

    assert shared.get("final_answer") == "4", (
        f"Expected final_answer='4', got {shared.get('final_answer')!r}"
    )

    assert shared.get("iter_total", 0) > 0, "iter_total should be > 0"

    bus: list[Any] = shared.get("broadcast_bus", [])
    decision_msgs = [m for m in bus if getattr(m, "kind", None) == MessageKind.DECISION]
    assert len(decision_msgs) >= 3, (
        f"Expected >= 3 DECISION messages on bus, got {len(decision_msgs)}"
    )


@pytest.mark.asyncio
async def test_mesh_max_rounds_path() -> None:
    """Max-rounds path: no consensus reached, graph exits via topology_max.

    Scenario:
      - 4 agents: planner (votes "4"), researcher (votes "5"),
        executor (votes "6"), critic (votes "7")
      - No agent agrees → no consensus
      - max_rounds=4, consensus_threshold=3 (never reached)
      - Graph exits when dispatch_round >= max_rounds
    """
    cfg = _make_cfg(consensus_threshold=3, max_rounds=4, max_iterations=50)

    agents = {
        "planner": _make_vote_agent("4", "planner"),
        "researcher": _make_vote_agent("5", "researcher"),
        "executor": _make_vote_agent("6", "executor"),
        "critic": _make_vote_agent("7", "critic"),
    }

    topology = MeshTopology()
    graph = topology.build(agents, cfg)

    initial_state = _make_initial_state()
    final_state = await graph.ainvoke(initial_state)

    shared: dict[str, Any] = final_state.get("shared", {})
    signals: dict[str, Any] = shared.get("signals", {})

    assert signals.get("consensus_reached") is not True, (
        f"Expected no consensus, but signals={signals}"
    )

    dispatch_round: int = int(signals.get("_mesh_dispatch_round", 0))
    assert dispatch_round >= 4, f"Expected dispatch_round >= 4 (max_rounds), got {dispatch_round}"

    assert shared.get("iter_total", 0) > 0, "iter_total should be > 0"

    bus: list[Any] = shared.get("broadcast_bus", [])
    assert len(bus) > 0, "broadcast_bus should not be empty after running"

    assert len(bus) <= 200, f"broadcast_bus exceeded cap: {len(bus)}"
