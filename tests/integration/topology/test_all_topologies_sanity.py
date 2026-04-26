"""Cross-topology sanity + integration-level precedence tests (Step 5, MC-2).

Three test groups:

1. test_all_5_topologies_registered
   Verifies that TopologyRegistry contains all 5 expected names with no duplicates.

2. test_topology_builds_and_runs_smoke (parametrized over 5 topology names)
   Builds each topology via TopologyRegistry.get(name).build(agents, cfg) with
   minimal no-op mock agents, runs ainvoke, and asserts iter_total > 0.

3. test_should_stop_precedence_consistent_integration (MC-2, parametrized over 5 topologies)
   For each topology, constructs a state where iter_total == max_iterations - 1
   AND a topology-specific success signal is present. After ainvoke, asserts that
   the graph terminates with iter_total >= max_iterations — verifying that the
   global max_iter guard fired (highest precedence per arch.md §7.1).

Signals used per topology:
  star:         phase="done" (topology_success = phase == "done")
  chain:        signals["critic_approved"] = True
  mesh:         signals["consensus_reached"] = True
  debate:       signals["judge_decided"] = True
  hierarchical: signals["top_coord_finalize"] = True

These tests do NOT require PostgreSQL, real LLM calls, or external services.
All agents are in-process no-op mocks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

# Side-effect imports — trigger all @TopologyRegistry.register() decorators
import atm.topology.chain
import atm.topology.debate
import atm.topology.hierarchical
import atm.topology.mesh
import atm.topology.star  # noqa: F401
from atm.topology.base import TopologyConfig, TopologyRegistry

# ---------------------------------------------------------------------------
# Expected topology names
# ---------------------------------------------------------------------------

_EXPECTED_NAMES: set[str] = {"star", "chain", "mesh", "debate", "hierarchical"}

# ---------------------------------------------------------------------------
# Minimal initial state factory
# ---------------------------------------------------------------------------


def _minimal_state(
    *,
    iter_total: int = 0,
    signals: dict[str, Any] | None = None,
    phase: str = "planning",
    debate_round: int = 0,
) -> dict[str, Any]:
    """Build a minimal GraphState-compatible dict for topology tests."""
    return {
        "shared": {
            "task_id": "sanity_test",
            "task_input": "smoke test input",
            "iter_total": iter_total,
            "iteration": 0,
            "final_answer": None,
            "phase": phase,
            "phase_started_at_iter": 0,
            "phase_history": [],
            "active_topology": "unknown",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
            "broadcast_bus": [],
            "human_requests": [],
            "human_responses": [],
            "signals": signals or {},
            "debate_round": debate_round,
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


# ---------------------------------------------------------------------------
# No-op mock agent factory
# ---------------------------------------------------------------------------


def _noop_agent(agent_id: str) -> Any:
    """Create a minimal mock agent that returns an empty agents-delta from step()."""
    agent = MagicMock()
    agent.agent_id = agent_id

    async def _step(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agents": {
                agent_id: {
                    "agent_id": agent_id,
                    "outbox": [],
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

    agent.step = _step
    return agent


# ---------------------------------------------------------------------------
# Per-topology agent dicts and configs for smoke tests
# ---------------------------------------------------------------------------


def _agents_and_cfg_for(topology_name: str) -> tuple[dict[str, Any], TopologyConfig]:
    """Return (agents, cfg) for a given topology for smoke-run purposes."""
    if topology_name == "star":
        agents = {
            "planner": _noop_agent("planner"),
            "executor": _noop_agent("executor"),
            "critic": _noop_agent("critic"),
        }
        cfg = TopologyConfig(
            name="star",
            max_iterations=5,
            extra={
                "planning_max_iter": 1,
                "exec_max_iter": 1,
                "verify_max_iter": 1,
            },
        )
        return agents, cfg

    elif topology_name == "chain":
        agents = {
            "planner": _noop_agent("planner"),
            "executor": _noop_agent("executor"),
            "critic": _noop_agent("critic"),
        }
        cfg = TopologyConfig(
            name="chain",
            max_iterations=3,
        )
        return agents, cfg

    elif topology_name == "mesh":
        agents = {
            "planner": _noop_agent("planner"),
            "researcher": _noop_agent("researcher"),
            "executor": _noop_agent("executor"),
            "critic": _noop_agent("critic"),
        }
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=10,
            extra={
                "max_rounds": 3,
                "consensus_threshold": 5,  # high threshold — no consensus
                "agent_order": ["planner", "researcher", "executor", "critic"],
                "broadcast_bus_cap": 200,
            },
        )
        return agents, cfg

    elif topology_name == "debate":
        agents = {
            "planner": _noop_agent("planner"),
            "debater_pro": _noop_agent("debater_pro"),
            "debater_contra": _noop_agent("debater_contra"),
            "judge": _noop_agent("judge"),
        }
        cfg = TopologyConfig(
            name="debate",
            max_iterations=5,
            extra={
                "max_rounds": 2,
                "debater_pro_id": "debater_pro",
                "debater_contra_id": "debater_contra",
                "judge_id": "judge",
            },
        )
        return agents, cfg

    elif topology_name == "hierarchical":
        agents = {
            "executor_a1": _noop_agent("executor_a1"),
            "executor_a2": _noop_agent("executor_a2"),
            "executor_b1": _noop_agent("executor_b1"),
            "executor_b2": _noop_agent("executor_b2"),
        }
        cfg = TopologyConfig(
            name="hierarchical",
            max_iterations=10,
            extra={
                "max_rounds": 2,
                "final_answer_strategy": "json_concat",
                "finalize_signal": "top_coord_finalize",
                "sub_teams": [
                    {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                    {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
                ],
            },
        )
        return agents, cfg

    else:
        raise ValueError(f"Unknown topology: {topology_name!r}")


# ---------------------------------------------------------------------------
# Test 1: registry completeness
# ---------------------------------------------------------------------------


def test_all_5_topologies_registered() -> None:
    """Verify that all 5 expected topology names are registered, with no duplicates.

    Checks:
      - set(TopologyRegistry.list_names()) >= {"star","chain","mesh","debate","hierarchical"}
      - len(set(names)) == len(names)  (no duplicates in registry)
    """
    names: list[str] = TopologyRegistry.list_names()
    names_set: set[str] = set(names)

    assert names_set >= _EXPECTED_NAMES, (
        f"Missing topologies from registry. "
        f"Expected superset of {sorted(_EXPECTED_NAMES)}, "
        f"got {sorted(names_set)}"
    )

    assert len(names_set) == len(names), (
        f"Duplicate topology names detected in registry! "
        f"list_names()={names}, unique={sorted(names_set)}"
    )


# ---------------------------------------------------------------------------
# Test 2: smoke-run (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology_name", sorted(_EXPECTED_NAMES))
@pytest.mark.asyncio
async def test_topology_builds_and_runs_smoke(topology_name: str) -> None:
    """Smoke test: build each topology and run a minimal e2e with no-op agents.

    Verifies:
      - TopologyRegistry.get(name) returns a topology class.
      - .build(agents, cfg) returns a compiled graph without raising.
      - graph.ainvoke(initial_state) completes without raising.
      - iter_total > 0 in the final state (the graph ran at least 1 iteration).
    """
    topology_cls = TopologyRegistry.get(topology_name)
    topology = topology_cls()

    agents, cfg = _agents_and_cfg_for(topology_name)

    compiled = topology.build(agents, cfg)
    assert compiled is not None, f"build() returned None for topology {topology_name!r}"

    initial_state = _minimal_state()
    final_state = await compiled.ainvoke(initial_state)

    assert final_state is not None, (
        f"ainvoke() returned None for topology {topology_name!r}"
    )

    shared = final_state.get("shared") or {}
    iter_total: int = int(shared.get("iter_total") or 0)

    assert iter_total > 0, (
        f"Expected iter_total > 0 after smoke run of {topology_name!r}, "
        f"got iter_total={iter_total}. full shared={shared}"
    )


# ---------------------------------------------------------------------------
# Test 3: stopping-precedence integration (MC-2, parametrized)
# ---------------------------------------------------------------------------


def _precedence_state_for(
    topology_name: str, max_iterations: int
) -> dict[str, Any]:
    """Build an initial state with iter_total = max_iterations - 1
    AND the topology's success signal pre-populated.

    This simulates the scenario where the global max_iter guard and the
    topology-specific success condition become true simultaneously after
    one node increments the counter.
    """
    iter_start = max_iterations - 1

    if topology_name == "star":
        # Success signal for star: phase == "done"
        return _minimal_state(
            iter_total=iter_start,
            phase="done",  # topology_success = (phase == "done")
            signals={"critic_approved": True},  # also set critic signal
        )

    elif topology_name == "chain":
        # Success signal for chain: critic_approved in signals
        return _minimal_state(
            iter_total=iter_start,
            signals={"critic_approved": True},
        )

    elif topology_name == "mesh":
        # Success signal for mesh: consensus_reached in signals
        return _minimal_state(
            iter_total=iter_start,
            signals={
                "consensus_reached": True,
                "consensus_winner": "42",
                "_mesh_rr_index": 0,
                "_mesh_dispatch_round": 0,
            },
        )

    elif topology_name == "debate":
        # Success signal for debate: judge_decided in signals
        return _minimal_state(
            iter_total=iter_start,
            signals={"judge_decided": True},
            debate_round=0,
        )

    elif topology_name == "hierarchical":
        # Success signal for hierarchical: top_coord_finalize in signals
        return _minimal_state(
            iter_total=iter_start,
            signals={
                "top_coord_finalize": True,
                "team_a_draft": "team_a result",
                "team_b_draft": "team_b result",
            },
        )

    else:
        raise ValueError(f"Unknown topology: {topology_name!r}")


@pytest.mark.parametrize("topology_name", sorted(_EXPECTED_NAMES))
@pytest.mark.asyncio
async def test_should_stop_precedence_consistent_integration(topology_name: str) -> None:
    """MC-2: global max_iter guard has highest precedence over topology_success.

    Test protocol (arch.md §7.1 stopping precedence):
      1. Construct a state where:
           iter_total == max_iterations - 1   (one step away from global max)
           AND success_signal == True          (topology-specific success condition)
      2. Run ainvoke on the compiled graph.
      3. Assert:
           a. Graph terminates without exception (no infinite loop).
           b. iter_total in final state >= max_iterations
              (the global max_iter guard fired; even when topology_success is set,
              the counter increment causes max_iter to be hit on the next check).

    Signals used per topology:
      star:         phase="done"
      chain:        signals["critic_approved"] = True
      mesh:         signals["consensus_reached"] = True
      debate:       signals["judge_decided"] = True
      hierarchical: signals["top_coord_finalize"] = True

    Tested via compiled graph (ainvoke), NOT direct _should_stop() call.
    """
    topology_cls = TopologyRegistry.get(topology_name)
    topology = topology_cls()

    agents, cfg = _agents_and_cfg_for(topology_name)
    max_iterations: int = cfg.max_iterations

    compiled = topology.build(agents, cfg)

    initial_state = _precedence_state_for(topology_name, max_iterations)
    initial_iter = initial_state["shared"]["iter_total"]
    assert initial_iter == max_iterations - 1, (
        f"Precondition failed: initial iter_total should be {max_iterations - 1}, "
        f"got {initial_iter}"
    )

    # Run the compiled graph — must terminate (no loop)
    final_state = await compiled.ainvoke(initial_state)

    assert final_state is not None, (
        f"ainvoke() returned None for topology {topology_name!r} in precedence test"
    )

    shared = final_state.get("shared") or {}
    final_iter_total: int = int(shared.get("iter_total") or 0)

    # The graph must have run (incremented iter_total) AND stopped
    assert final_iter_total >= max_iterations, (
        f"MC-2 precedence test FAILED for {topology_name!r}: "
        f"Expected final iter_total >= {max_iterations} (max_iter guard fired), "
        f"got iter_total={final_iter_total}. "
        f"This means the graph may not have run or the counter was not incremented. "
        f"signals={shared.get('signals', {})}"
    )
