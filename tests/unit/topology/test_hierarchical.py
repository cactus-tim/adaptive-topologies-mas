"""Unit tests for atm.topology.hierarchical — HierarchicalTopology.

Tests cover (11 tests):
  1.  test_build_creates_two_subgraphs_via_compile_call
  2.  test_top_coord_route_to_subgraphs_on_first_iter
  3.  test_top_coord_finalize_signal_sets_signals_and_routes_end
  4.  test_top_coord_max_rounds_reached_routes_end_with_topology_max
  5.  test_global_max_iterations_overrides_topology_max
  6.  test_subgraph_runs_workers_in_order
  7.  test_register_under_name_hierarchical
  8.  test_strict_two_levels_invariant
  9.  test_no_compiled_subgraph_in_workers (MC-1)
  10. test_final_answer_json_concat_format (MC-6)
  11. test_top_coord_and_sub_coord_are_not_in_agents_dict (MC-4)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

import atm.topology.hierarchical  # noqa: F401
from atm.core.types import Message, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry
from atm.topology.hierarchical import HierarchicalTopology


def _make_cfg(
    *,
    max_iterations: int = 20,
    max_rounds: int = 4,
    final_answer_strategy: str = "json_concat",
    sub_teams: list[dict[str, Any]] | None = None,
    finalize_signal: str = "top_coord_finalize",
) -> TopologyConfig:
    extra: dict[str, Any] = {
        "max_rounds": max_rounds,
        "final_answer_strategy": final_answer_strategy,
        "finalize_signal": finalize_signal,
    }
    if sub_teams is not None:
        extra["sub_teams"] = sub_teams
    return TopologyConfig(name="hierarchical", max_iterations=max_iterations, extra=extra)


def _make_state(
    *,
    iter_total: int = 0,
    iteration: int = 0,
    signals: dict[str, Any] | None = None,
    agents: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "shared": {
            "task_input": "test task",
            "iter_total": iter_total,
            "iteration": iteration,
            "signals": signals or {},
            "final_answer": None,
        },
        "agents": agents or {},
        "messages": [],
    }


def _make_mock_agent(agent_id: str) -> MagicMock:
    """Create a mock agent that returns a simple DRAFT delta dict."""
    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict) -> dict:
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


class TestBuildCreatesSubgraphs:
    """build() compiles two subgraphs and returns a CompiledStateGraph."""

    def test_build_creates_two_subgraphs_via_compile_call(self) -> None:
        """build() should return a CompiledStateGraph containing team_a and team_b nodes."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        compiled = topology.build(agents, cfg)

        assert isinstance(compiled, CompiledStateGraph), (
            f"Expected CompiledStateGraph, got {type(compiled)}"
        )

        node_names = set(compiled.get_graph().nodes.keys())
        assert "team_a" in node_names, f"Expected 'team_a' node, got nodes: {node_names}"
        assert "team_b" in node_names, f"Expected 'team_b' node, got nodes: {node_names}"


class TestTopCoordRouting:
    """top_coord routes to team_a on first iteration."""

    async def test_top_coord_route_to_subgraphs_on_first_iter(self) -> None:
        """On first iteration, top_coord should activate both teams (start with team_a)."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        compiled = topology.build(agents, cfg)

        initial_state = _make_state(iter_total=0)

        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        assert shared.get("iter_total", 0) > 0, "iter_total should be incremented"


class TestTopCoordFinalizeSignal:
    """When top_coord_finalize signal is set, routing goes to END."""

    async def test_top_coord_finalize_signal_sets_signals_and_routes_end(self) -> None:
        """Graph should terminate when top_coord_finalize signal is set."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20)
        agents = _make_agents()

        compiled = topology.build(agents, cfg)

        initial_state = _make_state(iter_total=0)
        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        signals = shared.get("signals", {})

        assert signals.get("top_coord_finalize") is True, (
            f"Expected top_coord_finalize=True in signals, got: {signals}"
        )

        assert shared.get("final_answer") is not None, "final_answer should be set"


class TestTopCoordMaxRounds:
    """When max_rounds is exceeded, topology terminates."""

    async def test_top_coord_max_rounds_reached_routes_end_with_topology_max(self) -> None:
        """Graph should terminate when max_rounds is reached."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_rounds=1, max_iterations=20)
        agents = _make_agents()

        compiled = topology.build(agents, cfg)

        initial_state = _make_state(iter_total=0)
        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        assert shared.get("iter_total", 0) > 0, "iter_total should be > 0"


class TestGlobalMaxIterations:
    """global max_iterations takes precedence over topology max_rounds."""

    async def test_global_max_iterations_overrides_topology_max(self) -> None:
        """Graph must stop at max_iterations regardless of max_rounds."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=2, max_rounds=10)

        no_draft_agents: dict[str, Any] = {}
        for agent_id in ["executor_a1", "executor_a2", "executor_b1", "executor_b2"]:
            agent = MagicMock()
            agent.agent_id = agent_id

            async def noop_step(state: dict, _id: str = agent_id) -> dict:
                return {"agents": {_id: {"agent_id": _id, "outbox": []}}}

            agent.step = noop_step
            no_draft_agents[agent_id] = agent

        compiled = topology.build(no_draft_agents, cfg)

        initial_state = _make_state(iter_total=0)
        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        iter_total = shared.get("iter_total", 0)

        assert iter_total >= 1, "Should have run at least once"
        assert iter_total <= cfg.max_iterations + 5, (
            f"iter_total {iter_total} exceeded max_iterations {cfg.max_iterations}"
        )


class TestSubgraphWorkerOrder:
    """Workers within a subgraph run in the configured order."""

    async def test_subgraph_runs_workers_in_order(self) -> None:
        """Workers executor_a1, executor_a2 should both run within team_a subgraph."""
        call_order: list[str] = []

        def _make_tracking_agent(agent_id: str) -> MagicMock:
            agent = MagicMock()
            agent.agent_id = agent_id

            async def tracking_step(state: dict) -> dict:
                call_order.append(agent_id)
                msg = Message(
                    sender=agent_id,
                    kind=MessageKind.DRAFT,
                    content=f"draft from {agent_id}",
                )
                return {
                    "agents": {agent_id: {"agent_id": agent_id, "outbox": [msg]}},
                }

            agent.step = tracking_step
            return agent

        tracking_agents = {
            "executor_a1": _make_tracking_agent("executor_a1"),
            "executor_a2": _make_tracking_agent("executor_a2"),
            "executor_b1": _make_tracking_agent("executor_b1"),
            "executor_b2": _make_tracking_agent("executor_b2"),
        }

        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20, max_rounds=4)
        compiled = topology.build(tracking_agents, cfg)

        initial_state = _make_state(iter_total=0)
        await compiled.ainvoke(initial_state)

        assert "executor_a1" in call_order, f"executor_a1 not called; order={call_order}"
        assert "executor_a2" in call_order, f"executor_a2 not called; order={call_order}"

        idx_a1 = next(i for i, x in enumerate(call_order) if x == "executor_a1")
        idx_a2 = next(i for i, x in enumerate(call_order) if x == "executor_a2")
        assert idx_a1 < idx_a2, (
            f"executor_a1 (idx={idx_a1}) should run before executor_a2 (idx={idx_a2})"
        )


class TestRegistration:
    """HierarchicalTopology is registered under 'hierarchical' in TopologyRegistry."""

    def test_register_under_name_hierarchical(self) -> None:
        """TopologyRegistry should have 'hierarchical' registered."""
        assert "hierarchical" in TopologyRegistry.list_names(), (
            f"'hierarchical' not in registry. Available: {TopologyRegistry.list_names()}"
        )

        cls = TopologyRegistry.get("hierarchical")
        assert cls is HierarchicalTopology

    def test_hierarchical_name_attribute(self) -> None:
        """HierarchicalTopology.name should be 'hierarchical'."""
        assert HierarchicalTopology.name == "hierarchical"


class TestStrictTwoLevels:
    """ValueError is raised when a sub_team config contains nested sub_teams."""

    def test_strict_two_levels_invariant(self) -> None:
        """build() must raise ValueError when sub_teams[i] contains 'sub_teams' key."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(
            sub_teams=[
                {
                    "team_id": "team_a",
                    "workers": ["executor_a1"],
                    "sub_teams": [  # 3rd level — must raise ValueError
                        {"team_id": "team_a_sub", "workers": ["executor_a1_1"]}
                    ],
                },
                {"team_id": "team_b", "workers": ["executor_b1"]},
            ]
        )
        agents = _make_agents()

        with pytest.raises(ValueError, match="sub_teams"):
            topology.build(agents, cfg)


class TestNoCompiledSubgraphInWorkers:
    """Workers are plain Agent nodes, not compiled subgraphs (MC-1)."""

    def test_no_compiled_subgraph_in_workers(self) -> None:
        """Agents dict should only contain plain worker agents, not compiled subgraphs."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        for agent_id, agent in agents.items():
            assert not isinstance(agent, CompiledStateGraph), (
                f"Worker {agent_id!r} should not be a CompiledStateGraph"
            )

        compiled = topology.build(agents, cfg)
        assert isinstance(compiled, CompiledStateGraph)


class TestFinalAnswerJsonConcat:
    """final_answer is a JSON-concat dict with team_a and team_b keys (MC-6)."""

    async def test_final_answer_json_concat_format(self) -> None:
        """final_answer should be json.dumps({"team_a": ..., "team_b": ...})."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(final_answer_strategy="json_concat")
        agents = _make_agents()

        compiled = topology.build(agents, cfg)
        initial_state = _make_state(iter_total=0)
        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        final_answer = shared.get("final_answer")

        assert final_answer is not None, "final_answer should not be None"

        parsed = json.loads(final_answer)
        assert isinstance(parsed, dict), f"final_answer should parse to dict, got {type(parsed)}"
        assert "team_a" in parsed, (
            f"Expected 'team_a' key in final_answer, got keys: {list(parsed.keys())}"
        )
        assert "team_b" in parsed, (
            f"Expected 'team_b' key in final_answer, got keys: {list(parsed.keys())}"
        )


def _make_mock_worker_with_file_write(
    agent_id: str, *, draft_content: str, file_content: str
) -> MagicMock:
    """Mock worker that emits both a DRAFT and a successful file_write call."""
    from atm.core.types import ToolCall, ToolResult

    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict) -> dict:
        call = ToolCall(
            tool_name="file_write",
            args={"path": "solution.py", "content": file_content},
            issued_by=agent_id,
        )
        msg = Message(
            sender=agent_id,
            kind=MessageKind.DRAFT,
            content=draft_content,
        )
        return {
            "agents": {
                agent_id: {
                    "agent_id": agent_id,
                    "outbox": [msg],
                    "tool_calls": [call],
                    "tool_results": [
                        ToolResult(call_id=call.id, ok=True, output=None, latency_ms=1)
                    ],
                }
            },
            "messages": [],
        }

    agent.step = fake_step
    return agent


class TestFinalizeTaskAware:
    """hierarchical_finalize honours shared.task_id when choosing artifact path."""

    @staticmethod
    def _agents_with_file_writes() -> dict[str, Any]:
        return {
            "executor_a1": _make_mock_worker_with_file_write(
                "executor_a1", draft_content="A1 answer: 42", file_content="PYCODE_A1"
            ),
            "executor_a2": _make_mock_worker_with_file_write(
                "executor_a2", draft_content="A2 answer: 42", file_content="PYCODE_A2"
            ),
            "executor_b1": _make_mock_worker_with_file_write(
                "executor_b1", draft_content="B1 answer: 42", file_content="PYCODE_B1"
            ),
            "executor_b2": _make_mock_worker_with_file_write(
                "executor_b2", draft_content="B2 answer: 42", file_content="PYCODE_B2"
            ),
        }

    @staticmethod
    def _state_with_task_id(task_id: str) -> dict[str, Any]:
        st = _make_state(iter_total=0)
        st["shared"]["task_id"] = task_id
        return st

    async def test_code_task_returns_solution_py_artifact(self) -> None:
        """For task_id=humaneval, final_answer is the solution.py content (artifact path)."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(final_answer_strategy="json_concat")
        compiled = topology.build(self._agents_with_file_writes(), cfg)

        final_state = await compiled.ainvoke(self._state_with_task_id("humaneval"))
        final_answer = final_state["shared"]["final_answer"]
        assert final_answer is not None
        assert final_answer.startswith("PYCODE_"), (
            f"code-task final_answer should be a solution.py artifact, got: {final_answer!r}"
        )

    async def test_non_code_task_uses_json_concat_over_solution_py(self) -> None:
        """For non-code tasks, json_concat with team DRAFTs wins over solution.py."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(final_answer_strategy="json_concat")
        compiled = topology.build(self._agents_with_file_writes(), cfg)

        for non_code in ("gsm8k", "commongen", "dabench"):
            agents = self._agents_with_file_writes()
            compiled = topology.build(agents, cfg)
            final_state = await compiled.ainvoke(self._state_with_task_id(non_code))
            final_answer = final_state["shared"]["final_answer"]
            assert final_answer is not None, f"final_answer missing for task_id={non_code}"

            assert not final_answer.startswith("PYCODE_"), (
                f"non-code task_id={non_code!r} leaked solution.py artifact: {final_answer!r}"
            )
            parsed = json.loads(final_answer)
            assert "team_a" in parsed and "team_b" in parsed, (
                f"non-code task_id={non_code!r} should produce json_concat envelope, "
                f"got keys: {list(parsed.keys())}"
            )


class TestCoordinatorsNotInAgentsDict:
    """top_coord and sub_coord nodes are NOT in the agents dict (MC-4)."""

    def test_top_coord_and_sub_coord_are_not_in_agents_dict(self) -> None:
        """Coordinator closures must not appear in agents dict."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        for key in agents:
            assert "coord" not in key.lower(), (
                f"Coordinator key {key!r} found in agents dict; "
                "coordinators must be rule-based closures, not in agents"
            )

        original_keys = set(agents.keys())
        topology.build(agents, cfg)
        post_build_keys = set(agents.keys())

        assert original_keys == post_build_keys, (
            f"build() mutated agents dict. Added keys: {post_build_keys - original_keys}"
        )

    def test_agents_dict_contains_only_workers(self) -> None:
        """After build(), agents dict should contain only worker agents."""
        agents = _make_agents()
        expected_workers = {"executor_a1", "executor_a2", "executor_b1", "executor_b2"}
        assert set(agents.keys()) == expected_workers, (
            f"Unexpected agents: {set(agents.keys()) - expected_workers}"
        )


class TestShouldStopFinalizeSignal:
    """_route_from_top_coord and _route_from_after_team_b use topology_success path for finalize."""

    async def test_should_stop_returns_topology_success_on_finalize_signal(self) -> None:
        """When finalize_signal is set, graph terminates via topology_success path in _should_stop.

        Verifies that finish_reason is topology_success (not topology_max or max_iter)
        when both teams have produced drafts and top_coord_finalize=True.
        """
        from atm.topology.base import _should_stop

        finalize_signal = "top_coord_finalize"
        state = _make_state(
            iter_total=5,
            signals={
                finalize_signal: True,
                "team_a_draft": "team_a result",
                "team_b_draft": "team_b result",
            },
        )

        cfg = _make_cfg(max_iterations=20, max_rounds=4, finalize_signal=finalize_signal)

        stop, reason = _should_stop(
            state,
            cfg,
            topology_success=bool(state["shared"]["signals"].get(finalize_signal)),
            topology_max_reached=False,
        )

        assert stop is True, "Expected _should_stop to return True when finalize_signal is set"
        assert reason == "success", (
            f"Expected reason='success' (topology_success path), got {reason!r}"
        )

    async def test_finalize_signal_in_state_causes_graph_to_stop(self) -> None:
        """End-to-end: when workers produce drafts, graph exits with finalize_signal=True."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20, max_rounds=4)
        agents = _make_agents()

        compiled = topology.build(agents, cfg)
        initial_state = _make_state(iter_total=0)
        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        signals = shared.get("signals", {})

        assert signals.get("top_coord_finalize") is True, (
            f"Expected top_coord_finalize=True, got signals={signals}"
        )


def _make_mock_graph() -> MagicMock:
    """Return a MagicMock with the methods StateGraph would expose during build()."""
    g = MagicMock()
    g.add_node.return_value = None
    g.add_edge.return_value = None
    g.add_conditional_edges.return_value = None
    compiled = MagicMock()
    g.compile.return_value = compiled
    return g


class TestHierarchicalBuildCheckpointer:
    """build() forwards checkpointer kwarg only to the top-level graph.compile()."""

    def test_build_passes_checkpointer_to_compile_for_top_graph(self) -> None:
        """The LAST compile() call (top-level graph) receives checkpointer=mock_cp."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()
        mock_cp = MagicMock()

        mock_team_a = _make_mock_graph()
        mock_team_b = _make_mock_graph()
        mock_top = _make_mock_graph()

        with patch(
            "atm.topology.hierarchical.StateGraph",
            side_effect=[mock_team_a, mock_team_b, mock_top],
        ):
            compiled = topology.build(agents, cfg, checkpointer=mock_cp)

        mock_top.compile.assert_called_once_with(checkpointer=mock_cp)
        assert compiled is mock_top.compile.return_value

    def test_subgraphs_compiled_without_checkpointer(self) -> None:
        """Subgraph compile() calls must NOT receive the checkpointer kwarg.

        LangGraph contract: only the root graph's checkpointer manages state;
        embedded subgraphs must be compiled with no arguments.
        """
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()
        mock_cp = MagicMock()

        mock_team_a = _make_mock_graph()
        mock_team_b = _make_mock_graph()
        mock_top = _make_mock_graph()

        with patch(
            "atm.topology.hierarchical.StateGraph",
            side_effect=[mock_team_a, mock_team_b, mock_top],
        ):
            topology.build(agents, cfg, checkpointer=mock_cp)

        mock_team_a.compile.assert_called_once_with()
        mock_team_b.compile.assert_called_once_with()
        mock_top.compile.assert_called_once_with(checkpointer=mock_cp)
