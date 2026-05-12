"""Unit tests for StarTopology HITL integration (M9.1 Step 2.2).

Tests cover (7+ required):
  1. Back-compat: no human_cfg → graph identical to M7 (no human_reviewer node).
  2. Back-compat: enabled=False → same as above.
  3. Graph structure: enabled=True → human_reviewer node inserted after critic_postprocess.
  4. Edge order: critic_postprocess → human_reviewer → coordinator.
  5. Approve response → shared["human_approved"]=True, continues to coordinator.
  6. Reject response → needs_rerun set, continues to coordinator (loop).
  7. Coordinator-override force-advance: signals["human_phase_override"]="advance"
     routes to the NEXT phase node.
  8. Coordinator-override force-stay: signals["human_phase_override"]="stay"
     routes same as normal (no change in routing).
  9. Coordinator-override force-finalize: signals["human_phase_override"]="finalize"
     routes to END.
  10. Back-compat route: no signal → routing unchanged (normal phase routing).
  11. Signal consumed (cleared to None) after _route_from_coord reads it.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

import atm.topology.star  # noqa: F401 — side-effect registers "star"
from atm.core.types import HumanResponse, HumanRole, Message, MessageKind, Phase
from atm.experiment.config import HumanCfg
from atm.topology.base import TopologyConfig, TopologyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    max_iterations: int = 20,
    planning_max_iter: int = 1,
    exec_max_iter: int = 1,
    verify_max_iter: int = 2,
) -> TopologyConfig:
    return TopologyConfig(
        name="star",
        max_iterations=max_iterations,
        extra={
            "planning_max_iter": planning_max_iter,
            "exec_max_iter": exec_max_iter,
            "verify_max_iter": verify_max_iter,
        },
    )


def _make_human_cfg(
    *,
    enabled: bool = True,
    gateway: str = "llm_simulated",
    role: HumanRole = HumanRole.REVIEWER,
    timeout_s: float | None = None,
    timeout_policy: str = "skip",
    extra: dict[str, Any] | None = None,
) -> HumanCfg:
    return HumanCfg(
        enabled=enabled,
        gateway=gateway,
        role=role,
        timeout_s=timeout_s,
        timeout_policy=timeout_policy,  # type: ignore[arg-type]
        extra=extra,
    )


def _make_mock_agent(agent_id: str) -> MagicMock:
    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agents": {agent_id: {"agent_id": agent_id, "outbox": []}},
            "messages": [],
        }

    agent.step = fake_step
    return agent


def _make_mock_agent_with_outbox(agent_id: str, outbox_messages: list[Any]) -> MagicMock:
    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "agents": {agent_id: {"agent_id": agent_id, "outbox": outbox_messages}},
            "messages": outbox_messages,
        }

    agent.step = fake_step
    return agent


def _make_initial_state(
    *,
    phase: str = Phase.PLANNING,
    iter_total: int = 0,
    signals: dict[str, Any] | None = None,
    run_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    _run_id = run_id or uuid.uuid4()
    return {
        "shared": {
            "run_id": _run_id,
            "task_input": "test task",
            "phase": phase,
            "iter_total": iter_total,
            "iteration": 0,
            "phase_started_at_iter": 0,
            "phase_history": [],
            "active_topology": "star",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
            "final_answer": None,
            "signals": signals or {},
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


def _make_approve_response() -> HumanResponse:
    return HumanResponse(action="approve", comment="LGTM", source="llm_sim", timed_out=False)


def _make_reject_response(comment: str = "Needs work") -> HumanResponse:
    return HumanResponse(action="reject", comment=comment, source="llm_sim", timed_out=False)


def _build_star_graph(
    *,
    agents: dict[str, Any] | None = None,
    cfg: TopologyConfig | None = None,
    human_cfg: HumanCfg | None = None,
    mock_graph: Any = None,
    mock_gw_cls: Any = None,
) -> Any:
    """Build star topology; with mock_graph, returns the mock directly."""
    cls = TopologyRegistry.get("star")
    topology = cls()
    if agents is None:
        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
    if cfg is None:
        cfg = _make_cfg()

    kwargs: dict[str, Any] = {}
    if human_cfg is not None:
        kwargs["human_cfg"] = human_cfg
    if mock_graph is None:
        kwargs["checkpointer"] = MemorySaver()

    if mock_graph is not None:
        with patch("atm.topology.star.StateGraph", return_value=mock_graph):
            if mock_gw_cls is not None:
                with patch("atm.topology.star.LLMSimulatedGateway", mock_gw_cls):
                    return topology.build(agents, cfg, **kwargs)
            return topology.build(agents, cfg, **kwargs)

    if mock_gw_cls is not None:
        with patch("atm.topology.star.LLMSimulatedGateway", mock_gw_cls):
            return topology.build(agents, cfg, **kwargs)

    return topology.build(agents, cfg, **kwargs)


# ---------------------------------------------------------------------------
# 1. Back-compat: no human_cfg → no human_reviewer node
# ---------------------------------------------------------------------------


class TestStarBackCompatNoHumanCfg:
    """Without human_cfg (or disabled), graph is identical to M7."""

    def test_no_human_cfg_no_human_reviewer_node(self) -> None:
        """Default (no human_cfg) — human_reviewer node NOT in graph."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        _build_star_graph(mock_graph=mock_graph)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" not in node_names

    def test_disabled_human_cfg_no_human_reviewer_node(self) -> None:
        """human_cfg.enabled=False — human_reviewer node NOT in graph."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        human_cfg = _make_human_cfg(enabled=False)
        _build_star_graph(human_cfg=human_cfg, mock_graph=mock_graph)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" not in node_names

    def test_no_human_cfg_critic_postprocess_to_coordinator_direct(self) -> None:
        """Back-compat: critic_postprocess edge goes directly to coordinator."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        _build_star_graph(mock_graph=mock_graph)

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        assert ("critic_postprocess", "coordinator") in edge_calls
        assert ("critic_postprocess", "human_reviewer") not in edge_calls


# ---------------------------------------------------------------------------
# 2. Graph structure with HITL enabled
# ---------------------------------------------------------------------------


class TestStarHITLNodeInsertion:
    """With human_cfg.enabled=True, human_reviewer node is added correctly."""

    def test_human_reviewer_node_added_when_enabled(self) -> None:
        """human_reviewer node is present in graph when enabled=True."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()

        human_cfg = _make_human_cfg(enabled=True)
        _build_star_graph(human_cfg=human_cfg, mock_graph=mock_graph, mock_gw_cls=mock_gw)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" in node_names

    def test_critic_postprocess_to_human_reviewer_edge(self) -> None:
        """Edge critic_postprocess → human_reviewer is added when enabled=True."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()

        human_cfg = _make_human_cfg(enabled=True)
        _build_star_graph(human_cfg=human_cfg, mock_graph=mock_graph, mock_gw_cls=mock_gw)

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        assert ("critic_postprocess", "human_reviewer") in edge_calls

    def test_human_reviewer_to_coordinator_edge(self) -> None:
        """Edge human_reviewer → coordinator is added when enabled=True."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()

        human_cfg = _make_human_cfg(enabled=True)
        _build_star_graph(human_cfg=human_cfg, mock_graph=mock_graph, mock_gw_cls=mock_gw)

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        assert ("human_reviewer", "coordinator") in edge_calls

    def test_dispatch_order_is_critic_postprocess_then_human_reviewer_then_coordinator(
        self,
    ) -> None:
        """Edge insertion order: critic_postprocess → human_reviewer before human_reviewer → coordinator."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()

        human_cfg = _make_human_cfg(enabled=True)
        _build_star_graph(human_cfg=human_cfg, mock_graph=mock_graph, mock_gw_cls=mock_gw)

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        cp_to_hr_idx = edge_calls.index(("critic_postprocess", "human_reviewer"))
        hr_to_coord_idx = edge_calls.index(("human_reviewer", "coordinator"))
        assert cp_to_hr_idx < hr_to_coord_idx


# ---------------------------------------------------------------------------
# 3. Reviewer mode: approve / reject
# ---------------------------------------------------------------------------


class TestStarReviewerApprove:
    """Reviewer approve: human_approved=True, run continues."""

    @pytest.mark.asyncio
    async def test_approve_sets_human_approved_and_terminates(self) -> None:
        """With approve response, run terminates with human_approved=True."""
        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True},
        )
        run_id = uuid.uuid4()
        initial = _make_initial_state(run_id=run_id)

        fake_gateway = AsyncMock()
        fake_gateway.request = AsyncMock(return_value=_make_approve_response())

        mock_gw_cls = MagicMock(return_value=fake_gateway)

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent_with_outbox("critic", [decision_msg]),
        }
        cfg = _make_cfg(planning_max_iter=1, exec_max_iter=1, verify_max_iter=3)

        async def _fake_dispatch(name: str, data: Any) -> None:
            pass

        with (
            patch("atm.topology.star.LLMSimulatedGateway", mock_gw_cls),
            patch(
                "atm.human._node_factory.adispatch_custom_event",
                side_effect=_fake_dispatch,
            ),
        ):
            human_cfg = _make_human_cfg(enabled=True)
            graph = _build_star_graph(agents=agents, cfg=cfg, human_cfg=human_cfg)

        result = await graph.ainvoke(
            initial,
            config={"configurable": {"thread_id": f"test-approve-{uuid.uuid4()}"}},
        )

        assert result is not None
        assert result["shared"].get("human_approved") is True


class TestStarReviewerReject:
    """Reviewer reject: gateway is called, graph eventually terminates."""

    @pytest.mark.asyncio
    async def test_reject_gateway_called_and_graph_terminates(self) -> None:
        """With reject response (first), graph continues looping until max_iter terminates it."""
        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="REJECT",
            payload={"approved": False},
        )
        run_id = uuid.uuid4()
        initial = _make_initial_state(run_id=run_id)

        # Reject on every call; graph will terminate via max_iterations cap
        reject_resp = _make_reject_response("Needs more work")
        call_count = 0

        async def _always_reject(ctx: Any, *, request_id: str) -> HumanResponse:
            nonlocal call_count
            call_count += 1
            return reject_resp

        fake_gateway = MagicMock()
        fake_gateway.request = _always_reject

        mock_gw_cls = MagicMock(return_value=fake_gateway)

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent_with_outbox("critic", [decision_msg]),
        }
        # Small max_iterations so test terminates quickly
        cfg = _make_cfg(
            max_iterations=8,
            planning_max_iter=1,
            exec_max_iter=1,
            verify_max_iter=2,
        )

        async def _fake_dispatch(name: str, data: Any) -> None:
            pass

        with (
            patch("atm.topology.star.LLMSimulatedGateway", mock_gw_cls),
            patch(
                "atm.human._node_factory.adispatch_custom_event",
                side_effect=_fake_dispatch,
            ),
        ):
            human_cfg = _make_human_cfg(enabled=True)
            graph = _build_star_graph(agents=agents, cfg=cfg, human_cfg=human_cfg)

        result = await graph.ainvoke(
            initial,
            config={"configurable": {"thread_id": f"test-reject-{uuid.uuid4()}"}},
        )

        # Graph terminates via max_iter; gateway was called at least once with reject
        assert result is not None
        assert call_count >= 1


# ---------------------------------------------------------------------------
# 4. Coordinator-override: _route_from_coord reads human_phase_override
# ---------------------------------------------------------------------------


class TestRouteFromCoordOverride:
    """_route_from_coord reads signals['human_phase_override'] as first check."""

    def _get_route_fn(self, phase: str = "verification") -> Any:
        """Build star and extract the _route_from_coord function via graph introspection.

        We test _route_from_coord by calling it directly on a state dict that
        has signals["human_phase_override"] set.
        """
        # Import the build_human_node_factory to build a real graph
        from atm.topology.star import StarTopology

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
        cfg = _make_cfg()
        topology = StarTopology()

        # We capture the _route_from_coord closure via a build call
        # by patching StateGraph to capture the conditional edges call
        captured_route_fn: list[Any] = []

        class CapturingGraph:
            def __init__(self, schema: Any) -> None:
                self._nodes: dict[str, Any] = {}
                self._edges: list[Any] = []
                self._conditional_edges: list[Any] = []

            def add_node(self, name: str, fn: Any) -> None:
                self._nodes[name] = fn

            def add_edge(self, src: str, dst: str) -> None:
                self._edges.append((src, dst))

            def add_conditional_edges(self, src: str, fn: Any, mapping: dict[str, Any]) -> None:
                self._conditional_edges.append((src, fn, mapping))
                if src == "coordinator":
                    captured_route_fn.append(fn)

            def compile(self, **kw: Any) -> Any:
                return MagicMock()

        capturing = CapturingGraph(None)

        with patch("atm.topology.star.StateGraph", return_value=capturing):
            topology.build(agents, cfg)

        assert captured_route_fn, "No conditional edges from coordinator found"
        return captured_route_fn[0]

    def test_force_finalize_returns_end(self) -> None:
        """signals['human_phase_override']='finalize' → routing returns END."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={"human_phase_override": "finalize"},
            phase=Phase.VERIFICATION,
        )
        result = route_fn(state)
        assert result == "__end__"

    def test_force_advance_from_planning_routes_to_executor(self) -> None:
        """signals['human_phase_override']='advance' in planning phase → executor."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={"human_phase_override": "advance"},
            phase=Phase.PLANNING,
        )
        result = route_fn(state)
        assert result == "executor"

    def test_force_advance_from_execution_routes_to_critic(self) -> None:
        """signals['human_phase_override']='advance' in execution phase → critic."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={"human_phase_override": "advance"},
            phase=Phase.EXECUTION,
        )
        result = route_fn(state)
        assert result == "critic"

    def test_force_stay_falls_through_to_normal_routing(self) -> None:
        """signals['human_phase_override']='stay' → same as no override (planning → planner)."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={"human_phase_override": "stay"},
            phase=Phase.PLANNING,
        )
        result = route_fn(state)
        # "stay" falls through to normal routing: planning → planner
        assert result == "planner"

    def test_no_override_signal_uses_normal_routing(self) -> None:
        """Without signals['human_phase_override'], routing is unchanged (planning → planner)."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={},  # no override
            phase=Phase.PLANNING,
        )
        result = route_fn(state)
        assert result == "planner"

    def test_none_override_uses_normal_routing(self) -> None:
        """signals['human_phase_override']=None → routing is unchanged."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={"human_phase_override": None},
            phase=Phase.PLANNING,
        )
        result = route_fn(state)
        assert result == "planner"

    def test_override_signal_cleared_after_read(self) -> None:
        """signals['human_phase_override'] is set to None after being read."""
        route_fn = self._get_route_fn()
        signals: dict[str, Any] = {"human_phase_override": "finalize"}
        state = _make_initial_state(
            signals=signals,
            phase=Phase.VERIFICATION,
        )
        route_fn(state)
        # The signals dict in shared should be cleared
        shared = state.get("shared", {})
        updated_signals = shared.get("signals", {})
        assert updated_signals.get("human_phase_override") is None

    def test_invalid_override_value_ignored(self) -> None:
        """signals['human_phase_override']='unknown_value' → falls through to normal routing."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={"human_phase_override": "unknown_value"},
            phase=Phase.PLANNING,
        )
        result = route_fn(state)
        # Should fall through to normal planning routing
        assert result == "planner"


# ---------------------------------------------------------------------------
# 5. Full graph run: back-compat (no HITL) — graph runs exactly like M7
# ---------------------------------------------------------------------------


class TestStarBackCompatFullRun:
    """Without human_cfg, full graph run produces same behavior as M7 StarTopology."""

    @pytest.mark.asyncio
    async def test_back_compat_approved_run_terminates(self) -> None:
        """Back-compat: no human_cfg, critic approves → run terminates normally."""
        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True},
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent_with_outbox("critic", [decision_msg]),
        }
        cfg = _make_cfg(planning_max_iter=1, exec_max_iter=1, verify_max_iter=3)

        cls = TopologyRegistry.get("star")
        topology = cls()
        graph = topology.build(agents, cfg, checkpointer=MemorySaver())

        initial = _make_initial_state()
        result = await graph.ainvoke(
            initial,
            config={"configurable": {"thread_id": f"back-compat-{uuid.uuid4()}"}},
        )

        assert result is not None
        phase = result["shared"].get("phase")
        assert str(phase) == "done" or result["shared"].get("final_answer") is not None
