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

        assert result is not None
        assert call_count >= 1


class TestRouteFromCoordOverride:
    """_route_from_coord reads signals['human_phase_override'] as first check."""

    def _get_route_fn(self, phase: str = "verification") -> Any:
        """Build star and extract the _route_from_coord function via graph introspection.

        We test _route_from_coord by calling it directly on a state dict that
        has signals["human_phase_override"] set.
        """
        from atm.topology.star import StarTopology

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
        cfg = _make_cfg()
        topology = StarTopology()

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
        assert result == "planner"

    def test_no_override_signal_uses_normal_routing(self) -> None:
        """Without signals['human_phase_override'], routing is unchanged (planning → planner)."""
        route_fn = self._get_route_fn()
        state = _make_initial_state(
            signals={},
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
        assert result == "planner"


class TestStarRoleRouter:
    """Verify role_router kwarg in StarTopology.build() is forwarded to factory."""

    def test_build_accepts_role_router_none(self) -> None:
        """build(role_router=None) builds successfully without error (back-compat)."""
        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()

        human_cfg = _make_human_cfg(enabled=True)
        _build_star_graph(human_cfg=human_cfg, mock_graph=mock_graph, mock_gw_cls=mock_gw)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_reviewer" in node_names

    def test_build_with_role_router_none_passes_none_to_factory(self) -> None:
        """build(role_router=None) → build_human_node_factory called with role_router=None."""
        from atm.topology.star import StarTopology

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()
        mock_gw_cls = MagicMock(return_value=mock_gw)

        human_cfg = _make_human_cfg(enabled=True, role=HumanRole.REVIEWER)
        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
        cfg = _make_cfg()

        captured_factory_calls: list[dict[str, Any]] = []

        def capturing_factory(
            topology_name: str,
            human_cfg: Any,
            gateway: Any,
            *,
            request_id_template: str,
            question_extractor: Any,
            apply_decision: Any = None,
            role_router: Any = None,
        ) -> Any:
            captured_factory_calls.append({"role_router": role_router})

            async def noop_node(state: Any) -> dict[str, Any]:
                return {}

            return noop_node

        with (
            patch("atm.topology.star.StateGraph", return_value=mock_graph),
            patch("atm.topology.star.LLMSimulatedGateway", mock_gw_cls),
            patch("atm.topology.star.build_human_node_factory", side_effect=capturing_factory),
        ):
            StarTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert len(captured_factory_calls) == 1
        assert captured_factory_calls[0]["role_router"] is None

    def test_build_with_role_router_dynamic_passes_router_to_factory(self) -> None:
        """build(role_router=FixedRoleRouter(JUDGE)) → factory receives the router."""
        from atm.human.role_router import FixedRoleRouter
        from atm.topology.star import StarTopology

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()
        mock_gw_cls = MagicMock(return_value=mock_gw)

        human_cfg = _make_human_cfg(enabled=True, role=HumanRole.REVIEWER)
        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
        cfg = _make_cfg()
        router = FixedRoleRouter(role=HumanRole.JUDGE)

        captured_factory_calls: list[dict[str, Any]] = []

        def capturing_factory(
            topology_name: str,
            human_cfg: Any,
            gateway: Any,
            *,
            request_id_template: str,
            question_extractor: Any,
            apply_decision: Any = None,
            role_router: Any = None,
        ) -> Any:
            captured_factory_calls.append({"role_router": role_router})

            async def noop_node(state: Any) -> dict[str, Any]:
                return {}

            return noop_node

        with (
            patch("atm.topology.star.StateGraph", return_value=mock_graph),
            patch("atm.topology.star.LLMSimulatedGateway", mock_gw_cls),
            patch("atm.topology.star.build_human_node_factory", side_effect=capturing_factory),
        ):
            StarTopology().build(agents, cfg, human_cfg=human_cfg, role_router=router)

        assert len(captured_factory_calls) == 1
        assert captured_factory_calls[0]["role_router"] is router

    @pytest.mark.asyncio
    async def test_dynamic_role_router_overrides_cfg_role_in_node(self) -> None:
        """End-to-end: role_router=FixedRoleRouter(JUDGE) → HumanContext.role == JUDGE."""
        from atm.core.types import HumanContext
        from atm.human.role_router import FixedRoleRouter

        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True},
        )
        run_id = uuid.uuid4()
        initial = _make_initial_state(run_id=run_id)

        captured_contexts: list[HumanContext] = []

        async def capturing_gateway_request(ctx: Any, *, request_id: str) -> Any:
            captured_contexts.append(ctx)
            return _make_approve_response()

        fake_gateway = MagicMock()
        fake_gateway.request = capturing_gateway_request
        mock_gw_cls = MagicMock(return_value=fake_gateway)

        router = FixedRoleRouter(role=HumanRole.JUDGE)

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
            human_cfg = _make_human_cfg(enabled=True, role=HumanRole.REVIEWER)
            graph = _build_star_graph(agents=agents, cfg=cfg, human_cfg=human_cfg)
            from langgraph.checkpoint.memory import MemorySaver

            from atm.topology.star import StarTopology

            graph = StarTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                role_router=router,
                checkpointer=MemorySaver(),
            )

        result = await graph.ainvoke(
            initial,
            config={"configurable": {"thread_id": f"test-router-{uuid.uuid4()}"}},
        )

        assert result is not None
        assert len(captured_contexts) >= 1
        assert captured_contexts[0].role == HumanRole.JUDGE

    @pytest.mark.asyncio
    async def test_back_compat_role_router_none_uses_cfg_role(self) -> None:
        """role_router=None → HumanContext.role == human_cfg.role (REVIEWER)."""
        from atm.core.types import HumanContext

        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True},
        )
        run_id = uuid.uuid4()
        initial = _make_initial_state(run_id=run_id)

        captured_contexts: list[HumanContext] = []

        async def capturing_gateway_request(ctx: Any, *, request_id: str) -> Any:
            captured_contexts.append(ctx)
            return _make_approve_response()

        fake_gateway = MagicMock()
        fake_gateway.request = capturing_gateway_request
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
            human_cfg = _make_human_cfg(enabled=True, role=HumanRole.REVIEWER)
            from langgraph.checkpoint.memory import MemorySaver

            from atm.topology.star import StarTopology

            graph = StarTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                role_router=None,
                checkpointer=MemorySaver(),
            )

        result = await graph.ainvoke(
            initial,
            config={"configurable": {"thread_id": f"test-back-compat-{uuid.uuid4()}"}},
        )

        assert result is not None
        assert len(captured_contexts) >= 1
        assert captured_contexts[0].role == HumanRole.REVIEWER


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
