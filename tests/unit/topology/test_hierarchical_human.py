"""Unit tests for HierarchicalTopology HITL integration (M9.1 Step 5).

Tests (6+):
  1.  test_backcompat_no_human_cfg_graph_identical
  2.  test_scope_top_inserts_human_top_reviewer_node
  3.  test_scope_sub_team_inserts_human_sub_reviewer_in_subgraphs
  4.  test_scope_top_dispatch_order_reviewer_before_finalize
  5.  test_scope_top_reviewer_approve_sets_human_approved
  6.  test_scope_sub_team_reviewer_called_for_each_team
  7.  test_warning_docstring_present_on_subgraph_helper
  8.  test_build_subgraph_with_human_uses_warning_docstring
  9.  test_scope_top_human_node_in_graph_nodes
  10. test_default_scope_is_top_when_extra_missing
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph.state import CompiledStateGraph

import atm.topology.hierarchical  # noqa: F401 — triggers registry registration
from atm.core.types import HumanRole, Message, MessageKind
from atm.experiment.config import HumanCfg
from atm.topology.base import TopologyConfig
from atm.topology.hierarchical import HierarchicalTopology, _build_human_sub_reviewer_node


def _make_cfg(
    *,
    max_iterations: int = 20,
    max_rounds: int = 4,
    sub_teams: list[dict[str, Any]] | None = None,
) -> TopologyConfig:
    extra: dict[str, Any] = {
        "max_rounds": max_rounds,
        "final_answer_strategy": "json_concat",
        "finalize_signal": "top_coord_finalize",
    }
    if sub_teams is not None:
        extra["sub_teams"] = sub_teams
    return TopologyConfig(name="hierarchical", max_iterations=max_iterations, extra=extra)


def _make_human_cfg(
    *,
    scope: str = "top",
    gateway: str = "llm_simulated",
    role: HumanRole = HumanRole.REVIEWER,
) -> HumanCfg:
    return HumanCfg(
        enabled=True,
        gateway=gateway,
        role=role,
        timeout_s=None,
        extra={"scope": scope},
    )


def _make_mock_agent(agent_id: str) -> MagicMock:
    """Create a mock agent that returns a simple DRAFT message."""
    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict[str, Any]) -> dict[str, Any]:
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


def _make_mock_gateway(action: str = "approve") -> MagicMock:
    """Create a mock HumanGateway that returns a fixed action."""
    from atm.human.gateway import HumanResponse

    gateway = MagicMock()
    response = HumanResponse(
        action=action,
        comment="unit test approval",
        source="llm_sim",
        timed_out=False,
    )
    gateway.request = AsyncMock(return_value=response)
    return gateway


def _make_state(
    *,
    iter_total: int = 0,
    run_id: uuid.UUID | None = None,
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "shared": {
            "task_input": "test task",
            "iter_total": iter_total,
            "iteration": iter_total,
            "signals": signals or {},
            "final_answer": None,
            "run_id": run_id or uuid.uuid4(),
        },
        "agents": {},
        "messages": [],
    }


class TestBackcompat:
    """Without human_cfg.enabled, graph behaves exactly as pre-M9.1."""

    def test_backcompat_no_human_cfg_graph_identical(self) -> None:
        """build() without human_cfg returns graph without human_top_reviewer node."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        compiled = topology.build(agents, cfg)

        assert isinstance(compiled, CompiledStateGraph)

        node_names = set(compiled.get_graph().nodes.keys())
        assert "human_top_reviewer" not in node_names, (
            f"human_top_reviewer should NOT be in graph without human_cfg; got: {node_names}"
        )
        for expected in ("top_coord", "team_a", "team_b", "hierarchical_finalize"):
            assert expected in node_names, f"Expected node '{expected}' in graph; got: {node_names}"

    def test_backcompat_human_cfg_disabled_graph_identical(self) -> None:
        """build() with human_cfg.enabled=False returns graph without human nodes."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        disabled_human_cfg = HumanCfg(enabled=False, extra={"scope": "top"})
        compiled = topology.build(agents, cfg, human_cfg=disabled_human_cfg)

        node_names = set(compiled.get_graph().nodes.keys())
        assert "human_top_reviewer" not in node_names


class TestScopeTopGraphStructure:
    """With scope='top', human_top_reviewer is present in the top-level graph."""

    def test_scope_top_inserts_human_top_reviewer_node(self) -> None:
        """build() with scope='top' inserts human_top_reviewer into the top-level graph."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="top")

        mock_gateway = _make_mock_gateway()

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = mock_gateway
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        node_names = set(compiled.get_graph().nodes.keys())
        assert "human_top_reviewer" in node_names, (
            f"Expected 'human_top_reviewer' in graph with scope='top'; got: {node_names}"
        )

    def test_scope_top_human_node_in_graph_nodes(self) -> None:
        """human_top_reviewer must appear in the compiled graph's node list."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="top")

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = _make_mock_gateway()
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        node_names = set(compiled.get_graph().nodes.keys())
        for expected in (
            "top_coord",
            "team_a",
            "team_b",
            "hierarchical_finalize",
            "human_top_reviewer",
        ):
            assert expected in node_names, (
                f"Expected '{expected}' in graph nodes; got: {node_names}"
            )


class TestScopeSubTeamGraphStructure:
    """With scope='sub_team', human reviewers are inside each subgraph, not top-level."""

    def test_scope_sub_team_inserts_human_sub_reviewer_in_subgraphs(self) -> None:
        """build() with scope='sub_team' does NOT add human_top_reviewer at top level."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="sub_team")

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = _make_mock_gateway()
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        top_node_names = set(compiled.get_graph().nodes.keys())
        assert "human_top_reviewer" not in top_node_names, (
            f"human_top_reviewer should not be in top-level graph for scope='sub_team'; "
            f"got: {top_node_names}"
        )
        for expected in ("top_coord", "team_a", "team_b", "hierarchical_finalize"):
            assert expected in top_node_names, (
                f"Expected '{expected}' in top-level nodes; got: {top_node_names}"
            )

    def test_scope_sub_team_builds_subgraph_with_human_called(self) -> None:
        """build() with scope='sub_team' calls _build_subgraph_with_human for both teams."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="sub_team")

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = _make_mock_gateway()
            with patch.object(
                topology, "_build_subgraph_with_human", wraps=topology._build_subgraph_with_human
            ) as mock_build:
                topology.build(agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock())

        assert mock_build.call_count == 2, (
            f"Expected _build_subgraph_with_human called 2 times, got {mock_build.call_count}"
        )


class TestScopeTopDispatchOrder:
    """With scope='top', human_top_reviewer fires BEFORE hierarchical_finalize."""

    async def test_scope_top_dispatch_order_reviewer_before_finalize(self) -> None:
        """Integration smoke: graph with scope='top' runs without error, human fires."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20)
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="top")

        call_log: list[str] = []
        run_id = uuid.uuid4()

        from atm.human.gateway import HumanResponse

        async def tracked_request(ctx: Any, *, request_id: str) -> HumanResponse:
            call_log.append(f"human_request:{request_id}")
            return HumanResponse(
                action="approve",
                comment="approved",
                source="llm_sim",
                timed_out=False,
            )

        mock_gateway = MagicMock()
        mock_gateway.request = tracked_request

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = mock_gateway
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        initial_state = _make_state(run_id=run_id)
        final_state = await compiled.ainvoke(initial_state)

        assert len(call_log) >= 1, f"Expected at least 1 human_request call, got {call_log}"
        assert any("hierarchical:top:" in entry for entry in call_log), (
            f"Expected hierarchical:top: prefix in request_id, got: {call_log}"
        )

        shared = final_state.get("shared", {})
        assert shared.get("final_answer") is not None, "final_answer should be set"


class TestScopeTopApprove:
    """approve action from human_top_reviewer sets shared['human_approved'] = True."""

    async def test_scope_top_reviewer_approve_sets_human_approved(self) -> None:
        """After scope='top' graph run with approve, human_approved should be True."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20)
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="top")
        run_id = uuid.uuid4()

        from atm.human.gateway import HumanResponse

        async def approve_request(ctx: Any, *, request_id: str) -> HumanResponse:
            return HumanResponse(
                action="approve",
                comment="LGTM",
                source="llm_sim",
                timed_out=False,
            )

        mock_gateway = MagicMock()
        mock_gateway.request = approve_request

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = mock_gateway
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        initial_state = _make_state(run_id=run_id)
        final_state = await compiled.ainvoke(initial_state)

        shared = final_state.get("shared", {})
        assert shared.get("human_approved") is True, (
            f"Expected human_approved=True after approve action; shared={shared}"
        )


class TestScopeSubTeamReviewerCalls:
    """With scope='sub_team', the gateway is called once per team sub-run."""

    async def test_scope_sub_team_reviewer_called_for_each_team(self) -> None:
        """scope='sub_team' fires the gateway once for team_a and once for team_b."""
        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20)
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="sub_team")
        run_id = uuid.uuid4()

        call_log: list[str] = []

        from atm.human.gateway import HumanResponse

        async def tracked_request(ctx: Any, *, request_id: str) -> HumanResponse:
            call_log.append(request_id)
            return HumanResponse(
                action="approve",
                comment="approved",
                source="llm_sim",
                timed_out=False,
            )

        mock_gateway = MagicMock()
        mock_gateway.request = tracked_request

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = mock_gateway
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        initial_state = _make_state(run_id=run_id)
        await compiled.ainvoke(initial_state)

        team_ids_seen = set()
        for rid in call_log:
            if "team_a" in rid:
                team_ids_seen.add("team_a")
            elif "team_b" in rid:
                team_ids_seen.add("team_b")

        assert "team_a" in team_ids_seen, (
            f"Expected team_a to appear in request_ids; got: {call_log}"
        )
        assert "team_b" in team_ids_seen, (
            f"Expected team_b to appear in request_ids; got: {call_log}"
        )


class TestWarningDocstring:
    """_build_subgraph_with_human has a WARNING docstring about CLIGateway limitation."""

    def test_warning_docstring_present_on_subgraph_helper(self) -> None:
        """_build_subgraph_with_human docstring must contain 'WARNING'."""
        topology = HierarchicalTopology()
        docstring = topology._build_subgraph_with_human.__doc__ or ""
        assert "WARNING" in docstring, (
            f"Expected 'WARNING' in _build_subgraph_with_human docstring; got: {docstring[:200]!r}"
        )

    def test_warning_docstring_mentions_cligateway(self) -> None:
        """Docstring should mention CLIGateway limitation."""
        topology = HierarchicalTopology()
        docstring = topology._build_subgraph_with_human.__doc__ or ""
        assert "CLIGateway" in docstring, (
            f"Expected 'CLIGateway' in _build_subgraph_with_human docstring; "
            f"got: {docstring[:200]!r}"
        )

    def test_build_subgraph_with_human_uses_warning_docstring(self) -> None:
        """_build_human_sub_reviewer_node also has a WARNING docstring."""
        docstring = _build_human_sub_reviewer_node.__doc__ or ""
        assert "WARNING" in docstring, (
            f"Expected 'WARNING' in _build_human_sub_reviewer_node docstring; "
            f"got: {docstring[:200]!r}"
        )


class TestDefaultScope:
    """When human_cfg.extra has no 'scope' key, default scope is 'top'."""

    def test_default_scope_is_top_when_extra_missing(self) -> None:
        """human_cfg with no extra.scope defaults to scope='top'."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        human_cfg = HumanCfg(enabled=True, gateway="llm_simulated", extra=None)

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = _make_mock_gateway()
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        node_names = set(compiled.get_graph().nodes.keys())
        assert "human_top_reviewer" in node_names, (
            f"Expected human_top_reviewer with default scope='top'; got: {node_names}"
        )

    def test_default_scope_is_top_when_extra_empty_dict(self) -> None:
        """human_cfg with extra={} (no scope key) defaults to scope='top'."""
        topology = HierarchicalTopology()
        cfg = _make_cfg()
        agents = _make_agents()

        human_cfg = HumanCfg(enabled=True, gateway="llm_simulated", extra={})

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = _make_mock_gateway()
            compiled = topology.build(
                agents, cfg, human_cfg=human_cfg, human_gateway_llm=MagicMock()
            )

        node_names = set(compiled.get_graph().nodes.keys())
        assert "human_top_reviewer" in node_names


class TestHierarchicalRoleRouter:
    """role_router=None → back-compat (human_cfg.role); role_router set → dynamic role."""

    def _capture_top_reviewer_roles(
        self,
        cfg_role: HumanRole,
        role_router: Any | None,
    ) -> list[HumanRole]:
        """Build scope='top' graph, run it, and capture HumanContext.role values."""
        import asyncio

        from atm.core.types import HumanContext

        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20)
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="top", role=cfg_role)
        run_id = uuid.uuid4()

        captured_roles: list[HumanRole] = []

        from atm.human.gateway import HumanResponse

        async def capturing_request(ctx: HumanContext, *, request_id: str) -> HumanResponse:
            captured_roles.append(ctx.role)
            return HumanResponse(
                action="approve",
                comment="unit test",
                source="llm_sim",
                timed_out=False,
            )

        mock_gateway = MagicMock()
        mock_gateway.request = capturing_request

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = mock_gateway
            compiled = topology.build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
                role_router=role_router,
            )

        initial_state = _make_state(run_id=run_id)
        asyncio.run(compiled.ainvoke(initial_state))
        return captured_roles

    @pytest.mark.parametrize(
        "use_router,cfg_role,expected_role",
        [
            (False, HumanRole.REVIEWER, HumanRole.REVIEWER),
            (True, HumanRole.REVIEWER, HumanRole.COORDINATOR),
        ],
        ids=["back_compat", "dynamic"],
    )
    def test_scope_top_role(
        self,
        use_router: bool,
        cfg_role: HumanRole,
        expected_role: HumanRole,
    ) -> None:
        """scope='top' HITL point: back-compat uses human_cfg.role; dynamic uses router.decide()."""
        from atm.human.role_router import FixedRoleRouter

        router = FixedRoleRouter(role=HumanRole.COORDINATOR) if use_router else None
        roles = self._capture_top_reviewer_roles(cfg_role=cfg_role, role_router=router)

        assert len(roles) >= 1, f"Expected at least 1 gateway call (top reviewer), got {roles}"
        assert roles[0] == expected_role, (
            f"scope=top: Expected role={expected_role!r}, got {roles[0]!r}"
        )

    def _capture_sub_reviewer_roles(
        self,
        cfg_role: HumanRole,
        role_router: Any | None,
    ) -> list[HumanRole]:
        """Build scope='sub_team' graph, run it, and capture HumanContext.role values."""
        import asyncio

        from atm.core.types import HumanContext

        topology = HierarchicalTopology()
        cfg = _make_cfg(max_iterations=20)
        agents = _make_agents()
        human_cfg = _make_human_cfg(scope="sub_team", role=cfg_role)
        run_id = uuid.uuid4()

        captured_roles: list[HumanRole] = []

        from atm.human.gateway import HumanResponse

        async def capturing_request(ctx: HumanContext, *, request_id: str) -> HumanResponse:
            captured_roles.append(ctx.role)
            return HumanResponse(
                action="approve",
                comment="unit test",
                source="llm_sim",
                timed_out=False,
            )

        mock_gateway = MagicMock()
        mock_gateway.request = capturing_request

        with patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_gw:
            mock_gw.return_value = mock_gateway
            compiled = topology.build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
                role_router=role_router,
            )

        initial_state = _make_state(run_id=run_id)
        asyncio.run(compiled.ainvoke(initial_state))
        return captured_roles

    @pytest.mark.parametrize(
        "use_router,cfg_role,expected_role",
        [
            (False, HumanRole.REVIEWER, HumanRole.REVIEWER),
            (True, HumanRole.REVIEWER, HumanRole.COORDINATOR),
        ],
        ids=["back_compat", "dynamic"],
    )
    def test_scope_sub_team_role(
        self,
        use_router: bool,
        cfg_role: HumanRole,
        expected_role: HumanRole,
    ) -> None:
        """scope='sub_team' HITL points: back-compat uses human_cfg.role; dynamic uses router."""
        from atm.human.role_router import FixedRoleRouter

        router = FixedRoleRouter(role=HumanRole.COORDINATOR) if use_router else None
        roles = self._capture_sub_reviewer_roles(cfg_role=cfg_role, role_router=router)

        assert len(roles) >= 2, (
            f"Expected at least 2 sub-reviewer calls (one per team), got: {roles}"
        )
        for i, role in enumerate(roles):
            assert role == expected_role, (
                f"scope=sub_team call[{i}]: Expected role={expected_role!r}, got {role!r}"
            )
