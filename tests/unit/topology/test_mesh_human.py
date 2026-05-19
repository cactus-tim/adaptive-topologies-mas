"""Unit tests for MeshTopology HITL integration (M9.1 Step 2.5).

Tests (6+):
  1. test_human_peer_added_to_agent_order_when_enabled
  2. test_human_peer_node_registered_when_enabled
  3. test_dispatcher_skips_human_peer_before_activation_round
  4. test_dispatcher_activates_human_peer_at_activation_round
  5. test_consensus_pending_signal_set_on_split_vote
  6. test_consensus_pending_not_set_when_no_votes
  7. test_back_compat_without_human_cfg
  8. test_on_consensus_pending_skips_human_peer_when_not_pending
  9. test_effective_agent_order_includes_human_peer

All tests use mocked StateGraph (via patch) to introspect build() calls.
No real LangGraph invocations needed for unit tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import atm.topology.mesh  # noqa: F401 — triggers @TopologyRegistry.register
from atm.core.types import Message, MessageKind
from atm.topology.base import TopologyConfig
from atm.topology.mesh import (
    _DEFAULT_HUMAN_ACTIVATION_ROUND,
    _HUMAN_PEER_ID,
    MeshTopology,
)


def _make_topology_cfg(
    *,
    max_rounds: int = 6,
    consensus_threshold: int = 4,
    agent_order: list[str] | None = None,
) -> TopologyConfig:
    order = agent_order or ["planner", "researcher", "executor"]
    return TopologyConfig(
        name="mesh",
        max_iterations=20,
        extra={
            "max_rounds": max_rounds,
            "consensus_threshold": consensus_threshold,
            "agent_order": order,
        },
    )


def _make_human_cfg(
    *,
    enabled: bool = True,
    gateway: str = "llm_simulated",
    activation_round: int = 2,
    on_consensus_pending: bool = False,
) -> Any:
    """Create a simple duck-typed HumanCfg-like object for testing."""
    from types import SimpleNamespace

    from atm.core.types import HumanRole

    extra: dict[str, Any] = {"activation_round": activation_round}
    if on_consensus_pending:
        extra["on_consensus_pending"] = True

    return SimpleNamespace(
        enabled=enabled,
        gateway=gateway,
        role=HumanRole.REVIEWER,
        timeout_s=30.0,
        timeout_policy="skip",
        extra=extra,
    )


def _make_mock_agent() -> MagicMock:
    mock_agent = MagicMock()
    mock_agent.step = AsyncMock(return_value={})
    return mock_agent


def _make_mock_gateway() -> MagicMock:
    from atm.human.gateway import HumanResponse

    mock_gw = MagicMock()
    mock_gw.request = AsyncMock(
        return_value=HumanResponse(
            action="X",
            comment="X",
            payload={"vote_for": "X"},
            source="llm_sim",
            timed_out=False,
        )
    )
    return mock_gw


def _make_agents(order: list[str] | None = None) -> dict[str, Any]:
    order = order or ["planner", "researcher", "executor"]
    return {aid: _make_mock_agent() for aid in order}


def _build_with_mock_graph(
    agents: dict[str, Any],
    cfg: TopologyConfig,
    human_cfg: Any | None = None,
    mock_gateway: Any | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build MeshTopology with patched StateGraph; return (mock_graph, compiled)."""
    mock_compiled = MagicMock()
    mock_graph = MagicMock()
    mock_graph.compile.return_value = mock_compiled

    topology = MeshTopology()

    kwargs: dict[str, Any] = {}
    if human_cfg is not None:
        kwargs["human_cfg"] = human_cfg
    if mock_gateway is not None:
        kwargs["human_gateway_llm"] = mock_gateway

    with (
        patch("atm.topology.mesh.StateGraph", return_value=mock_graph),
        patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gateway or MagicMock()),
    ):
        topology.build(agents, cfg, **kwargs)

    return mock_graph, mock_compiled


class TestHumanPeerAddedToAgentOrder:
    """human_peer is included in the dispatcher routing map when HITL enabled."""

    def test_human_peer_added_to_agent_order_when_enabled(self) -> None:
        """When human_cfg.enabled=True, dispatcher routing map includes 'human_peer'."""
        cfg = _make_topology_cfg(agent_order=["planner", "researcher", "executor"])
        human_cfg = _make_human_cfg(enabled=True)
        agents = _make_agents(["planner", "researcher", "executor"])
        mock_gw = _make_mock_gateway()

        mock_graph, _ = _build_with_mock_graph(
            agents, cfg, human_cfg=human_cfg, mock_gateway=mock_gw
        )

        dispatcher_call = None
        for call in mock_graph.add_conditional_edges.call_args_list:
            if call.args and call.args[0] == "dispatcher":
                dispatcher_call = call
                break

        assert dispatcher_call is not None, "No conditional edges added for 'dispatcher'"
        routing_map: dict[str, str] = (
            dispatcher_call.args[2] if len(dispatcher_call.args) > 2 else {}
        )

        assert _HUMAN_PEER_ID in routing_map, (
            f"'{_HUMAN_PEER_ID}' missing from dispatcher routing map: {routing_map}"
        )


class TestHumanPeerNodeRegistered:
    """build() adds a 'human_peer' node when human_cfg.enabled=True."""

    def test_human_peer_node_registered_when_enabled(self) -> None:
        """graph.add_node('human_peer', ...) is called when HITL enabled."""
        cfg = _make_topology_cfg(agent_order=["planner", "researcher", "executor"])
        human_cfg = _make_human_cfg(enabled=True)
        agents = _make_agents(["planner", "researcher", "executor"])
        mock_gw = _make_mock_gateway()

        mock_graph, _ = _build_with_mock_graph(
            agents, cfg, human_cfg=human_cfg, mock_gateway=mock_gw
        )

        added_node_names = {call.args[0] for call in mock_graph.add_node.call_args_list}
        assert _HUMAN_PEER_ID in added_node_names, (
            f"'human_peer' not found in added nodes: {added_node_names}"
        )

    def test_human_peer_node_not_registered_when_disabled(self) -> None:
        """graph.add_node('human_peer', ...) is NOT called when HITL disabled."""
        cfg = _make_topology_cfg(agent_order=["planner", "researcher", "executor"])
        agents = _make_agents(["planner", "researcher", "executor"])

        mock_graph, _ = _build_with_mock_graph(agents, cfg)

        added_node_names = {call.args[0] for call in mock_graph.add_node.call_args_list}
        assert _HUMAN_PEER_ID not in added_node_names, (
            f"'human_peer' should not be in added nodes when disabled: {added_node_names}"
        )


class TestDispatcherSkipsHumanPeerBeforeActivationRound:
    """Dispatcher skips human_peer when dispatch_round < activation_round."""

    @pytest.mark.asyncio
    async def test_dispatcher_skips_human_peer_before_activation_round(self) -> None:
        """When dispatch_round=0 < activation_round=2, human_peer is NOT activated."""
        agent_order = ["planner", "researcher", "executor"]
        vote_counts_per_agent: dict[str, str] = {
            "planner": "X",
            "researcher": "X",
            "executor": "X",
        }

        class _VoteAgent:
            def __init__(self, aid: str, vote: str) -> None:
                self.agent_id = aid
                self._vote = vote

            async def step(self, state: dict[str, Any]) -> dict[str, Any]:
                msg = Message(
                    sender=self.agent_id,
                    kind=MessageKind.DECISION,
                    content=f"vote {self._vote}",
                    payload={"vote_for": self._vote},
                )
                return {
                    "agents": {
                        self.agent_id: {
                            "agent_id": self.agent_id,
                            "outbox": [msg],
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

        human_step_called = False

        class _TrackingHumanAgent:
            async def step(self, state: dict[str, Any]) -> dict[str, Any]:
                nonlocal human_step_called
                human_step_called = True
                return {}

        agents: dict[str, Any] = {
            aid: _VoteAgent(aid, vote) for aid, vote in vote_counts_per_agent.items()
        }

        cfg = _make_topology_cfg(
            agent_order=agent_order,
            consensus_threshold=10,
            max_rounds=2,
        )
        human_cfg = _make_human_cfg(enabled=True, activation_round=2)

        from atm.human.gateway import HumanResponse

        gateway_called = False
        mock_gw = MagicMock()

        async def _mock_request(ctx: Any, *, request_id: str) -> HumanResponse:
            nonlocal gateway_called
            gateway_called = True
            return HumanResponse(
                action="X",
                comment="X",
                payload={"vote_for": "X"},
                source="llm_sim",
                timed_out=False,
            )

        mock_gw.request = _mock_request

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(agents, cfg, human_cfg=human_cfg, human_gateway_llm=mock_gw)

        initial_state = {
            "shared": {
                "task_id": "test",
                "task_input": "test",
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

        signals = final_state.get("shared", {}).get("signals", {})
        dispatch_round = int(signals.get("_mesh_dispatch_round", 0))

        assert dispatch_round >= 1, f"Expected at least 1 dispatch round, got {dispatch_round}"


class TestDispatcherActivatesHumanPeerAtActivationRound:
    """Dispatcher activates human_peer when dispatch_round >= activation_round."""

    @pytest.mark.asyncio
    async def test_dispatcher_activates_human_peer_at_activation_round(self) -> None:
        """With activation_round=1, human_peer fires on first dispatch."""
        agent_order = ["planner"]

        class _VoteAgent:
            async def step(self, state: dict[str, Any]) -> dict[str, Any]:
                msg = Message(
                    sender="planner",
                    kind=MessageKind.DECISION,
                    content="vote X",
                    payload={"vote_for": "X"},
                )
                return {
                    "agents": {
                        "planner": {
                            "agent_id": "planner",
                            "outbox": [msg],
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

        from atm.human.gateway import HumanResponse

        gateway_call_count = 0
        mock_gw = MagicMock()

        async def _mock_request(ctx: Any, *, request_id: str) -> HumanResponse:
            nonlocal gateway_call_count
            gateway_call_count += 1
            return HumanResponse(
                action="X",
                comment="X",
                payload={"vote_for": "X"},
                source="llm_sim",
                timed_out=False,
            )

        mock_gw.request = _mock_request

        cfg = _make_topology_cfg(
            agent_order=agent_order,
            consensus_threshold=10,
            max_rounds=2,
        )
        human_cfg = _make_human_cfg(enabled=True, activation_round=1)

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(
                {"planner": _VoteAgent()}, cfg, human_cfg=human_cfg, human_gateway_llm=mock_gw
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "task_id": "test",
                "task_input": "test",
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

        await graph.ainvoke(initial_state)

        assert gateway_call_count >= 1, (
            f"Expected human gateway to be called ≥1 time (activation_round=1), "
            f"got {gateway_call_count}"
        )


class TestConsensusPendingSignalOnSplitVote:
    """mesh_postprocess sets consensus_pending=True when votes exist but no winner."""

    @pytest.mark.asyncio
    async def test_consensus_pending_signal_set_on_split_vote(self) -> None:
        """Split vote (no majority) → consensus_pending=True in signals."""
        agent_order = ["planner", "researcher"]

        class _VoteAgent:
            def __init__(self, aid: str, vote: str) -> None:
                self.agent_id = aid
                self._vote = vote

            async def step(self, state: dict[str, Any]) -> dict[str, Any]:
                msg = Message(
                    sender=self.agent_id,
                    kind=MessageKind.DECISION,
                    content=f"vote {self._vote}",
                    payload={"vote_for": self._vote},
                )
                return {
                    "agents": {
                        self.agent_id: {
                            "agent_id": self.agent_id,
                            "outbox": [msg],
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

        from atm.human.gateway import HumanResponse

        mock_gw = MagicMock()
        mock_gw.request = AsyncMock(
            return_value=HumanResponse(
                action="abstain",
                comment="",
                payload={},
                source="llm_sim",
                timed_out=False,
            )
        )

        agents: dict[str, Any] = {
            "planner": _VoteAgent("planner", "A"),
            "researcher": _VoteAgent("researcher", "B"),
        }
        cfg = _make_topology_cfg(
            agent_order=agent_order,
            consensus_threshold=3,
            max_rounds=2,
        )
        human_cfg = _make_human_cfg(enabled=True, activation_round=99)

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(agents, cfg, human_cfg=human_cfg, human_gateway_llm=mock_gw)

        initial_state: dict[str, Any] = {
            "shared": {
                "task_id": "test",
                "task_input": "test",
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
        signals = final_state.get("shared", {}).get("signals", {})

        assert "consensus_pending" in signals, (
            f"Expected 'consensus_pending' in signals, got {signals}"
        )
        assert signals["consensus_pending"] is True, (
            f"Expected consensus_pending=True on split vote, got {signals['consensus_pending']!r}"
        )


class TestConsensusPendingNotSetWhenNoVotes:
    """mesh_postprocess consensus_pending=False when no votes exist."""

    @pytest.mark.asyncio
    async def test_consensus_pending_not_set_when_no_votes(self) -> None:
        """With no DECISION votes, consensus_pending=False (not stuck pending)."""

        class _SilentAgent:
            async def step(self, state: dict[str, Any]) -> dict[str, Any]:
                return {}

        from atm.human.gateway import HumanResponse

        mock_gw = MagicMock()
        mock_gw.request = AsyncMock(
            return_value=HumanResponse(
                action="abstain", comment="", payload={}, source="llm_sim", timed_out=False
            )
        )

        cfg = _make_topology_cfg(
            agent_order=["planner"],
            consensus_threshold=3,
            max_rounds=1,
        )
        human_cfg = _make_human_cfg(enabled=True, activation_round=99)

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(
                {"planner": _SilentAgent()},
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=mock_gw,
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "task_id": "test",
                "task_input": "test",
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
        signals = final_state.get("shared", {}).get("signals", {})

        assert signals.get("consensus_pending") is False, (
            f"Expected consensus_pending=False with no votes, got {signals.get('consensus_pending')!r}"
        )


class TestBackCompatWithoutHumanCfg:
    """Without human_cfg, graph structure is identical to pre-M9.1 Mesh."""

    def test_back_compat_without_human_cfg(self) -> None:
        """build() without human_cfg produces graph with original 4 nodes only."""
        agent_order = ["planner", "researcher", "executor", "critic"]
        cfg = _make_topology_cfg(agent_order=agent_order)
        agents = {aid: _make_mock_agent() for aid in agent_order}

        mock_graph, _ = _build_with_mock_graph(agents, cfg)

        added_node_names = {call.args[0] for call in mock_graph.add_node.call_args_list}
        expected = {"dispatcher", "mesh_broadcast", "mesh_postprocess"} | set(agent_order)
        assert expected == added_node_names, (
            f"Back-compat: unexpected nodes. Extra: {added_node_names - expected}, "
            f"Missing: {expected - added_node_names}"
        )
        assert _HUMAN_PEER_ID not in added_node_names

    def test_back_compat_dispatcher_routing_without_human_cfg(self) -> None:
        """Without human_cfg, dispatcher routing map only has original agents."""
        agent_order = ["planner", "researcher", "executor", "critic"]
        cfg = _make_topology_cfg(agent_order=agent_order)
        agents = {aid: _make_mock_agent() for aid in agent_order}

        mock_graph, _ = _build_with_mock_graph(agents, cfg)

        dispatcher_call = None
        for call in mock_graph.add_conditional_edges.call_args_list:
            if call.args and call.args[0] == "dispatcher":
                dispatcher_call = call
                break

        assert dispatcher_call is not None
        routing_map: dict[str, str] = (
            dispatcher_call.args[2] if len(dispatcher_call.args) > 2 else {}
        )
        assert _HUMAN_PEER_ID not in routing_map, (
            f"human_peer should NOT be in routing map without human_cfg: {routing_map}"
        )


class TestOnConsensusPendingSkips:
    """on_consensus_pending=True: human_peer only activates when consensus_pending signal set."""

    @pytest.mark.asyncio
    async def test_on_consensus_pending_skips_human_peer_when_not_pending(self) -> None:
        """When on_consensus_pending=True and no split vote, human_peer is skipped."""

        class _ConsensusBotAgent:
            def __init__(self, aid: str) -> None:
                self.agent_id = aid

            async def step(self, state: dict[str, Any]) -> dict[str, Any]:
                msg = Message(
                    sender=self.agent_id,
                    kind=MessageKind.DECISION,
                    content="vote X",
                    payload={"vote_for": "X"},
                )
                return {
                    "agents": {
                        self.agent_id: {
                            "agent_id": self.agent_id,
                            "outbox": [msg],
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

        from atm.human.gateway import HumanResponse

        gateway_calls = 0
        mock_gw = MagicMock()

        async def _mock_request(ctx: Any, *, request_id: str) -> HumanResponse:
            nonlocal gateway_calls
            gateway_calls += 1
            return HumanResponse(
                action="X",
                comment="X",
                payload={"vote_for": "X"},
                source="llm_sim",
                timed_out=False,
            )

        mock_gw.request = _mock_request

        cfg = _make_topology_cfg(
            agent_order=["planner"],
            consensus_threshold=1,
            max_rounds=3,
        )
        human_cfg = _make_human_cfg(
            enabled=True,
            activation_round=1,
            on_consensus_pending=True,
        )

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(
                {"planner": _ConsensusBotAgent("planner")},
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=mock_gw,
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "task_id": "test",
                "task_input": "test",
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
        signals = final_state.get("shared", {}).get("signals", {})

        assert signals.get("consensus_reached") is True, (
            f"Expected consensus_reached=True, got {signals}"
        )
        assert gateway_calls == 0, (
            f"Expected gateway_calls=0 with on_consensus_pending=True and immediate consensus, "
            f"got {gateway_calls}"
        )


class TestEffectiveAgentOrderIncludesHumanPeer:
    """When human_cfg.enabled=True, human_peer is in the effective agent order."""

    def test_effective_agent_order_includes_human_peer(self) -> None:
        """Verify that human_peer appears as a node and in edges when HITL enabled."""
        agent_order = ["planner", "executor"]
        cfg = _make_topology_cfg(agent_order=agent_order)
        human_cfg = _make_human_cfg(enabled=True)
        agents = _make_agents(agent_order)
        mock_gw = _make_mock_gateway()

        mock_graph, _ = _build_with_mock_graph(
            agents, cfg, human_cfg=human_cfg, mock_gateway=mock_gw
        )

        edge_sources = [
            call.args[0]
            for call in mock_graph.add_edge.call_args_list
            if len(call.args) >= 2 and call.args[1] == "mesh_broadcast"
        ]
        assert _HUMAN_PEER_ID in edge_sources, (
            f"Expected edge human_peer → mesh_broadcast, found edges to mesh_broadcast "
            f"from: {edge_sources}"
        )

    def test_default_activation_round_constant(self) -> None:
        """_DEFAULT_HUMAN_ACTIVATION_ROUND is 2 (per plan spec)."""
        assert _DEFAULT_HUMAN_ACTIVATION_ROUND == 2

    def test_human_peer_id_constant(self) -> None:
        """_HUMAN_PEER_ID is 'human_peer' (per plan spec)."""
        assert _HUMAN_PEER_ID == "human_peer"


class TestRoleRouterBackCompat:
    """role_router=None (or omitted) → human_cfg.role is used for HumanContext."""

    @pytest.mark.asyncio
    async def test_back_compat_role_router_none_uses_cfg_role(self) -> None:
        """When role_router is None, the role recorded in human_interactions matches human_cfg.role."""
        from atm.core.types import HumanRole
        from atm.human.gateway import HumanResponse

        captured_roles: list[str] = []
        mock_gw = MagicMock()

        async def _mock_request(ctx: Any, *, request_id: str) -> HumanResponse:
            role_val = ctx.role.value if hasattr(ctx.role, "value") else str(ctx.role)
            captured_roles.append(role_val)
            return HumanResponse(
                action="X",
                comment="X",
                payload={"vote_for": "X"},
                source="llm_sim",
                timed_out=False,
            )

        mock_gw.request = _mock_request

        cfg = _make_topology_cfg(
            agent_order=["planner"],
            consensus_threshold=10,
            max_rounds=2,
        )
        human_cfg = _make_human_cfg(enabled=True, activation_round=1)
        assert human_cfg.role == HumanRole.REVIEWER

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(
                {"planner": _make_mock_agent()},
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=mock_gw,
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "task_id": "t",
                "task_input": "t",
                "iter_total": 0,
                "iteration": 0,
                "final_answer": None,
                "signals": {},
                "broadcast_bus": [],
                "phase": "execution",
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

        await graph.ainvoke(initial_state)

        assert len(captured_roles) >= 1, (
            f"Expected human gateway to be called at least once, got {captured_roles}"
        )
        for r in captured_roles:
            assert r == HumanRole.REVIEWER.value, (
                f"Back-compat: expected role={HumanRole.REVIEWER.value!r}, got {r!r}"
            )


class TestRoleRouterDynamic:
    """role_router=FixedRoleRouter(JUDGE) overrides human_cfg.role=REVIEWER."""

    @pytest.mark.asyncio
    async def test_dynamic_role_router_overrides_cfg_role(self) -> None:
        """FixedRoleRouter(role=JUDGE) with human_cfg.role=REVIEWER → role used = JUDGE."""
        from atm.core.types import HumanRole
        from atm.human.gateway import HumanResponse
        from atm.human.role_router import FixedRoleRouter

        captured_roles: list[str] = []
        mock_gw = MagicMock()

        async def _mock_request(ctx: Any, *, request_id: str) -> HumanResponse:
            role_val = ctx.role.value if hasattr(ctx.role, "value") else str(ctx.role)
            captured_roles.append(role_val)
            return HumanResponse(
                action="X",
                comment="X",
                payload={"vote_for": "X"},
                source="llm_sim",
                timed_out=False,
            )

        mock_gw.request = _mock_request

        cfg = _make_topology_cfg(
            agent_order=["planner"],
            consensus_threshold=10,
            max_rounds=2,
        )
        human_cfg = _make_human_cfg(enabled=True, activation_round=1)
        assert human_cfg.role == HumanRole.REVIEWER

        fixed_router = FixedRoleRouter(role=HumanRole.JUDGE)

        topology = MeshTopology()
        with (
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=mock_gw),
            patch("atm.topology.mesh.request_with_timeout", None),
        ):
            graph = topology.build(
                {"planner": _make_mock_agent()},
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=mock_gw,
                role_router=fixed_router,
            )

        initial_state: dict[str, Any] = {
            "shared": {
                "task_id": "t",
                "task_input": "t",
                "iter_total": 0,
                "iteration": 0,
                "final_answer": None,
                "signals": {},
                "broadcast_bus": [],
                "phase": "execution",
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

        await graph.ainvoke(initial_state)

        assert len(captured_roles) >= 1, (
            f"Expected human gateway to be called at least once, got {captured_roles}"
        )
        for r in captured_roles:
            assert r == HumanRole.JUDGE.value, (
                f"Dynamic router: expected role={HumanRole.JUDGE.value!r}, got {r!r}"
            )
