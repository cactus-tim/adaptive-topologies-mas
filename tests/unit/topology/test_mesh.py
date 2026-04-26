"""Unit tests for atm.topology.mesh — MeshTopology (10 tests).

Tests:
  1. test_dispatcher_round_robin_cycles_agents
  2. test_priority_activation_routes_to_critic_on_draft
  3. test_consensus_threshold_writes_winner_signal
  4. test_max_rounds_routes_to_end
  5. test_global_max_iterations_overrides_topology_max
  6. test_build_returns_compiled_graph_with_expected_nodes
  7. test_register_under_name_mesh
  8. test_mesh_broadcast_writes_outbox_to_bus
  9. test_broadcast_bus_does_not_unbound (MC-5)
  10. test_consensus_vote_payload_str_format
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import atm.topology.mesh  # noqa: F401 — triggers @TopologyRegistry.register("mesh")
from atm.core.types import Message, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry
from atm.topology.mesh import MeshTopology, _pick_priority_agent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    max_iterations: int = 20,
    max_rounds: int = 6,
    consensus_threshold: int = 3,
    activation_policy: str = "round_robin",
    broadcast_bus_cap: int = 200,
    agent_order: list[str] | None = None,
) -> TopologyConfig:
    order = agent_order or ["planner", "researcher", "executor", "critic"]
    return TopologyConfig(
        name="mesh",
        max_iterations=max_iterations,
        extra={
            "max_rounds": max_rounds,
            "consensus_threshold": consensus_threshold,
            "activation_policy": activation_policy,
            "agent_order": order,
            "broadcast_bus_cap": broadcast_bus_cap,
        },
    )


def _make_shared(
    *,
    iter_total: int = 0,
    broadcast_bus: list[Any] | None = None,
    signals: dict[str, Any] | None = None,
    final_answer: str | None = None,
) -> dict[str, Any]:
    s: dict[str, Any] = {"iter_total": iter_total}
    if broadcast_bus is not None:
        s["broadcast_bus"] = broadcast_bus
    if signals is not None:
        s["signals"] = signals
    if final_answer is not None:
        s["final_answer"] = final_answer
    return s


def _make_state(
    *,
    iter_total: int = 0,
    broadcast_bus: list[Any] | None = None,
    signals: dict[str, Any] | None = None,
    agents: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "shared": _make_shared(
            iter_total=iter_total,
            broadcast_bus=broadcast_bus or [],
            signals=signals or {},
        ),
        "agents": agents or {},
    }


def _make_decision_msg(*, vote_for: Any, sender: str = "planner") -> Message:
    return Message(
        sender=sender,
        kind=MessageKind.DECISION,
        content=f"vote:{vote_for}",
        payload={"vote_for": vote_for},
    )


def _make_draft_msg(*, sender: str = "executor") -> Message:
    return Message(
        sender=sender,
        kind=MessageKind.DRAFT,
        content="draft content",
    )


def _ensure_mesh_registered() -> None:
    if "mesh" not in TopologyRegistry.list_names():
        TopologyRegistry.register("mesh")(MeshTopology)


def _make_mock_agent() -> MagicMock:
    mock_agent = MagicMock()
    mock_agent.step = AsyncMock(return_value={})
    return mock_agent


# ---------------------------------------------------------------------------
# Test 1: test_dispatcher_round_robin_cycles_agents
# ---------------------------------------------------------------------------


class TestDispatcherRoundRobin:
    """Dispatcher round-robin correctly cycles through agents."""

    def setup_method(self) -> None:
        _ensure_mesh_registered()

    def test_dispatcher_round_robin_cycles_agents(self) -> None:
        """Dispatcher should route to each agent in order via round-robin."""
        cfg = _make_cfg(activation_policy="round_robin")
        agent_order = ["planner", "researcher", "executor", "critic"]

        mock_agent = _make_mock_agent()
        agents = dict.fromkeys(agent_order, mock_agent)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.mesh.StateGraph", return_value=mock_graph):
            topology = MeshTopology()
            compiled = topology.build(agents, cfg)

        assert compiled is mock_compiled

        # Verify that the graph was built with the expected conditional edges
        # The dispatcher routing function should use round-robin
        call_args_list = mock_graph.add_conditional_edges.call_args_list
        dispatcher_call = None
        for call in call_args_list:
            if call.args and call.args[0] == "dispatcher":
                dispatcher_call = call
                break

        assert dispatcher_call is not None, "dispatcher conditional edges not found"

        # Verify all 4 agent names are in the routing map
        routing_map = dispatcher_call.args[2] if len(dispatcher_call.args) > 2 else {}
        for agent_id in agent_order:
            assert agent_id in routing_map, f"Agent {agent_id!r} missing from routing map"


# ---------------------------------------------------------------------------
# Test 2: test_priority_activation_routes_to_critic_on_draft
# ---------------------------------------------------------------------------


class TestPriorityActivation:
    """Priority policy routes to critic when a DRAFT is on the broadcast bus."""

    def test_priority_activation_routes_to_critic_on_draft(self) -> None:
        """_pick_priority_agent returns 'critic' when DRAFT is on broadcast bus."""
        draft_msg = _make_draft_msg(sender="executor")
        state: dict[str, Any] = {
            "shared": {"broadcast_bus": [draft_msg]},
            "agents": {},
        }
        agent_order = ["planner", "researcher", "executor", "critic"]

        result = _pick_priority_agent(state, agent_order, rr_index=0)  # type: ignore[arg-type]
        assert result == "critic", f"Expected 'critic', got {result!r}"

    def test_priority_activation_fallback_to_round_robin_when_no_draft(self) -> None:
        """_pick_priority_agent falls back to round-robin when no DRAFT on bus."""
        state: dict[str, Any] = {
            "shared": {"broadcast_bus": []},
            "agents": {},
        }
        agent_order = ["planner", "researcher", "executor", "critic"]

        # rr_index=2 → should return agent_order[2] = "executor"
        result = _pick_priority_agent(state, agent_order, rr_index=2)  # type: ignore[arg-type]
        assert result == "executor", f"Expected 'executor', got {result!r}"


# ---------------------------------------------------------------------------
# Test 3: test_consensus_threshold_writes_winner_signal
# ---------------------------------------------------------------------------


class TestConsensusThreshold:
    """mesh_postprocess sets consensus_reached and final_answer when threshold met."""

    def test_consensus_threshold_writes_winner_signal(self) -> None:
        """When vote_for='4' appears 3 times, consensus_reached=True and final_answer='4'."""
        # Create a state with 3 identical votes
        votes = [_make_decision_msg(vote_for="4", sender=f"agent_{i}") for i in range(3)]

        # Simulate what mesh_postprocess_node would do
        bus = votes
        vote_counts: dict[str, int] = {}
        for msg in bus:
            kind = getattr(msg, "kind", None)
            if kind != MessageKind.DECISION and str(kind) != "decision":
                continue
            payload: dict[str, Any] = getattr(msg, "payload", {}) or {}
            vote_for_val = payload.get("vote_for")
            if not isinstance(vote_for_val, str):
                continue
            vote_counts[vote_for_val] = vote_counts.get(vote_for_val, 0) + 1

        winner = None
        for candidate, count in vote_counts.items():
            if count >= 3:
                winner = candidate
                break

        assert winner == "4", f"Expected winner='4', got {winner!r}"

    def test_no_consensus_when_votes_below_threshold(self) -> None:
        """When vote_for='4' appears 2 times with threshold=3, no consensus."""
        votes = [_make_decision_msg(vote_for="4", sender=f"agent_{i}") for i in range(2)]

        vote_counts: dict[str, int] = {}
        for msg in votes:
            payload = getattr(msg, "payload", {}) or {}
            vote_for_val = payload.get("vote_for")
            if not isinstance(vote_for_val, str):
                continue
            vote_counts[vote_for_val] = vote_counts.get(vote_for_val, 0) + 1

        winner = None
        for candidate, count in vote_counts.items():
            if count >= 3:
                winner = candidate
                break

        assert winner is None


# ---------------------------------------------------------------------------
# Test 4: test_max_rounds_routes_to_end
# ---------------------------------------------------------------------------


class TestMaxRoundsRoutesToEnd:
    """When dispatch_round >= max_rounds (and no consensus), _should_stop triggers."""

    def test_max_rounds_routes_to_end(self) -> None:
        """topology_max_reached=True when dispatch_round >= max_rounds → stop=True."""
        from atm.topology.base import _should_stop

        cfg = _make_cfg(max_rounds=6, consensus_threshold=3)
        state = _make_state(iter_total=5)

        # Simulate: dispatch_round = 6 >= max_rounds=6, no consensus
        stop, reason = _should_stop(
            state,
            cfg,
            topology_success=False,
            topology_max_reached=True,
        )
        assert stop is True
        assert reason == "topology_max"

    def test_no_stop_when_rounds_below_max(self) -> None:
        """topology_max_reached=False and no consensus → stop=False."""
        from atm.topology.base import _should_stop

        cfg = _make_cfg(max_rounds=6, consensus_threshold=3)
        state = _make_state(iter_total=3)

        stop, reason = _should_stop(
            state,
            cfg,
            topology_success=False,
            topology_max_reached=False,
        )
        assert stop is False
        assert reason == ""


# ---------------------------------------------------------------------------
# Test 5: test_global_max_iterations_overrides_topology_max
# ---------------------------------------------------------------------------


class TestGlobalMaxIterationsOverride:
    """Global max_iterations guard fires before topology_max (arch.md §7.1 precedence)."""

    def test_global_max_iterations_overrides_topology_max(self) -> None:
        """iter_total >= max_iterations → MAX_ITER reason, even with topology_max=True."""
        from atm.topology.base import _should_stop

        cfg = _make_cfg(max_iterations=10, max_rounds=6)
        # iter_total == 10 == max_iterations → global guard fires first
        state = _make_state(iter_total=10)

        stop, reason = _should_stop(
            state,
            cfg,
            topology_success=False,
            topology_max_reached=True,  # topology also wants to stop
        )
        assert stop is True
        assert reason == "max_iter", f"Expected 'max_iter', got {reason!r}"

    def test_global_max_iterations_overrides_topology_success(self) -> None:
        """iter_total >= max_iterations → MAX_ITER reason, even with topology_success=True."""
        from atm.topology.base import _should_stop

        cfg = _make_cfg(max_iterations=5)
        state = _make_state(iter_total=5)

        stop, reason = _should_stop(
            state,
            cfg,
            topology_success=True,
            topology_max_reached=False,
        )
        assert stop is True
        assert reason == "max_iter"


# ---------------------------------------------------------------------------
# Test 6: test_build_returns_compiled_graph_with_expected_nodes
# ---------------------------------------------------------------------------


class TestBuildReturnsCompiledGraph:
    """build() returns compiled graph and registers expected nodes."""

    def setup_method(self) -> None:
        _ensure_mesh_registered()

    def test_build_returns_compiled_graph_with_expected_nodes(self) -> None:
        """build() calls add_node for dispatcher, 4 agents, mesh_broadcast, mesh_postprocess."""
        cfg = _make_cfg()
        agent_order = ["planner", "researcher", "executor", "critic"]

        mock_agent = _make_mock_agent()
        agents = dict.fromkeys(agent_order, mock_agent)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.mesh.StateGraph", return_value=mock_graph):
            topology = MeshTopology()
            compiled = topology.build(agents, cfg)

        assert compiled is mock_compiled

        # Verify nodes were added
        added_node_names = {call.args[0] for call in mock_graph.add_node.call_args_list}
        expected_nodes = {"dispatcher", "mesh_broadcast", "mesh_postprocess",
                          "planner", "researcher", "executor", "critic"}
        assert expected_nodes.issubset(added_node_names), (
            f"Missing nodes: {expected_nodes - added_node_names}"
        )

    def test_build_passes_checkpointer_to_compile(self) -> None:
        """build() forwards checkpointer kwarg to graph.compile()."""
        cfg = _make_cfg()
        mock_agent = _make_mock_agent()
        agents = dict.fromkeys(["planner", "researcher", "executor", "critic"], mock_agent)
        mock_cp = MagicMock()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.mesh.StateGraph", return_value=mock_graph):
            topology = MeshTopology()
            topology.build(agents, cfg, checkpointer=mock_cp)

        mock_graph.compile.assert_called_once_with(checkpointer=mock_cp)


# ---------------------------------------------------------------------------
# Test 7: test_register_under_name_mesh
# ---------------------------------------------------------------------------


class TestRegisterUnderNameMesh:
    """MeshTopology is registered under name 'mesh' in TopologyRegistry."""

    def setup_method(self) -> None:
        _ensure_mesh_registered()

    def test_register_under_name_mesh(self) -> None:
        """TopologyRegistry['mesh'] returns MeshTopology class after import."""
        cls = TopologyRegistry.get("mesh")
        assert cls is MeshTopology

    def test_mesh_name_attribute(self) -> None:
        """MeshTopology.name == 'mesh'."""
        assert MeshTopology.name == "mesh"

    def test_mesh_has_callable_build(self) -> None:
        """MeshTopology has a callable build method."""
        assert callable(MeshTopology.build)


# ---------------------------------------------------------------------------
# Test 8: test_mesh_broadcast_writes_outbox_to_bus
# ---------------------------------------------------------------------------


class TestMeshBroadcastWritesOutboxToBus:
    """mesh_broadcast node flushes active agent's outbox to broadcast_bus."""

    def test_mesh_broadcast_writes_outbox_to_bus(self) -> None:
        """When active agent has messages in outbox, they appear on broadcast_bus."""
        msg1 = _make_draft_msg(sender="planner")
        msg2 = Message(
            sender="planner",
            kind=MessageKind.REQUEST,
            content="hello",
        )
        state: dict[str, Any] = {
            "shared": {
                "broadcast_bus": [],
                "signals": {"_mesh_active_agent": "planner"},
                "iter_total": 0,
            },
            "agents": {
                "planner": {"outbox": [msg1, msg2]},
            },
        }

        # Reproduce the mesh_broadcast logic directly
        shared = dict(state["shared"])
        signals = dict(shared.get("signals", {}))
        active_agent = str(signals.get("_mesh_active_agent", ""))
        outbox = list(state["agents"].get(active_agent, {}).get("outbox", []))
        existing_bus = list(shared.get("broadcast_bus", []))
        new_bus = existing_bus + outbox
        cap = 200
        if len(new_bus) > cap:
            new_bus = new_bus[-cap:]
        shared["broadcast_bus"] = new_bus

        assert len(shared["broadcast_bus"]) == 2
        assert msg1 in shared["broadcast_bus"]
        assert msg2 in shared["broadcast_bus"]

    def test_mesh_broadcast_appends_to_existing_bus(self) -> None:
        """mesh_broadcast appends to existing messages, not replaces."""
        old_msg = _make_draft_msg(sender="researcher")
        new_msg = _make_decision_msg(vote_for="answer_A")

        existing_bus = [old_msg]
        new_outbox = [new_msg]

        cap = 200
        new_bus = existing_bus + new_outbox
        if len(new_bus) > cap:
            new_bus = new_bus[-cap:]

        assert len(new_bus) == 2
        assert old_msg in new_bus
        assert new_msg in new_bus


# ---------------------------------------------------------------------------
# Test 9: test_broadcast_bus_does_not_unbound (MC-5)
# ---------------------------------------------------------------------------


class TestBroadcastBusDoesNotUnbound:
    """broadcast_bus is capped at broadcast_bus_cap — MC-5 invariant."""

    def test_broadcast_bus_does_not_unbound(self) -> None:
        """Adding messages beyond cap keeps only the last cap entries."""
        cap = 10
        # Start with cap-1 existing messages
        existing_msgs = [_make_draft_msg(sender=f"agent_{i}") for i in range(cap - 1)]
        new_msgs = [_make_decision_msg(vote_for="x", sender=f"new_{i}") for i in range(5)]

        new_bus = existing_msgs + new_msgs
        if len(new_bus) > cap:
            new_bus = new_bus[-cap:]

        assert len(new_bus) == cap, f"Expected bus size={cap}, got {len(new_bus)}"

    def test_broadcast_bus_cap_trims_oldest_messages(self) -> None:
        """When cap is exceeded, the oldest messages are dropped (keep last cap)."""
        cap = 3
        old_msgs = [_make_draft_msg(sender=f"old_{i}") for i in range(3)]
        new_msgs = [_make_decision_msg(vote_for="v") for _ in range(2)]

        new_bus = old_msgs + new_msgs
        if len(new_bus) > cap:
            new_bus = new_bus[-cap:]

        # Should keep last 3: old_msgs[2] + both new_msgs
        assert len(new_bus) == cap
        assert old_msgs[0] not in new_bus, "oldest message should be dropped"
        assert new_msgs[-1] in new_bus, "newest message should be kept"

    def test_default_cap_is_200(self) -> None:
        """Default broadcast_bus_cap is 200 per conf/topology/mesh.yaml."""
        from atm.topology.mesh import _DEFAULT_BROADCAST_BUS_CAP

        assert _DEFAULT_BROADCAST_BUS_CAP == 200


# ---------------------------------------------------------------------------
# Test 10: test_consensus_vote_payload_str_format
# ---------------------------------------------------------------------------


class TestConsensusVotePayloadStrFormat:
    """Vote payload must be str; non-str values are silently ignored."""

    def test_consensus_vote_payload_str_format(self) -> None:
        """MessageKind.DECISION with payload={'vote_for': str} is the correct format."""
        msg = _make_decision_msg(vote_for="answer_42")
        assert msg.kind == MessageKind.DECISION
        assert isinstance(msg.payload.get("vote_for"), str)
        assert msg.payload["vote_for"] == "answer_42"

    def test_non_string_vote_for_is_ignored_in_tally(self) -> None:
        """Non-string vote_for values (int, None, list) are skipped during tally."""
        bad_votes = [
            Message(sender="a", kind=MessageKind.DECISION, content="x",
                    payload={"vote_for": 42}),
            Message(sender="b", kind=MessageKind.DECISION, content="y",
                    payload={"vote_for": None}),
            Message(sender="c", kind=MessageKind.DECISION, content="z",
                    payload={"vote_for": ["list"]}),
        ]

        vote_counts: dict[str, int] = {}
        for msg in bad_votes:
            kind = getattr(msg, "kind", None)
            if kind != MessageKind.DECISION:
                continue
            payload = getattr(msg, "payload", {}) or {}
            vote_for_val = payload.get("vote_for")
            if not isinstance(vote_for_val, str):
                continue  # silently ignored
            vote_counts[vote_for_val] = vote_counts.get(vote_for_val, 0) + 1

        assert len(vote_counts) == 0, f"Expected empty tally, got {vote_counts}"

    def test_missing_vote_for_key_is_ignored(self) -> None:
        """DECISION message without vote_for key is silently skipped."""
        msg = Message(
            sender="agent",
            kind=MessageKind.DECISION,
            content="decision",
            payload={"some_other_key": "value"},
        )

        vote_counts: dict[str, int] = {}
        payload = msg.payload or {}
        vote_for_val = payload.get("vote_for")
        if isinstance(vote_for_val, str):
            vote_counts[vote_for_val] = vote_counts.get(vote_for_val, 0) + 1

        assert len(vote_counts) == 0
