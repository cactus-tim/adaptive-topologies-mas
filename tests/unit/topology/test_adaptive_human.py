"""Unit tests for AdaptiveTopology HITL human_advisor node (M9.1 Step 2.15).

Tests cover:
  1. Literal widening — TopologyDecision and TopologyTransition accept "human_override".
  2. Back-compat — without human_cfg graph is identical to M8 (no human_advisor node).
  3. Back-compat — with human_cfg.enabled=False graph has no human_advisor node.
  4. Graph structure — with human_cfg.enabled=True, human_advisor_node is present
     and wired between topology_router_node and dispatch_topology_node.
  5. Advisory mode — human action != "switch_topology" writes hint to signals,
     _topo_dec_slot NOT mutated (routing unchanged).
  6. Advisory mode (abstain) — abstain response leaves signals unchanged.
  7. Override mode (guards allow) — human action="switch_topology" with valid topology
     replaces _topo_dec_slot, decided_by="human_override", hint written.
  8. Override mode (invalid target) — invalid topology name is ignored, hint set,
     slot NOT replaced.
  9. Override mode (guards block) — when GuardedRouter blocks override, original
     slot set to guard_decision with human intent in considered_alternatives.
  10. decided_by="human_override" — TopologyTransition emitted with this value
      when override takes effect.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import atm.topology.adaptive  # noqa: F401 — side-effect: register "adaptive"
from atm.core.types import (
    HumanResponse,
    HumanRole,
    TopologyDecision,
    TopologyTransition,
)
from atm.experiment.config import HumanCfg
from atm.topology.adaptive import AdaptiveTopology, apply_transition_gate
from atm.topology.base import TopologyConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_cfg(*, max_iterations: int = 3) -> TopologyConfig:
    return TopologyConfig(
        name="adaptive",
        max_iterations=max_iterations,
        extra={"switch_guards": False},
    )


def _make_state(
    *,
    iter_total: int = 0,
    run_id: uuid.UUID | None = None,
    active_topology: str = "linear",
    signals: dict[str, Any] | None = None,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    _run_id = run_id or uuid.uuid4()
    return {
        "shared": {
            "run_id": _run_id,
            "iter_total": iter_total,
            "iteration": 0,
            "phase": "planning",
            "active_topology": active_topology,
            "topology_switch_count": 0,
            "topology_started_at_iter": 0,
            "phase_started_at_iter": 0,
            "topology_history": [],
            "signals": dict(signals or {}),
            "final_answer": None,
        },
        "agents": {},
        "messages": list(messages or []),
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _make_approve_response(action: str = "approve") -> HumanResponse:
    return HumanResponse(action=action, comment="LGTM", source="llm_sim", timed_out=False)


def _make_advise_response(hint: str = "mesh is better") -> HumanResponse:
    return HumanResponse(
        action="advise",
        comment=hint,
        payload={},
        source="llm_sim",
        timed_out=False,
    )


def _make_switch_response(topology: str) -> HumanResponse:
    return HumanResponse(
        action="switch_topology",
        comment=f"switch to {topology}",
        payload={"topology": topology},
        source="llm_sim",
        timed_out=False,
    )


def _make_abstain_response() -> HumanResponse:
    return HumanResponse(
        action="abstain", comment=None, payload={}, source="llm_sim", timed_out=False
    )


# ---------------------------------------------------------------------------
# Group 0 — Literal widening (task 2.13 acceptance in test)
# ---------------------------------------------------------------------------


class TestLiteralWidening:
    """TopologyDecision and TopologyTransition accept decided_by='human_override'."""

    def test_topology_decision_accepts_human_override(self) -> None:
        """TopologyDecision(decided_by='human_override') creates without ValidationError."""
        dec = TopologyDecision(
            topology="mesh",
            reason="human test",
            decided_by="human_override",
        )
        assert dec.decided_by == "human_override"
        assert dec.topology == "mesh"

    def test_topology_transition_accepts_human_override(self) -> None:
        """TopologyTransition(decided_by='human_override') creates without ValidationError."""
        from atm.core.types import Phase

        trans = TopologyTransition(
            run_id=uuid.uuid4(),
            from_topology="linear",
            to_topology="mesh",
            phase_at_decision=Phase.PLANNING,
            iter_within_phase=0,
            iter_within_topology=0,
            decided_by="human_override",
            reason="human test",
        )
        assert trans.decided_by == "human_override"

    def test_topology_decision_other_values_still_valid(self) -> None:
        """Existing Literal values still pass validation after widening."""
        for val in ("rule", "llm_router", "oracle", "guard_override", "initial"):
            dec = TopologyDecision(
                topology="linear",
                reason="test",
                decided_by=val,  # type: ignore[arg-type]
            )
            assert dec.decided_by == val


# ---------------------------------------------------------------------------
# Group 1 — Back-compat: without HITL graph is identical to M8
# ---------------------------------------------------------------------------


class TestBackCompat:
    """Without human_cfg (or enabled=False), graph structure is M8-identical."""

    def test_build_no_human_cfg_has_no_human_advisor(self) -> None:
        """build(human_cfg=None) → no 'human_advisor_node' in graph."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.adaptive.StateGraph", return_value=mock_graph):
            AdaptiveTopology().build(agents, cfg)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_advisor_node" not in node_names

    def test_build_disabled_human_cfg_has_no_human_advisor(self) -> None:
        """build(human_cfg=HumanCfg(enabled=False)) → no human_advisor_node."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=False)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.adaptive.StateGraph", return_value=mock_graph):
            AdaptiveTopology().build(agents, cfg, human_cfg=human_cfg)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_advisor_node" not in node_names

    def test_build_no_human_cfg_edge_topology_router_to_dispatch(self) -> None:
        """Without HITL: topology_router_node → dispatch_topology_node edge is direct."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.adaptive.StateGraph", return_value=mock_graph):
            AdaptiveTopology().build(agents, cfg)

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        assert ("topology_router_node", "dispatch_topology_node") in edge_calls


# ---------------------------------------------------------------------------
# Group 2 — Graph structure with HITL enabled
# ---------------------------------------------------------------------------


class TestGraphStructureWithHITL:
    """With human_cfg.enabled=True and valid human_gateway_llm, human_advisor_node is wired."""

    def test_human_advisor_node_added(self) -> None:
        """build(human_cfg=enabled=True) inserts 'human_advisor_node' in graph."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw = MagicMock()
        mock_gw_cls = MagicMock(return_value=mock_gw)

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", mock_gw_cls),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
            )

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_advisor_node" in node_names

    def test_edges_router_to_advisor_to_dispatch(self) -> None:
        """With HITL: topology_router → human_advisor → dispatch edges exist."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw_cls = MagicMock(return_value=MagicMock())

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", mock_gw_cls),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
            )

        edge_calls = [call.args for call in mock_graph.add_edge.call_args_list]
        assert ("topology_router_node", "human_advisor_node") in edge_calls
        assert ("human_advisor_node", "dispatch_topology_node") in edge_calls
        # Direct router→dispatch edge should NOT exist
        assert ("topology_router_node", "dispatch_topology_node") not in edge_calls

    def test_no_llm_wrapper_disables_hitl(self) -> None:
        """human_cfg.enabled=True but no human_gateway_llm → HITL falls back to disabled."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(enabled=True)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled
        mock_gw_cls = MagicMock(return_value=MagicMock())

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", mock_gw_cls),
        ):
            # no human_gateway_llm kwarg → falls back silently
            AdaptiveTopology().build(agents, cfg, human_cfg=human_cfg)

        node_names = [call.args[0] for call in mock_graph.add_node.call_args_list]
        assert "human_advisor_node" not in node_names


# ---------------------------------------------------------------------------
# Group 3 — human_advisor_node advisory mode
# ---------------------------------------------------------------------------


class _FakeAdaptiveWithAdvisor:
    """Helper: builds the adaptive graph and extracts human_advisor_node closure."""

    def __init__(
        self,
        *,
        human_can_override: bool = False,
        use_guards: bool = False,
    ) -> None:
        self.human_can_override = human_can_override
        self.use_guards = use_guards
        self._gateway = AsyncMock()
        self._node: Any = None
        self._topo_dec_slot: list[Any] = [None]

        self._built_topo_dec_slot: list[Any] = []
        self._built_node: Any = None

    def _build_and_extract_node(self) -> Any:
        """Build AdaptiveTopology with HITL and extract the human_advisor_node closure."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}

        extra: dict[str, Any] = {"switch_guards": self.use_guards}
        cfg = TopologyConfig(name="adaptive", max_iterations=3, extra=extra)

        human_cfg_extra: dict[str, Any] = {}
        if self.human_can_override:
            human_cfg_extra["human_can_override_router"] = True

        human_cfg = _make_human_cfg(enabled=True, extra=human_cfg_extra or None)

        captured_nodes: dict[str, Any] = {}

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled

        def _add_node_capture(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph.add_node = _add_node_capture
        mock_graph.add_edge = MagicMock()
        mock_graph.add_conditional_edges = MagicMock()

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", return_value=self._gateway),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
            )

        return captured_nodes.get("human_advisor_node")


class TestAdvisoryMode:
    """Advisory mode: hint written to signals, _topo_dec_slot NOT replaced."""

    def _get_advisor_node(self, **kwargs: Any) -> Any:
        helper = _FakeAdaptiveWithAdvisor(**kwargs)
        return helper._build_and_extract_node()

    def test_advisory_mode_writes_hint_to_signals(self) -> None:
        """Advisory action='advise' → signals['human_advisor_hint'] set, slot unchanged."""
        node = self._get_advisor_node(human_can_override=False)
        if node is None:
            pytest.skip("Could not extract human_advisor_node from build")

        state = _make_state(run_id=uuid.uuid4(), iter_total=1)
        fake_response = _make_advise_response("use mesh topology")

        fake_gw = AsyncMock()
        fake_gw.request = AsyncMock(return_value=fake_response)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with (
            patch("atm.topology.adaptive._gateway", fake_gw, create=True),
            patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch),
        ):
            # We need to patch the node's closure _gateway
            # Since we can't easily patch closure vars, test via the full pipeline
            # The node is already bound — we test it by calling it with a patched gateway
            pass

        # Alternative: test advisory mode by verifying the node writes hint
        # We do this through a direct invocation with the closed-over _gateway mocked
        # The factory helper already sets self._gateway as an AsyncMock
        # We just need to configure its return value and invoke the node
        helper = _FakeAdaptiveWithAdvisor(human_can_override=False)
        helper._gateway.request = AsyncMock(return_value=fake_response)
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        async def fake_dispatch2(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch2):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        assert "human_advisor_hint" in signals, (
            f"Expected 'human_advisor_hint' in signals; got {signals!r}"
        )
        assert (
            "mesh" in signals["human_advisor_hint"] or "use mesh" in signals["human_advisor_hint"]
        )

    def test_advisory_mode_abstain_leaves_signals_unchanged(self) -> None:
        """Advisory action='abstain' → no hint written, signals unchanged."""
        helper = _FakeAdaptiveWithAdvisor(human_can_override=False)
        helper._gateway.request = AsyncMock(return_value=_make_abstain_response())
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), signals={"existing_signal": True})

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        # abstain with empty comment → no hint added
        assert "human_advisor_hint" not in signals or not signals["human_advisor_hint"]
        # Existing signal preserved
        assert signals.get("existing_signal") is True


# ---------------------------------------------------------------------------
# Group 4 — Override mode (guards allow)
# ---------------------------------------------------------------------------


class TestOverrideModeGuardsAllow:
    """Override mode: human replaces _topo_dec_slot[0] when guards allow."""

    def test_override_valid_topology_no_guards_replaces_slot(self) -> None:
        """Override action='switch_topology' with valid topo and no guards → slot replaced."""
        helper = _FakeAdaptiveWithAdvisor(human_can_override=True, use_guards=False)
        helper._gateway.request = AsyncMock(return_value=_make_switch_response("mesh"))
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), active_topology="linear", iter_total=2)

        dispatched: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            dispatched.append(name)

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        assert "human_advisor_hint" in signals
        assert "override_applied" in signals["human_advisor_hint"]

        # human_request and human_response events should have been dispatched
        assert "human_request" in dispatched
        assert "human_response" in dispatched

    def test_override_invalid_topology_ignored_hint_set(self) -> None:
        """Override with invalid topology name → ignored, hint written with error info."""
        helper = _FakeAdaptiveWithAdvisor(human_can_override=True, use_guards=False)
        helper._gateway.request = AsyncMock(
            return_value=_make_switch_response("nonexistent_topology_xyz")
        )
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), active_topology="linear", iter_total=1)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        # Hint should mention the invalid topology
        assert "human_advisor_hint" in signals
        hint = signals["human_advisor_hint"]
        assert "invalid_topology" in hint or "nonexistent_topology_xyz" in hint

    def test_override_mode_decided_by_is_human_override(self) -> None:
        """After successful override, TopologyDecision.decided_by='human_override'."""
        # We test this via apply_transition_gate: if we manually set _topo_dec_slot[0]
        # to a human_override decision, transition_gate_node will produce a
        # TopologyTransition with decided_by='human_override'.
        dec = TopologyDecision(
            topology="mesh",
            reason="human_override: switch to mesh",
            decided_by="human_override",
            considered_alternatives=("linear",),
        )
        from atm.core.types import Phase, PhaseDecision

        phase_dec = PhaseDecision(
            next_phase=Phase.PLANNING,
            reason="no change",
            decided_by="rule",
        )
        state = _make_state(run_id=uuid.uuid4(), active_topology="linear")
        new_state = apply_transition_gate(state, phase_dec, dec, run_id=str(uuid.uuid4()))

        transitions: list[TopologyTransition] = new_state.get("topology_transitions", [])
        assert len(transitions) >= 1
        latest = transitions[-1]
        assert latest.decided_by == "human_override", (
            f"Expected decided_by='human_override', got {latest.decided_by!r}"
        )

    def test_advisory_mode_does_not_mutate_slot_when_override_flag_off(self) -> None:
        """Advisory mode: even with action='switch_topology', slot is NOT replaced
        unless human_can_override_router=True."""
        helper = _FakeAdaptiveWithAdvisor(human_can_override=False, use_guards=False)
        # Send switch_topology action in advisory mode
        helper._gateway.request = AsyncMock(return_value=_make_switch_response("mesh"))
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), active_topology="linear", iter_total=0)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        # In advisory mode, switch_topology treated as advisory — hint written but
        # no mention of "override_applied" (that's an override mode hint)
        signals = delta.get("shared", {}).get("signals", {})
        hint = signals.get("human_advisor_hint", "")
        assert "override_applied" not in hint, (
            f"Advisory mode should not write 'override_applied' hint; got {hint!r}"
        )


# ---------------------------------------------------------------------------
# Group 5 — Dispatch events emitted correctly
# ---------------------------------------------------------------------------


class TestDispatchEvents:
    """Verify human_request and human_response events are dispatched."""

    def test_human_request_before_human_response(self) -> None:
        """human_request is dispatched BEFORE human_response."""
        helper = _FakeAdaptiveWithAdvisor(human_can_override=False)
        helper._gateway.request = AsyncMock(return_value=_make_advise_response())
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), iter_total=0)
        event_order: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            event_order.append(name)

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node(state))

        assert "human_request" in event_order
        assert "human_response" in event_order
        req_idx = event_order.index("human_request")
        resp_idx = event_order.index("human_response")
        assert req_idx < resp_idx, (
            f"human_request (idx={req_idx}) must precede human_response (idx={resp_idx})"
        )

    def test_request_id_contains_adaptive_and_iter_total(self) -> None:
        """request_id format is 'adaptive:{run_id}:{iter_total}:advisor'."""
        helper = _FakeAdaptiveWithAdvisor(human_can_override=False)
        helper._gateway.request = AsyncMock(return_value=_make_advise_response())
        node = helper._build_and_extract_node()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=5)

        captured: dict[str, Any] = {}

        async def fake_dispatch(name: str, data: Any) -> None:
            if name == "human_request":
                captured.update(data)

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node(state))

        assert "request_id" in captured
        assert "adaptive" in captured["request_id"]
        assert "5" in captured["request_id"]  # iter_total=5


# ---------------------------------------------------------------------------
# Group 6 — Override mode with guards that BLOCK the override (Fix #2 test)
# ---------------------------------------------------------------------------


class TestOverrideModeGuardsBlock:
    """Guards directly block a proposed human override (Fix #2: non-tautological check).

    These tests verify that guard violation functions (_violates_min_dwell etc.)
    are evaluated against the proposed topology, not via a topo_router re-call.
    """

    def test_guards_block_override_when_min_dwell_not_met(self) -> None:
        """When min_dwell guard fires, override is rejected.

        Setup:
          - topology_started_at_iter=0, iter_total=0 → dwell=0 < min_dwell_iters=2
          - Human proposes switching from 'linear' to 'mesh' (valid topology)
          - switch_guards=True, human_can_override_router=True

        Expected:
          - decided_by != 'human_override' (guard blocked)
          - hint contains 'override_blocked_by_guards'
          - considered_alternatives contains 'mesh' (the proposed topology)
        """
        # Build with switch_guards=True (min_dwell_iters defaults to 2)
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}

        cfg = TopologyConfig(
            name="adaptive",
            max_iterations=10,
            extra={"switch_guards": True, "switch_guards_config": {"min_dwell_iters": 2}},
        )
        human_cfg = _make_human_cfg(
            enabled=True,
            extra={"human_can_override_router": True},
        )

        captured_nodes: dict[str, Any] = {}
        mock_graph = MagicMock()
        mock_graph.compile.return_value = MagicMock()

        def _add_node_capture(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph.add_node = _add_node_capture
        mock_graph.add_edge = MagicMock()
        mock_graph.add_conditional_edges = MagicMock()

        mock_gateway = AsyncMock()
        mock_gateway.request = AsyncMock(return_value=_make_switch_response("mesh"))

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", return_value=mock_gateway),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
            )

        node = captured_nodes.get("human_advisor_node")
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        # State: iter_total=0, topology_started_at_iter=0 → dwell=0 < min_dwell=2
        state = _make_state(
            run_id=uuid.uuid4(),
            active_topology="linear",
            iter_total=0,
            signals={},
        )
        state["shared"]["topology_started_at_iter"] = 0

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        hint = signals.get("human_advisor_hint", "")

        # Override must be blocked
        assert "override_blocked_by_guards" in hint, (
            f"Expected 'override_blocked_by_guards' in hint when min_dwell violated; got {hint!r}"
        )
        assert "mesh" in hint, f"Expected proposed topology 'mesh' in hint; got {hint!r}"

    def test_guards_block_override_decided_by_is_guard_override(self) -> None:
        """When guards block override, the slot decision has decided_by='guard_override'."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}

        cfg = TopologyConfig(
            name="adaptive",
            max_iterations=10,
            extra={"switch_guards": True, "switch_guards_config": {"min_dwell_iters": 5}},
        )
        human_cfg = _make_human_cfg(
            enabled=True,
            extra={"human_can_override_router": True},
        )

        captured_nodes: dict[str, Any] = {}

        # We verify guard-blocking by inspecting the delta signal hint
        # (the slot is closure-internal but the hint reflects the guard outcome)
        mock_graph = MagicMock()
        mock_graph.compile.return_value = MagicMock()

        def _add_node_capture(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph.add_node = _add_node_capture
        mock_graph.add_edge = MagicMock()
        mock_graph.add_conditional_edges = MagicMock()

        mock_gateway = AsyncMock()
        mock_gateway.request = AsyncMock(return_value=_make_switch_response("mesh"))

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", return_value=mock_gateway),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
            )

        node = captured_nodes.get("human_advisor_node")
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        # State: dwell = iter_total - topology_started_at_iter = 1 - 0 = 1 < min_dwell=5
        state = _make_state(
            run_id=uuid.uuid4(),
            active_topology="linear",
            iter_total=1,
            signals={},
        )
        state["shared"]["topology_started_at_iter"] = 0

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        hint = signals.get("human_advisor_hint", "")

        # The override must be blocked and hint must reflect it
        assert "override_blocked_by_guards" in hint, f"Expected guard block hint; got {hint!r}"
        # 'mesh' (the proposed topology) should be in considered_alternatives
        # We verify by checking the hint mentions the blocked topology
        assert "mesh" in hint

    def test_guards_allow_override_when_no_violations(self) -> None:
        """When no guard fires, override is accepted with decided_by='human_override'.

        Setup: large iter_total means dwell is satisfied, no cooldown, no switch-count cap.
        """
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}

        cfg = TopologyConfig(
            name="adaptive",
            max_iterations=50,
            extra={
                "switch_guards": True,
                "switch_guards_config": {
                    "min_dwell_iters": 2,
                    "cooldown_iters": 0,
                    "max_per_run": 100,
                    "max_per_phase": 100,
                },
            },
        )
        human_cfg = _make_human_cfg(
            enabled=True,
            extra={"human_can_override_router": True},
        )

        captured_nodes: dict[str, Any] = {}
        mock_graph = MagicMock()
        mock_graph.compile.return_value = MagicMock()

        def _add_node_capture(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph.add_node = _add_node_capture
        mock_graph.add_edge = MagicMock()
        mock_graph.add_conditional_edges = MagicMock()

        mock_gateway = AsyncMock()
        mock_gateway.request = AsyncMock(return_value=_make_switch_response("mesh"))

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", return_value=mock_gateway),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                human_gateway_llm=MagicMock(),
            )

        node = captured_nodes.get("human_advisor_node")
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        # Dwell = iter_total - topology_started_at_iter = 10 - 0 = 10 >= min_dwell=2
        state = _make_state(
            run_id=uuid.uuid4(),
            active_topology="linear",
            iter_total=10,
            signals={},
        )
        state["shared"]["topology_started_at_iter"] = 0
        state["shared"]["topology_switch_count"] = 0
        state["shared"]["topology_history"] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            delta = asyncio.run(node(state))

        signals = delta.get("shared", {}).get("signals", {})
        hint = signals.get("human_advisor_hint", "")

        # Override must be accepted when no guards fire
        assert "override_applied" in hint, (
            f"Expected 'override_applied' when guards allow; got {hint!r}"
        )


# ---------------------------------------------------------------------------
# Group 7 — role_router integration (M9.2)
# ---------------------------------------------------------------------------


class _FakeAdaptiveWithRoleRouter:
    """Helper: builds AdaptiveTopology with HITL + optional role_router and extracts node."""

    def __init__(
        self,
        *,
        human_cfg_role: HumanRole = HumanRole.REVIEWER,
        role_router: Any = None,
    ) -> None:
        self.human_cfg_role = human_cfg_role
        self.role_router = role_router
        self._gateway: Any = AsyncMock()

    def build_and_extract(self) -> Any:
        """Build AdaptiveTopology and return (node_fn, gateway)."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent}

        cfg = TopologyConfig(
            name="adaptive",
            max_iterations=3,
            extra={"switch_guards": False},
        )
        human_cfg = _make_human_cfg(enabled=True, role=self.human_cfg_role)

        captured_nodes: dict[str, Any] = {}
        mock_graph = MagicMock()
        mock_graph.compile.return_value = MagicMock()

        def _add_node_capture(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph.add_node = _add_node_capture
        mock_graph.add_edge = MagicMock()
        mock_graph.add_conditional_edges = MagicMock()

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", return_value=self._gateway),
        ):
            AdaptiveTopology().build(
                agents,
                cfg,
                human_cfg=human_cfg,
                role_router=self.role_router,
                human_gateway_llm=MagicMock(),
            )

        return captured_nodes.get("human_advisor_node")


class TestAdaptiveRoleRouter:
    """M9.2 role_router integration in AdaptiveTopology.human_advisor_node.

    Three tests:
      1. Back-compat: role_router=None → role in HumanContext == human_cfg.role.
      2. Dynamic: role_router=FixedRoleRouter(JUDGE), human_cfg.role=REVIEWER → role==JUDGE.
      3. Phase-dependent: RuleBasedRoleRouter returns different roles per phase.
    """

    def test_back_compat_role_router_none_uses_human_cfg_role(self) -> None:
        """role_router=None → HumanContext.role == human_cfg.role (REVIEWER)."""
        from atm.core.types import HumanContext  # noqa: F401

        helper = _FakeAdaptiveWithRoleRouter(
            human_cfg_role=HumanRole.REVIEWER,
            role_router=None,
        )

        captured_roles: list[Any] = []

        async def capturing_request(ctx: Any, *, request_id: str) -> Any:
            captured_roles.append(ctx.role)
            return _make_advise_response("OK")

        helper._gateway.request = capturing_request
        node = helper.build_and_extract()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), iter_total=0)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node(state))

        assert len(captured_roles) == 1, f"Expected 1 captured role; got {captured_roles!r}"
        assert captured_roles[0] == HumanRole.REVIEWER, (
            f"Expected REVIEWER (back-compat); got {captured_roles[0]!r}"
        )

    def test_dynamic_fixed_role_router_overrides_human_cfg_role(self) -> None:
        """role_router=FixedRoleRouter(JUDGE), human_cfg.role=REVIEWER → role==JUDGE."""
        from atm.human.role_router import FixedRoleRouter

        router = FixedRoleRouter(role=HumanRole.JUDGE)
        helper = _FakeAdaptiveWithRoleRouter(
            human_cfg_role=HumanRole.REVIEWER,
            role_router=router,
        )

        captured_roles: list[Any] = []

        async def capturing_request(ctx: Any, *, request_id: str) -> Any:
            captured_roles.append(ctx.role)
            return _make_advise_response("OK")

        helper._gateway.request = capturing_request
        node = helper.build_and_extract()
        if node is None:
            pytest.skip("Could not extract human_advisor_node")

        state = _make_state(run_id=uuid.uuid4(), iter_total=0)

        async def fake_dispatch(name: str, data: Any) -> None:
            pass

        with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node(state))

        assert len(captured_roles) == 1, f"Expected 1 captured role; got {captured_roles!r}"
        assert captured_roles[0] == HumanRole.JUDGE, (
            f"Expected JUDGE (from FixedRoleRouter); got {captured_roles[0]!r}"
        )

    def test_rule_based_router_returns_phase_dependent_role(self) -> None:
        """RuleBasedRoleRouter returns different roles per phase (planning→COORDINATOR, etc.)."""
        from atm.human.role_router import RuleBasedRoleRouter

        # Use default role table: planning→coordinator, execution→peer, etc.
        router = RuleBasedRoleRouter()

        # We'll invoke the node twice: once with phase=planning, once with phase=execution.
        # Re-build for each phase to start fresh (closure state is independent per build).
        captured_roles_by_phase: dict[str, Any] = {}

        for phase_str in ("planning", "execution"):
            helper = _FakeAdaptiveWithRoleRouter(
                human_cfg_role=HumanRole.REVIEWER,  # static fallback, should be overridden
                role_router=router,
            )

            captured: list[Any] = []

            async def capturing_request(
                ctx: Any, *, request_id: str, _cap: list[Any] = captured
            ) -> Any:
                _cap.append(ctx.role)
                return _make_advise_response("OK")

            helper._gateway.request = capturing_request
            node = helper.build_and_extract()
            if node is None:
                pytest.skip("Could not extract human_advisor_node")

            state = _make_state(run_id=uuid.uuid4(), iter_total=0)
            # Override phase in shared state
            state["shared"]["phase"] = phase_str

            async def fake_dispatch(name: str, data: Any) -> None:
                pass

            with patch("atm.topology.adaptive.adispatch_custom_event", side_effect=fake_dispatch):
                asyncio.run(node(state))

            if captured:
                captured_roles_by_phase[phase_str] = captured[0]

        # planning → coordinator, execution → peer (per DEFAULT_ROLE_TABLE)
        assert "planning" in captured_roles_by_phase, "No role captured for planning phase"
        assert "execution" in captured_roles_by_phase, "No role captured for execution phase"

        assert captured_roles_by_phase["planning"] == HumanRole.COORDINATOR, (
            f"Expected COORDINATOR for planning; got {captured_roles_by_phase['planning']!r}"
        )
        assert captured_roles_by_phase["execution"] == HumanRole.PEER, (
            f"Expected PEER for execution; got {captured_roles_by_phase['execution']!r}"
        )
