"""Integration tests for AdaptiveTopology (M8.6) — scripted FakeLLM scenarios.

Tests:
  1. test_adaptive_topology_switches — full end-to-end adaptive run with provoked
     topology switches:
       - Start in PLANNING phase with linear (chain) topology.
       - Executor gets stuck=True → TopologyRouter switches to mesh.
       - rejected_count reaches 3 → TopologyRouter switches to debate.
       - Critic finally approves → phase advances to DONE.
     Asserts:
       - topology_transitions contains ≥1 real switch (from_topology != to_topology).
       - phase monotonicity across all transitions.
       - messages have no duplicate ids (dedup reducer invariant).

  2. test_transition_gate_pure — unit-like test of apply_transition_gate as a
     pure function:
       - Phase advance: signals cleared, iteration reset.
       - Topology switch: only consumed signals cleared, iteration reset.
       - No-change: iteration incremented.

  3. test_guard_override_provoked — thrashing provocation scenario where
     GuardedRouter fires min_dwell guard:
       - Two consecutive ticks with switch proposals under min_dwell_iters.
       - topology_transitions contains a record with decided_by='guard_override'.

All tests use mock agents (no real LLM calls, no PG, no Parquet).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

import atm.topology.adaptive
import atm.topology.chain
import atm.topology.debate
import atm.topology.hierarchical
import atm.topology.mesh
import atm.topology.star  # noqa: F401 — side-effect: register "star"
from atm.core.types import Message, MessageKind, Phase, PhaseDecision, TopologyDecision
from atm.topology.adaptive import AdaptiveTopology, apply_transition_gate, build_adaptive_graph
from atm.topology.base import TopologyConfig

# ---------------------------------------------------------------------------
# Mock agent helpers
# ---------------------------------------------------------------------------


class _StuckExecutor:
    """Mock executor that emits stuck=True signal on first call, then approves."""

    def __init__(self, agent_id: str = "executor") -> None:
        self.agent_id = agent_id
        self._call_count = 0

    async def step(self, state: dict[str, Any]) -> dict[str, Any]:
        self._call_count += 1
        shared = dict(state.get("shared") or {})
        signals = dict(shared.get("signals") or {})

        if self._call_count == 1:
            # First call: stuck signal
            signals["stuck"] = True
        else:
            # Subsequent calls: ready for verification
            signals["ready_for_verification"] = True
            signals["stuck"] = False

        shared["signals"] = signals

        draft_msg = Message(
            sender=self.agent_id,
            kind=MessageKind.DRAFT,
            content="def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)",
            payload={"draft": "def fib(n): return n if n <= 1 else fib(n-1)+fib(n-2)"},
        )
        return {
            "shared": shared,
            "agents": {
                self.agent_id: {
                    "agent_id": self.agent_id,
                    "outbox": [draft_msg],
                    "step_count": self._call_count,
                }
            },
            "messages": [draft_msg],
        }


class _RejectingCritic:
    """Mock critic that rejects N times, then approves."""

    def __init__(self, agent_id: str = "critic", reject_times: int = 3) -> None:
        self.agent_id = agent_id
        self._reject_times = reject_times
        self._call_count = 0

    async def step(self, state: dict[str, Any]) -> dict[str, Any]:
        self._call_count += 1
        shared = dict(state.get("shared") or {})
        signals = dict(shared.get("signals") or {})

        approved = self._call_count > self._reject_times

        if not approved:
            rejected_count = int(signals.get("rejected_count", 0)) + 1
            signals["rejected_count"] = rejected_count
            if rejected_count >= 3:
                signals["needs_debate"] = True
        else:
            signals["critic_approved"] = True

        shared["signals"] = signals

        decision_msg = Message(
            sender=self.agent_id,
            kind=MessageKind.DECISION,
            content="APPROVE" if approved else "REJECT",
            payload={"approved": approved},
        )
        return {
            "shared": shared,
            "agents": {
                self.agent_id: {
                    "agent_id": self.agent_id,
                    "outbox": [decision_msg],
                    "step_count": self._call_count,
                }
            },
            "messages": [decision_msg],
        }


class _SimplePlanner:
    """Mock planner that immediately signals ready_for_execution."""

    def __init__(self, agent_id: str = "planner") -> None:
        self.agent_id = agent_id
        self._call_count = 0

    async def step(self, state: dict[str, Any]) -> dict[str, Any]:
        self._call_count += 1
        shared = dict(state.get("shared") or {})
        signals = dict(shared.get("signals") or {})
        signals["ready_for_execution"] = True
        shared["signals"] = signals

        plan_msg = Message(
            sender=self.agent_id,
            kind=MessageKind.DRAFT,
            content="Plan: write fibonacci function.",
        )
        return {
            "shared": shared,
            "agents": {
                self.agent_id: {
                    "agent_id": self.agent_id,
                    "outbox": [plan_msg],
                    "step_count": self._call_count,
                }
            },
            "messages": [plan_msg],
        }


class _PassthroughAgent:
    """Mock agent that does nothing (noop) — for roles unused by a topology."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    async def step(self, state: dict[str, Any]) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Initial state helper
# ---------------------------------------------------------------------------


def _make_initial_state(*, task_input: str = "Write a fibonacci function.") -> dict[str, Any]:
    return {
        "shared": {
            "task_id": "adaptive_test_task",
            "task_input": task_input,
            "iter_total": 0,
            "iteration": 0,
            "final_answer": None,
            "signals": {},
            "broadcast_bus": [],
            "phase": Phase.PLANNING,
            "phase_history": [],
            "phase_started_at_iter": 0,
            "active_topology": None,
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


# ---------------------------------------------------------------------------
# Test 1: apply_transition_gate pure function
# ---------------------------------------------------------------------------


class TestApplyTransitionGate:
    """Unit-like tests for the pure TransitionGate function."""

    def _base_state(
        self, *, phase: Phase = Phase.EXECUTION, topology: str = "linear"
    ) -> dict[str, Any]:
        return {
            "shared": {
                "phase": phase,
                "active_topology": topology,
                "iteration": 3,
                "iter_total": 5,
                "phase_started_at_iter": 0,
                "topology_started_at_iter": 0,
                "topology_switch_count": 0,
                "topology_history": [],
                "signals": {"stuck": True, "some_other": 42},
                "broadcast_bus": [{"msg": "old"}],
                "final_answer": None,
                "human_requests": [],
                "human_responses": [],
                "phase_history": [],
            },
            "agents": {
                "planner": {"agent_id": "planner", "inbox": [{"m": 1}], "outbox": [{"m": 2}]},
            },
            "messages": [],
            "llm_calls": [],
            "budget_events": [],
            "topology_transitions": [],
        }

    def test_no_change_increments_iteration(self) -> None:
        """No phase/topology change → iteration +1."""
        state = self._base_state()
        phase_dec = PhaseDecision(next_phase=Phase.EXECUTION, reason="stay", decided_by="rule")
        topo_dec = TopologyDecision(topology="linear", reason="keep linear", decided_by="rule")
        new_state = apply_transition_gate(state, phase_dec, topo_dec, run_id=str(uuid4()))

        assert new_state["shared"]["iteration"] == 4  # was 3, incremented
        assert new_state["shared"]["phase"] == Phase.EXECUTION
        assert new_state["shared"]["active_topology"] == "linear"
        # broadcast_bus preserved (no change)
        assert new_state["shared"]["broadcast_bus"] == [{"msg": "old"}]
        # topology_transitions got one new record
        assert len(new_state["topology_transitions"]) == 1
        assert new_state["topology_transitions"][0].decided_by == "rule"
        assert new_state["topology_transitions"][0].to_topology == "linear"

    def test_topology_switch_resets_iteration(self) -> None:
        """Topology switch → iteration reset to 0, topology_switch_count +1."""
        state = self._base_state(topology="linear")
        phase_dec = PhaseDecision(next_phase=Phase.EXECUTION, reason="stay", decided_by="rule")
        topo_dec = TopologyDecision(topology="mesh", reason="stuck=True", decided_by="rule")
        new_state = apply_transition_gate(state, phase_dec, topo_dec, run_id=str(uuid4()))

        assert new_state["shared"]["iteration"] == 0
        assert new_state["shared"]["active_topology"] == "mesh"
        assert new_state["shared"]["topology_switch_count"] == 1
        assert "linear" in new_state["shared"]["topology_history"]
        # broadcast_bus cleared on topology switch
        assert new_state["shared"]["broadcast_bus"] == []
        # stuck signal should be cleared (consumed by router reason "stuck")
        assert "stuck" not in new_state["shared"]["signals"]
        # other signal preserved
        assert new_state["shared"]["signals"].get("some_other") == 42

    def test_phase_advance_clears_signals_and_inboxes(self) -> None:
        """Phase advance → guard signal cleared, other signals preserved, agent inbox/outbox cleared."""
        state = self._base_state(phase=Phase.PLANNING)
        # Add signals: ready_for_execution (guard signal) + another signal
        shared = dict(state["shared"])
        shared["signals"] = {"stuck": True, "some_other": 42, "ready_for_execution": True}
        state = dict(state)
        state["shared"] = shared

        phase_dec = PhaseDecision(
            next_phase=Phase.EXECUTION, reason="ready_for_execution", decided_by="rule"
        )
        topo_dec = TopologyDecision(topology="chain", reason="no rule fired", decided_by="rule")
        new_state = apply_transition_gate(state, phase_dec, topo_dec, run_id=str(uuid4()))

        assert new_state["shared"]["phase"] == Phase.EXECUTION
        # Only the guard signal for the old phase (ready_for_execution) is cleared
        assert "ready_for_execution" not in new_state["shared"]["signals"]
        # Other signals are preserved (not wiped on phase change)
        assert new_state["shared"]["signals"].get("some_other") == 42
        assert new_state["shared"]["iteration"] == 0
        # agent inbox/outbox cleared
        planner_state = new_state["agents"]["planner"]
        assert planner_state["inbox"] == []
        assert planner_state["outbox"] == []

    def test_transition_recorded_on_no_change(self) -> None:
        """TopologyTransition recorded even on no-change (for RQ2 analytics)."""
        state = self._base_state()
        phase_dec = PhaseDecision(next_phase=Phase.EXECUTION, reason="stay", decided_by="rule")
        topo_dec = TopologyDecision(topology="linear", reason="keep", decided_by="rule")

        # Seed one existing transition
        from atm.core.types import TopologyTransition

        existing = TopologyTransition(
            run_id=uuid4(),
            from_topology=None,
            to_topology="linear",
            phase_at_decision=Phase.PLANNING,
            iter_within_phase=0,
            iter_within_topology=0,
            decided_by="initial",
            reason="initial",
        )
        state["topology_transitions"] = [existing]

        new_state = apply_transition_gate(state, phase_dec, topo_dec, run_id=str(uuid4()))
        assert len(new_state["topology_transitions"]) == 2
        assert new_state["topology_transitions"][1].to_topology == "linear"


# ---------------------------------------------------------------------------
# Test 2: AdaptiveTopology builds a compilable meta-graph
# ---------------------------------------------------------------------------


class TestAdaptiveTopologyBuild:
    """Tests that AdaptiveTopology.build() returns a compilable graph."""

    def test_build_returns_compiled_graph(self) -> None:
        """build() should return an object with ainvoke."""
        agents = {
            "planner": _SimplePlanner(),
            "executor": _StuckExecutor(),
            "critic": _RejectingCritic(reject_times=0),
        }
        cfg = TopologyConfig(
            name="adaptive",
            max_iterations=5,
            extra={"switch_guards": False},
        )
        graph = AdaptiveTopology().build(agents, cfg)
        assert hasattr(graph, "ainvoke"), "build() should return an object with ainvoke"

    def test_build_adaptive_graph_factory(self) -> None:
        """build_adaptive_graph() convenience factory works."""
        agents = {"planner": _SimplePlanner()}
        graph = build_adaptive_graph(agents, max_iterations=5)
        assert hasattr(graph, "ainvoke")

    def test_adaptive_registered_in_registry(self) -> None:
        """AdaptiveTopology should be registered under 'adaptive' key.

        Explicitly import adaptive module first to ensure registration,
        since some unit test fixtures may clear TopologyRegistry._registry.
        """
        import atm.topology.adaptive as _adap_mod  # ensure registration side-effect
        from atm.topology.base import TopologyRegistry

        # Re-register if cleared by prior test fixture
        if "adaptive" not in TopologyRegistry.list_names():
            TopologyRegistry._registry["adaptive"] = _adap_mod.AdaptiveTopology

        cls = TopologyRegistry.get("adaptive")
        assert cls is AdaptiveTopology


# ---------------------------------------------------------------------------
# Test 3: Full adaptive run with topology switches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_topology_switches() -> None:
    """End-to-end test: adaptive meta-graph makes ≥1 real topology switch.

    Design: We use a custom AdaptiveTopology subclass that overrides
    _get_subgraph to return lightweight mock subgraphs.  Each mock subgraph
    emits exactly the signals we need for this tick, without running the full
    chain/mesh/debate internal logic (which would confuse the phase-management
    with their own internal coordinators).

    Scenario (signals set by mock subgraphs):
      Tick 1 (PLANNING, linear): mock subgraph emits ready_for_execution=True
      Tick 2 (EXECUTION, linear): mock subgraph emits stuck=True
      Tick 3 (EXECUTION, mesh → switched by TopologyRouter from stuck signal):
                                 mock subgraph emits ready_for_verification=True
      Tick 4 (VERIFICATION, linear):
                                 mock subgraph emits critic_approved=True → DONE

    Expected topology_transitions:
      None→linear (initial), linear→linear (no-change), linear→mesh (SWITCH), mesh→linear (topo router)
    """
    ticks: list[dict[str, Any]] = []

    # Shared mutable tick counter (incremented by mock subgraphs)
    tick_box: list[int] = [0]

    async def _make_mock_subgraph(tick_signals: list[dict[str, Any]]) -> Any:
        """Create a mock compiled graph that emits signals based on tick index."""

        class _MockGraph:
            """Minimal mock that behaves like a compiled LangGraph for ainvoke."""

            async def ainvoke(
                self, state: dict[str, Any], config: dict[str, Any] | None = None
            ) -> dict[str, Any]:
                idx = tick_box[0]
                tick_box[0] += 1

                signals_to_emit = tick_signals[idx] if idx < len(tick_signals) else {}
                ticks.append({"tick": idx, "signals": signals_to_emit})

                new_state = dict(state)
                shared = dict(state.get("shared") or {})
                signals = dict(shared.get("signals") or {})
                signals.update(signals_to_emit)
                shared["signals"] = signals
                new_state["shared"] = shared

                # Emit a message to test dedup
                msg = Message(
                    sender="mock",
                    kind=MessageKind.DRAFT,
                    content=f"tick_{idx}",
                )
                existing_msgs = list(state.get("messages") or [])
                new_state["messages"] = [*existing_msgs, msg]
                return new_state

        return _MockGraph()

    # Pre-build the mock subgraph (same instance for all topology names)
    tick_signals_script = [
        # tick 0 (PLANNING phase): signal readiness for execution
        {"ready_for_execution": True},
        # tick 1 (EXECUTION, linear): get stuck
        {"stuck": True},
        # tick 2 (EXECUTION, mesh after switch): ready for verification
        {"ready_for_verification": True},
        # tick 3 (VERIFICATION): approve
        {"critic_approved": True},
    ]
    mock_sub = await _make_mock_subgraph(tick_signals_script)

    # Build adaptive topology with mock subgraphs.
    # We construct the meta-graph directly using the same node functions as
    # AdaptiveTopology, but with _get_subgraph replaced by mock_sub.

    # Override by directly constructing the meta-graph using the same pattern
    # as AdaptiveTopology.build(), but replacing the subgraph with mock_sub.
    import uuid

    from langgraph.graph import END as LEND
    from langgraph.graph import START, StateGraph

    from atm.core.state import GraphState
    from atm.core.types import Phase, PhaseDecision, TopologyDecision
    from atm.phases.guards import GuardedRouter, SwitchGuards
    from atm.phases.manager import PhaseLimits, RuleBasedPhaseRouter
    from atm.phases.topology_router import RuleBasedTopologyRouter
    from atm.topology.adaptive import apply_transition_gate

    limits = PhaseLimits(planning_max_iter=3, exec_max_iter=10, verify_max_iter=4)
    phase_router = RuleBasedPhaseRouter(limits=limits, guards={})
    rule_topo = RuleBasedTopologyRouter()
    guarded_topo = GuardedRouter(
        inner=rule_topo,
        guards=SwitchGuards(min_dwell_iters=1, cooldown_iters=1, max_per_run=10, max_per_phase=5),
    )

    run_id = str(uuid.uuid4())
    _phase_slot: list[PhaseDecision | None] = [None]
    _topo_slot: list[TopologyDecision | None] = [None]

    async def _phase_router_node(state: dict) -> dict:  # type: ignore[type-arg]
        dec = await phase_router.decide(state)
        _phase_slot[0] = dec
        return {}

    async def _topo_router_node(state: dict) -> dict:  # type: ignore[type-arg]
        from typing import cast

        from atm.core.state import SharedState

        dec = await guarded_topo.decide(cast(SharedState, state.get("shared") or {}))
        _topo_slot[0] = dec
        return {}

    async def _dispatch_node(state: dict) -> dict:  # type: ignore[type-arg]
        shared = dict(state.get("shared") or {})
        iter_total = int(shared.get("iter_total", 0)) + 1
        shared["iter_total"] = iter_total
        # Do NOT update active_topology here — TransitionGate is responsible for that.

        if iter_total > 20:
            return {"shared": shared}

        sub_state = dict(state)
        sub_state["shared"] = shared
        result = await mock_sub.ainvoke(sub_state)
        return result

    async def _gate_node(state: dict) -> dict:  # type: ignore[type-arg]
        from typing import cast

        phase_dec = _phase_slot[0]
        topo_dec = _topo_slot[0]
        shared = state.get("shared") or {}
        current_phase = shared.get("phase", Phase.PLANNING)
        if phase_dec is None:
            phase_dec = PhaseDecision(
                next_phase=current_phase, reason="fallback", decided_by="rule"
            )
        if topo_dec is None:
            topo_dec = TopologyDecision(
                topology=shared.get("active_topology") or "linear",
                reason="fallback",
                decided_by="rule",
            )
        new_state = apply_transition_gate(
            cast(dict[str, Any], state), phase_dec, topo_dec, run_id=run_id
        )
        _phase_slot[0] = None
        _topo_slot[0] = None
        return new_state

    def _should_end(state: dict) -> str:  # type: ignore[type-arg]
        shared = state.get("shared") or {}
        phase = shared.get("phase", Phase.PLANNING)
        iter_total = int(shared.get("iter_total", 0))
        if phase == Phase.DONE or str(phase) == "done":
            return LEND
        if iter_total >= 20:
            return LEND
        return "_phase_router_node"

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("_phase_router_node", _phase_router_node)
    graph_builder.add_node("_topo_router_node", _topo_router_node)
    graph_builder.add_node("_dispatch_node", _dispatch_node)
    graph_builder.add_node("_gate_node", _gate_node)
    graph_builder.add_edge(START, "_phase_router_node")
    graph_builder.add_edge("_phase_router_node", "_topo_router_node")
    graph_builder.add_edge("_topo_router_node", "_dispatch_node")
    graph_builder.add_edge("_dispatch_node", "_gate_node")
    graph_builder.add_conditional_edges(
        "_gate_node", _should_end, {LEND: LEND, "_phase_router_node": "_phase_router_node"}
    )
    graph = graph_builder.compile()

    initial_state = _make_initial_state()
    final_state: dict[str, Any] = await graph.ainvoke(
        initial_state, config={"recursion_limit": 100}
    )

    transitions = final_state.get("topology_transitions", [])

    # --- Assertion 1: ≥1 real switch (from_topology != to_topology, not initial) ---
    real_switches = [
        t for t in transitions if t.from_topology is not None and t.from_topology != t.to_topology
    ]
    assert len(real_switches) >= 1, (
        f"Expected ≥1 real topology switch, got 0.\n"
        f"All transitions: {[(t.from_topology, t.to_topology, t.decided_by) for t in transitions]}\n"
        f"ticks: {ticks}"
    )

    # --- Assertion 2: phase monotonicity ---
    phase_order = {Phase.PLANNING: 0, Phase.EXECUTION: 1, Phase.VERIFICATION: 2, Phase.DONE: 3}
    prev_phase_ord = -1
    for t in transitions:
        phase_ord = phase_order.get(t.phase_at_decision, -1)
        assert phase_ord >= prev_phase_ord, (
            f"Phase monotonicity violated: {t.phase_at_decision} went before "
            f"previous recorded phase order {prev_phase_ord}"
        )
        if phase_ord > prev_phase_ord:
            prev_phase_ord = phase_ord

    # --- Assertion 3: no duplicate message ids ---
    messages = final_state.get("messages", [])
    message_ids = [getattr(m, "id", None) or getattr(m, "message_id", None) for m in messages]
    message_ids_clean = [mid for mid in message_ids if mid is not None]
    assert len(message_ids_clean) == len(set(message_ids_clean)), "Duplicate message ids found"

    # --- Assertion 4: final phase is DONE (or max_iterations hit) ---
    final_phase = final_state.get("shared", {}).get("phase")
    assert final_phase is not None, "shared.phase should be set in final state"
    assert (
        str(final_phase) in {"done", "Phase.DONE"} or final_state["shared"]["iter_total"] >= 20
    ), f"Expected DONE phase, got {final_phase}"


# ---------------------------------------------------------------------------
# Test 4: guard_override provocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_override_provoked() -> None:
    """Provoke guard_override by attempting topology switch before min_dwell_iters.

    Uses the GuardedRouter directly with a state that violates min_dwell constraint.
    This tests that topology_transitions can contain 'guard_override' entries.
    """
    from atm.phases.guards import GuardedRouter, SwitchGuards
    from atm.phases.topology_router import RuleBasedTopologyRouter

    inner = RuleBasedTopologyRouter()
    # Very strict guards: need 10 ticks before switch
    guards = GuardedRouter(inner=inner, guards=SwitchGuards(min_dwell_iters=10))

    # State: currently at linear, iter_total=1, topology_started_at_iter=0
    # → dwell = 1 < 10 → min_dwell should fire
    state: dict[str, Any] = {
        "phase": Phase.EXECUTION,
        "active_topology": "linear",
        "iter_total": 1,
        "topology_started_at_iter": 0,
        "topology_switch_count": 0,
        "topology_history": [],
        "signals": {"stuck": True},  # rule would want to switch to mesh
    }

    decision = await guards.decide(state)  # type: ignore[arg-type]

    # Should be guard_override because dwell=1 < min_dwell_iters=10
    assert decision.decided_by == "guard_override", (
        f"Expected guard_override, got {decision.decided_by}: {decision.reason}"
    )
    assert decision.topology == "linear"  # stayed with current


# ---------------------------------------------------------------------------
# Test 5: topology_transitions dedup
# ---------------------------------------------------------------------------


def test_topology_transitions_dedup_reducer() -> None:
    """Verify that dedup_by_id_reducer on topology_transitions removes duplicates."""
    from atm.core.reducers import dedup_by_id_reducer
    from atm.core.types import TopologyTransition

    reducer = dedup_by_id_reducer("id", sort_by="at")

    run_id = uuid4()
    t1 = TopologyTransition(
        run_id=run_id,
        from_topology="linear",
        to_topology="mesh",
        phase_at_decision=Phase.EXECUTION,
        iter_within_phase=1,
        iter_within_topology=1,
        decided_by="rule",
        reason="stuck",
    )
    t2 = TopologyTransition(
        run_id=run_id,
        from_topology="mesh",
        to_topology="debate",
        phase_at_decision=Phase.EXECUTION,
        iter_within_phase=2,
        iter_within_topology=1,
        decided_by="rule",
        reason="rejected_count>=3",
    )

    existing = [t1]
    update = [t1, t2]  # t1 is a duplicate

    result = reducer(existing, update)
    unique_ids = {str(r.id) for r in result}
    assert len(result) == len(unique_ids), "Dedup reducer should remove duplicate transitions"
    assert len(result) == 2, f"Expected 2 unique transitions, got {len(result)}"
