"""Unit tests for atm.phases.topology_router — three TopologyRouter implementations.

TDD: these tests define the contract for RuleBasedTopologyRouter,
LLMTopologyRouter, and OracleTopologyRouter.

Coverage:
  - RuleBasedTopologyRouter: each phase x signal combination from arch.md §7.7
  - LLMTopologyRouter: happy-path, malformed-JSON→fallback, unknown-topology→fallback,
    missing-field→fallback, cost bookkeeping
  - OracleTopologyRouter: known task_id, fallback to task_type, fallback to default
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from atm.core.state import SharedState
from atm.core.types import LLMResponse, Phase, TokenUsage, TopologyDecision

# ---------------------------------------------------------------------------
# Helpers to build minimal SharedState for testing
# ---------------------------------------------------------------------------

ORACLE_FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "oracle_table.json"


def _make_state(
    phase: Phase = Phase.EXECUTION,
    signals: dict[str, object] | None = None,
    active_topology: str = "linear",
    iter_total: int = 5,
    topology_started_at_iter: int = 0,
    topology_switch_count: int = 0,
    topology_history: list[str] | None = None,
    task_id: str = "",
    task_type: str = "",
) -> SharedState:
    """Build a minimal SharedState for topology router tests."""
    state: SharedState = {
        "phase": phase,
        "signals": signals if signals is not None else {},
        "active_topology": active_topology,
        "iter_total": iter_total,
        "topology_started_at_iter": topology_started_at_iter,
        "topology_switch_count": topology_switch_count,
        "topology_history": topology_history if topology_history is not None else [],
    }
    if task_id:
        state["task_id"] = task_id
    if task_type:
        state["task_type"] = task_type  # type: ignore[typeddict-unknown-key]
    return state


# ---------------------------------------------------------------------------
# LLM helper — mirrors pattern from test_manager.py
# ---------------------------------------------------------------------------


def _make_fake_llm(text: str, cost_usd: float = 0.001):
    """Create an AsyncMock that returns an LLMResponse with the given text.

    Matches the real LLMWrapper.ainvoke signature: accepts list[Message].
    """
    mock = AsyncMock()
    usage = TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    mock.ainvoke.return_value = LLMResponse(
        model="fake:test",
        text=text,
        usage=usage,
        cost_usd=cost_usd,
        latency_ms=50,
        finish_reason="stop",
    )
    return mock


def _make_rule_fallback():
    """Create a RuleBasedTopologyRouter suitable for fallback tests."""
    from atm.phases.topology_router import RuleBasedTopologyRouter

    return RuleBasedTopologyRouter()


# ===========================================================================
# TestRuleBasedTopologyRouter
# ===========================================================================


class TestRuleBasedTopologyRouter:
    """Tests for RuleBasedTopologyRouter.decide() per arch.md §7.7 table."""

    # -----------------------------------------------------------------------
    # PLANNING phase — no topology switch rules
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_planning_no_signals_keeps_current(self) -> None:
        """In PLANNING with no signals, keep current topology (no rules fire)."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(phase=Phase.PLANNING, signals={}, active_topology="linear")
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "linear"
        assert decision.decided_by == "rule"

    @pytest.mark.asyncio
    async def test_planning_with_signals_no_switch(self) -> None:
        """In PLANNING, even with execution signals, topology router does not switch."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.PLANNING,
            signals={"stuck": True, "rejected_count": 5},
            active_topology="supervisor",
        )
        decision: TopologyDecision = await router.decide(state)

        # PLANNING has no topology-switch rules
        assert decision.topology == "supervisor"
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # EXECUTION phase — Rule 1: stuck + early in topology → mesh
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execution_stuck_early_switches_to_mesh(self) -> None:
        """stuck=True + iter_within_topology <= 5 → switch to mesh (brainstorm)."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        # iter_within_topology = iter_total - topology_started_at_iter = 5 - 2 = 3
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"stuck": True},
            active_topology="linear",
            iter_total=5,
            topology_started_at_iter=2,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "mesh"
        assert decision.decided_by == "rule"
        assert "mesh" in decision.reason

    @pytest.mark.asyncio
    async def test_execution_stuck_late_no_mesh_switch(self) -> None:
        """stuck=True but iter_within_topology > 5 → rule 1 does NOT fire."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        # iter_within_topology = 10 - 0 = 10 > 5
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"stuck": True},
            active_topology="linear",
            iter_total=10,
            topology_started_at_iter=0,
        )
        decision: TopologyDecision = await router.decide(state)

        # Rule 1 doesn't fire (too late); rule 2: rejected_count=0 → no switch
        assert decision.topology == "linear"
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # EXECUTION phase — Rule 2: rejected_count >= 3 → debate
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execution_rejected_count_3_switches_to_debate(self) -> None:
        """rejected_count >= 3 → switch to debate."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"rejected_count": 3},
            active_topology="linear",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "debate"
        assert decision.decided_by == "rule"
        assert "debate" in decision.reason

    @pytest.mark.asyncio
    async def test_execution_rejected_count_2_no_debate(self) -> None:
        """rejected_count < 3 → no switch to debate."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"rejected_count": 2},
            active_topology="linear",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "linear"
        assert decision.decided_by == "rule"

    @pytest.mark.asyncio
    async def test_execution_rejected_count_5_switches_to_debate(self) -> None:
        """rejected_count >= 3 (here 5) → switch to debate."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"rejected_count": 5},
            active_topology="supervisor",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "debate"
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # EXECUTION phase — Rule priority: stuck fires before rejected_count
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execution_stuck_priority_over_rejected(self) -> None:
        """When both stuck=True (early) and rejected_count>=3, stuck rule wins (first match)."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        # iter_within_topology = 5 - 2 = 3 <= 5 → stuck rule fires first
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"stuck": True, "rejected_count": 3},
            active_topology="linear",
            iter_total=5,
            topology_started_at_iter=2,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "mesh"
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # VERIFICATION phase — needs_revision → linear
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_verification_needs_revision_switches_to_linear(self) -> None:
        """needs_revision=True in verification → switch to linear for rewrite."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.VERIFICATION,
            signals={"needs_revision": True},
            active_topology="debate",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "linear"
        assert decision.decided_by == "rule"

    @pytest.mark.asyncio
    async def test_verification_no_signals_keeps_current(self) -> None:
        """In VERIFICATION with no revision signal, keep current topology."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.VERIFICATION,
            signals={},
            active_topology="debate",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "debate"
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # DONE phase — terminal, no switches
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_done_phase_stays_current(self) -> None:
        """In DONE phase, topology router always keeps current topology."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state = _make_state(
            phase=Phase.DONE,
            signals={"stuck": True, "rejected_count": 10, "needs_revision": True},
            active_topology="hierarchical",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "hierarchical"
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Default topology when none is active
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_active_topology_defaults_to_linear(self) -> None:
        """When active_topology is None, default to 'linear'."""
        from atm.phases.topology_router import RuleBasedTopologyRouter

        router = RuleBasedTopologyRouter()
        state: SharedState = {
            "phase": Phase.PLANNING,
            "signals": {},
            "iter_total": 0,
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
        }
        # active_topology is missing / None
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "linear"
        assert decision.decided_by == "rule"


# ===========================================================================
# TestLLMTopologyRouter
# ===========================================================================


class TestLLMTopologyRouter:
    """Unit tests for LLMTopologyRouter.decide() using AsyncMock."""

    # -----------------------------------------------------------------------
    # Test 1: happy-path — valid JSON → correct TopologyDecision
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_happy_path_valid_json(self) -> None:
        """LLM returns valid JSON with valid topology → TopologyDecision(decided_by='llm_router').

        Verifies ainvoke is called with list[Message] (not list[str]).
        """
        from atm.core.types import Message
        from atm.phases.topology_router import LLMTopologyRouter

        llm = _make_fake_llm('{"topology": "mesh", "reason": "brainstorm needed"}')
        fallback = _make_rule_fallback()
        router = LLMTopologyRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "mesh"
        assert decision.decided_by == "llm_router"
        assert decision.reason == "brainstorm needed"
        assert decision.router_cost_usd == pytest.approx(0.001)

        # Verify ainvoke called with list[Message]
        llm.ainvoke.assert_awaited_once()
        call_args = llm.ainvoke.call_args
        messages_arg = call_args[0][0]
        assert isinstance(messages_arg, list)
        assert len(messages_arg) == 1
        assert isinstance(messages_arg[0], Message)

    # -----------------------------------------------------------------------
    # Test 2: malformed JSON → fallback to rule
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_to_rule(self) -> None:
        """LLM returns non-JSON text → fallback (decided_by='rule')."""
        from atm.phases.topology_router import LLMTopologyRouter

        llm = _make_fake_llm("I recommend switching to mesh topology immediately!")
        fallback = _make_rule_fallback()
        router = LLMTopologyRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.router_cost_usd == 0.0
        llm.ainvoke.assert_awaited_once()

    # -----------------------------------------------------------------------
    # Test 3: unknown topology → fallback to rule
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unknown_topology_falls_back_to_rule(self) -> None:
        """LLM returns a topology name not in valid set → fallback (decided_by='rule')."""
        from atm.phases.topology_router import LLMTopologyRouter

        llm = _make_fake_llm('{"topology": "star", "reason": "use star topology"}')
        fallback = _make_rule_fallback()
        router = LLMTopologyRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.router_cost_usd == 0.0
        llm.ainvoke.assert_awaited_once()

    # -----------------------------------------------------------------------
    # Test 4: missing 'topology' field → fallback to rule
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_missing_topology_field_falls_back_to_rule(self) -> None:
        """LLM returns JSON without 'topology' key → fallback (decided_by='rule')."""
        from atm.phases.topology_router import LLMTopologyRouter

        llm = _make_fake_llm('{"reason": "need to reconsider approach"}')
        fallback = _make_rule_fallback()
        router = LLMTopologyRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        llm.ainvoke.assert_awaited_once()

    # -----------------------------------------------------------------------
    # Test 5: cost bookkeeping — router_cost_usd reflects LLM call cost
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cost_bookkeeping_on_success(self) -> None:
        """router_cost_usd must reflect the actual LLM call cost on success."""
        from atm.phases.topology_router import LLMTopologyRouter

        expensive_llm = _make_fake_llm(
            '{"topology": "debate", "reason": "debate needed"}',
            cost_usd=0.042,
        )
        fallback = _make_rule_fallback()
        router = LLMTopologyRouter(llm=expensive_llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "llm_router"
        assert decision.router_cost_usd == pytest.approx(0.042)

    # -----------------------------------------------------------------------
    # Test 6: LLM call exception → fallback to rule
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_rule(self) -> None:
        """LLM raises an exception → fallback to rule (decided_by='rule')."""
        from atm.phases.topology_router import LLMTopologyRouter

        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("connection timeout")
        fallback = _make_rule_fallback()
        router = LLMTopologyRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"


# ===========================================================================
# TestOracleTopologyRouter
# ===========================================================================


class TestOracleTopologyRouter:
    """Unit tests for OracleTopologyRouter using the fixture table."""

    # -----------------------------------------------------------------------
    # Test 1: known task_id → stable deterministic decision
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_known_task_id_returns_stable_decision(self) -> None:
        """Known task_id in oracle table → returns the mapped topology consistently."""
        from atm.phases.topology_router import OracleTopologyRouter

        router = OracleTopologyRouter(oracle_table=ORACLE_FIXTURE_PATH)
        state = _make_state(
            phase=Phase.EXECUTION,
            task_id="task_abc123",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "debate"  # from fixture: task_abc123, execution
        assert decision.decided_by == "oracle"
        assert decision.router_cost_usd == 0.0
        assert "task_abc123" in decision.reason

        # Call again — must be deterministic
        decision2: TopologyDecision = await router.decide(state)
        assert decision2.topology == decision.topology

    # -----------------------------------------------------------------------
    # Test 2: unknown task_id but known task_type → fallback to task_type
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unknown_task_id_falls_back_to_task_type(self) -> None:
        """Unknown task_id → try task_type → returns task_type mapping."""
        from atm.phases.topology_router import OracleTopologyRouter

        router = OracleTopologyRouter(oracle_table=ORACLE_FIXTURE_PATH)
        state = _make_state(
            phase=Phase.EXECUTION,
            task_id="unknown_task_999",
            task_type="programming",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "mesh"  # fixture: by_task_type.programming.execution
        assert decision.decided_by == "oracle"
        assert "programming" in decision.reason

    # -----------------------------------------------------------------------
    # Test 3: unknown task_id + unknown task_type → fallback to _default
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_match_falls_back_to_default(self) -> None:
        """No task_id or task_type match → returns '_default' topology from table."""
        from atm.phases.topology_router import OracleTopologyRouter

        router = OracleTopologyRouter(oracle_table=ORACLE_FIXTURE_PATH)
        state = _make_state(
            phase=Phase.EXECUTION,
            task_id="completely_unknown",
            task_type="",
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "linear"  # fixture: _default = "linear"
        assert decision.decided_by == "oracle"

    # -----------------------------------------------------------------------
    # Test 4: dict constructor works (for testing without file I/O)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dict_constructor(self) -> None:
        """OracleTopologyRouter can be constructed from a dict directly."""
        from atm.phases.topology_router import OracleTopologyRouter

        table = {
            "by_task_id": {"myid": {"execution": "hierarchical"}},
            "by_task_type": {},
            "_default": "supervisor",
        }
        router = OracleTopologyRouter(oracle_table=table)
        state = _make_state(phase=Phase.EXECUTION, task_id="myid")
        decision: TopologyDecision = await router.decide(state)

        assert decision.topology == "hierarchical"
        assert decision.decided_by == "oracle"

    # -----------------------------------------------------------------------
    # Test 5: different phases → different topology from oracle
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_different_phase_returns_different_topology(self) -> None:
        """Oracle returns phase-specific topology for the same task_type."""
        from atm.phases.topology_router import OracleTopologyRouter

        router = OracleTopologyRouter(oracle_table=ORACLE_FIXTURE_PATH)

        state_plan = _make_state(
            phase=Phase.PLANNING,
            task_id="no_match",
            task_type="programming",
        )
        state_verify = _make_state(
            phase=Phase.VERIFICATION,
            task_id="no_match",
            task_type="programming",
        )

        decision_plan = await router.decide(state_plan)
        decision_verify = await router.decide(state_verify)

        # Fixture: programming.planning = "linear", programming.verification = "debate"
        assert decision_plan.topology == "linear"
        assert decision_verify.topology == "debate"


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


def test_import_public_symbols() -> None:
    """All public topology router symbols must be importable from atm.phases."""
    from atm.phases import (  # noqa: F401
        GuardedRouter,
        LLMTopologyRouter,
        OracleTopologyRouter,
        RuleBasedTopologyRouter,
        SwitchGuards,
        TopologyRouter,
    )
