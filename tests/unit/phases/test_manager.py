"""Unit tests for atm.phases.manager — RuleBasedPhaseRouter and LLMPhaseRouter (M8).

TDD: these tests define the contract for both routers.

Coverage:
  - 4 guards: ready_for_execution, ready_for_verification, critic_approved, iter caps
  - Each guard: positive (advance) and negative (stay) branches
  - Terminal phase 'done': no transitions possible
  - Stay in planning if no guard fires and iter cap not exceeded
  - Monotonicity: decided_by='rule'
  - LLMPhaseRouter: happy-path, rollback-attempt, malformed-JSON, unknown-phase,
    missing-field (all via AsyncMock)
"""

from __future__ import annotations

import pytest

from atm.core.state import GraphState, SharedState
from atm.core.types import Phase, PhaseDecision

# ---------------------------------------------------------------------------
# Helpers to build minimal GraphState for testing
# ---------------------------------------------------------------------------


def _make_state(
    phase: Phase = Phase.PLANNING,
    signals: dict | None = None,
    iteration: int = 0,
) -> GraphState:
    """Build a minimal GraphState with given phase, signals, and iteration count."""
    shared: SharedState = {
        "phase": phase,
        "iteration": iteration,
        "signals": signals if signals is not None else {},
    }
    state: GraphState = {"shared": shared}
    return state


# ---------------------------------------------------------------------------
# Helpers to build PhaseLimits with tight caps for testing
# ---------------------------------------------------------------------------


def _default_limits():
    """Import and create PhaseLimits with default values for testing."""
    from atm.phases.manager import PhaseLimits

    return PhaseLimits(planning_max_iter=3, exec_max_iter=5, verify_max_iter=4)


def _tight_limits():
    """PhaseLimits with very low caps so iter-cap tests trigger at iteration=1."""
    from atm.phases.manager import PhaseLimits

    return PhaseLimits(planning_max_iter=1, exec_max_iter=1, verify_max_iter=1)


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


def test_import_public_symbols() -> None:
    """All public symbols from phases.manager must be importable."""
    from atm.phases.manager import (  # noqa: F401
        LLMPhaseRouter,
        PhaseGuard,
        PhaseLimits,
        PhaseRouter,
        RuleBasedPhaseRouter,
    )


class TestRuleBased:
    """Unit tests for RuleBasedPhaseRouter.decide()."""

    # -----------------------------------------------------------------------
    # Guard 1: ready_for_execution (planning → execution)
    # -----------------------------------------------------------------------

    def test_ready_for_execution_positive(self) -> None:
        """When ready_for_execution=True in planning, router advances to execution."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(phase=Phase.PLANNING, signals={"ready_for_execution": True})
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.EXECUTION
        assert decision.decided_by == "rule"
        assert "ready_for_execution" in decision.reason.lower() or decision.reason

    def test_ready_for_execution_negative(self) -> None:
        """When ready_for_execution=False and iter cap not reached, stay in planning."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(
            phase=Phase.PLANNING,
            signals={"ready_for_execution": False},
            iteration=1,
        )
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.PLANNING
        assert decision.decided_by == "rule"

    def test_ready_for_execution_missing_signal_stays(self) -> None:
        """When signal is absent and iter cap not reached, stay in planning."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(phase=Phase.PLANNING, signals={}, iteration=0)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.PLANNING
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Guard 2: iter cap in planning
    # -----------------------------------------------------------------------

    def test_planning_iter_cap_advances(self) -> None:
        """When iteration >= planning_max_iter, advance planning → execution."""
        from atm.phases.manager import RuleBasedPhaseRouter

        limits = _tight_limits()  # planning_max_iter=1
        router = RuleBasedPhaseRouter(limits=limits, guards={})
        # iteration=1 >= planning_max_iter=1 → advance
        state = _make_state(phase=Phase.PLANNING, signals={}, iteration=1)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.EXECUTION
        assert decision.decided_by == "rule"

    def test_planning_iter_cap_not_reached_stays(self) -> None:
        """When iteration < planning_max_iter and no signal, stay in planning."""
        from atm.phases.manager import RuleBasedPhaseRouter

        limits = _default_limits()  # planning_max_iter=3
        router = RuleBasedPhaseRouter(limits=limits, guards={})
        state = _make_state(phase=Phase.PLANNING, signals={}, iteration=2)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.PLANNING

    # -----------------------------------------------------------------------
    # Guard 3: ready_for_verification (execution → verification)
    # -----------------------------------------------------------------------

    def test_ready_for_verification_positive(self) -> None:
        """When ready_for_verification=True in execution, advance to verification."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"ready_for_verification": True},
        )
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.VERIFICATION
        assert decision.decided_by == "rule"

    def test_ready_for_verification_negative(self) -> None:
        """When ready_for_verification is False and iter cap not reached, stay in execution."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(
            phase=Phase.EXECUTION,
            signals={"ready_for_verification": False},
            iteration=0,
        )
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.EXECUTION
        assert decision.decided_by == "rule"

    def test_exec_iter_cap_advances(self) -> None:
        """When iteration >= exec_max_iter in execution, advance to verification."""
        from atm.phases.manager import RuleBasedPhaseRouter

        limits = _tight_limits()  # exec_max_iter=1
        router = RuleBasedPhaseRouter(limits=limits, guards={})
        state = _make_state(phase=Phase.EXECUTION, signals={}, iteration=1)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.VERIFICATION
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Guard 4: critic_approved (verification → done)
    # -----------------------------------------------------------------------

    def test_critic_approved_positive(self) -> None:
        """When critic_approved=True in verification, advance to done."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(
            phase=Phase.VERIFICATION,
            signals={"critic_approved": True},
        )
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.DONE
        assert decision.decided_by == "rule"

    def test_critic_approved_negative(self) -> None:
        """When critic_approved=False and iter cap not reached, stay in verification."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(
            phase=Phase.VERIFICATION,
            signals={"critic_approved": False},
            iteration=0,
        )
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.VERIFICATION
        assert decision.decided_by == "rule"

    def test_verify_iter_cap_advances_to_done(self) -> None:
        """When iteration >= verify_max_iter in verification, advance to done."""
        from atm.phases.manager import RuleBasedPhaseRouter

        limits = _tight_limits()  # verify_max_iter=1
        router = RuleBasedPhaseRouter(limits=limits, guards={})
        state = _make_state(phase=Phase.VERIFICATION, signals={}, iteration=1)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.DONE
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Terminal phase: done → no transition
    # -----------------------------------------------------------------------

    def test_done_stays_done(self) -> None:
        """In terminal phase 'done', router always returns done (no transitions)."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        state = _make_state(phase=Phase.DONE, signals={"critic_approved": True}, iteration=99)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.DONE
        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Monotonicity: decided_by is always "rule"
    # -----------------------------------------------------------------------

    def test_decided_by_is_always_rule(self) -> None:
        """All RuleBasedPhaseRouter decisions must have decided_by='rule'."""
        from atm.phases.manager import RuleBasedPhaseRouter

        router = RuleBasedPhaseRouter(limits=_default_limits(), guards={})
        scenarios = [
            _make_state(Phase.PLANNING, {"ready_for_execution": True}),
            _make_state(Phase.PLANNING, {}),
            _make_state(Phase.EXECUTION, {"ready_for_verification": True}),
            _make_state(Phase.VERIFICATION, {"critic_approved": True}),
            _make_state(Phase.DONE),
        ]
        for state in scenarios:
            decision = router.decide(state)
            assert decision.decided_by == "rule", (
                f"Expected decided_by='rule' for state phase={state['shared']['phase']}, "
                f"got {decision.decided_by!r}"
            )

    # -----------------------------------------------------------------------
    # Custom guards override: if custom guard provided, use it
    # -----------------------------------------------------------------------

    def test_custom_guard_overrides_signal(self) -> None:
        """Custom guard in guards dict overrides the built-in signal check."""
        from atm.phases.manager import PhaseGuard, RuleBasedPhaseRouter

        # Guard that always returns True (advance planning)
        always_true: PhaseGuard = lambda state: True  # noqa: E731

        router = RuleBasedPhaseRouter(
            limits=_default_limits(),
            guards={Phase.PLANNING: [always_true]},
        )
        # No signal set, but custom guard fires
        state = _make_state(phase=Phase.PLANNING, signals={}, iteration=0)
        decision: PhaseDecision = router.decide(state)

        assert decision.next_phase == Phase.EXECUTION
        assert decision.decided_by == "rule"

    def test_custom_guard_false_stays(self) -> None:
        """Custom guard returning False keeps phase unchanged (when iter cap not hit)."""
        from atm.phases.manager import PhaseGuard, RuleBasedPhaseRouter

        never_advance: PhaseGuard = lambda state: False  # noqa: E731

        router = RuleBasedPhaseRouter(
            limits=_default_limits(),
            guards={Phase.PLANNING: [never_advance]},
        )
        state = _make_state(phase=Phase.PLANNING, signals={"ready_for_execution": True}, iteration=0)
        decision: PhaseDecision = router.decide(state)

        # Custom guard overrides signal → stays in planning
        assert decision.next_phase == Phase.PLANNING

    # -----------------------------------------------------------------------
    # PhaseLimits: immutable (frozen Pydantic model)
    # -----------------------------------------------------------------------

    def test_phase_limits_frozen(self) -> None:
        """PhaseLimits must be immutable (Pydantic frozen model)."""
        import pydantic

        from atm.phases.manager import PhaseLimits

        limits = PhaseLimits(planning_max_iter=3, exec_max_iter=10, verify_max_iter=4)
        with pytest.raises((TypeError, pydantic.ValidationError)):
            limits.planning_max_iter = 99  # type: ignore[misc]

    def test_phase_limits_defaults(self) -> None:
        """PhaseLimits has sensible positive integer caps."""
        from atm.phases.manager import PhaseLimits

        limits = PhaseLimits(planning_max_iter=3, exec_max_iter=10, verify_max_iter=4)
        assert limits.planning_max_iter == 3
        assert limits.exec_max_iter == 10
        assert limits.verify_max_iter == 4


# ---------------------------------------------------------------------------
# LLMPhaseRouter helpers
# ---------------------------------------------------------------------------


def _make_fake_llm(text: str):
    """Create an AsyncMock that returns an LLMResponse with the given text."""
    from unittest.mock import AsyncMock

    from atm.core.types import LLMResponse, TokenUsage

    mock = AsyncMock()
    usage = TokenUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
    mock.ainvoke.return_value = LLMResponse(
        model="fake:test",
        text=text,
        usage=usage,
        cost_usd=0.001,
        latency_ms=50,
        finish_reason="stop",
    )
    return mock


def _make_rule_fallback():
    """Create a RuleBasedPhaseRouter suitable for fallback tests."""
    from atm.phases.manager import PhaseLimits, RuleBasedPhaseRouter

    limits = PhaseLimits(planning_max_iter=5, exec_max_iter=5, verify_max_iter=5)
    return RuleBasedPhaseRouter(limits=limits, guards={})


class TestLLMRouter:
    """Unit tests for LLMPhaseRouter.decide() using AsyncMock."""

    # -----------------------------------------------------------------------
    # Test 1: happy-path — LLM returns valid JSON → correct PhaseDecision
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_happy_path_valid_json(self) -> None:
        """LLM returns valid JSON with valid next_phase → PhaseDecision(decided_by='llm_router')."""
        from atm.phases.manager import LLMPhaseRouter

        llm = _make_fake_llm('{"next_phase": "execution", "reason": "ready to execute"}')
        fallback = _make_rule_fallback()
        router = LLMPhaseRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.PLANNING, signals={})
        decision = await router.decide(state)

        assert decision.next_phase == Phase.EXECUTION
        assert decision.decided_by == "llm_router"
        assert decision.reason == "ready to execute"
        llm.ainvoke.assert_awaited_once()

    # -----------------------------------------------------------------------
    # Test 2: rollback-attempt — LLM returns lower phase → fallback to rule
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rollback_attempt_falls_back_to_rule(self) -> None:
        """LLM tries to return a phase earlier than current → fallback (decided_by='rule')."""
        from atm.phases.manager import LLMPhaseRouter

        # Current is EXECUTION, LLM tries to return PLANNING (rollback)
        llm = _make_fake_llm('{"next_phase": "planning", "reason": "go back"}')
        fallback = _make_rule_fallback()
        router = LLMPhaseRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.EXECUTION, signals={})
        decision = await router.decide(state)

        # Fallback rule: no signals, iter=0 < cap=5 → stay in execution
        assert decision.decided_by == "rule"
        assert decision.next_phase == Phase.EXECUTION

    # -----------------------------------------------------------------------
    # Test 3: malformed JSON — LLM returns non-JSON → fallback to rule
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_to_rule(self) -> None:
        """LLM returns non-JSON text → fallback (decided_by='rule')."""
        from atm.phases.manager import LLMPhaseRouter

        llm = _make_fake_llm("I think you should move to execution phase now!")
        fallback = _make_rule_fallback()
        router = LLMPhaseRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.PLANNING, signals={})
        decision = await router.decide(state)

        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Test 4: unknown phase — LLM returns phase not in enum → fallback
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_unknown_phase_falls_back_to_rule(self) -> None:
        """LLM returns a phase string not in Phase enum → fallback (decided_by='rule')."""
        from atm.phases.manager import LLMPhaseRouter

        llm = _make_fake_llm('{"next_phase": "review", "reason": "need review"}')
        fallback = _make_rule_fallback()
        router = LLMPhaseRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.PLANNING, signals={})
        decision = await router.decide(state)

        assert decision.decided_by == "rule"

    # -----------------------------------------------------------------------
    # Test 5: missing field — JSON without next_phase → fallback
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_missing_next_phase_field_falls_back_to_rule(self) -> None:
        """LLM returns JSON without 'next_phase' key → fallback (decided_by='rule')."""
        from atm.phases.manager import LLMPhaseRouter

        llm = _make_fake_llm('{"reason": "looks good but no phase specified"}')
        fallback = _make_rule_fallback()
        router = LLMPhaseRouter(llm=llm, rule_fallback=fallback)

        state = _make_state(phase=Phase.PLANNING, signals={})
        decision = await router.decide(state)

        assert decision.decided_by == "rule"
