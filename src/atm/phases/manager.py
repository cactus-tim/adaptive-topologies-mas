"""Phase Manager — RuleBasedPhaseRouter and supporting types.

Implements the monotonic Phase FSM described in arch.md §8.1:
    planning → execution → verification → done

Public API:
    PhaseRouter    — Protocol (decide method)
    PhaseGuard     — Callable type alias: (GraphState) -> bool
    PhaseLimits    — Frozen Pydantic model with per-phase iteration caps
    RuleBasedPhaseRouter — Deterministic rule-based implementation

Architecture decisions (arch.md §8.1, §8.2 sketch = obsolete):
    - PhaseGuard is bool-returning (§8.1 line 1615, final)
    - Guards dict overrides built-in signal checks when provided
    - decided_by='rule' for all RuleBasedPhaseRouter decisions
    - Phase enum imported strictly from atm.core.types (not storage.models)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from atm.core.state import GraphState, SharedState
from atm.core.types import Phase, PhaseDecision

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhaseGuard — type alias for pure guard functions
# ---------------------------------------------------------------------------

#: Guard function type: takes full GraphState, returns True to advance phase.
#: Signature matches arch.md §8.1 line 1615 (bool-semantics, FINAL).
PhaseGuard = Callable[[GraphState], bool]


# ---------------------------------------------------------------------------
# _phase_order — monotonicity helper
# ---------------------------------------------------------------------------

_PHASE_ORDER: dict[Phase, int] = {
    Phase.PLANNING: 0,
    Phase.EXECUTION: 1,
    Phase.VERIFICATION: 2,
    Phase.DONE: 3,
}


def _phase_order(phase: Phase) -> int:
    """Return integer ordering of phase for monotonicity checks.

    planning=0 < execution=1 < verification=2 < done=3
    """
    return _PHASE_ORDER[phase]


def _next_phase(current: Phase) -> Phase:
    """Return the next phase in the monotonic sequence, or DONE if terminal."""
    sequence = [Phase.PLANNING, Phase.EXECUTION, Phase.VERIFICATION, Phase.DONE]
    idx = sequence.index(current)
    if idx < len(sequence) - 1:
        return sequence[idx + 1]
    return Phase.DONE


# ---------------------------------------------------------------------------
# PhaseLimits — per-phase iteration caps (frozen Pydantic model)
# ---------------------------------------------------------------------------


class PhaseLimits(BaseModel):
    """Iteration caps for each phase.

    When iteration >= cap, PhaseRouter forces advance regardless of signals.
    All caps must be positive integers. Model is frozen (immutable after creation).
    """

    model_config = ConfigDict(frozen=True)

    planning_max_iter: int
    exec_max_iter: int
    verify_max_iter: int

    def cap_for(self, phase: Phase) -> int | None:
        """Return the max iteration cap for the given phase, or None for DONE."""
        if phase == Phase.PLANNING:
            return self.planning_max_iter
        if phase == Phase.EXECUTION:
            return self.exec_max_iter
        if phase == Phase.VERIFICATION:
            return self.verify_max_iter
        return None  # DONE has no cap


# ---------------------------------------------------------------------------
# PhaseRouter — Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PhaseRouter(Protocol):
    """Protocol for all PhaseRouter implementations.

    Contract (arch.md §8.2):
    - decide() is called exactly once per meta-graph tick, after subgraph END.
    - Returns a PhaseDecision where next_phase >= state.shared.phase (monotonic).
    - Must NOT mutate state or produce side-effects (DB writes, LLM calls for 'rule').
    """

    def decide(self, state: GraphState) -> PhaseDecision:
        """Decide whether to advance or stay in current phase.

        Args:
            state: Full GraphState after current subgraph run.

        Returns:
            PhaseDecision with monotonic next_phase and decided_by='rule'
            (or 'llm_router' for LLMPhaseRouter).
        """
        ...


# ---------------------------------------------------------------------------
# Built-in signal guards (default behavior per arch.md §8.1, Table on line 1569)
# ---------------------------------------------------------------------------

_BUILTIN_SIGNAL_KEYS: dict[Phase, str] = {
    Phase.PLANNING: "ready_for_execution",
    Phase.EXECUTION: "ready_for_verification",
    Phase.VERIFICATION: "critic_approved",
}


def _builtin_guard(phase: Phase) -> PhaseGuard:
    """Return the built-in signal guard for the given phase.

    Reads state.shared.signals[signal_key] and returns bool.
    If the key is absent, returns False (do not advance).
    """
    signal_key = _BUILTIN_SIGNAL_KEYS.get(phase)

    def _guard(state: GraphState) -> bool:
        if signal_key is None:
            return False
        raw_shared: Any = state.get("shared", {})
        shared = cast(SharedState, raw_shared)
        signals: dict[str, Any] = shared.get("signals", {})
        value = signals.get(signal_key, False)
        return bool(value)

    return _guard


# ---------------------------------------------------------------------------
# RuleBasedPhaseRouter
# ---------------------------------------------------------------------------


class RuleBasedPhaseRouter:
    """Deterministic rule-based phase router.

    Decision logic (per phase):
    1. If current phase is DONE → stay DONE (terminal, no transitions).
    2. If custom guards provided for this phase → use them (any True → advance).
    3. Else → use built-in signal guard (reads state.shared.signals).
    4. If iteration >= phase cap → force advance (timeout guard).
    5. Otherwise → stay in current phase.

    All decisions have decided_by='rule'.

    Args:
        limits: PhaseLimits with per-phase iteration caps.
        guards: Optional dict mapping Phase → list of PhaseGuard functions.
                When provided for a phase, these replace (not supplement)
                the built-in signal guard for that phase.
    """

    def __init__(
        self,
        limits: PhaseLimits,
        guards: dict[Phase, list[PhaseGuard]],
    ) -> None:
        self._limits = limits
        self._guards = guards

    def decide(self, state: GraphState) -> PhaseDecision:
        """Evaluate guards and return a PhaseDecision.

        Monotonicity invariant: next_phase >= current_phase (enforced here).
        """
        raw_shared: Any = state.get("shared", {})
        shared = cast(SharedState, raw_shared)
        current_phase: Phase = shared.get("phase", Phase.PLANNING)
        iteration: int = shared.get("iteration", 0)

        # Terminal phase: no transitions
        if current_phase == Phase.DONE:
            return PhaseDecision(
                next_phase=Phase.DONE,
                reason="terminal phase — no further transitions",
                decided_by="rule",
            )

        # Check if custom guards are registered for this phase
        custom_guard_list = self._guards.get(current_phase)

        if custom_guard_list is not None:
            # Custom guards override built-in signal check
            should_advance = any(g(state) for g in custom_guard_list)
            guard_source = "custom_guard"
        else:
            # Use built-in signal guard
            builtin = _builtin_guard(current_phase)
            should_advance = builtin(state)
            guard_source = _BUILTIN_SIGNAL_KEYS.get(current_phase, "signal")

        # Determine next phase if advancing
        next_ph = _next_phase(current_phase)

        if should_advance:
            return PhaseDecision(
                next_phase=next_ph,
                reason=f"{guard_source} fired — advancing {current_phase} → {next_ph}",
                decided_by="rule",
            )

        # Check iteration cap (timeout guard)
        cap = self._limits.cap_for(current_phase)
        if cap is not None and iteration >= cap:
            return PhaseDecision(
                next_phase=next_ph,
                reason=f"iter cap reached ({iteration} >= {cap}) — advancing {current_phase} → {next_ph}",
                decided_by="rule",
            )

        # Stay in current phase
        return PhaseDecision(
            next_phase=current_phase,
            reason=f"no guard fired, iter={iteration} < cap={cap} — staying in {current_phase}",
            decided_by="rule",
        )


# ---------------------------------------------------------------------------
# LLMPhaseRouter
# ---------------------------------------------------------------------------


class LLMPhaseRouter:
    """LLM-based phase router with JSON parsing, monotonicity validation, and fallback.

    Decision flow:
    1. Build prompt from template + current state.
    2. Call llm.ainvoke(messages, agent_id='phase_router').
    3. Parse JSON from LLMResponse.text.
    4. Validate: next_phase in Phase enum, next_phase >= current_phase (monotonic).
    5. On any failure (JSON parse error, missing field, unknown phase, rollback):
       log WARNING and delegate to rule_fallback.decide(state).
    6. On success: return PhaseDecision(decided_by='llm_router', router_cost_usd=...).
    7. On fallback: decided_by='rule' (from rule_fallback), router_cost_usd=0.0.

    Architecture note (arch.md §8.2):
    - decided_by='rule' for fallback decisions — this is intentional.
      The decided_by field reflects who made the final decision, not who attempted.

    Args:
        llm: Any object with async ainvoke(messages, *, agent_id) -> LLMResponse.
        rule_fallback: RuleBasedPhaseRouter for fallback.
        prompt_template: String template. Must contain {current_phase} and {signals}
                         placeholders; formatted with str.format_map().
    """

    _DEFAULT_PROMPT = (
        "You are a phase router. Current phase: {current_phase}. "
        "Signals: {signals}. "
        "Respond with JSON: {{\"next_phase\": \"<phase>\", \"reason\": \"<reason>\"}}. "
        "Valid phases (monotonic order): planning, execution, verification, done."
    )

    def __init__(
        self,
        llm: Any,
        rule_fallback: RuleBasedPhaseRouter,
        prompt_template: str | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = rule_fallback
        self._prompt_template = prompt_template or self._DEFAULT_PROMPT

    async def decide(self, state: GraphState) -> PhaseDecision:
        """Invoke LLM to decide phase transition; fallback to rule on any error.

        Returns:
            PhaseDecision with decided_by='llm_router' on success,
            or decided_by='rule' (from fallback) on any parse/validation error.
        """
        raw_shared: Any = state.get("shared", {})
        shared = cast(SharedState, raw_shared)
        current_phase: Phase = shared.get("phase", Phase.PLANNING)
        signals: dict[str, Any] = shared.get("signals", {})

        prompt = self._prompt_template.format_map(
            {"current_phase": current_phase, "signals": signals}
        )

        try:
            response = await self._llm.ainvoke([prompt], agent_id="phase_router")
        except Exception as exc:
            _log.warning("LLMPhaseRouter: LLM call failed (%s), falling back to rule.", exc)
            return self._fallback.decide(state)

        raw_text = response.text or ""
        _log.debug("LLMPhaseRouter: LLM call cost=%.6f USD", response.cost_usd)

        # Parse JSON
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            _log.warning(
                "LLMPhaseRouter: malformed JSON from LLM (%s), raw=%r, falling back to rule.",
                exc,
                raw_text,
            )
            return self._fallback.decide(state)

        # Validate 'next_phase' field present
        if "next_phase" not in data:
            _log.warning(
                "LLMPhaseRouter: missing 'next_phase' in LLM response %r, falling back to rule.",
                data,
            )
            return self._fallback.decide(state)

        # Validate next_phase is a known Phase member
        raw_next = data["next_phase"]
        try:
            next_phase = Phase(raw_next)
        except ValueError:
            _log.warning(
                "LLMPhaseRouter: unknown phase %r from LLM, falling back to rule.",
                raw_next,
            )
            return self._fallback.decide(state)

        # Validate monotonicity: next_phase >= current_phase
        if _phase_order(next_phase) < _phase_order(current_phase):
            _log.warning(
                "LLMPhaseRouter: rollback attempt %r -> %r (violates monotonicity), "
                "falling back to rule.",
                current_phase,
                next_phase,
            )
            return self._fallback.decide(state)

        reason: str = str(data.get("reason", "llm_router decision"))

        return PhaseDecision(
            next_phase=next_phase,
            reason=reason,
            decided_by="llm_router",
        )
