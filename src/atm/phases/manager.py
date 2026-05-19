"""Phase Manager — RuleBasedPhaseRouter, PhaseGuard, PhaseLimits, PhaseRouter Protocol."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from atm.core.state import GraphState, SharedState
from atm.core.types import Message, MessageKind, Phase, PhaseDecision

_log = logging.getLogger(__name__)


PhaseGuard = Callable[[GraphState], bool]

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
        return None


@runtime_checkable
class PhaseRouter(Protocol):
    """Protocol for all PhaseRouter implementations."""

    async def decide(self, state: GraphState) -> PhaseDecision:
        """Decide whether to advance or stay in current phase."""
        ...


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


class RuleBasedPhaseRouter:
    """Deterministic rule-based phase router (signal guards + iteration caps)."""

    def __init__(
        self,
        limits: PhaseLimits,
        guards: dict[Phase, list[PhaseGuard]],
    ) -> None:
        self._limits = limits
        self._guards = guards

    async def decide(self, state: GraphState) -> PhaseDecision:
        """Evaluate guards and return a PhaseDecision."""
        raw_shared: Any = state.get("shared", {})
        shared = cast(SharedState, raw_shared)
        current_phase: Phase = shared.get("phase", Phase.PLANNING)
        iteration: int = shared.get("iteration", 0)

        if current_phase == Phase.DONE:
            return PhaseDecision(
                next_phase=Phase.DONE,
                reason="terminal phase — no further transitions",
                decided_by="rule",
            )

        custom_guard_list = self._guards.get(current_phase)

        if custom_guard_list is not None:
            should_advance = any(g(state) for g in custom_guard_list)
            guard_source = "custom_guard"
        else:
            builtin = _builtin_guard(current_phase)
            should_advance = builtin(state)
            guard_source = _BUILTIN_SIGNAL_KEYS.get(current_phase, "signal")

        next_ph = _next_phase(current_phase)

        if should_advance:
            return PhaseDecision(
                next_phase=next_ph,
                reason=f"{guard_source} fired — advancing {current_phase} → {next_ph}",
                decided_by="rule",
            )

        cap = self._limits.cap_for(current_phase)
        if cap is not None and iteration >= cap:
            return PhaseDecision(
                next_phase=next_ph,
                reason=f"iter cap reached ({iteration} >= {cap}) — advancing {current_phase} → {next_ph}",
                decided_by="rule",
            )

        return PhaseDecision(
            next_phase=current_phase,
            reason=f"no guard fired, iter={iteration} < cap={cap} — staying in {current_phase}",
            decided_by="rule",
        )


class LLMPhaseRouter:
    """LLM-based phase router with JSON parsing, monotonicity validation, and rule fallback."""

    _DEFAULT_PROMPT = (
        "You are a phase router. Current phase: {current_phase}. "
        "Signals: {signals}. "
        'Respond with JSON: {{"next_phase": "<phase>", "reason": "<reason>"}}. '
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
        """Invoke LLM to decide phase transition; fallback to rule on any error."""
        raw_shared: Any = state.get("shared", {})
        shared = cast(SharedState, raw_shared)
        current_phase: Phase = shared.get("phase", Phase.PLANNING)
        signals: dict[str, Any] = shared.get("signals", {})

        prompt = self._prompt_template.format_map(
            {"current_phase": current_phase, "signals": signals}
        )

        messages = [
            Message(
                sender="phase_router",
                kind=MessageKind.REQUEST,
                content=prompt,
            )
        ]

        try:
            response = await self._llm.ainvoke(messages, agent_id="phase_router")
        except Exception as exc:
            _log.warning("LLMPhaseRouter: LLM call failed (%s), falling back to rule.", exc)
            return await self._fallback.decide(state)

        raw_text = response.text or ""
        _log.debug("LLMPhaseRouter: LLM call cost=%.6f USD", response.cost_usd)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            _log.warning(
                "LLMPhaseRouter: malformed JSON from LLM (%s), "
                "raw_preview=%r (len=%d), falling back to rule.",
                exc,
                raw_text[:80],
                len(raw_text),
            )
            return await self._fallback.decide(state)

        if "next_phase" not in data:
            _log.warning(
                "LLMPhaseRouter: missing 'next_phase' in LLM response, "
                "keys=%s, falling back to rule.",
                list(data.keys()),
            )
            return await self._fallback.decide(state)

        raw_next = data["next_phase"]
        try:
            next_phase = Phase(raw_next)
        except ValueError:
            _log.warning(
                "LLMPhaseRouter: unknown phase %r from LLM, falling back to rule.",
                raw_next,
            )
            return await self._fallback.decide(state)

        if _phase_order(next_phase) < _phase_order(current_phase):
            _log.warning(
                "LLMPhaseRouter: rollback attempt %r -> %r (violates monotonicity), "
                "falling back to rule.",
                current_phase,
                next_phase,
            )
            return await self._fallback.decide(state)

        reason: str = str(data.get("reason", "llm_router decision"))

        return PhaseDecision(
            next_phase=next_phase,
            reason=reason,
            decided_by="llm_router",
        )
