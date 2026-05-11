"""Public API for atm.phases — Phase Manager (M8 scope).

Exports the Phase FSM router and supporting types.

Symbols:
    PhaseRouter           — Protocol for all PhaseRouter implementations
    PhaseGuard            — Callable type alias: (GraphState) -> bool
    PhaseLimits           — Frozen Pydantic model with per-phase iteration caps
    RuleBasedPhaseRouter  — Deterministic rule-based implementation (decided_by='rule')
"""

from atm.phases.manager import PhaseGuard, PhaseLimits, PhaseRouter, RuleBasedPhaseRouter

__all__ = [
    "PhaseGuard",
    "PhaseLimits",
    "PhaseRouter",
    "RuleBasedPhaseRouter",
]
