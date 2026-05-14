"""Public API for atm.phases — Phase Manager + Topology Router + Signals (M8 scope).

Exports the Phase FSM router, Topology Router, signal helpers, and supporting types.

Symbols:
    PhaseRouter              — Protocol for all PhaseRouter implementations
    PhaseGuard               — Callable type alias: (GraphState) -> bool
    PhaseLimits              — Frozen Pydantic model with per-phase iteration caps
    RuleBasedPhaseRouter     — Deterministic rule-based implementation (decided_by='rule')
    LLMPhaseRouter           — LLM-based router with fallback (decided_by='llm_router'/'rule')

    TopologyRouter           — Protocol for all TopologyRouter implementations
    RuleBasedTopologyRouter  — Deterministic rule table (phase x signals) -> topology
    LLMTopologyRouter        — LLM-based topology selection with fallback
    OracleTopologyRouter     — Pre-built oracle table lookup (upper-bound for E3)

    SwitchGuards             — Frozen Pydantic config for topology switch guards
    GuardedRouter            — Decorator: wraps TopologyRouter, enforces SwitchGuards

    SignalKey                — Literal union type of all defined signal key strings
    emit_signal              — Pure-function signal emitter (returns updated SharedState)
    increment_signal         — Convenience helper for integer counter signals

    STUCK                    — Signal key constant "stuck"
    REJECTED_COUNT           — Signal key constant "rejected_count"
    NEEDS_DEBATE             — Signal key constant "needs_debate"
    READY_FOR_EXECUTION      — Signal key constant "ready_for_execution"
    READY_FOR_VERIFICATION   — Signal key constant "ready_for_verification"
    CRITIC_APPROVED          — Signal key constant "critic_approved"
"""

from atm.phases.guards import GuardedRouter, SwitchGuards
from atm.phases.manager import (
    LLMPhaseRouter,
    PhaseGuard,
    PhaseLimits,
    PhaseRouter,
    RuleBasedPhaseRouter,
)
from atm.phases.signals import (
    CRITIC_APPROVED,
    NEEDS_DEBATE,
    READY_FOR_EXECUTION,
    READY_FOR_VERIFICATION,
    REJECTED_COUNT,
    STUCK,
    SignalKey,
    emit_signal,
    increment_signal,
)
from atm.phases.topology_router import (
    LLMTopologyRouter,
    OracleTopologyRouter,
    RuleBasedTopologyRouter,
    TopologyRouter,
)

__all__ = [
    "CRITIC_APPROVED",
    "NEEDS_DEBATE",
    "READY_FOR_EXECUTION",
    "READY_FOR_VERIFICATION",
    "REJECTED_COUNT",
    "STUCK",
    "GuardedRouter",
    "LLMPhaseRouter",
    "LLMTopologyRouter",
    "OracleTopologyRouter",
    "PhaseGuard",
    "PhaseLimits",
    "PhaseRouter",
    "RuleBasedPhaseRouter",
    "RuleBasedTopologyRouter",
    "SignalKey",
    "SwitchGuards",
    "TopologyRouter",
    "emit_signal",
    "increment_signal",
]
