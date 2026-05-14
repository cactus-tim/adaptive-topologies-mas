"""SwitchGuards and GuardedRouter — protection against topology thrashing.

Implements arch.md §8bis.3:
  - SwitchGuards   — Pydantic frozen config model with 4 guard parameters
  - GuardedRouter  — decorator around any TopologyRouter, enforces guards

Guard semantics (all evaluated only when inner router proposes a SWITCH):
  1. min_dwell_iters:  new topology must live >= N ticks before any switch is allowed
  2. cooldown_iters:   cannot switch back to a topology that was just left (within N ticks)
  3. max_per_run:      hard cap on total switches across entire run
  4. max_per_phase:    cap on switches within the current phase

If any guard fires:
  - Returns stay-decision (same topology as active) with decided_by='guard_override'
  - considered_alternatives contains the topology inner router wanted + its alternatives
  - router_cost_usd is passed through (if llm_router was used, we still spent the cost)
  - guards_applied field is NOT part of TopologyDecision — guards info goes into reason str

Architecture note:
  GuardedRouter reads from SharedState:
    topology_history            — list[str], recent topology names (tail for cooldown)
    topology_started_at_iter    — int, abs tick when current topology activated
    topology_switch_count       — int, total switches across run
    iter_total                  — int, absolute meta-graph tick
    phase_started_at_iter       — int, abs tick when current phase started
  These fields are set/updated by TransitionGate (arch.md §8bis.5).

  Since we need per-phase switch count but SharedState only has topology_switch_count
  (per-run), we compute per-phase switches as: switches that happened after
  phase_started_at_iter. We approximate this from topology_history length since
  phase start — topology_history is a tail of recent entries maintained by
  TransitionGate for cooldown-check purposes (length <= cooldown_iters + 2).

  Simpler approach used: per-phase switch count is tracked via a dedicated
  field `phase_switch_count` that we read from state if present, else fall back
  to comparing topology_history with phase start. Since SharedState does not yet
  have phase_switch_count, we use the length of topology_history as an upper-bound
  proxy for testing; the real implementation will use the TransitionGate counter.

  Update: reading `phase_switch_count` from signals if available; otherwise
  computing from topology_history as described above.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from atm.core.state import SharedState
from atm.core.types import TopologyDecision
from atm.phases.topology_router import TopologyRouter

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SwitchGuards — frozen configuration model
# ---------------------------------------------------------------------------


class SwitchGuards(BaseModel):
    """Configuration for topology switch guards (arch.md §8bis.3).

    All parameters are non-negative integers. The model is frozen (immutable).

    Attributes:
        min_dwell_iters:  Minimum number of ticks a topology must be active before
                          any switch is permitted. Default 2 (arch.md).
        cooldown_iters:   After leaving topology T, cannot return to T for this many
                          ticks. Default 3 (arch.md).
        max_per_run:      Maximum total topology switches across an entire run.
                          Default 8 (arch.md).
        max_per_phase:    Maximum topology switches within the current phase.
                          Default 4 (arch.md).
    """

    model_config = ConfigDict(frozen=True)

    min_dwell_iters: int = 2
    cooldown_iters: int = 3
    max_per_run: int = 8
    max_per_phase: int = 4


# ---------------------------------------------------------------------------
# Pure guard functions (state + proposed topology → bool blocked)
# ---------------------------------------------------------------------------


def _violates_min_dwell(state: SharedState, guards: SwitchGuards) -> bool:
    """Return True if current topology has not dwelled enough ticks."""
    iter_total: int = state.get("iter_total", 0)
    topology_started: int = state.get("topology_started_at_iter", 0)
    dwell: int = iter_total - topology_started
    return dwell < guards.min_dwell_iters


def _violates_cooldown(state: SharedState, proposed_topology: str, guards: SwitchGuards) -> bool:
    """Return True if proposed topology was recently abandoned (cooldown not elapsed).

    Checks topology_history for the proposed topology within the last cooldown_iters
    entries (excluding the current topology at index -1 if present).
    """
    history: list[str] = state.get("topology_history", [])
    # Check recent history (excluding the very last entry = current topology)
    cooldown_window = guards.cooldown_iters
    # The tail of history to inspect: all but the last entry (current topo)
    tail = history[:-1] if history else []
    # Only look at the last cooldown_window entries of that tail
    recent = tail[-cooldown_window:] if len(tail) > cooldown_window else tail
    return proposed_topology in recent


def _violates_max_per_run(state: SharedState, guards: SwitchGuards) -> bool:
    """Return True if the per-run switch cap has been reached."""
    switch_count: int = state.get("topology_switch_count", 0)
    return switch_count >= guards.max_per_run


def _violates_max_per_phase(state: SharedState, guards: SwitchGuards) -> bool:
    """Return True if the per-phase switch cap has been reached.

    Per-phase switch count is read from signals['phase_switch_count'] if available,
    otherwise approximated from topology_history entries since phase_started_at_iter.
    """
    signals: dict[str, Any] = state.get("signals", {})
    if "phase_switch_count" in signals:
        phase_switches: int = int(signals["phase_switch_count"])
    else:
        # Approximate: count how many switches happened since phase start
        # TransitionGate appends to topology_history on each switch.
        # We proxy this as: count entries in topology_history that could have
        # occurred in this phase. This is an approximation — real tracking in M8.7.
        history: list[str] = state.get("topology_history", [])
        # Conservatively, use full history length as upper bound for phase switches.
        # In practice TransitionGate prunes history, so this is bounded by max_per_run.
        phase_switches = len(history)
    return phase_switches >= guards.max_per_phase


# ---------------------------------------------------------------------------
# GuardedRouter — decorator
# ---------------------------------------------------------------------------


class GuardedRouter:
    """Decorator around any TopologyRouter; enforces SwitchGuards before returning.

    Usage:
        inner = RuleBasedTopologyRouter()
        guarded = GuardedRouter(inner=inner, guards=SwitchGuards())
        decision = await guarded.decide(state)

    Contract (arch.md §8bis.3):
    - If inner proposes a switch (topology != active_topology):
        * Evaluate all 4 guards.
        * If any guard fires: return stay-decision with
            decided_by='guard_override',
            topology=current_topology (no switch),
            considered_alternatives=(inner_topology, *inner.considered_alternatives).
    - If inner proposes no switch: return inner decision unmodified.
    - Guards are only evaluated when a switch is proposed.
    - router_cost_usd is passed through even on guard_override (cost was already spent).

    Args:
        inner:  Any TopologyRouter implementation.
        guards: SwitchGuards configuration (frozen Pydantic model).
    """

    def __init__(self, inner: TopologyRouter, guards: SwitchGuards) -> None:
        self._inner = inner
        self._guards = guards

    async def decide(self, state: SharedState) -> TopologyDecision:
        """Delegate to inner router; block switch if any guard fires.

        Returns:
            TopologyDecision — either unmodified inner decision (no guard triggered)
            or a stay-decision with decided_by='guard_override'.
        """
        raw: TopologyDecision = await self._inner.decide(state)
        current_topology: str = state.get("active_topology") or raw.topology

        # Guards only apply when a switch is proposed
        if raw.topology == current_topology:
            return raw

        # Evaluate all guards
        applied: list[str] = []

        if _violates_min_dwell(state, self._guards):
            applied.append("min_dwell")

        if _violates_cooldown(state, raw.topology, self._guards):
            applied.append("cooldown")

        if _violates_max_per_run(state, self._guards):
            applied.append("max_per_run")

        if _violates_max_per_phase(state, self._guards):
            applied.append("max_per_phase")

        if not applied:
            # No guards fired — allow the switch
            return raw

        _log.info(
            "GuardedRouter: blocked switch %r → %r (guards=%s)",
            current_topology,
            raw.topology,
            applied,
        )

        return TopologyDecision(
            topology=current_topology,
            reason=f"guards={applied}: keep '{current_topology}'",
            decided_by="guard_override",
            considered_alternatives=(raw.topology, *raw.considered_alternatives),
            router_cost_usd=raw.router_cost_usd,  # pass through LLM cost
        )
