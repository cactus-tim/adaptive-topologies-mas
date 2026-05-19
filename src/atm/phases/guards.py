"""SwitchGuards and GuardedRouter — protection against topology thrashing."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from atm.core.state import SharedState
from atm.core.types import TopologyDecision
from atm.phases.topology_router import TopologyRouter

_log = logging.getLogger(__name__)


class SwitchGuards(BaseModel):
    """Frozen config for topology switch guards (min_dwell, cooldown, max_per_run, max_per_phase)."""

    model_config = ConfigDict(frozen=True)

    min_dwell_iters: int = 2
    cooldown_iters: int = 3
    max_per_run: int = 8
    max_per_phase: int = 4


def _violates_min_dwell(state: SharedState, guards: SwitchGuards) -> bool:
    """Return True if current topology has not dwelled enough ticks."""
    iter_total: int = state.get("iter_total", 0)
    topology_started: int = state.get("topology_started_at_iter", 0)
    dwell: int = iter_total - topology_started
    return dwell < guards.min_dwell_iters


def _violates_cooldown(state: SharedState, proposed_topology: str, guards: SwitchGuards) -> bool:
    """Return True if proposed topology was recently abandoned (within cooldown window)."""
    history: list[str] = list(state.get("topology_history", []) or [])
    if not history:
        return False
    cooldown_window = guards.cooldown_iters
    if cooldown_window <= 0:
        return False
    recent = history[-cooldown_window:]
    return proposed_topology in recent


def _violates_max_per_run(state: SharedState, guards: SwitchGuards) -> bool:
    """Return True if the per-run switch cap has been reached."""
    switch_count: int = state.get("topology_switch_count", 0)
    return switch_count >= guards.max_per_run


def _violates_max_per_phase(state: SharedState, guards: SwitchGuards) -> bool:
    """Return True if the per-phase switch cap has been reached."""
    signals: dict[str, Any] = state.get("signals", {})
    if "phase_switch_count" in signals:
        phase_switches: int = int(signals["phase_switch_count"])
    else:
        history: list[str] = state.get("topology_history", [])
        phase_switches = len(history)
    return phase_switches >= guards.max_per_phase


class GuardedRouter:
    """Decorator around any TopologyRouter; enforces SwitchGuards before returning."""

    def __init__(self, inner: TopologyRouter, guards: SwitchGuards) -> None:
        self._inner = inner
        self._guards = guards

    async def decide(self, state: SharedState) -> TopologyDecision:
        """Delegate to inner router; return stay-decision with guard_override if any guard fires."""
        raw: TopologyDecision = await self._inner.decide(state)
        current_topology: str = state.get("active_topology") or raw.topology

        if raw.topology == current_topology:
            return raw

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
            router_cost_usd=raw.router_cost_usd,
        )
