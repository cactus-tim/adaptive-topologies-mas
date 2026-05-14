"""Unit tests for atm.phases.guards — SwitchGuards and GuardedRouter.

TDD: these tests define the contract for topology switch protection.

Coverage:
  - SwitchGuards: frozen Pydantic model with correct defaults
  - Each guard blocks as expected (trigger / no-trigger pairs):
      min_dwell, cooldown, max_per_run, max_per_phase
  - considered_alternatives preserved when guard blocks
  - decided_by='guard_override' when guard blocks
  - router_cost_usd passes through from inner router on guard_override
  - No guard → inner decision returned unmodified
  - Only one guard needs to fire to block (any-guard semantics)
"""

from __future__ import annotations

import pytest

from atm.core.state import SharedState
from atm.core.types import Phase, TopologyDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    active_topology: str = "linear",
    topology_history: list[str] | None = None,
    topology_started_at_iter: int = 0,
    topology_switch_count: int = 0,
    iter_total: int = 10,
    phase: Phase = Phase.EXECUTION,
    phase_started_at_iter: int = 0,
    signals: dict[str, object] | None = None,
) -> SharedState:
    """Build minimal SharedState for guard tests."""
    return {
        "active_topology": active_topology,
        "topology_history": topology_history if topology_history is not None else [active_topology],
        "topology_started_at_iter": topology_started_at_iter,
        "topology_switch_count": topology_switch_count,
        "iter_total": iter_total,
        "phase": phase,
        "phase_started_at_iter": phase_started_at_iter,
        "signals": signals if signals is not None else {},
    }


def _make_switch_decision(topology: str = "mesh", cost_usd: float = 0.0) -> TopologyDecision:
    """Build a TopologyDecision proposing a switch to the given topology."""
    return TopologyDecision(
        topology=topology,
        reason="inner router decision",
        decided_by="rule",
        router_cost_usd=cost_usd,
    )


class _AlwaysSwitchRouter:
    """Test double: always proposes switching to 'mesh' with given cost."""

    def __init__(self, target: str = "mesh", cost_usd: float = 0.0) -> None:
        self._target = target
        self._cost = cost_usd

    async def decide(self, state: SharedState) -> TopologyDecision:
        return TopologyDecision(
            topology=self._target,
            reason="always switch",
            decided_by="rule",
            router_cost_usd=self._cost,
        )


class _AlwaysStayRouter:
    """Test double: always proposes staying with current topology."""

    async def decide(self, state: SharedState) -> TopologyDecision:
        topology = state.get("active_topology") or "linear"
        return TopologyDecision(
            topology=topology,
            reason="always stay",
            decided_by="rule",
        )


# ===========================================================================
# TestSwitchGuards — configuration model
# ===========================================================================


class TestSwitchGuards:
    """Tests for SwitchGuards Pydantic model."""

    def test_default_values(self) -> None:
        """SwitchGuards has correct default values from arch.md §8bis.3."""
        from atm.phases.guards import SwitchGuards

        guards = SwitchGuards()
        assert guards.min_dwell_iters == 2
        assert guards.cooldown_iters == 3
        assert guards.max_per_run == 8
        assert guards.max_per_phase == 4

    def test_custom_values(self) -> None:
        """SwitchGuards accepts custom values."""
        from atm.phases.guards import SwitchGuards

        guards = SwitchGuards(
            min_dwell_iters=5,
            cooldown_iters=10,
            max_per_run=20,
            max_per_phase=8,
        )
        assert guards.min_dwell_iters == 5
        assert guards.cooldown_iters == 10
        assert guards.max_per_run == 20
        assert guards.max_per_phase == 8

    def test_frozen_immutable(self) -> None:
        """SwitchGuards is immutable (frozen Pydantic model)."""
        import pydantic

        from atm.phases.guards import SwitchGuards

        guards = SwitchGuards()
        with pytest.raises((TypeError, pydantic.ValidationError)):
            guards.min_dwell_iters = 99  # type: ignore[misc]


# ===========================================================================
# TestGuardedRouter — individual guard behaviors
# ===========================================================================


class TestGuardedRouter:
    """Tests for GuardedRouter.decide() -- 4 guards x trigger/no-trigger."""

    # -----------------------------------------------------------------------
    # Guard 1: min_dwell — topology must live >= N ticks
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_min_dwell_triggers_blocks_switch(self) -> None:
        """min_dwell guard blocks switch when topology has not dwelled long enough."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=5, cooldown_iters=0, max_per_run=99, max_per_phase=99)
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        # iter_within_topology = iter_total - topology_started = 10 - 8 = 2 < 5
        state = _make_state(
            active_topology="linear",
            iter_total=10,
            topology_started_at_iter=8,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        assert decision.topology == "linear"  # stayed
        assert "mesh" in decision.considered_alternatives
        assert "min_dwell" in decision.reason

    @pytest.mark.asyncio
    async def test_min_dwell_no_trigger_allows_switch(self) -> None:
        """min_dwell guard does NOT block when dwell >= threshold."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=2, cooldown_iters=0, max_per_run=99, max_per_phase=99)
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        # dwell = 10 - 0 = 10 >= 2 → passes min_dwell
        state = _make_state(
            active_topology="linear",
            iter_total=10,
            topology_started_at_iter=0,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.topology == "mesh"

    # -----------------------------------------------------------------------
    # Guard 2: cooldown — cannot return to recently abandoned topology
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cooldown_triggers_blocks_return(self) -> None:
        """cooldown guard blocks switching back to a recently abandoned topology."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=0, cooldown_iters=3, max_per_run=99, max_per_phase=99)
        # Inner wants to switch to "mesh"; "mesh" appears in recent history
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        # history shows mesh was recently used (within cooldown window)
        state = _make_state(
            active_topology="linear",
            topology_history=["mesh", "debate", "linear"],  # mesh was recent
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        assert decision.topology == "linear"
        assert "mesh" in decision.considered_alternatives
        assert "cooldown" in decision.reason

    @pytest.mark.asyncio
    async def test_cooldown_no_trigger_when_not_recent(self) -> None:
        """cooldown guard does NOT block when topology was not recently used."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=0, cooldown_iters=2, max_per_run=99, max_per_phase=99)
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        # "mesh" is NOT in recent history (only "debate" and "linear")
        state = _make_state(
            active_topology="linear",
            topology_history=["supervisor", "debate", "linear"],
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.topology == "mesh"

    # -----------------------------------------------------------------------
    # Guard 3: max_per_run — total switch cap across run
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_max_per_run_triggers_blocks_switch(self) -> None:
        """max_per_run guard blocks switch when per-run cap reached."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=0, cooldown_iters=0, max_per_run=3, max_per_phase=99)
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        # switch_count = 3 >= max_per_run = 3
        state = _make_state(
            active_topology="linear",
            topology_switch_count=3,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        assert decision.topology == "linear"
        assert "mesh" in decision.considered_alternatives
        assert "max_per_run" in decision.reason

    @pytest.mark.asyncio
    async def test_max_per_run_no_trigger_when_under_cap(self) -> None:
        """max_per_run guard does NOT block when under the cap."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=0, cooldown_iters=0, max_per_run=5, max_per_phase=99)
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        # switch_count = 2 < max_per_run = 5
        state = _make_state(
            active_topology="linear",
            topology_switch_count=2,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.topology == "mesh"

    # -----------------------------------------------------------------------
    # Guard 4: max_per_phase — switch cap within current phase
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_max_per_phase_triggers_via_signals(self) -> None:
        """max_per_phase guard blocks when phase_switch_count in signals >= cap."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=0, cooldown_iters=0, max_per_run=99, max_per_phase=2)
        inner = _AlwaysSwitchRouter(target="debate")
        router = GuardedRouter(inner=inner, guards=guards)

        # phase_switch_count = 2 >= max_per_phase = 2
        state = _make_state(
            active_topology="linear",
            signals={"phase_switch_count": 2},
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        assert decision.topology == "linear"
        assert "debate" in decision.considered_alternatives
        assert "max_per_phase" in decision.reason

    @pytest.mark.asyncio
    async def test_max_per_phase_no_trigger_when_under_cap(self) -> None:
        """max_per_phase guard does NOT block when phase switches are under cap."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(min_dwell_iters=0, cooldown_iters=0, max_per_run=99, max_per_phase=4)
        inner = _AlwaysSwitchRouter(target="debate")
        router = GuardedRouter(inner=inner, guards=guards)

        # phase_switch_count = 1 < max_per_phase = 4
        state = _make_state(
            active_topology="linear",
            signals={"phase_switch_count": 1},
            topology_history=["linear"],  # short history → approx < cap
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.topology == "debate"

    # -----------------------------------------------------------------------
    # Guard override: considered_alternatives preserved
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_considered_alternatives_preserved_on_block(self) -> None:
        """On guard_override, considered_alternatives contains the blocked topology."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(
            min_dwell_iters=99, cooldown_iters=0, max_per_run=99, max_per_phase=99
        )
        inner = _AlwaysSwitchRouter(target="hierarchical")
        router = GuardedRouter(inner=inner, guards=guards)

        state = _make_state(active_topology="linear", iter_total=5, topology_started_at_iter=4)
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        assert "hierarchical" in decision.considered_alternatives

    # -----------------------------------------------------------------------
    # No switch proposed → inner decision returned as-is
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_switch_passes_through(self) -> None:
        """When inner proposes no switch, GuardedRouter returns inner decision unchanged."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        # Very strict guards — but inner stays, so guards never fire
        guards = SwitchGuards(min_dwell_iters=99, cooldown_iters=99, max_per_run=0, max_per_phase=0)
        inner = _AlwaysStayRouter()
        router = GuardedRouter(inner=inner, guards=guards)

        state = _make_state(active_topology="linear")
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "rule"
        assert decision.topology == "linear"
        # No guard_override since no switch was proposed
        assert "guard_override" not in decision.decided_by

    # -----------------------------------------------------------------------
    # Cost passthrough on guard_override
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cost_passes_through_on_guard_override(self) -> None:
        """router_cost_usd from inner (LLM cost) is preserved even on guard_override."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(
            min_dwell_iters=99, cooldown_iters=0, max_per_run=99, max_per_phase=99
        )
        # Inner is an LLM router that cost 0.05 USD but gets blocked by min_dwell
        inner = _AlwaysSwitchRouter(target="mesh", cost_usd=0.05)
        router = GuardedRouter(inner=inner, guards=guards)

        state = _make_state(active_topology="linear", iter_total=5, topology_started_at_iter=4)
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        assert decision.router_cost_usd == pytest.approx(0.05)

    # -----------------------------------------------------------------------
    # Multiple guards fire — all listed in reason
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_multiple_guards_fire_all_in_reason(self) -> None:
        """When multiple guards fire simultaneously, all are reflected in reason."""
        from atm.phases.guards import GuardedRouter, SwitchGuards

        guards = SwitchGuards(
            min_dwell_iters=10,  # fires: dwell=2 < 10
            cooldown_iters=5,  # fires: mesh in recent history
            max_per_run=1,  # fires: switch_count=1 >= 1
            max_per_phase=0,  # fires: phase_switch_count=0 >= 0... depends on impl
        )
        inner = _AlwaysSwitchRouter(target="mesh")
        router = GuardedRouter(inner=inner, guards=guards)

        state = _make_state(
            active_topology="linear",
            topology_history=["mesh", "debate", "linear"],
            topology_switch_count=1,
            iter_total=5,
            topology_started_at_iter=3,
        )
        decision: TopologyDecision = await router.decide(state)

        assert decision.decided_by == "guard_override"
        # At least min_dwell and cooldown and max_per_run should fire
        assert "min_dwell" in decision.reason
        assert "cooldown" in decision.reason
        assert "max_per_run" in decision.reason
