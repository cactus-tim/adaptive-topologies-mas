"""HumanRoleRouter Protocol and concrete implementations.

Public API:
  - HumanRoleRouter  — @runtime_checkable Protocol; async decide(phase, state) -> HumanRole
  - FixedRoleRouter  — returns a constant HumanRole regardless of phase/state (back-compat default)

Pattern mirrors the Topology Protocol in atm.topology.base with async decide semantics.

Design notes (m9.2-plan.md §Approach):
  - _build_role_router returns None for role_router="fixed" (short-circuit, step 4.1).
    FixedRoleRouter is kept as an explicit class for test-doubles and explicit usage.
  - RuleBasedRoleRouter (step 2.1) and LLMRoleRouter (step 3.1) are added in later waves.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from atm.core.types import HumanRole, Phase

# ---------------------------------------------------------------------------
# HumanRoleRouter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HumanRoleRouter(Protocol):
    """Duck-typing contract for all role router implementations.

    Concrete classes do NOT inherit from this Protocol — structural
    subtyping (same pattern as Topology in atm.topology.base).

    Contract:
      async decide(phase, state) -> HumanRole

    Args:
        phase: Current execution phase of the run.
        state: SharedState-compatible dict (may be empty).

    Returns:
        The HumanRole the topology should use for this interaction.
    """

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the HumanRole for the current phase and state."""
        ...


# ---------------------------------------------------------------------------
# FixedRoleRouter — back-compat constant-role implementation
# ---------------------------------------------------------------------------


class FixedRoleRouter:
    """Role router that always returns the same configured HumanRole.

    This is the back-compat default (m9.2-plan.md §Architecture Decisions):
    when role_router="fixed", _build_role_router returns None (topology
    uses cfg.human.role directly). FixedRoleRouter is available for
    explicit use in tests and for ad-hoc wiring.

    Satisfies HumanRoleRouter Protocol via structural subtyping.

    Example::

        router = FixedRoleRouter(role=HumanRole.REVIEWER)
        active_role = await router.decide(Phase.EXECUTION, state)
        # active_role is always HumanRole.REVIEWER

    Args:
        role: The HumanRole returned on every decide() call.
    """

    def __init__(self, role: HumanRole) -> None:
        self.role = role

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the configured role, ignoring phase and state.

        Args:
            phase: Ignored — kept for protocol compliance.
            state: Ignored — kept for protocol compliance.

        Returns:
            The HumanRole passed at construction time.
        """
        return self.role
