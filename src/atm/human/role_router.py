"""HumanRoleRouter Protocol and concrete implementations.

Public API:
  - HumanRoleRouter      — @runtime_checkable Protocol; async decide(phase, state) -> HumanRole
  - FixedRoleRouter      — returns a constant HumanRole regardless of phase/state (back-compat default)
  - RuleBasedRoleRouter  — table-driven Phase → HumanRole lookup with YAML support

Pattern mirrors the Topology Protocol in atm.topology.base with async decide semantics.

Design notes (m9.2-plan.md §Approach):
  - _build_role_router returns None for role_router="fixed" (short-circuit, step 4.1).
    FixedRoleRouter is kept as an explicit class for test-doubles and explicit usage.
  - RuleBasedRoleRouter (step 2.1) and LLMRoleRouter (step 3.1) are added in later waves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

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


# ---------------------------------------------------------------------------
# RuleBasedRoleRouter — table-driven Phase → HumanRole implementation
# ---------------------------------------------------------------------------

#: Default mapping used when no custom table is provided.
DEFAULT_ROLE_TABLE: dict[Phase, HumanRole] = {
    Phase.PLANNING: HumanRole.COORDINATOR,
    Phase.EXECUTION: HumanRole.PEER,
    Phase.VERIFICATION: HumanRole.REVIEWER,
    Phase.DONE: HumanRole.REVIEWER,
}


class RuleBasedRoleRouter:
    """Role router driven by a static Phase → HumanRole lookup table.

    Uses DEFAULT_ROLE_TABLE when no custom table is provided. Unknown phases
    fall back to the configured ``fallback`` role.

    Satisfies HumanRoleRouter Protocol via structural subtyping.

    Example::

        router = RuleBasedRoleRouter()
        role = await router.decide(Phase.PLANNING, {})
        # role == HumanRole.COORDINATOR

        router = RuleBasedRoleRouter.from_yaml("conf/human/role_table.yaml")

    Args:
        table:    Optional override mapping. Values are normalised to
                  ``dict[Phase, HumanRole]`` at construction time.
        fallback: Role returned when *phase* is absent from *table*.
                  Defaults to ``HumanRole.REVIEWER``.
    """

    def __init__(
        self,
        table: dict[Phase, HumanRole] | None = None,
        fallback: HumanRole = HumanRole.REVIEWER,
    ) -> None:
        self._table: dict[Phase, HumanRole] = dict(table) if table is not None else dict(DEFAULT_ROLE_TABLE)
        self._fallback = fallback

    async def decide(self, phase: Phase, state: dict[str, Any]) -> HumanRole:
        """Return the HumanRole for *phase*, or *fallback* if not in table.

        Args:
            phase: Current execution phase of the run.
            state: SharedState-compatible dict (unused by this implementation).

        Returns:
            Mapped HumanRole or the fallback role.
        """
        return self._table.get(phase, self._fallback)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuleBasedRoleRouter:
        """Build a RuleBasedRoleRouter from a YAML mapping file.

        The YAML file must contain a flat mapping of phase strings to role
        strings, e.g.::

            planning: coordinator
            execution: peer
            verification: reviewer
            done: reviewer

        Args:
            path: Path to the YAML file.

        Returns:
            A new RuleBasedRoleRouter with the table read from the file.

        Raises:
            ValueError: If the file contains an unknown phase or role string.
        """
        path = Path(path)
        raw: dict[str, str] = yaml.safe_load(path.read_text())

        table: dict[Phase, HumanRole] = {}
        for phase_str, role_str in raw.items():
            try:
                phase = Phase(phase_str)
            except ValueError:
                valid = [p.value for p in Phase]
                raise ValueError(
                    f"Unknown phase {phase_str!r} in {path}. Valid values: {valid}"
                ) from None
            try:
                role = HumanRole(role_str)
            except ValueError:
                valid = [r.value for r in HumanRole]
                raise ValueError(
                    f"Unknown role {role_str!r} for phase {phase_str!r} in {path}. Valid values: {valid}"
                ) from None
            table[phase] = role

        return cls(table=table)
