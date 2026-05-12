"""Tests for HumanRoleRouter Protocol and FixedRoleRouter implementation.

Coverage (step 1.1 — fixed subset):
  - Protocol shape: HumanRoleRouter is runtime_checkable and has async decide().
  - FixedRoleRouter.decide() returns the configured role regardless of phase/state.
  - FixedRoleRouter satisfies isinstance(x, HumanRoleRouter) check.
  - FixedRoleRouter works for every HumanRole enum value.
  - Public __init__ exports are present.
"""

from __future__ import annotations

import inspect

import pytest

from atm.core.types import HumanRole, Phase
from atm.human.role_router import FixedRoleRouter, HumanRoleRouter

# ---------------------------------------------------------------------------
# Protocol shape tests
# ---------------------------------------------------------------------------


class TestHumanRoleRouterProtocol:
    """Verify the structural contract of HumanRoleRouter."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """isinstance() must not raise TypeError — protocol must be @runtime_checkable."""
        # This would raise TypeError if not @runtime_checkable
        result = isinstance(object(), HumanRoleRouter)
        assert isinstance(result, bool)

    def test_protocol_has_decide_method(self) -> None:
        """HumanRoleRouter must expose an async 'decide' method."""
        assert hasattr(HumanRoleRouter, "decide")
        # The method should be in the protocol's own annotations/members
        members = {name for name, _ in inspect.getmembers(HumanRoleRouter)}
        assert "decide" in members

    def test_decide_is_coroutinefunction_on_fixed(self) -> None:
        """decide() on a concrete impl must be an async method."""
        router = FixedRoleRouter(role=HumanRole.REVIEWER)
        assert inspect.iscoroutinefunction(router.decide)


# ---------------------------------------------------------------------------
# FixedRoleRouter — behavioural tests
# ---------------------------------------------------------------------------


class TestFixedRoleRouter:
    """Verify FixedRoleRouter returns the configured role for all inputs."""

    @pytest.mark.asyncio
    async def test_fixed_returns_configured_role(self) -> None:
        """decide() should return the role passed at construction."""
        router = FixedRoleRouter(role=HumanRole.REVIEWER)
        result = await router.decide(Phase.EXECUTION, {})
        assert result == HumanRole.REVIEWER

    @pytest.mark.asyncio
    async def test_fixed_ignores_phase(self) -> None:
        """decide() must return the same role regardless of which Phase is passed."""
        router = FixedRoleRouter(role=HumanRole.COORDINATOR)
        for phase in Phase:
            result = await router.decide(phase, {})
            assert result == HumanRole.COORDINATOR, f"Expected COORDINATOR for phase {phase}"

    @pytest.mark.asyncio
    async def test_fixed_ignores_state(self) -> None:
        """decide() must return the same role regardless of state dict content."""
        router = FixedRoleRouter(role=HumanRole.JUDGE)
        states: list[dict] = [
            {},
            {"iter_total": 5, "phase": "execution"},
            {"some_key": [1, 2, 3]},
        ]
        for state in states:
            result = await router.decide(Phase.PLANNING, state)
            assert result == HumanRole.JUDGE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("role", list(HumanRole))
    async def test_fixed_all_human_roles(self, role: HumanRole) -> None:
        """FixedRoleRouter must work correctly for every HumanRole enum value."""
        router = FixedRoleRouter(role=role)
        result = await router.decide(Phase.VERIFICATION, {})
        assert result == role

    def test_fixed_satisfies_protocol(self) -> None:
        """FixedRoleRouter instance must pass isinstance check against HumanRoleRouter."""
        router = FixedRoleRouter(role=HumanRole.PEER)
        assert isinstance(router, HumanRoleRouter)

    def test_fixed_role_stored_as_attribute(self) -> None:
        """The configured role should be accessible as router.role."""
        router = FixedRoleRouter(role=HumanRole.MONITOR)
        assert router.role == HumanRole.MONITOR


# ---------------------------------------------------------------------------
# Public API exports test
# ---------------------------------------------------------------------------


class TestPublicExports:
    """Verify that atm.human.__init__ re-exports the required symbols."""

    def test_human_role_router_exported(self) -> None:
        import atm.human as human_pkg

        assert hasattr(human_pkg, "HumanRoleRouter")

    def test_fixed_role_router_exported(self) -> None:
        import atm.human as human_pkg

        assert hasattr(human_pkg, "FixedRoleRouter")
