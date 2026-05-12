"""Tests for HumanRoleRouter Protocol and FixedRoleRouter / RuleBasedRoleRouter implementations.

Coverage (step 1.1 — fixed subset):
  - Protocol shape: HumanRoleRouter is runtime_checkable and has async decide().
  - FixedRoleRouter.decide() returns the configured role regardless of phase/state.
  - FixedRoleRouter satisfies isinstance(x, HumanRoleRouter) check.
  - FixedRoleRouter works for every HumanRole enum value.
  - Public __init__ exports are present.

Coverage (step 2.1 — rule subset):
  - DEFAULT_ROLE_TABLE maps all four Phase values to expected roles.
  - RuleBasedRoleRouter with default table returns correct role per phase.
  - Explicit override table: returns the overridden role.
  - Fallback role used when phase is missing from a custom table.
  - from_yaml() with conf/human/role_table.yaml builds the default mapping.
  - from_yaml() with unknown phase raises ValueError.
  - from_yaml() with unknown role raises ValueError.
  - isinstance(rule_router, HumanRoleRouter) is True.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from atm.core.types import HumanRole, Phase
from atm.human.role_router import (
    DEFAULT_ROLE_TABLE,
    FixedRoleRouter,
    HumanRoleRouter,
    RuleBasedRoleRouter,
)

# Absolute path to the default role_table YAML shipped with the project.
_CONF_ROLE_TABLE = Path(__file__).parent.parent.parent.parent / "conf" / "human" / "role_table.yaml"

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

    def test_rule_based_role_router_exported(self) -> None:
        import atm.human as human_pkg

        assert hasattr(human_pkg, "RuleBasedRoleRouter")


# ---------------------------------------------------------------------------
# DEFAULT_ROLE_TABLE constant
# ---------------------------------------------------------------------------


class TestDefaultRoleTable:
    """Verify DEFAULT_ROLE_TABLE maps every Phase to the expected role."""

    def test_planning_maps_to_coordinator(self) -> None:
        assert DEFAULT_ROLE_TABLE[Phase.PLANNING] == HumanRole.COORDINATOR

    def test_execution_maps_to_peer(self) -> None:
        assert DEFAULT_ROLE_TABLE[Phase.EXECUTION] == HumanRole.PEER

    def test_verification_maps_to_reviewer(self) -> None:
        assert DEFAULT_ROLE_TABLE[Phase.VERIFICATION] == HumanRole.REVIEWER

    def test_done_maps_to_reviewer(self) -> None:
        assert DEFAULT_ROLE_TABLE[Phase.DONE] == HumanRole.REVIEWER

    def test_all_phases_covered(self) -> None:
        """DEFAULT_ROLE_TABLE must cover all four Phase values."""
        assert set(DEFAULT_ROLE_TABLE.keys()) == set(Phase)


# ---------------------------------------------------------------------------
# RuleBasedRoleRouter — behavioural tests
# ---------------------------------------------------------------------------


class TestRuleBasedRoleRouter:
    """Verify RuleBasedRoleRouter returns correct roles from the table."""

    @pytest.mark.asyncio
    async def test_rule_default_table_planning(self) -> None:
        router = RuleBasedRoleRouter()
        result = await router.decide(Phase.PLANNING, {})
        assert result == HumanRole.COORDINATOR

    @pytest.mark.asyncio
    async def test_rule_default_table_execution(self) -> None:
        router = RuleBasedRoleRouter()
        result = await router.decide(Phase.EXECUTION, {})
        assert result == HumanRole.PEER

    @pytest.mark.asyncio
    async def test_rule_default_table_verification(self) -> None:
        router = RuleBasedRoleRouter()
        result = await router.decide(Phase.VERIFICATION, {})
        assert result == HumanRole.REVIEWER

    @pytest.mark.asyncio
    async def test_rule_default_table_done(self) -> None:
        router = RuleBasedRoleRouter()
        result = await router.decide(Phase.DONE, {})
        assert result == HumanRole.REVIEWER

    @pytest.mark.asyncio
    async def test_rule_override_table(self) -> None:
        """Custom table should override the default mapping."""
        custom = {Phase.PLANNING: HumanRole.JUDGE, Phase.EXECUTION: HumanRole.MONITOR}
        router = RuleBasedRoleRouter(table=custom)
        assert await router.decide(Phase.PLANNING, {}) == HumanRole.JUDGE
        assert await router.decide(Phase.EXECUTION, {}) == HumanRole.MONITOR

    @pytest.mark.asyncio
    async def test_rule_fallback_when_phase_missing(self) -> None:
        """Phases absent from custom table must return the fallback role."""
        partial_table = {Phase.PLANNING: HumanRole.COORDINATOR}
        router = RuleBasedRoleRouter(table=partial_table, fallback=HumanRole.MONITOR)
        # EXECUTION is not in partial_table → fallback
        result = await router.decide(Phase.EXECUTION, {})
        assert result == HumanRole.MONITOR

    @pytest.mark.asyncio
    async def test_rule_default_fallback_is_reviewer(self) -> None:
        """Default fallback role must be HumanRole.REVIEWER."""
        router = RuleBasedRoleRouter(table={})
        result = await router.decide(Phase.PLANNING, {})
        assert result == HumanRole.REVIEWER

    def test_rule_satisfies_protocol(self) -> None:
        """RuleBasedRoleRouter instance must pass isinstance check against HumanRoleRouter."""
        router = RuleBasedRoleRouter()
        assert isinstance(router, HumanRoleRouter)

    def test_rule_decide_is_coroutine(self) -> None:
        """decide() must be an async method."""
        router = RuleBasedRoleRouter()
        assert inspect.iscoroutinefunction(router.decide)

    def test_rule_table_is_independent_copy(self) -> None:
        """Mutating the input dict after construction must not affect the router."""
        original: dict[Phase, HumanRole] = {Phase.PLANNING: HumanRole.COORDINATOR}
        router = RuleBasedRoleRouter(table=original)
        original[Phase.PLANNING] = HumanRole.JUDGE
        # Router should still have the original value
        assert router._table[Phase.PLANNING] == HumanRole.COORDINATOR


# ---------------------------------------------------------------------------
# RuleBasedRoleRouter.from_yaml
# ---------------------------------------------------------------------------


class TestRuleBasedRoleRouterFromYaml:
    """Verify from_yaml() correctly loads and validates the YAML config."""

    def test_from_yaml_builds_default_mapping(self) -> None:
        """from_yaml() with the shipped role_table.yaml should match DEFAULT_ROLE_TABLE."""
        router = RuleBasedRoleRouter.from_yaml(_CONF_ROLE_TABLE)
        assert router._table == DEFAULT_ROLE_TABLE

    @pytest.mark.asyncio
    async def test_from_yaml_decide_planning(self) -> None:
        router = RuleBasedRoleRouter.from_yaml(_CONF_ROLE_TABLE)
        assert await router.decide(Phase.PLANNING, {}) == HumanRole.COORDINATOR

    @pytest.mark.asyncio
    async def test_from_yaml_decide_execution(self) -> None:
        router = RuleBasedRoleRouter.from_yaml(_CONF_ROLE_TABLE)
        assert await router.decide(Phase.EXECUTION, {}) == HumanRole.PEER

    def test_from_yaml_unknown_phase_raises_value_error(self, tmp_path: Path) -> None:
        """An unknown phase string in the YAML must raise ValueError."""
        bad_yaml = tmp_path / "bad_phase.yaml"
        bad_yaml.write_text("invalid_phase: reviewer\n")
        with pytest.raises(ValueError, match="Unknown phase"):
            RuleBasedRoleRouter.from_yaml(bad_yaml)

    def test_from_yaml_unknown_role_raises_value_error(self, tmp_path: Path) -> None:
        """An unknown role string in the YAML must raise ValueError."""
        bad_yaml = tmp_path / "bad_role.yaml"
        bad_yaml.write_text("planning: superadmin\n")
        with pytest.raises(ValueError, match="Unknown role"):
            RuleBasedRoleRouter.from_yaml(bad_yaml)
