"""Unit tests for atm.topology.base — Protocol, Registry, _should_stop helper.

Tests cover:
  - _should_stop precedence: max_iter > topology_success > topology_max > continue
  - Budget detection is NOT done here (budget raises from LLMWrapper)
  - TopologyRegistry: register, lookup, validation of build/name
  - Topology Protocol: runtime_checkable isinstance check
  - Reason strings match FinishReason.value constants
"""

from __future__ import annotations

import importlib

import pytest

from atm.storage.models import FinishReason
from atm.topology.base import (
    Topology,
    TopologyConfig,
    TopologyRegistry,
    _should_stop,
)

# ---------------------------------------------------------------------------
# Helpers — minimal fake state and config for _should_stop
# ---------------------------------------------------------------------------


def _make_state(*, iter_total: int = 0) -> dict:
    """Minimal GraphState-like dict with shared.iter_total."""
    return {"shared": {"iter_total": iter_total}}


def _make_cfg(*, max_iterations: int = 10) -> TopologyConfig:
    return TopologyConfig(name="test", max_iterations=max_iterations)


# ---------------------------------------------------------------------------
# Test group 1: _should_stop — max_iter branch (highest priority)
# ---------------------------------------------------------------------------


class TestShouldStopMaxIter:
    """max_iter takes priority over topology_success and topology_max."""

    def test_max_iter_reached_returns_stop_true(self) -> None:
        state = _make_state(iter_total=10)
        cfg = _make_cfg(max_iterations=10)
        stop, _reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert stop is True

    def test_max_iter_reason_matches_finish_reason(self) -> None:
        state = _make_state(iter_total=10)
        cfg = _make_cfg(max_iterations=10)
        _, reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert reason == FinishReason.MAX_ITER.value

    def test_max_iter_overrides_topology_success(self) -> None:
        """Even if topology_success=True, max_iter wins when iter_total >= max_iterations."""
        state = _make_state(iter_total=10)
        cfg = _make_cfg(max_iterations=10)
        stop, reason = _should_stop(state, cfg, topology_success=True, topology_max_reached=True)
        assert stop is True
        assert reason == FinishReason.MAX_ITER.value

    def test_max_iter_not_reached_does_not_stop(self) -> None:
        """iter_total < max_iterations → max_iter branch not triggered."""
        state = _make_state(iter_total=9)
        cfg = _make_cfg(max_iterations=10)
        stop, _ = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert stop is False

    def test_max_iter_zero_triggers_immediately(self) -> None:
        """max_iterations=0 → stop on iter_total=0."""
        state = _make_state(iter_total=0)
        cfg = _make_cfg(max_iterations=0)
        stop, reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert stop is True
        assert reason == FinishReason.MAX_ITER.value


# ---------------------------------------------------------------------------
# Test group 2: _should_stop — topology_success branch (second priority)
# ---------------------------------------------------------------------------


class TestShouldStopTopologySuccess:
    """topology_success=True → SUCCESS reason when max_iter not reached."""

    def test_topology_success_returns_stop_true(self) -> None:
        state = _make_state(iter_total=3)
        cfg = _make_cfg(max_iterations=10)
        stop, _reason = _should_stop(state, cfg, topology_success=True, topology_max_reached=False)
        assert stop is True

    def test_topology_success_reason_matches_finish_reason(self) -> None:
        state = _make_state(iter_total=3)
        cfg = _make_cfg(max_iterations=10)
        _, reason = _should_stop(state, cfg, topology_success=True, topology_max_reached=False)
        assert reason == FinishReason.SUCCESS.value

    def test_topology_success_overrides_topology_max(self) -> None:
        """topology_success has higher priority than topology_max_reached."""
        state = _make_state(iter_total=3)
        cfg = _make_cfg(max_iterations=10)
        stop, reason = _should_stop(state, cfg, topology_success=True, topology_max_reached=True)
        assert stop is True
        assert reason == FinishReason.SUCCESS.value


# ---------------------------------------------------------------------------
# Test group 3: _should_stop — topology_max branch (third priority)
# ---------------------------------------------------------------------------


class TestShouldStopTopologyMax:
    """topology_max_reached=True (and success=False, max_iter not hit) → TOPOLOGY_MAX."""

    def test_topology_max_returns_stop_true(self) -> None:
        state = _make_state(iter_total=3)
        cfg = _make_cfg(max_iterations=10)
        stop, _reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=True)
        assert stop is True

    def test_topology_max_reason_matches_finish_reason(self) -> None:
        state = _make_state(iter_total=3)
        cfg = _make_cfg(max_iterations=10)
        _, reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=True)
        assert reason == FinishReason.TOPOLOGY_MAX.value


# ---------------------------------------------------------------------------
# Test group 4: _should_stop — continue branch (no stop)
# ---------------------------------------------------------------------------


class TestShouldStopContinue:
    """When none of the stop conditions are met → (False, '')."""

    def test_no_conditions_returns_false(self) -> None:
        state = _make_state(iter_total=3)
        cfg = _make_cfg(max_iterations=10)
        stop, reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert stop is False
        assert reason == ""

    def test_continue_reason_is_empty_string(self) -> None:
        state = _make_state(iter_total=0)
        cfg = _make_cfg(max_iterations=100)
        stop, reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert not stop
        assert reason == ""


# ---------------------------------------------------------------------------
# Test group 5: Budget NOT detected in _should_stop
# ---------------------------------------------------------------------------


class TestBudgetNotDetectedHere:
    """Budget is detected in LLMWrapper (raises BudgetExceededError).
    _should_stop does NOT inspect budget fields — it never returns budget_exceeded."""

    def test_should_stop_never_returns_budget_exceeded(self) -> None:
        """Even with very high iter_total that might hint budget, reason is max_iter."""
        state = _make_state(iter_total=999)
        cfg = _make_cfg(max_iterations=5)
        stop, reason = _should_stop(state, cfg, topology_success=False, topology_max_reached=False)
        assert stop is True
        assert reason != FinishReason.BUDGET_EXCEEDED.value
        assert reason == FinishReason.MAX_ITER.value


# ---------------------------------------------------------------------------
# Test group 6: TopologyConfig validation
# ---------------------------------------------------------------------------


class TestTopologyConfig:
    """TopologyConfig is a Pydantic model with name and max_iterations."""

    def test_create_minimal(self) -> None:
        cfg = TopologyConfig(name="star", max_iterations=20)
        assert cfg.name == "star"
        assert cfg.max_iterations == 20

    def test_extra_field(self) -> None:
        cfg = TopologyConfig(name="chain", max_iterations=12, extra={})
        assert cfg.extra == {}

    def test_extra_with_values(self) -> None:
        cfg = TopologyConfig(
            name="star",
            max_iterations=20,
            extra={"planning_max_iter": 2, "verify_max_iter": 3},
        )
        assert cfg.extra["planning_max_iter"] == 2


# ---------------------------------------------------------------------------
# Test group 7: TopologyRegistry — register and lookup
# ---------------------------------------------------------------------------


class TestTopologyRegistry:
    """Registry stores and retrieves topology classes by name."""

    def setup_method(self) -> None:
        """Clear registry before each test to avoid state leakage."""
        TopologyRegistry._registry.clear()

    def teardown_method(self) -> None:
        """Restore registry to original state after each test."""
        TopologyRegistry._registry.clear()

    def test_register_and_get(self) -> None:
        class FakeStar:
            name = "star"

            def build(self, agents: dict, cfg: TopologyConfig) -> None: ...

        TopologyRegistry.register("star")(FakeStar)
        result = TopologyRegistry.get("star")
        assert result is FakeStar

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            TopologyRegistry.get("unknown_topology")

    def test_register_validates_build_attribute(self) -> None:
        """Registry must reject a class without a callable 'build' attribute."""

        class NoBuild:
            name = "no_build"

        with pytest.raises((TypeError, AttributeError, ValueError)):
            TopologyRegistry.register("no_build")(NoBuild)

    def test_register_validates_name_attribute(self) -> None:
        """Registry must reject a class without a 'name' attribute."""

        class NoName:
            def build(self, agents: dict, cfg: TopologyConfig) -> None: ...

        with pytest.raises((TypeError, AttributeError, ValueError)):
            TopologyRegistry.register("no_name")(NoName)

    def test_list_names_returns_registered(self) -> None:
        class FakeChain:
            name = "chain"

            def build(self, agents: dict, cfg: TopologyConfig) -> None: ...

        TopologyRegistry.register("chain")(FakeChain)
        names = TopologyRegistry.list_names()
        assert "chain" in names

    def test_decorator_returns_class_unchanged(self) -> None:
        """@TopologyRegistry.register must return the original class."""

        class FakeTopology:
            name = "fake"

            def build(self, agents: dict, cfg: TopologyConfig) -> None: ...

        result = TopologyRegistry.register("fake")(FakeTopology)
        assert result is FakeTopology


# ---------------------------------------------------------------------------
# Test group 8: Topology Protocol — isinstance check (runtime_checkable)
# ---------------------------------------------------------------------------


class TestTopologyProtocol:
    """Topology is a @runtime_checkable Protocol with name: str and build()."""

    def test_conforming_class_isinstance(self) -> None:
        class GoodTopology:
            name = "good"

            def build(self, agents: dict, cfg: TopologyConfig) -> None: ...

        assert isinstance(GoodTopology(), Topology)

    def test_missing_name_not_isinstance(self) -> None:
        class NoNameTopology:
            def build(self, agents: dict, cfg: TopologyConfig) -> None: ...

        assert not isinstance(NoNameTopology(), Topology)

    def test_missing_build_not_isinstance(self) -> None:
        class NoBuildTopology:
            name = "no_build"

        assert not isinstance(NoBuildTopology(), Topology)


# ---------------------------------------------------------------------------
# Test group 9: Public API — topology/__init__.py exports
# ---------------------------------------------------------------------------


class TestPublicApi:
    """Verify that the public symbols are importable from atm.topology."""

    def test_import_topology(self) -> None:
        from atm.topology import Topology  # noqa: F401

    def test_import_topology_registry(self) -> None:
        from atm.topology import TopologyRegistry  # noqa: F401

    def test_import_should_stop(self) -> None:
        from atm.topology import _should_stop  # noqa: F401

    def test_import_topology_config(self) -> None:
        from atm.topology import TopologyConfig  # noqa: F401

    def test_star_chain_import_guards_do_not_raise(self) -> None:
        """Importing atm.topology must not raise even without star.py / chain.py."""
        import atm.topology as mod

        importlib.reload(mod)
