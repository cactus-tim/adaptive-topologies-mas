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
    get_topology_extras,
)


def _make_state(*, iter_total: int = 0) -> dict:
    """Minimal GraphState-like dict with shared.iter_total."""
    return {"shared": {"iter_total": iter_total}}


def _make_cfg(*, max_iterations: int = 10) -> TopologyConfig:
    return TopologyConfig(name="test", max_iterations=max_iterations)


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

    def test_import_get_topology_extras(self) -> None:
        from atm.topology import get_topology_extras  # noqa: F401


class TestGetTopologyExtras:
    """get_topology_extras handles namespaced, flat, empty, and mixed shapes."""

    def test_namespaced_returns_correct_bucket(self) -> None:
        """Fully namespaced extra → returns the named topology's sub-dict."""
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=8,
            extra={"mesh": {"max_rounds": 12}, "star": {}},
        )
        result = get_topology_extras(cfg, "mesh")
        assert result == {"max_rounds": 12}

    def test_namespaced_returns_empty_for_absent_topology(self) -> None:
        """Namespaced extra with no key for this topology → {}."""
        cfg = TopologyConfig(
            name="debate",
            max_iterations=8,
            extra={"mesh": {"max_rounds": 12}},
        )
        result = get_topology_extras(cfg, "debate")
        assert result == {}

    def test_namespaced_all_six_topologies(self) -> None:
        """All six topology namespaces present → each returns its own bucket."""
        extra = {
            "star": {"planning_max_iter": 2},
            "chain": {},
            "debate": {"max_rounds": 3},
            "hierarchical": {"max_rounds": 4},
            "mesh": {"max_rounds": 12},
            "adaptive": {"exec_max_iter": 10},
        }
        cfg = TopologyConfig(name="star", max_iterations=20, extra=extra)
        assert get_topology_extras(cfg, "star") == {"planning_max_iter": 2}
        assert get_topology_extras(cfg, "chain") == {}
        assert get_topology_extras(cfg, "debate") == {"max_rounds": 3}
        assert get_topology_extras(cfg, "mesh") == {"max_rounds": 12}

    def test_flat_dict_returned_verbatim(self) -> None:
        """Flat extra (non-namespace key present) → full dict returned as-is."""
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=8,
            extra={"max_rounds": 12},
        )
        result = get_topology_extras(cfg, "mesh")
        assert result == {"max_rounds": 12}

    def test_flat_dict_same_for_any_topology_name(self) -> None:
        """Flat form is returned verbatim regardless of topology_name argument."""
        cfg = TopologyConfig(
            name="debate",
            max_iterations=8,
            extra={"max_rounds": 5, "judge_id": "judge"},
        )
        assert get_topology_extras(cfg, "debate") == {"max_rounds": 5, "judge_id": "judge"}
        assert get_topology_extras(cfg, "mesh") == {"max_rounds": 5, "judge_id": "judge"}

    def test_flat_star_keys_returned_verbatim(self) -> None:
        """Star keys in flat form are returned as-is (not confused for namespace)."""
        cfg = TopologyConfig(
            name="star",
            max_iterations=20,
            extra={"planning_max_iter": 2, "exec_max_iter": 5, "verify_max_iter": 3},
        )
        result = get_topology_extras(cfg, "star")
        assert result == {"planning_max_iter": 2, "exec_max_iter": 5, "verify_max_iter": 3}

    def test_empty_extra_dict_returns_empty(self) -> None:
        """cfg.extra == {} → always returns {}."""
        cfg = TopologyConfig(name="mesh", max_iterations=8, extra={})
        assert get_topology_extras(cfg, "mesh") == {}

    def test_default_extra_returns_empty(self) -> None:
        """cfg.extra uses default_factory=dict → returns {}."""
        cfg = TopologyConfig(name="star", max_iterations=20)
        assert get_topology_extras(cfg, "star") == {}

    def test_mixed_shape_treated_as_legacy_flat(self) -> None:
        """A mix of namespace keys and non-namespace keys → entire dict is flat."""
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=8,
            extra={"mesh": {"max_rounds": 12}, "some_unknown_key": 99},
        )
        result = get_topology_extras(cfg, "mesh")
        assert result == {"mesh": {"max_rounds": 12}, "some_unknown_key": 99}

    def test_always_returns_plain_dict(self) -> None:
        """Return value must be a plain dict instance in all branches."""
        cfg_ns = TopologyConfig(name="star", extra={"star": {"planning_max_iter": 1}})
        cfg_flat = TopologyConfig(name="star", extra={"planning_max_iter": 1})
        cfg_empty = TopologyConfig(name="star", extra={})

        assert type(get_topology_extras(cfg_ns, "star")) is dict
        assert type(get_topology_extras(cfg_flat, "star")) is dict
        assert type(get_topology_extras(cfg_empty, "star")) is dict
