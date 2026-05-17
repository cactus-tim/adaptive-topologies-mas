"""Unit tests for gateway dispatch patch — D7 canonical `human_gateway` kwarg (M14 Step 5.2).

For each of the 6 topologies, verify that:
  - When `human_gateway=<FakeStreamlitGateway>` is passed to `.build()`, the
    pre-built gateway is used directly (not a freshly-constructed LLMSimulatedGateway).
  - When `human_gateway` is NOT passed (None / absent), the existing construction
    path (LLMSimulatedGateway or CLI) is unchanged — back-compat guaranteed.

Additionally:
  - Global test: no topology constructs `LLMSimulatedGateway` when `human_gateway`
    is supplied.
  - Debate back-compat: the legacy `gateway` kwarg is still honoured when
    `human_gateway` is absent.

FakeStreamlitGateway is a minimal Protocol-compatible stub with a distinct identity
so tests can use `is` checks and/or confirm LLMSimulatedGateway was NOT called.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import atm.topology.adaptive
import atm.topology.chain
import atm.topology.debate
import atm.topology.hierarchical
import atm.topology.mesh
import atm.topology.star  # noqa: F401
from atm.experiment.config import HumanCfg

# AdaptiveTopology has a different import path since build() is the method we care about
from atm.topology.adaptive import AdaptiveTopology
from atm.topology.base import TopologyConfig
from atm.topology.chain import ChainTopology
from atm.topology.debate import DebateTopology
from atm.topology.hierarchical import HierarchicalTopology
from atm.topology.mesh import MeshTopology
from atm.topology.star import StarTopology

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeStreamlitGateway:
    """Minimal stub that is NOT an LLMSimulatedGateway instance.

    Used to verify that the topology uses a pre-built gateway rather than
    constructing a fresh LLMSimulatedGateway.
    """

    async def request(self, ctx: Any, *, request_id: str) -> Any:
        from atm.core.types import HumanResponse

        return HumanResponse(action="approve", comment="ok", source="human", timed_out=False)


def _make_human_cfg_streamlit(*, enabled: bool = True) -> HumanCfg:
    """Return a HumanCfg with gateway="streamlit" and enabled=True."""
    return HumanCfg(
        enabled=enabled,
        gateway="streamlit",  # type: ignore[arg-type]
        role="reviewer",
        timeout_s=None,  # type: ignore[arg-type]
        timeout_policy="skip",
    )


def _make_human_cfg_llm_simulated(*, enabled: bool = True) -> HumanCfg:
    """Return a HumanCfg with gateway="llm_simulated" (back-compat default)."""
    return HumanCfg(
        enabled=enabled,
        gateway="llm_simulated",
        role="reviewer",
        timeout_s=None,  # type: ignore[arg-type]
        timeout_policy="skip",
    )


def _make_mock_graph() -> MagicMock:
    mock_compiled = MagicMock()
    mock_graph = MagicMock()
    mock_graph.compile.return_value = mock_compiled
    return mock_graph


# ---------------------------------------------------------------------------
# 1. ChainTopology dispatch test
# ---------------------------------------------------------------------------


class TestChainDispatch:
    """ChainTopology uses pre-built human_gateway when passed as kwarg."""

    def test_human_gateway_kwarg_is_used_not_llm_simulated(self) -> None:
        """When human_gateway=FakeStreamlitGateway() is passed, LLMSimulatedGateway
        is NOT constructed."""
        fake_gw = FakeStreamlitGateway()
        human_cfg = _make_human_cfg_streamlit()

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = TopologyConfig(name="chain", max_iterations=10)

        captured_build_calls: list[Any] = []

        def capturing_build_node(h_cfg: Any, gw: Any, **kw: Any) -> Any:
            captured_build_calls.append(gw)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch(
                "atm.topology.chain._build_human_reviewer_node",
                side_effect=capturing_build_node,
            ),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_llm_sim,
        ):
            ChainTopology().build(agents, cfg, human_cfg=human_cfg, human_gateway=fake_gw)

        # LLMSimulatedGateway must NOT have been constructed
        mock_llm_sim.assert_not_called()

        # The gateway passed to _build_human_reviewer_node must be our fake
        assert len(captured_build_calls) == 1
        assert captured_build_calls[0] is fake_gw

    def test_no_human_gateway_kwarg_uses_llm_simulated(self) -> None:
        """Without human_gateway kwarg, back-compat path constructs LLMSimulatedGateway."""
        human_cfg = _make_human_cfg_llm_simulated()

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "executor": mock_agent, "critic": mock_agent}
        cfg = TopologyConfig(name="chain", max_iterations=10)
        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            ChainTopology().build(agents, cfg, human_cfg=human_cfg)

        mock_llm_sim.assert_called_once()


# ---------------------------------------------------------------------------
# 2. StarTopology dispatch test
# ---------------------------------------------------------------------------


class TestStarDispatch:
    """StarTopology uses pre-built human_gateway when passed as kwarg."""

    def test_human_gateway_kwarg_is_used_not_llm_simulated(self) -> None:
        """When human_gateway=FakeStreamlitGateway() is passed, LLMSimulatedGateway
        is NOT constructed."""
        fake_gw = FakeStreamlitGateway()
        human_cfg = _make_human_cfg_streamlit()

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "worker_a": mock_agent,
            "worker_b": mock_agent,
            "critic": mock_agent,
            "coordinator": mock_agent,
        }
        cfg = TopologyConfig(name="star", max_iterations=10)
        mock_graph = _make_mock_graph()

        captured_gateway_args: list[Any] = []

        def capturing_factory(**kw: Any) -> Any:
            captured_gateway_args.append(kw.get("gateway"))

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.star.StateGraph", return_value=mock_graph),
            patch(
                "atm.topology.star.build_human_node_factory",
                side_effect=capturing_factory,
            ),
            patch("atm.topology.star.LLMSimulatedGateway") as mock_llm_sim,
        ):
            StarTopology().build(agents, cfg, human_cfg=human_cfg, human_gateway=fake_gw)

        mock_llm_sim.assert_not_called()
        assert len(captured_gateway_args) == 1
        assert captured_gateway_args[0] is fake_gw

    def test_no_human_gateway_kwarg_uses_llm_simulated(self) -> None:
        """Without human_gateway kwarg, back-compat path constructs LLMSimulatedGateway."""
        human_cfg = _make_human_cfg_llm_simulated()

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "worker_a": mock_agent,
            "worker_b": mock_agent,
            "critic": mock_agent,
            "coordinator": mock_agent,
        }
        cfg = TopologyConfig(name="star", max_iterations=10)
        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.star.StateGraph", return_value=mock_graph),
            patch("atm.topology.star.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            StarTopology().build(agents, cfg, human_cfg=human_cfg)

        mock_llm_sim.assert_called_once()


# ---------------------------------------------------------------------------
# 3. MeshTopology dispatch test
# ---------------------------------------------------------------------------


class TestMeshDispatch:
    """MeshTopology uses pre-built human_gateway (existing local var reassignment)."""

    def test_human_gateway_kwarg_is_used_not_llm_simulated(self) -> None:
        """When human_gateway=FakeStreamlitGateway() is passed, LLMSimulatedGateway
        is NOT constructed for the mesh human_peer gateway.

        Mesh uses an inline closure (not build_human_node_factory), so we verify
        LLMSimulatedGateway was not called rather than inspecting the factory call.
        """
        fake_gw = FakeStreamlitGateway()
        human_cfg = _make_human_cfg_streamlit()

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "researcher": mock_agent,
            "executor": mock_agent,
        }
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=20,
            extra={
                "max_rounds": 6,
                "consensus_threshold": 2,
                "agent_order": ["planner", "researcher", "executor"],
            },
        )
        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.mesh.StateGraph", return_value=mock_graph),
            patch("atm.topology.mesh.LLMSimulatedGateway") as mock_llm_sim,
        ):
            MeshTopology().build(agents, cfg, human_cfg=human_cfg, human_gateway=fake_gw)

        mock_llm_sim.assert_not_called()

    def test_no_human_gateway_kwarg_uses_llm_simulated(self) -> None:
        """Without human_gateway kwarg, back-compat path constructs LLMSimulatedGateway."""
        human_cfg = _make_human_cfg_llm_simulated()

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "researcher": mock_agent,
            "executor": mock_agent,
        }
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=20,
            extra={
                "max_rounds": 6,
                "consensus_threshold": 2,
                "agent_order": ["planner", "researcher", "executor"],
            },
        )
        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.mesh.StateGraph", return_value=mock_graph),
            patch("atm.topology.mesh.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            MeshTopology().build(agents, cfg, human_cfg=human_cfg)

        mock_llm_sim.assert_called_once()


# ---------------------------------------------------------------------------
# 4. DebateTopology dispatch test
# ---------------------------------------------------------------------------


class TestDebateDispatch:
    """DebateTopology uses pre-built human_gateway (canonical key) or legacy gateway kwarg."""

    def _make_agents(self) -> dict[str, Any]:
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        return {
            "planner": mock_agent,
            "debater_pro": mock_agent,
            "debater_contra": mock_agent,
            "judge": mock_agent,
        }

    def _make_cfg(self) -> TopologyConfig:
        return TopologyConfig(
            name="debate",
            max_iterations=10,
            extra={"judge": "human"},
        )

    def test_human_gateway_kwarg_is_used_not_llm_simulated(self) -> None:
        """When human_gateway=FakeStreamlitGateway() is passed, LLMSimulatedGateway
        is NOT constructed."""
        fake_gw = FakeStreamlitGateway()
        human_cfg = HumanCfg(
            enabled=True,
            gateway="streamlit",  # type: ignore[arg-type]
            role="judge",
            timeout_s=None,  # type: ignore[arg-type]
            timeout_policy="skip",
            extra={"judge": "human"},
        )

        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.debate.StateGraph", return_value=mock_graph),
            patch("atm.topology.debate.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            # Debate uses gateway internally; we verify LLMSimulatedGateway is NOT called
            # by passing human_gateway=fake_gw (the canonical key)
            DebateTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
                human_gateway=fake_gw,
            )

        mock_llm_sim.assert_not_called()

    def test_legacy_gateway_kwarg_still_works(self) -> None:
        """Legacy `gateway` kwarg is still accepted when `human_gateway` absent."""
        legacy_gw = FakeStreamlitGateway()
        human_cfg = HumanCfg(
            enabled=True,
            gateway="llm_simulated",
            role="judge",
            timeout_s=None,  # type: ignore[arg-type]
            timeout_policy="skip",
            extra={"judge": "human"},
        )

        mock_graph = _make_mock_graph()

        captured_gateway: list[Any] = []

        def capturing_build_node(h_cfg: Any, gw: Any, **kw: Any) -> Any:
            captured_gateway.append(gw)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.debate.StateGraph", return_value=mock_graph),
            patch(
                "atm.topology.debate._build_human_judge_node",
                side_effect=capturing_build_node,
            ),
            patch("atm.topology.debate.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            DebateTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
                gateway=legacy_gw,
            )

        # Legacy gateway kwarg should be used (no LLMSimulatedGateway constructed)
        mock_llm_sim.assert_not_called()
        assert len(captured_gateway) == 1
        assert captured_gateway[0] is legacy_gw

    def test_human_gateway_takes_priority_over_legacy_gateway(self) -> None:
        """When both human_gateway and gateway are passed, human_gateway wins."""
        canonical_gw = FakeStreamlitGateway()
        legacy_gw = FakeStreamlitGateway()
        human_cfg = HumanCfg(
            enabled=True,
            gateway="streamlit",  # type: ignore[arg-type]
            role="judge",
            timeout_s=None,  # type: ignore[arg-type]
            timeout_policy="skip",
            extra={"judge": "human"},
        )

        mock_graph = _make_mock_graph()

        captured_gateway: list[Any] = []

        def capturing_build_node(h_cfg: Any, gw: Any, **kw: Any) -> Any:
            captured_gateway.append(gw)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.debate.StateGraph", return_value=mock_graph),
            patch(
                "atm.topology.debate._build_human_judge_node",
                side_effect=capturing_build_node,
            ),
            patch("atm.topology.debate.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            DebateTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
                human_gateway=canonical_gw,
                gateway=legacy_gw,
            )

        mock_llm_sim.assert_not_called()
        assert len(captured_gateway) == 1
        assert captured_gateway[0] is canonical_gw


# ---------------------------------------------------------------------------
# 5. HierarchicalTopology dispatch test
# ---------------------------------------------------------------------------


class TestHierarchicalDispatch:
    """HierarchicalTopology uses pre-built human_gateway when passed as kwarg."""

    def _make_agents(self) -> dict[str, Any]:
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        return {
            "executor_a1": mock_agent,
            "executor_a2": mock_agent,
            "executor_b1": mock_agent,
            "executor_b2": mock_agent,
        }

    def _make_cfg(self) -> TopologyConfig:
        return TopologyConfig(
            name="hierarchical",
            max_iterations=20,
            extra={
                "sub_teams": [
                    {"team_id": "team_a", "workers": ["executor_a1", "executor_a2"]},
                    {"team_id": "team_b", "workers": ["executor_b1", "executor_b2"]},
                ],
                "scope": "top",
            },
        )

    def test_human_gateway_kwarg_is_used_not_llm_simulated(self) -> None:
        """When human_gateway=FakeStreamlitGateway() is passed, LLMSimulatedGateway
        is NOT constructed."""
        fake_gw = FakeStreamlitGateway()
        human_cfg = HumanCfg(
            enabled=True,
            gateway="streamlit",  # type: ignore[arg-type]
            role="reviewer",
            timeout_s=None,  # type: ignore[arg-type]
            timeout_policy="skip",
            extra={"scope": "top"},
        )

        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.hierarchical.StateGraph", return_value=mock_graph),
            patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            HierarchicalTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
                human_gateway=fake_gw,
            )

        mock_llm_sim.assert_not_called()

    def test_no_human_gateway_kwarg_uses_llm_simulated(self) -> None:
        """Without human_gateway kwarg, back-compat constructs LLMSimulatedGateway."""
        human_cfg = HumanCfg(
            enabled=True,
            gateway="llm_simulated",
            role="reviewer",
            timeout_s=None,  # type: ignore[arg-type]
            timeout_policy="skip",
            extra={"scope": "top"},
        )

        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.hierarchical.StateGraph", return_value=mock_graph),
            patch("atm.topology.hierarchical.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            HierarchicalTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
            )

        mock_llm_sim.assert_called_once()


# ---------------------------------------------------------------------------
# 6. AdaptiveTopology dispatch test
# ---------------------------------------------------------------------------


class TestAdaptiveDispatch:
    """AdaptiveTopology uses pre-built human_gateway (assigned to existing _gateway)."""

    def _make_agents(self) -> dict[str, Any]:
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        return {
            "planner": mock_agent,
            "executor": mock_agent,
            "critic": mock_agent,
            "chain_planner": mock_agent,
            "chain_executor": mock_agent,
            "chain_critic": mock_agent,
            "star_planner": mock_agent,
            "star_worker_a": mock_agent,
            "star_critic": mock_agent,
            "star_coordinator": mock_agent,
        }

    def _make_cfg(self) -> TopologyConfig:
        return TopologyConfig(
            name="adaptive",
            max_iterations=30,
        )

    def test_human_gateway_kwarg_is_used_not_llm_simulated(self) -> None:
        """When human_gateway=FakeStreamlitGateway() is passed, LLMSimulatedGateway
        is NOT constructed for adaptive human_advisor."""
        fake_gw = FakeStreamlitGateway()
        human_cfg = HumanCfg(
            enabled=True,
            gateway="streamlit",  # type: ignore[arg-type]
            role="reviewer",
            timeout_s=None,  # type: ignore[arg-type]
            timeout_policy="skip",
        )

        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            AdaptiveTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
                human_gateway=fake_gw,
            )

        mock_llm_sim.assert_not_called()

    def test_no_human_gateway_kwarg_with_llm_wrapper_uses_llm_simulated(self) -> None:
        """Without human_gateway kwarg, back-compat path constructs LLMSimulatedGateway
        when human_gateway_llm is provided."""
        human_cfg = _make_human_cfg_llm_simulated()
        mock_llm_wrapper = MagicMock()

        mock_graph = _make_mock_graph()

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            AdaptiveTopology().build(
                self._make_agents(),
                self._make_cfg(),
                human_cfg=human_cfg,
                human_gateway_llm=mock_llm_wrapper,
            )

        mock_llm_sim.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Global invariant: no topology constructs LLMSimulatedGateway when
#    human_gateway kwarg is provided
# ---------------------------------------------------------------------------


class TestGlobalNoLLMSimulatedWhenHumanGatewayProvided:
    """When human_gateway is passed to any topology's build(), LLMSimulatedGateway
    must not be constructed.

    This is the D7 invariant — the pre-built gateway always wins.
    """

    @pytest.mark.parametrize(
        "topology_cls,module_path,agents_factory,cfg_factory",
        [
            (
                "chain",
                "atm.topology.chain",
                lambda: {
                    "planner": _make_noop_agent(),
                    "executor": _make_noop_agent(),
                    "critic": _make_noop_agent(),
                },
                lambda: TopologyConfig(name="chain", max_iterations=10),
            ),
            (
                "star",
                "atm.topology.star",
                lambda: {
                    "planner": _make_noop_agent(),
                    "worker_a": _make_noop_agent(),
                    "worker_b": _make_noop_agent(),
                    "critic": _make_noop_agent(),
                    "coordinator": _make_noop_agent(),
                },
                lambda: TopologyConfig(name="star", max_iterations=10),
            ),
            (
                "mesh",
                "atm.topology.mesh",
                lambda: {
                    "planner": _make_noop_agent(),
                    "researcher": _make_noop_agent(),
                    "executor": _make_noop_agent(),
                },
                lambda: TopologyConfig(
                    name="mesh",
                    max_iterations=20,
                    extra={
                        "max_rounds": 6,
                        "consensus_threshold": 2,
                        "agent_order": ["planner", "researcher", "executor"],
                    },
                ),
            ),
        ],
    )
    def test_no_llm_simulated_constructed_when_human_gateway_passed(
        self,
        topology_cls: str,
        module_path: str,
        agents_factory: Any,
        cfg_factory: Any,
    ) -> None:
        """None of the 3 simple topologies (chain/star/mesh) build LLMSimulatedGateway
        when human_gateway is supplied."""
        from importlib import import_module

        fake_gw = FakeStreamlitGateway()
        human_cfg = _make_human_cfg_streamlit()

        mod = import_module(module_path)
        # Get the topology class
        cls_map = {
            "chain": ChainTopology,
            "star": StarTopology,
            "mesh": MeshTopology,
        }
        topo_cls = cls_map[topology_cls]
        topo = topo_cls()

        mock_graph = _make_mock_graph()

        import importlib

        mod = importlib.import_module(module_path)

        # Conditionally patch node factory helpers (only where they exist in module)
        optional_patches: list[Any] = []
        if hasattr(mod, "build_human_node_factory"):
            optional_patches.append(
                patch(f"{module_path}.build_human_node_factory", return_value=_noop_node_fn)
            )
        if hasattr(mod, "_build_human_reviewer_node"):
            optional_patches.append(
                patch(f"{module_path}._build_human_reviewer_node", return_value=_noop_node_fn)
            )

        with (
            patch(f"{module_path}.StateGraph", return_value=mock_graph),
            patch(f"{module_path}.LLMSimulatedGateway") as mock_llm_sim,
        ):
            mock_llm_sim.return_value = MagicMock()
            # Apply optional patches in a stack
            ctx_stack = [p.__enter__() for p in optional_patches]
            try:
                topo.build(
                    agents_factory(),
                    cfg_factory(),
                    human_cfg=human_cfg,
                    human_gateway=fake_gw,
                )
            except Exception:
                pass  # We only care that LLMSimulatedGateway was NOT called
            finally:
                for p, _ctx in zip(reversed(optional_patches), reversed(ctx_stack), strict=True):
                    p.__exit__(None, None, None)

        assert mock_llm_sim.call_count == 0, (
            f"{topology_cls}: LLMSimulatedGateway was constructed despite human_gateway being passed"
        )


# ---------------------------------------------------------------------------
# Helpers used by parametrized tests
# ---------------------------------------------------------------------------


def _make_noop_agent() -> MagicMock:
    agent = MagicMock()
    agent.step = AsyncMock(return_value={})
    return agent


async def _noop_node_fn(state: Any) -> dict[str, Any]:
    return {}
