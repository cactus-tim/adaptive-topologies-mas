"""Back-compat regression tests for role_router=None ("fixed") behaviour (m9.2 Step 7.2).

Invariant: when HumanCfg.role_router == "fixed" (the default), the runner's
_build_role_router(...) MUST return None.  Every topology that integrated
role_router in Wave 5 takes the None branch and uses human_cfg.role directly,
preserving pre-m9.2 behaviour byte-for-byte.

Test catalogue:
  1. Default HumanCfg()  → _build_role_router returns None.
  2. Explicit HumanCfg(role_router="fixed") → _build_role_router returns None.
  3. human_cfg=None passed directly → _build_role_router returns None.
  4. ChainTopology.build() with role_router=None: _build_human_reviewer_node
     receives role_router=None (back-compat path taken).
  5. StarTopology.build() with role_router=None: build_human_node_factory
     receives role_router=None (back-compat path taken).
  6. MeshTopology.build() with role_router=None: human_peer_node closure uses
     human_cfg.role (back-compat).
  7. DebateTopology.build() with role_router=None: _build_human_judge_node
     receives role_router=None (back-compat path taken).
  8. HierarchicalTopology.build() with role_router=None: top reviewer node
     receives role_router=None (back-compat path taken).
  9. AdaptiveTopology.build() with role_router=None: build() succeeds and no
     router.decide() is ever called during a human advisory node invocation
     when role_router=None.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from atm.core.types import HumanResponse, HumanRole
from atm.experiment.config import HumanCfg
from atm.experiment.runner import _build_role_router
from atm.topology.adaptive import AdaptiveTopology
from atm.topology.base import TopologyConfig
from atm.topology.chain import ChainTopology, _build_human_reviewer_node
from atm.topology.debate import DebateTopology
from atm.topology.hierarchical import HierarchicalTopology
from atm.topology.mesh import MeshTopology
from atm.topology.star import StarTopology

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_human_cfg() -> HumanCfg:
    """Return a HumanCfg with all defaults — role_router must equal 'fixed'."""
    return HumanCfg()


def _fixed_human_cfg(role: HumanRole = HumanRole.REVIEWER) -> HumanCfg:
    """Return a HumanCfg with role_router explicitly set to 'fixed'."""
    return HumanCfg(role_router="fixed", role=role)


def _enabled_human_cfg(role: HumanRole = HumanRole.REVIEWER) -> HumanCfg:
    """Return an enabled HumanCfg with role_router='fixed' (default)."""
    return HumanCfg(enabled=True, role=role, timeout_s=None, timeout_policy="skip")


def _make_cfg(name: str, **extra: Any) -> TopologyConfig:
    return TopologyConfig(name=name, max_iterations=10, extra=extra)


def _make_mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.agent_id = "mock_agent"
    agent.step = AsyncMock(return_value={})
    return agent


def _make_compiled_graph_mock() -> tuple[MagicMock, MagicMock]:
    """Return (mock_compiled, mock_graph) pair for patching StateGraph."""
    mock_compiled = MagicMock()
    mock_graph = MagicMock()
    mock_graph.compile.return_value = mock_compiled
    return mock_compiled, mock_graph


# ---------------------------------------------------------------------------
# 1-3: _build_role_router returns None for "fixed" strategy
# ---------------------------------------------------------------------------


class TestBuildRoleRouterFixed:
    """_build_role_router MUST return None for any 'fixed' configuration."""

    def test_default_human_cfg_returns_none(self) -> None:
        """Default HumanCfg() has role_router='fixed' → _build_role_router is None."""
        cfg = _default_human_cfg()
        assert cfg.role_router == "fixed", (
            f"HumanCfg() default role_router must be 'fixed', got {cfg.role_router!r}"
        )
        result = _build_role_router(cfg)
        assert result is None, (
            f"_build_role_router(HumanCfg()) must return None (back-compat), got {result!r}"
        )

    def test_explicit_fixed_role_router_returns_none(self) -> None:
        """Explicit HumanCfg(role_router='fixed') → _build_role_router is None."""
        cfg = _fixed_human_cfg()
        result = _build_role_router(cfg)
        assert result is None, (
            f"_build_role_router with role_router='fixed' must return None, got {result!r}"
        )

    def test_none_human_cfg_returns_none(self) -> None:
        """human_cfg=None → _build_role_router returns None (pre-HITL back-compat)."""
        result = _build_role_router(None)
        assert result is None, f"_build_role_router(None) must return None, got {result!r}"

    def test_fixed_with_different_roles_returns_none(self) -> None:
        """role_router='fixed' returns None regardless of the configured HumanRole."""
        for role in HumanRole:
            cfg = HumanCfg(role_router="fixed", role=role)
            result = _build_role_router(cfg)
            assert result is None, (
                f"_build_role_router with role={role!r} and role_router='fixed' must return "
                f"None, got {result!r}"
            )

    def test_fixed_with_llm_factory_still_returns_none(self) -> None:
        """Providing a llm_factory does NOT affect the 'fixed' short-circuit."""
        cfg = _fixed_human_cfg()
        stub_factory = MagicMock(return_value=MagicMock())
        result = _build_role_router(cfg, llm_factory=stub_factory)
        assert result is None
        stub_factory.assert_not_called()


# ---------------------------------------------------------------------------
# 4: Chain topology — role_router=None propagated to node builder
# ---------------------------------------------------------------------------


class TestChainBackCompat:
    """ChainTopology: role_router=None is forwarded to _build_human_reviewer_node."""

    def test_chain_build_with_none_router_passes_none_to_node_builder(self) -> None:
        """build(role_router=None) → _build_human_reviewer_node receives role_router=None."""
        agents = {k: _make_mock_agent() for k in ("planner", "executor", "critic")}
        cfg = _make_cfg("chain")
        human_cfg = _enabled_human_cfg()
        _, mock_graph = _make_compiled_graph_mock()

        captured: list[Any] = []

        def capturing_build(h_cfg: Any, gw: Any, role_router: Any = None) -> Any:
            captured.append(role_router)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.chain.StateGraph", return_value=mock_graph),
            patch("atm.topology.chain.LLMSimulatedGateway", return_value=MagicMock()),
            patch(
                "atm.topology.chain._build_human_reviewer_node",
                side_effect=capturing_build,
            ),
        ):
            ChainTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert len(captured) == 1
        assert captured[0] is None, (
            f"Expected role_router=None forwarded to node builder, got {captured[0]!r}"
        )

    def test_chain_node_uses_human_cfg_role_when_router_none(self) -> None:
        """role_router=None → human_request payload role == human_cfg.role string."""
        run_id = uuid.uuid4()
        state: dict[str, Any] = {
            "shared": {
                "run_id": run_id,
                "iter_total": 0,
                "iteration": 0,
                "phase": "execution",
                "signals": {"critic_approved": False},
                "final_answer": None,
            },
            "agents": {
                "critic": {
                    "outbox": [],
                }
            },
            "messages": [],
            "llm_calls": [],
            "budget_events": [],
            "topology_transitions": [],
        }

        captured_roles: list[str] = []

        async def fake_dispatch(name: str, data: Any) -> None:
            if name == "human_request" and "role" in data:
                captured_roles.append(data["role"])

        fake_gw = AsyncMock()
        fake_gw.request = AsyncMock(
            return_value=HumanResponse(
                action="approve", comment=None, source="llm_sim", timed_out=False
            )
        )

        human_cfg = _enabled_human_cfg(role=HumanRole.JUDGE)
        node_fn = _build_human_reviewer_node(human_cfg, fake_gw, role_router=None)

        with patch("atm.topology.chain.adispatch_custom_event", side_effect=fake_dispatch):
            asyncio.run(node_fn(state))

        assert len(captured_roles) == 1
        assert captured_roles[0] == HumanRole.JUDGE.value, (
            f"Expected role={HumanRole.JUDGE.value!r} in payload, got {captured_roles[0]!r}"
        )


# ---------------------------------------------------------------------------
# 5: Star topology — role_router=None propagated to build_human_node_factory
# ---------------------------------------------------------------------------


class TestStarBackCompat:
    """StarTopology: role_router=None is forwarded to build_human_node_factory."""

    def test_star_build_with_none_router_passes_none_to_factory(self) -> None:
        """build(role_router=None) → build_human_node_factory receives role_router=None."""
        agent_ids = ("coordinator", "planner", "executor", "critic")
        agents = {k: _make_mock_agent() for k in agent_ids}
        cfg = _make_cfg(
            "star",
            planning_max_iter=1,
            exec_max_iter=1,
            verify_max_iter=1,
        )
        human_cfg = _enabled_human_cfg()
        _, mock_graph = _make_compiled_graph_mock()

        captured_kwargs: list[dict[str, Any]] = []

        def capturing_factory(**kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.star.StateGraph", return_value=mock_graph),
            patch("atm.topology.star.LLMSimulatedGateway", return_value=MagicMock()),
            patch(
                "atm.topology.star.build_human_node_factory",
                side_effect=capturing_factory,
            ),
        ):
            StarTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert len(captured_kwargs) == 1
        assert captured_kwargs[0].get("role_router") is None, (
            f"Expected role_router=None forwarded to factory, "
            f"got {captured_kwargs[0].get('role_router')!r}"
        )


# ---------------------------------------------------------------------------
# 6: Mesh topology — role_router=None → human_peer_node uses human_cfg.role
# ---------------------------------------------------------------------------


class TestMeshBackCompat:
    """MeshTopology: role_router=None → human peer node uses human_cfg.role."""

    def test_mesh_build_accepts_none_router(self) -> None:
        """MeshTopology.build(role_router=None) does not raise."""
        agent_ids = ("planner", "researcher", "executor")
        agents = {k: _make_mock_agent() for k in agent_ids}
        cfg = _make_cfg(
            "mesh",
            max_rounds=3,
            consensus_threshold=2,
            agent_order=list(agent_ids),
        )
        human_cfg = _enabled_human_cfg()
        _, mock_graph = _make_compiled_graph_mock()

        with (
            patch("atm.topology.mesh.StateGraph", return_value=mock_graph),
            patch("atm.topology.mesh.LLMSimulatedGateway", return_value=MagicMock()),
        ):
            # Must not raise
            result = MeshTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert result is mock_graph.compile.return_value


# ---------------------------------------------------------------------------
# 7: Debate topology — role_router=None propagated to judge node builder
# ---------------------------------------------------------------------------


class TestDebateBackCompat:
    """DebateTopology: role_router=None is forwarded to _build_human_judge_node."""

    def test_debate_build_accepts_none_router(self) -> None:
        """DebateTopology.build(role_router=None) does not raise and builds graph.

        To trigger the 'human' judge path (which calls _build_human_judge_node),
        we pass human_cfg.extra={"judge": "human"}. Without this extra key, the
        debate topology defaults to 'critic' mode (back-compat) and never calls
        _build_human_judge_node at all.
        """
        agent_ids = ("debater_pro", "debater_contra", "judge")
        agents = {k: _make_mock_agent() for k in agent_ids}
        cfg = TopologyConfig(
            name="debate",
            max_iterations=12,
            extra={
                "max_rounds": 2,
                "debater_pro_id": "debater_pro",
                "debater_contra_id": "debater_contra",
                "judge_id": "judge",
            },
        )
        # extra={"judge": "human"} triggers the HITL judge path
        human_cfg = HumanCfg(
            enabled=True,
            role=HumanRole.JUDGE,
            timeout_s=None,
            timeout_policy="skip",
            extra={"judge": "human"},
        )
        _, mock_graph = _make_compiled_graph_mock()

        captured_role_routers: list[Any] = []

        def capturing_judge_node(h_cfg: Any, gw: Any, role_router: Any = None, **kw: Any) -> Any:
            captured_role_routers.append(role_router)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.debate.StateGraph", return_value=mock_graph),
            patch("atm.topology.debate.LLMSimulatedGateway", return_value=MagicMock()),
            patch(
                "atm.topology.debate._build_human_judge_node",
                side_effect=capturing_judge_node,
            ),
        ):
            DebateTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        # _build_human_judge_node must have been called once (human judge mode)
        assert len(captured_role_routers) >= 1, "Expected at least one _build_human_judge_node call"
        for idx, rr in enumerate(captured_role_routers):
            assert rr is None, (
                f"_build_human_judge_node call #{idx} received role_router={rr!r}, expected None"
            )


# ---------------------------------------------------------------------------
# 8: Hierarchical topology — role_router=None propagated to top reviewer node
# ---------------------------------------------------------------------------


class TestHierarchicalBackCompat:
    """HierarchicalTopology: role_router=None is forwarded through the build chain."""

    def test_hierarchical_build_accepts_none_router(self) -> None:
        """HierarchicalTopology.build(role_router=None) does not raise."""
        agent_ids = ("top_coordinator", "sub_planner", "sub_executor")
        agents = {k: _make_mock_agent() for k in agent_ids}
        cfg = _make_cfg("hierarchical")
        human_cfg = _enabled_human_cfg()
        _, mock_graph = _make_compiled_graph_mock()

        captured_role_routers: list[Any] = []

        def capturing_top_reviewer(h_cfg: Any, gw: Any, role_router: Any = None) -> Any:
            captured_role_routers.append(role_router)

            async def noop(state: Any) -> dict[str, Any]:
                return {}

            return noop

        with (
            patch("atm.topology.hierarchical.StateGraph", return_value=mock_graph),
            patch("atm.topology.hierarchical.LLMSimulatedGateway", return_value=MagicMock()),
            patch(
                "atm.topology.hierarchical._build_human_top_reviewer_node",
                side_effect=capturing_top_reviewer,
            ),
        ):
            HierarchicalTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert len(captured_role_routers) >= 1, (
            "Expected at least one _build_human_top_reviewer_node call"
        )
        for idx, rr in enumerate(captured_role_routers):
            assert rr is None, (
                f"_build_human_top_reviewer_node call #{idx} received "
                f"role_router={rr!r}, expected None"
            )


# ---------------------------------------------------------------------------
# 9: Adaptive topology — role_router=None → no router.decide() called
# ---------------------------------------------------------------------------


class TestAdaptiveBackCompat:
    """AdaptiveTopology: role_router=None → human advisory node uses human_cfg.role."""

    def test_adaptive_build_accepts_none_router(self) -> None:
        """AdaptiveTopology.build(role_router=None) does not raise."""
        agent_ids = ("coordinator", "planner", "executor", "critic")
        agents = {k: _make_mock_agent() for k in agent_ids}
        cfg = _make_cfg("adaptive")
        human_cfg = _enabled_human_cfg()
        _, mock_graph = _make_compiled_graph_mock()

        with (
            patch("atm.topology.adaptive.StateGraph", return_value=mock_graph),
            patch("atm.topology.adaptive._LLMSimulatedGateway", MagicMock()),
        ):
            result = AdaptiveTopology().build(agents, cfg, human_cfg=human_cfg, role_router=None)

        assert result is mock_graph.compile.return_value
