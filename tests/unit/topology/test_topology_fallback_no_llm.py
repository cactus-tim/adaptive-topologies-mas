"""Unit tests for fallback_llm threading — Step 5.3 (M14).

For each of the 6 topologies, we verify that:
  1. When a FakeStreamlitGateway (no ._llm attribute) is the primary gateway,
     and human_cfg.timeout_s is very small + timeout_policy="llm_fallback",
     NO AttributeError is raised during fallback resolution.
  2. When fallback_llm is NOT injected (and the gateway has no ._llm), the
     guarded pattern `LLMSimulatedGateway(llm=_fb_llm) if _fb_llm is not None else None`
     results in _fallback_gateway=None, which then either:
     a. Triggers ValueError from request_with_timeout (expected — guard fires),
     b. OR succeeds when fallback_llm IS injected (LLMSimulatedGateway built correctly).

Test strategy:
  - We do NOT invoke the full graph; we invoke only the HITL node function
    directly with a minimal fake state.
  - The primary gateway is FakeStreamlitGatewayNoLLM (no ._llm attribute at all).
  - We patch request_with_timeout to intercept the call and inspect
    llm_fallback_gateway — checking that it is either None (guard fired) or
    a non-None object (fallback_llm correctly threaded).
  - The test passes if NO AttributeError is raised on fallback resolution.

All 6 topologies:
  A. _node_factory.py (used by star.py)
  B. chain.py  _build_human_reviewer_node
  C. mesh.py   human_peer_node inline closure
  D. debate.py _build_human_judge_node (judge_mode="human")
  E. debate.py _both_judge_postprocess (judge_mode="both")
  F. hierarchical.py _build_human_top_reviewer_node
  G. hierarchical.py _build_human_sub_reviewer_node
  H. adaptive.py human_advisor_node inline closure
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from atm.experiment.config import HumanCfg
from atm.topology.base import TopologyConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeStreamlitGatewayNoLLM:
    """Minimal gateway stub that has NO ._llm attribute — simulates StreamlitHumanGateway."""

    async def request(self, ctx: Any, *, request_id: str) -> Any:
        # This will always "time out" because we patch request_with_timeout anyway.
        # But if called directly, return a valid response so tests don't hang.
        from atm.core.types import HumanResponse

        return HumanResponse(action="approve", comment="ok", source="human", timed_out=False)


def _make_timeout_human_cfg(*, extra: dict[str, Any] | None = None) -> HumanCfg:
    """Return a HumanCfg that forces llm_fallback timeout policy with tiny timeout."""
    return HumanCfg(
        enabled=True,
        gateway="streamlit",  # type: ignore[arg-type]
        role="reviewer",
        timeout_s=0.01,  # very short so timeout fires
        timeout_policy="llm_fallback",
        extra=extra,
    )


def _make_run_id() -> uuid.UUID:
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


def _minimal_state(run_id: uuid.UUID | None = None) -> dict[str, Any]:
    """Return a minimal GraphState-like dict for node invocation."""
    _id = run_id or _make_run_id()
    return {
        "shared": {
            "run_id": _id,
            "iter_total": 0,
            "phase": "execution",
            "signals": {},
            "broadcast_bus": [],
        },
        "agents": {
            "critic": {"outbox": []},
            "judge": {"outbox": []},
            "debater_pro": {"outbox": []},
            "debater_contra": {"outbox": []},
        },
        "messages": [],
    }


def _make_fake_llm() -> MagicMock:
    """Return a minimal LLM-like mock that satisfies LLMSimulatedGateway construction."""
    return MagicMock(name="fake_llm")


# ---------------------------------------------------------------------------
# Site A: _node_factory.py (via build_human_node_factory) — used by star.py
# ---------------------------------------------------------------------------


class TestFallbackSiteANodeFactory:
    """build_human_node_factory honours fallback_llm kwarg; no AttributeError on StreamlitGW."""

    def test_no_llm_attr_with_fallback_llm_injected_no_attribute_error(self) -> None:
        """When fallback_llm is injected, LLMSimulatedGateway is built from it (not gateway._llm)."""
        from atm.human._node_factory import build_human_node_factory

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg()
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = build_human_node_factory(
            topology_name="star",
            human_cfg=human_cfg,
            gateway=fake_gw,
            request_id_template="star:{run_id}:{iter_total}:reviewer",
            question_extractor=lambda state: "test question",
            fallback_llm=fake_llm,
        )

        state = _minimal_state()

        with patch(
            "atm.human._node_factory.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            # Should not raise AttributeError
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        # fallback_gateway should NOT be None (fallback_llm was injected)
        assert intercepted[0] is not None

    def test_no_llm_attr_no_fallback_llm_guard_fires(self) -> None:
        """Without fallback_llm and no ._llm on gateway, guard fires: _fb_llm=None → fallback=None."""
        from atm.human._node_factory import build_human_node_factory

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = build_human_node_factory(
            topology_name="star",
            human_cfg=human_cfg,
            gateway=fake_gw,
            request_id_template="star:{run_id}:{iter_total}:reviewer",
            question_extractor=lambda state: "test question",
            # fallback_llm NOT provided — guard should fire, _fb_llm=None → _fallback=None
        )

        state = _minimal_state()

        with patch(
            "atm.human._node_factory.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            # Should not raise AttributeError (that was the bug)
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        # Guard fired: _fb_llm is None → _fallback_gateway is None
        assert intercepted[0] is None


# ---------------------------------------------------------------------------
# Site B: chain.py _build_human_reviewer_node
# ---------------------------------------------------------------------------


class TestFallbackSiteBChain:
    """Chain's _build_human_reviewer_node honours fallback_llm kwarg."""

    def test_no_llm_attr_with_fallback_llm_no_attribute_error(self) -> None:
        from atm.topology.chain import _build_human_reviewer_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg()
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_reviewer_node(human_cfg, fake_gw, fallback_llm=fake_llm)

        state = _minimal_state()

        with patch(
            "atm.topology.chain.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            # Must not raise AttributeError
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None  # built from fake_llm

    def test_no_llm_attr_no_fallback_llm_guard_fires(self) -> None:
        from atm.topology.chain import _build_human_reviewer_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_reviewer_node(
            human_cfg,
            fake_gw,
            # no fallback_llm
        )

        state = _minimal_state()

        with patch(
            "atm.topology.chain.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None  # guard fired


# ---------------------------------------------------------------------------
# Site C: mesh.py human_peer_node inline closure
# ---------------------------------------------------------------------------


class TestFallbackSiteCMesh:
    """Mesh human_peer_node closure honours gateway_llm fallback; no AttributeError."""

    def _build_peer_node(
        self,
        fake_gw: Any,
        human_cfg: HumanCfg,
        gateway_llm: Any = None,
    ) -> Any:
        """Extract human_peer_node by building MeshTopology with mocked graph."""
        from atm.topology.mesh import MeshTopology

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {"planner": mock_agent, "researcher": mock_agent, "executor": mock_agent}
        cfg = TopologyConfig(
            name="mesh",
            max_iterations=20,
            extra={
                "max_rounds": 6,
                "consensus_threshold": 2,
                "agent_order": ["planner", "researcher", "executor"],
            },
        )

        captured_nodes: dict[str, Any] = {}

        class CapturingGraph:
            def add_node(self, name: str, fn: Any) -> None:
                captured_nodes[name] = fn

            def add_edge(self, *args: Any) -> None:
                pass

            def add_conditional_edges(self, *args: Any, **kwargs: Any) -> None:
                pass

            def compile(self, **kwargs: Any) -> Any:
                return MagicMock()

        with patch("atm.topology.mesh.StateGraph", return_value=CapturingGraph()):
            if gateway_llm is not None:
                MeshTopology().build(
                    agents,
                    cfg,
                    human_cfg=human_cfg,
                    human_gateway=fake_gw,
                    human_gateway_llm=gateway_llm,
                )
            else:
                MeshTopology().build(
                    agents,
                    cfg,
                    human_cfg=human_cfg,
                    human_gateway=fake_gw,
                )

        return captured_nodes.get("human_peer")

    def test_no_llm_attr_with_gateway_llm_no_attribute_error(self) -> None:
        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"activation_round": 0})
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        peer_node = self._build_peer_node(fake_gw, human_cfg, gateway_llm=fake_llm)
        assert peer_node is not None, "human_peer node was not registered in the graph"

        state = _minimal_state()

        with patch("atm.topology.mesh.request_with_timeout", side_effect=fake_request_with_timeout):
            asyncio.run(peer_node(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None  # built from gateway_llm

    def test_no_llm_attr_no_gateway_llm_guard_fires(self) -> None:
        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"activation_round": 0})

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        peer_node = self._build_peer_node(fake_gw, human_cfg, gateway_llm=None)
        assert peer_node is not None, "human_peer node was not registered in the graph"

        state = _minimal_state()

        with patch("atm.topology.mesh.request_with_timeout", side_effect=fake_request_with_timeout):
            asyncio.run(peer_node(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None  # guard fired: no llm available


# ---------------------------------------------------------------------------
# Site D: debate.py _build_human_judge_node (judge_mode="human")
# ---------------------------------------------------------------------------


class TestFallbackSiteDDebateHumanJudge:
    """Debate's _build_human_judge_node honours fallback_llm; no AttributeError."""

    def test_no_llm_attr_with_fallback_llm_no_attribute_error(self) -> None:
        from atm.topology.debate import _build_human_judge_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"judge": "human"})
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_judge_node(
            human_cfg,
            fake_gw,
            judge_id="judge",
            debater_pro_id="debater_pro",
            debater_contra_id="debater_contra",
            fallback_llm=fake_llm,
        )

        state = _minimal_state()

        with patch(
            "atm.topology.debate.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None  # built from fake_llm

    def test_no_llm_attr_no_fallback_llm_guard_fires(self) -> None:
        from atm.topology.debate import _build_human_judge_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"judge": "human"})

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_judge_node(
            human_cfg,
            fake_gw,
            judge_id="judge",
            debater_pro_id="debater_pro",
            debater_contra_id="debater_contra",
            # no fallback_llm
        )

        state = _minimal_state()

        with patch(
            "atm.topology.debate.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None  # guard fired


# ---------------------------------------------------------------------------
# Site E: debate.py _both_judge_postprocess (judge_mode="both")
# ---------------------------------------------------------------------------


class TestFallbackSiteEDebateBothMode:
    """Debate's _both_judge_postprocess honours _gateway_llm; no AttributeError."""

    def _build_combined_node(
        self,
        fake_gw: Any,
        human_cfg: HumanCfg,
        gateway_llm: Any = None,
    ) -> Any:
        """Extract judge_combined node by building DebateTopology with mocked graph."""
        from atm.topology.debate import DebateTopology

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "debater_pro": mock_agent,
            "debater_contra": mock_agent,
            "judge": mock_agent,
        }
        cfg = TopologyConfig(name="debate", max_iterations=10, extra={"judge": "both"})

        captured_nodes: dict[str, Any] = {}

        class CapturingGraph:
            def add_node(self, name: str, fn: Any) -> None:
                captured_nodes[name] = fn

            def add_edge(self, *args: Any) -> None:
                pass

            def add_conditional_edges(self, *args: Any, **kwargs: Any) -> None:
                pass

            def compile(self, **kwargs: Any) -> Any:
                return MagicMock()

        with patch("atm.topology.debate.StateGraph", return_value=CapturingGraph()):
            kw: dict[str, Any] = {
                "human_cfg": human_cfg,
                "human_gateway": fake_gw,
            }
            if gateway_llm is not None:
                kw["human_gateway_llm"] = gateway_llm
            DebateTopology().build(agents, cfg, **kw)

        return captured_nodes.get("judge_combined")

    def test_no_llm_attr_with_gateway_llm_no_attribute_error(self) -> None:
        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"judge": "both"})
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_judge_step(state: Any) -> dict[str, Any]:
            return {}

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        combined_node = self._build_combined_node(fake_gw, human_cfg, gateway_llm=fake_llm)
        assert combined_node is not None

        state = _minimal_state()

        with (
            patch(
                "atm.topology.debate.request_with_timeout", side_effect=fake_request_with_timeout
            ),
            patch.object(MagicMock, "step", side_effect=fake_judge_step, create=True),
        ):
            asyncio.run(combined_node(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None

    def test_no_llm_attr_no_gateway_llm_guard_fires(self) -> None:
        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"judge": "both"})

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        combined_node = self._build_combined_node(fake_gw, human_cfg, gateway_llm=None)
        assert combined_node is not None

        state = _minimal_state()

        with patch(
            "atm.topology.debate.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(combined_node(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None  # guard fired


# ---------------------------------------------------------------------------
# Site F: hierarchical.py _build_human_top_reviewer_node
# ---------------------------------------------------------------------------


class TestFallbackSiteFHierarchicalTop:
    """Hierarchical's _build_human_top_reviewer_node honours fallback_llm; no AttributeError."""

    def test_no_llm_attr_with_fallback_llm_no_attribute_error(self) -> None:
        from atm.topology.hierarchical import _build_human_top_reviewer_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"scope": "top"})
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_top_reviewer_node(human_cfg, fake_gw, fallback_llm=fake_llm)

        state = _minimal_state()

        with patch(
            "atm.topology.hierarchical.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None

    def test_no_llm_attr_no_fallback_llm_guard_fires(self) -> None:
        from atm.topology.hierarchical import _build_human_top_reviewer_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"scope": "top"})

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_top_reviewer_node(human_cfg, fake_gw)

        state = _minimal_state()

        with patch(
            "atm.topology.hierarchical.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None


# ---------------------------------------------------------------------------
# Site G: hierarchical.py _build_human_sub_reviewer_node
# ---------------------------------------------------------------------------


class TestFallbackSiteGHierarchicalSub:
    """Hierarchical's _build_human_sub_reviewer_node honours fallback_llm; no AttributeError."""

    def test_no_llm_attr_with_fallback_llm_no_attribute_error(self) -> None:
        from atm.topology.hierarchical import _build_human_sub_reviewer_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"scope": "sub_team"})
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_sub_reviewer_node(
            "team_a", human_cfg, fake_gw, fallback_llm=fake_llm
        )

        state = _minimal_state()

        with patch(
            "atm.topology.hierarchical.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None

    def test_no_llm_attr_no_fallback_llm_guard_fires(self) -> None:
        from atm.topology.hierarchical import _build_human_sub_reviewer_node

        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg(extra={"scope": "sub_team"})

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="approve", comment="ok", source="fallback", timed_out=True)

        node_fn = _build_human_sub_reviewer_node("team_a", human_cfg, fake_gw)

        state = _minimal_state()

        with patch(
            "atm.topology.hierarchical.request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(node_fn(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None


# ---------------------------------------------------------------------------
# Site H: adaptive.py human_advisor_node inline closure
# ---------------------------------------------------------------------------


class TestFallbackSiteHAdaptive:
    """Adaptive's human_advisor_node honours llm_wrapper fallback; no AttributeError."""

    def _build_advisor_node(
        self,
        fake_gw: Any,
        human_cfg: HumanCfg,
        gateway_llm: Any = None,
    ) -> Any:
        """Extract human_advisor_node by building AdaptiveTopology with mocked graph."""
        from atm.topology.adaptive import AdaptiveTopology

        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
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
        cfg = TopologyConfig(name="adaptive", max_iterations=30)

        captured_nodes: dict[str, Any] = {}

        class CapturingGraph:
            def add_node(self, name: str, fn: Any) -> None:
                captured_nodes[name] = fn

            def add_edge(self, *args: Any) -> None:
                pass

            def add_conditional_edges(self, *args: Any, **kwargs: Any) -> None:
                pass

            def compile(self, **kwargs: Any) -> Any:
                return MagicMock()

            def set_entry_point(self, name: str) -> None:
                pass

        with patch("atm.topology.adaptive.StateGraph", return_value=CapturingGraph()):
            kw: dict[str, Any] = {
                "human_cfg": human_cfg,
                "human_gateway": fake_gw,
            }
            if gateway_llm is not None:
                kw["human_gateway_llm"] = gateway_llm
            AdaptiveTopology().build(agents, cfg, **kw)

        return captured_nodes.get("human_advisor_node")

    def test_no_llm_attr_with_gateway_llm_no_attribute_error(self) -> None:
        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg()
        fake_llm = _make_fake_llm()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="advise", comment="hint", source="fallback", timed_out=True)

        advisor_node = self._build_advisor_node(fake_gw, human_cfg, gateway_llm=fake_llm)
        assert advisor_node is not None, "human_advisor_node was not registered in the graph"

        state = _minimal_state()
        # Inject a topo decision slot via the closure for human_advisor_node
        # (adaptive stores it in a closure-level slot; we only need the node itself)

        with patch(
            "atm.topology.adaptive._request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(advisor_node(state))

        assert len(intercepted) == 1
        assert intercepted[0] is not None

    def test_no_llm_attr_no_gateway_llm_guard_fires(self) -> None:
        fake_gw = FakeStreamlitGatewayNoLLM()
        human_cfg = _make_timeout_human_cfg()

        intercepted: list[Any] = []

        async def fake_request_with_timeout(
            gateway: Any,
            ctx: Any,
            *,
            request_id: str,
            timeout_s: Any,
            policy: Any,
            llm_fallback_gateway: Any = None,
        ) -> Any:
            intercepted.append(llm_fallback_gateway)
            from atm.core.types import HumanResponse

            return HumanResponse(action="advise", comment="hint", source="fallback", timed_out=True)

        # For adaptive, we need human_gateway_llm to build successfully (or gateway passed directly)
        # Pass fake_gw as human_gateway and no gateway_llm — _hitl_enabled stays True
        advisor_node = self._build_advisor_node(fake_gw, human_cfg, gateway_llm=None)
        assert advisor_node is not None, "human_advisor_node was not registered in the graph"

        state = _minimal_state()

        with patch(
            "atm.topology.adaptive._request_with_timeout", side_effect=fake_request_with_timeout
        ):
            asyncio.run(advisor_node(state))

        assert len(intercepted) == 1
        assert intercepted[0] is None  # guard fired: no llm available
