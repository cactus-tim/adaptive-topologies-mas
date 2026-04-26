"""Unit tests for atm.topology.chain — ChainTopology, _critic_postprocess, _route_from_critic.

Tests cover (≥8):
  1. ChainTopology is registered under name "chain" in TopologyRegistry.
  2. ChainTopology.name == "chain".
  3. _critic_postprocess sets critic_approved=True when DECISION payload["approved"]=True.
  4. _critic_postprocess sets critic_approved=False when DECISION payload["approved"]=False.
  5. _critic_postprocess handles malformed payload (no "approved" key) → approved=False.
  6. _critic_postprocess populates shared["final_answer"] from executor DRAFT when approved.
  7. _critic_postprocess uses msg.payload["draft"] when present, falls back to msg.content.
  8. _critic_postprocess sets final_answer to "<incomplete>" when no executor DRAFT exists.
  9. _route_from_critic increments shared["iter_total"] and shared["iteration"].
  10. _route_from_critic returns "executor" when critic_approved=False and stop=False.
  11. _route_from_critic returns END-sentinel when critic_approved=True.
  12. _route_from_critic returns END-sentinel when max_iterations reached.
  13. shared.phase is pinned to "execution" after _critic_postprocess runs.
  14. build() compiles successfully with mocked agents and mocked StateGraph.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Side-effect import: triggers @TopologyRegistry.register("chain")
import atm.topology.chain  # noqa: F401
from atm.core.types import Message, MessageKind
from atm.topology.base import TopologyConfig, TopologyRegistry
from atm.topology.chain import (
    CHAIN_END,
    ChainTopology,
    _critic_postprocess,
    _route_from_critic,
)

# ---------------------------------------------------------------------------
# Helpers — minimal GraphState factories
# ---------------------------------------------------------------------------


def _make_shared(
    *,
    iter_total: int = 0,
    iteration: int = 0,
    critic_approved: bool | None = None,
    phase: str = "execution",
) -> dict[str, Any]:
    """Build a minimal SharedState-like dict."""
    signals: dict[str, Any] = {}
    if critic_approved is not None:
        signals["critic_approved"] = critic_approved
    return {
        "iter_total": iter_total,
        "iteration": iteration,
        "phase": phase,
        "signals": signals,
        "final_answer": None,
    }


def _make_state(
    *,
    iter_total: int = 0,
    iteration: int = 0,
    critic_approved: bool | None = None,
    critic_outbox: list[Message] | None = None,
    executor_outbox: list[Message] | None = None,
) -> dict[str, Any]:
    """Build a minimal GraphState-like dict."""
    agents: dict[str, Any] = {}
    if critic_outbox is not None:
        agents["critic"] = {"outbox": list(critic_outbox)}
    if executor_outbox is not None:
        agents["executor"] = {"outbox": list(executor_outbox)}
    return {
        "shared": _make_shared(
            iter_total=iter_total,
            iteration=iteration,
            critic_approved=critic_approved,
        ),
        "agents": agents,
    }


def _make_decision_msg(*, approved: bool, comment: str = "ok") -> Message:
    """Build a DECISION-kind message with proper payload."""
    return Message(
        sender="critic",
        kind=MessageKind.DECISION,
        content="APPROVE" if approved else "REJECT",
        payload={"approved": approved, "comment": comment},
    )


def _make_malformed_decision_msg() -> Message:
    """Build a DECISION-kind message without 'approved' key."""
    return Message(
        sender="critic",
        kind=MessageKind.DECISION,
        content="SOME TEXT",
        payload={"comment": "missing approved key"},
    )


def _make_draft_msg(*, draft: str | None = None, content: str = "draft content") -> Message:
    """Build a DRAFT-kind message from executor."""
    payload: dict[str, Any] = {}
    if draft is not None:
        payload["draft"] = draft
    return Message(
        sender="executor",
        kind=MessageKind.DRAFT,
        content=content,
        payload=payload,
    )


def _make_cfg(*, max_iterations: int = 10) -> TopologyConfig:
    return TopologyConfig(name="chain", max_iterations=max_iterations)


# ---------------------------------------------------------------------------
# Fixture — ensure ChainTopology is registered before each test.
# test_base.py clears the TopologyRegistry between tests (setup_method/teardown_method).
# Since Python caches module imports, re-importing atm.topology.chain does NOT
# re-run the @TopologyRegistry.register("chain") decorator. We restore it manually.
# ---------------------------------------------------------------------------


def _ensure_chain_registered() -> None:
    """Re-register ChainTopology if the registry was cleared by another test suite."""
    if "chain" not in TopologyRegistry.list_names():
        TopologyRegistry.register("chain")(ChainTopology)


# ---------------------------------------------------------------------------
# Test group 1: Registry registration
# ---------------------------------------------------------------------------


class TestChainRegistration:
    """ChainTopology registers itself under name 'chain' as a side-effect import."""

    def setup_method(self) -> None:
        """Ensure chain is registered before each test."""
        _ensure_chain_registered()

    def test_chain_registered_in_registry(self) -> None:
        """TopologyRegistry['chain'] returns ChainTopology class after import."""
        cls = TopologyRegistry.get("chain")
        assert cls is ChainTopology

    def test_chain_name_attribute(self) -> None:
        """ChainTopology.name == 'chain'."""
        assert ChainTopology.name == "chain"

    def test_chain_has_build_callable(self) -> None:
        """ChainTopology has callable 'build'."""
        assert callable(ChainTopology.build)

    def test_chain_instance_satisfies_topology_protocol(self) -> None:
        """ChainTopology() isinstance check passes for Topology Protocol."""
        from atm.topology.base import Topology

        topology = ChainTopology()
        assert isinstance(topology, Topology)


# ---------------------------------------------------------------------------
# Test group 2: _critic_postprocess — approved paths
# ---------------------------------------------------------------------------


class TestCriticPostprocessApproved:
    """_critic_postprocess correctly handles approved=True DECISION messages."""

    def test_sets_critic_approved_true(self) -> None:
        """critic_approved=True in signals when DECISION has payload['approved']=True."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=True)],
            executor_outbox=[_make_draft_msg(content="my draft")],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["signals"]["critic_approved"] is True

    def test_populates_final_answer_from_draft_payload(self) -> None:
        """final_answer is set from executor DRAFT msg.payload['draft'] when present."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=True)],
            executor_outbox=[_make_draft_msg(draft="fib(10)=55", content="other")],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["final_answer"] == "fib(10)=55"

    def test_populates_final_answer_from_msg_content_fallback(self) -> None:
        """final_answer falls back to msg.content when payload['draft'] absent."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=True)],
            executor_outbox=[_make_draft_msg(content="content_fallback")],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["final_answer"] == "content_fallback"

    def test_final_answer_incomplete_when_no_executor_draft(self) -> None:
        """final_answer == '<incomplete>' when no executor DRAFT message exists."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=True)],
            executor_outbox=[],  # no DRAFT messages
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["final_answer"] == "<incomplete>"

    def test_phase_pinned_to_execution(self) -> None:
        """shared['phase'] remains 'execution' after postprocess (M6 pin)."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=True)],
            executor_outbox=[_make_draft_msg(content="x")],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["phase"] == "execution"

    def test_uses_last_decision_message_from_outbox(self) -> None:
        """When multiple messages in outbox, uses last DECISION-kind one."""
        earlier_reject = _make_decision_msg(approved=False)
        later_approve = _make_decision_msg(approved=True)
        state = _make_state(
            critic_outbox=[earlier_reject, later_approve],
            executor_outbox=[_make_draft_msg(draft="result_55")],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["signals"]["critic_approved"] is True
        assert delta["shared"]["final_answer"] == "result_55"


# ---------------------------------------------------------------------------
# Test group 3: _critic_postprocess — rejected/malformed paths
# ---------------------------------------------------------------------------


class TestCriticPostprocessRejected:
    """_critic_postprocess correctly handles approved=False and malformed messages."""

    def test_sets_critic_approved_false(self) -> None:
        """critic_approved=False in signals when DECISION has payload['approved']=False."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=False)],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["signals"]["critic_approved"] is False

    def test_malformed_payload_treated_as_rejected(self) -> None:
        """Malformed DECISION message (no 'approved' key) → approved=False."""
        state = _make_state(
            critic_outbox=[_make_malformed_decision_msg()],
        )
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["signals"]["critic_approved"] is False

    def test_no_decision_message_treated_as_rejected(self) -> None:
        """Empty outbox → no DECISION found → critic_approved=False."""
        state = _make_state(critic_outbox=[])
        delta = asyncio.run(_critic_postprocess(state))
        assert delta["shared"]["signals"]["critic_approved"] is False

    def test_rejected_does_not_set_final_answer(self) -> None:
        """When approved=False, final_answer is NOT populated (stays None)."""
        state = _make_state(
            critic_outbox=[_make_decision_msg(approved=False)],
        )
        delta = asyncio.run(_critic_postprocess(state))
        # final_answer should be None — not set on rejection
        assert delta["shared"].get("final_answer") is None


# ---------------------------------------------------------------------------
# Test group 4: _route_from_critic counter increments and routing
# ---------------------------------------------------------------------------


class TestRouteFromCritic:
    """_route_from_critic increments counters and returns correct routing string."""

    def test_returns_executor_when_not_approved(self) -> None:
        """Returns 'executor' when critic_approved=False and max not reached."""
        state = _make_state(iter_total=0, critic_approved=False)
        cfg = _make_cfg(max_iterations=10)
        result = _route_from_critic(state, cfg)
        assert result == "executor"

    def test_returns_end_sentinel_when_approved(self) -> None:
        """Returns CHAIN_END sentinel when critic_approved=True."""
        state = _make_state(iter_total=0, critic_approved=True)
        cfg = _make_cfg(max_iterations=10)
        result = _route_from_critic(state, cfg)
        assert result == CHAIN_END

    def test_returns_end_sentinel_when_max_iterations_reached(self) -> None:
        """Returns CHAIN_END sentinel when iter_total >= max_iterations."""
        state = _make_state(iter_total=10, critic_approved=False)
        cfg = _make_cfg(max_iterations=10)
        result = _route_from_critic(state, cfg)
        assert result == CHAIN_END

    def test_increments_iter_total_in_place(self) -> None:
        """_route_from_critic increments state['shared']['iter_total'] by 1."""
        state = _make_state(iter_total=5, iteration=2, critic_approved=False)
        cfg = _make_cfg(max_iterations=20)
        _route_from_critic(state, cfg)
        assert state["shared"]["iter_total"] == 6

    def test_increments_iteration_in_place(self) -> None:
        """_route_from_critic increments state['shared']['iteration'] by 1."""
        state = _make_state(iter_total=5, iteration=2, critic_approved=False)
        cfg = _make_cfg(max_iterations=20)
        _route_from_critic(state, cfg)
        assert state["shared"]["iteration"] == 3

    def test_stop_reason_max_iter_returns_end(self) -> None:
        """After increment, iter_total==max_iterations → CHAIN_END."""
        # After increment: iter_total=10 >= max_iterations=10 → stop
        state = _make_state(iter_total=9, critic_approved=False)
        cfg = _make_cfg(max_iterations=10)
        result = _route_from_critic(state, cfg)
        assert result == CHAIN_END


# ---------------------------------------------------------------------------
# Test group 5: build() — graph compilation with mocked StateGraph
# ---------------------------------------------------------------------------


class TestChainBuild:
    """ChainTopology.build() produces a compiled graph (StateGraph mocked)."""

    def test_build_returns_non_none_with_mock(self) -> None:
        """build() returns a non-None object representing the compiled graph."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "executor": mock_agent,
            "critic": mock_agent,
        }
        cfg = _make_cfg(max_iterations=5)

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.add_node.return_value = None
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.set_entry_point.return_value = None
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.chain.StateGraph", return_value=mock_graph):
            topology = ChainTopology()
            compiled = topology.build(agents, cfg)

        assert compiled is not None
        assert compiled is mock_compiled

    def test_build_with_checkpointer_keyword(self) -> None:
        """build() passes checkpointer kwarg to compile() without raising."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "executor": mock_agent,
            "critic": mock_agent,
        }
        cfg = _make_cfg(max_iterations=5)
        mock_checkpointer = MagicMock()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.add_node.return_value = None
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.set_entry_point.return_value = None
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.chain.StateGraph", return_value=mock_graph):
            topology = ChainTopology()
            compiled = topology.build(agents, cfg, checkpointer=mock_checkpointer)

        assert compiled is not None
        # compile() should have been called with the checkpointer
        mock_graph.compile.assert_called_once_with(checkpointer=mock_checkpointer)
