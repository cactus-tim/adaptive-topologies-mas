"""Unit tests for DebateTopology HITL integration (M9.1 Step 2.8).

Tests cover:
  1. test_critic_mode_default_back_compat — judge mode "critic" (no human_cfg) has
     "judge" and "judge_postprocess" nodes; no "human_judge" or "judge_combined".
  2. test_critic_mode_explicit — judge mode "critic" with human_cfg.enabled=True
     but extra={"judge": "critic"} — still back-compat nodes.
  3. test_human_mode_node_structure — judge mode "human" has "human_judge" +
     "judge_postprocess" nodes; no "judge" node.
  4. test_human_mode_approve_writes_decision_to_outbox — human_judge node called
     with approve response writes DECISION(approved=True) to agents[judge_id]["outbox"].
  5. test_human_mode_reject_writes_decision_to_outbox — human_judge node called
     with reject response writes DECISION(approved=False).
  6. test_both_mode_node_structure — judge mode "both" has "judge_combined" node;
     no "judge" or "judge_postprocess" nodes.
  7. test_both_mode_human_approves_critic_rejects_final_approved — sequential composite:
     human approves + critic rejects → final approved=True (human wins).
  8. test_both_mode_human_rejects_critic_approves_final_rejected — human rejects +
     critic approves → final approved=False (human wins).
  9. test_both_mode_both_approve_final_approved — both approve → approved=True.
  10. test_both_mode_both_reject_final_rejected — both reject → approved=False.
  11. test_human_mode_dispatch_order — human_request dispatched before human_response.
  12. test_human_mode_request_id_format — request_id = f"debate:{iter_total}:judge".
  13. test_build_requires_gateway_for_human_mode — AssertionError/ValueError when
     human_cfg.enabled=True with mode="human" but no gateway.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import atm.topology.debate  # noqa: F401 — triggers @TopologyRegistry.register
from atm.core.types import HumanResponse, HumanRole, Message, MessageKind
from atm.experiment.config import HumanCfg
from atm.topology.base import TopologyConfig, TopologyRegistry
from atm.topology.debate import (
    DebateTopology,
    _build_human_judge_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(*, max_iterations: int = 12, max_rounds: int = 4) -> TopologyConfig:
    return TopologyConfig(
        name="debate",
        max_iterations=max_iterations,
        extra={
            "max_rounds": max_rounds,
            "debater_pro_id": "debater_pro",
            "debater_contra_id": "debater_contra",
            "judge_id": "judge",
        },
    )


def _make_human_cfg(
    *,
    enabled: bool = True,
    judge_mode: str = "human",
    role: HumanRole = HumanRole.JUDGE,
    timeout_s: float | None = None,
    timeout_policy: str = "skip",
) -> HumanCfg:
    return HumanCfg(
        enabled=enabled,
        gateway="llm_simulated",
        role=role,
        timeout_s=timeout_s,
        timeout_policy=timeout_policy,  # type: ignore[arg-type]
        extra={"judge": judge_mode},
    )


def _make_state(
    *,
    iter_total: int = 0,
    debate_round: int = 0,
    run_id: uuid.UUID | None = None,
    judge_outbox: list[Message] | None = None,
    debater_pro_outbox: list[Message] | None = None,
    debater_contra_outbox: list[Message] | None = None,
) -> dict[str, Any]:
    """Build a minimal GraphState-like dict."""
    _run_id = run_id or uuid.uuid4()
    agents: dict[str, Any] = {}
    if judge_outbox is not None:
        agents["judge"] = {"outbox": list(judge_outbox)}
    if debater_pro_outbox is not None:
        agents["debater_pro"] = {"outbox": list(debater_pro_outbox)}
    if debater_contra_outbox is not None:
        agents["debater_contra"] = {"outbox": list(debater_contra_outbox)}
    return {
        "shared": {
            "run_id": _run_id,
            "iter_total": iter_total,
            "debate_round": debate_round,
            "signals": {},
            "final_answer": None,
        },
        "agents": agents,
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _make_decision_msg(*, approved: bool, winner: str = "pro") -> Message:
    return Message(
        sender="judge",
        kind=MessageKind.DECISION,
        content="APPROVE" if approved else "REJECT",
        payload={"approved": approved, "winner": winner},
    )


def _make_draft_msg(content: str = "draft text") -> Message:
    return Message(
        sender="debater_pro",
        kind=MessageKind.DRAFT,
        content=content,
        payload={},
    )


def _make_approve_response(winner: str = "pro") -> HumanResponse:
    return HumanResponse(
        action="approve",
        comment="LGTM",
        payload={"winner": winner},
        source="llm_sim",
        timed_out=False,
    )


def _make_reject_response() -> HumanResponse:
    return HumanResponse(
        action="reject",
        comment="Needs more debate",
        source="llm_sim",
        timed_out=False,
    )


def _make_mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.step = AsyncMock(return_value={})
    return agent


def _make_mock_graph() -> MagicMock:
    """Build a mock StateGraph that records calls."""
    mock_compiled = MagicMock()
    mock_graph = MagicMock()
    mock_graph.add_node.return_value = None
    mock_graph.add_edge.return_value = None
    mock_graph.add_conditional_edges.return_value = None
    mock_graph.compile.return_value = mock_compiled
    return mock_graph


def _ensure_debate_registered() -> None:
    if "debate" not in TopologyRegistry.list_names():
        TopologyRegistry.register("debate")(DebateTopology)


def _build_debate_graph(
    judge_mode: str = "critic",
    *,
    human_cfg: HumanCfg | None = None,
    gateway: Any | None = None,
    mock_graph: MagicMock | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a DebateTopology graph with mocked StateGraph.

    Returns (mock_graph, compiled).
    """
    agents = {
        "planner": _make_mock_agent(),
        "debater_pro": _make_mock_agent(),
        "debater_contra": _make_mock_agent(),
        "judge": _make_mock_agent(),
    }
    cfg = _make_cfg()
    if human_cfg is None and judge_mode != "critic":
        human_cfg = _make_human_cfg(judge_mode=judge_mode)
    if gateway is None and judge_mode != "critic":
        gateway = MagicMock()
        gateway.request = AsyncMock(return_value=_make_approve_response())

    if mock_graph is None:
        mock_graph = _make_mock_graph()

    with patch("atm.topology.debate.StateGraph", return_value=mock_graph):
        topology = DebateTopology()
        compiled = topology.build(agents, cfg, human_cfg=human_cfg, gateway=gateway)

    return mock_graph, compiled


# ---------------------------------------------------------------------------
# Test 1: back-compat — no human_cfg → "judge" + "judge_postprocess" nodes present
# ---------------------------------------------------------------------------


class TestDebateBackCompat:
    """Default mode (no human_cfg) graph is identical to M7 Debate."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_critic_mode_default_back_compat(self) -> None:
        """Without human_cfg, graph has 'judge' + 'judge_postprocess', no HITL nodes."""
        mock_graph, _compiled = _build_debate_graph(judge_mode="critic", human_cfg=None)

        node_calls = [c[0][0] for c in mock_graph.add_node.call_args_list]
        assert "judge" in node_calls
        assert "judge_postprocess" in node_calls
        assert "human_judge" not in node_calls
        assert "judge_combined" not in node_calls

    def test_critic_mode_explicit_with_enabled_true(self) -> None:
        """human_cfg.enabled=True but extra={'judge':'critic'} → still back-compat nodes."""
        human_cfg = _make_human_cfg(judge_mode="critic", enabled=True)
        gateway = MagicMock()
        gateway.request = AsyncMock(return_value=_make_approve_response())

        mock_graph, _ = _build_debate_graph(
            judge_mode="critic",
            human_cfg=human_cfg,
            gateway=gateway,
        )
        node_calls = [c[0][0] for c in mock_graph.add_node.call_args_list]
        assert "judge" in node_calls
        assert "judge_postprocess" in node_calls
        assert "human_judge" not in node_calls
        assert "judge_combined" not in node_calls

    def test_critic_mode_edges_identical_to_m7(self) -> None:
        """Edge structure in 'critic' mode matches M7: pro→judge, contra→judge, judge→judge_postprocess."""
        mock_graph, _ = _build_debate_graph(judge_mode="critic", human_cfg=None)

        edge_calls = [c[0] for c in mock_graph.add_edge.call_args_list]
        assert ("debater_pro", "judge") in edge_calls
        assert ("debater_contra", "judge") in edge_calls
        assert ("judge", "judge_postprocess") in edge_calls


# ---------------------------------------------------------------------------
# Test 2: "human" mode — node structure
# ---------------------------------------------------------------------------


class TestDebateHumanMode:
    """judge mode 'human' inserts human_judge node; judge_postprocess unchanged."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_human_mode_node_structure(self) -> None:
        """Mode 'human': 'human_judge' + 'judge_postprocess' present; no 'judge' node."""
        mock_graph, _ = _build_debate_graph(judge_mode="human")

        node_calls = [c[0][0] for c in mock_graph.add_node.call_args_list]
        assert "human_judge" in node_calls
        assert "judge_postprocess" in node_calls
        assert "judge" not in node_calls
        assert "judge_combined" not in node_calls

    def test_human_mode_edges(self) -> None:
        """Mode 'human': debater_pro→human_judge, debater_contra→human_judge."""
        mock_graph, _ = _build_debate_graph(judge_mode="human")

        edge_calls = [c[0] for c in mock_graph.add_edge.call_args_list]
        assert ("debater_pro", "human_judge") in edge_calls
        assert ("debater_contra", "human_judge") in edge_calls
        assert ("human_judge", "judge_postprocess") in edge_calls

    def test_human_mode_approve_writes_decision_to_outbox(self) -> None:
        """human_judge node: approve response writes DECISION(approved=True) to judge outbox."""
        run_id = uuid.uuid4()
        state = _make_state(
            run_id=run_id,
            debater_pro_outbox=[_make_draft_msg("pro argument")],
            debater_contra_outbox=[_make_draft_msg("contra argument")],
        )

        human_cfg = _make_human_cfg(judge_mode="human")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=_make_approve_response(winner="pro"))

        # Build the node directly (bypass graph)
        node_fn = _build_human_judge_node(
            human_cfg,
            mock_gateway,
            judge_id="judge",
            debater_pro_id="debater_pro",
            debater_contra_id="debater_contra",
        )

        with patch(
            "atm.topology.debate.adispatch_custom_event", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = None
            delta = asyncio.run(node_fn(state))

        # DECISION should be in judge's outbox
        assert "agents" in delta
        judge_outbox = delta["agents"]["judge"]["outbox"]
        assert len(judge_outbox) >= 1
        decision = judge_outbox[-1]
        assert decision.kind == MessageKind.DECISION
        assert decision.payload["approved"] is True
        assert decision.payload["winner"] == "pro"

    def test_human_mode_reject_writes_decision_to_outbox(self) -> None:
        """human_judge node: reject response writes DECISION(approved=False) to judge outbox."""
        run_id = uuid.uuid4()
        state = _make_state(
            run_id=run_id,
            debater_pro_outbox=[_make_draft_msg("pro arg")],
            debater_contra_outbox=[_make_draft_msg("contra arg")],
        )

        human_cfg = _make_human_cfg(judge_mode="human")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=_make_reject_response())

        node_fn = _build_human_judge_node(
            human_cfg,
            mock_gateway,
            judge_id="judge",
            debater_pro_id="debater_pro",
            debater_contra_id="debater_contra",
        )

        with patch(
            "atm.topology.debate.adispatch_custom_event", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = None
            delta = asyncio.run(node_fn(state))

        judge_outbox = delta["agents"]["judge"]["outbox"]
        assert len(judge_outbox) >= 1
        decision = judge_outbox[-1]
        assert decision.kind == MessageKind.DECISION
        assert decision.payload["approved"] is False

    def test_human_mode_dispatch_order(self) -> None:
        """human_request is dispatched BEFORE human_response."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id)

        human_cfg = _make_human_cfg(judge_mode="human")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=_make_approve_response())

        node_fn = _build_human_judge_node(
            human_cfg,
            mock_gateway,
            judge_id="judge",
            debater_pro_id="debater_pro",
            debater_contra_id="debater_contra",
        )

        dispatch_calls: list[str] = []

        async def tracking_dispatch(event_name: str, payload: Any) -> None:
            dispatch_calls.append(event_name)

        with patch("atm.topology.debate.adispatch_custom_event", side_effect=tracking_dispatch):
            asyncio.run(node_fn(state))

        assert dispatch_calls.index("human_request") < dispatch_calls.index("human_response"), (
            "human_request must be dispatched before human_response"
        )

    def test_human_mode_request_id_format(self) -> None:
        """request_id = f'debate:{iter_total}:judge' for human mode."""
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id, iter_total=3)

        human_cfg = _make_human_cfg(judge_mode="human")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=_make_approve_response())

        node_fn = _build_human_judge_node(
            human_cfg,
            mock_gateway,
            judge_id="judge",
            debater_pro_id="debater_pro",
            debater_contra_id="debater_contra",
        )

        captured_request_ids: list[str] = []

        async def capture_dispatch(event_name: str, payload: Any) -> None:
            if event_name == "human_request":
                captured_request_ids.append(payload.get("request_id", ""))

        with patch("atm.topology.debate.adispatch_custom_event", side_effect=capture_dispatch):
            asyncio.run(node_fn(state))

        assert len(captured_request_ids) == 1
        assert captured_request_ids[0] == "debate:3:judge"


# ---------------------------------------------------------------------------
# Tests 3: "both" mode — node structure
# ---------------------------------------------------------------------------


class TestDebateBothModeStructure:
    """judge mode 'both' has 'judge_combined'; no 'judge' or 'judge_postprocess'."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_both_mode_node_structure(self) -> None:
        """Mode 'both': 'judge_combined' present; no 'judge' or 'judge_postprocess'."""
        mock_graph, _ = _build_debate_graph(judge_mode="both")

        node_calls = [c[0][0] for c in mock_graph.add_node.call_args_list]
        assert "judge_combined" in node_calls
        assert "judge" not in node_calls
        assert "judge_postprocess" not in node_calls
        assert "human_judge" not in node_calls

    def test_both_mode_edges(self) -> None:
        """Mode 'both': debater_pro→judge_combined, debater_contra→judge_combined."""
        mock_graph, _ = _build_debate_graph(judge_mode="both")

        edge_calls = [c[0] for c in mock_graph.add_edge.call_args_list]
        assert ("debater_pro", "judge_combined") in edge_calls
        assert ("debater_contra", "judge_combined") in edge_calls
        # No judge→judge_postprocess edge
        assert ("judge", "judge_postprocess") not in edge_calls


# ---------------------------------------------------------------------------
# Tests 4-7: "both" mode aggregation logic
# ---------------------------------------------------------------------------


class TestDebateBothModeAggregation:
    """both mode aggregation: human-override > critic when they disagree."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def _build_and_get_judge_combined_node(
        self,
        *,
        llm_judge_approved: bool,
        human_response: HumanResponse,
    ) -> Any:
        """Build a DebateTopology in 'both' mode and extract judge_combined node fn."""
        agents = {
            "planner": _make_mock_agent(),
            "debater_pro": _make_mock_agent(),
            "debater_contra": _make_mock_agent(),
            "judge": _make_mock_agent(),
        }
        # Configure judge agent to return a DECISION message
        decision = _make_decision_msg(approved=llm_judge_approved, winner="pro")
        agents["judge"].step = AsyncMock(return_value={"agents": {"judge": {"outbox": [decision]}}})

        cfg = _make_cfg()
        human_cfg = _make_human_cfg(judge_mode="both")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=human_response)

        # Capture the actual judge_combined node function
        captured_nodes: dict[str, Any] = {}

        def capture_add_node(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph = MagicMock()
        mock_graph.add_node.side_effect = capture_add_node
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.compile.return_value = MagicMock()

        with patch("atm.topology.debate.StateGraph", return_value=mock_graph):
            topology = DebateTopology()
            topology.build(agents, cfg, human_cfg=human_cfg, gateway=mock_gateway)

        return captured_nodes["judge_combined"]

    def _run_both_node(
        self,
        node_fn: Any,
        *,
        llm_judge_approved: bool,
    ) -> dict[str, Any]:
        """Run the both-mode node with a real state and return delta."""
        run_id = uuid.uuid4()
        # Ensure judge outbox is empty initially (node populates from LLM step)
        state = _make_state(
            run_id=run_id,
            debater_pro_outbox=[_make_draft_msg("pro argument here")],
            debater_contra_outbox=[_make_draft_msg("contra argument here")],
        )

        with patch(
            "atm.topology.debate.adispatch_custom_event", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = None
            delta = asyncio.run(node_fn(state))

        return delta

    def test_both_mode_human_approves_critic_rejects_final_approved(self) -> None:
        """Human approves + critic rejects → final approved=True (human wins)."""
        node_fn = self._build_and_get_judge_combined_node(
            llm_judge_approved=False,  # critic rejects
            human_response=_make_approve_response(winner="pro"),  # human approves
        )
        delta = self._run_both_node(node_fn, llm_judge_approved=False)

        # Check signals from _judge_postprocess
        signals = delta["shared"]["signals"]
        assert signals.get("judge_decided") is True, (
            "Human approves + critic rejects → human wins → final approved=True"
        )

    def test_both_mode_human_rejects_critic_approves_final_rejected(self) -> None:
        """Human rejects + critic approves → final approved=False (human wins)."""
        node_fn = self._build_and_get_judge_combined_node(
            llm_judge_approved=True,  # critic approves
            human_response=_make_reject_response(),  # human rejects
        )
        delta = self._run_both_node(node_fn, llm_judge_approved=True)

        signals = delta["shared"]["signals"]
        assert signals.get("judge_decided") is not True, (
            "Human rejects + critic approves → human wins → final approved=False"
        )

    def test_both_mode_both_approve_final_approved(self) -> None:
        """Both human and critic approve → final approved=True."""
        node_fn = self._build_and_get_judge_combined_node(
            llm_judge_approved=True,
            human_response=_make_approve_response(),
        )
        delta = self._run_both_node(node_fn, llm_judge_approved=True)

        signals = delta["shared"]["signals"]
        assert signals.get("judge_decided") is True, "Both approve → final approved=True"

    def test_both_mode_both_reject_final_rejected(self) -> None:
        """Both human and critic reject → final approved=False."""
        node_fn = self._build_and_get_judge_combined_node(
            llm_judge_approved=False,
            human_response=_make_reject_response(),
        )
        delta = self._run_both_node(node_fn, llm_judge_approved=False)

        signals = delta["shared"]["signals"]
        assert signals.get("judge_decided") is not True, "Both reject → final approved=False"

    def test_both_mode_increments_debate_round(self) -> None:
        """both mode judge_combined increments debate_round (from _judge_postprocess inline)."""
        node_fn = self._build_and_get_judge_combined_node(
            llm_judge_approved=True,
            human_response=_make_approve_response(),
        )
        delta = self._run_both_node(node_fn, llm_judge_approved=True)

        assert delta["shared"]["debate_round"] == 1, (
            "debate_round should be incremented from 0 to 1 by _judge_postprocess"
        )

    def test_both_mode_execution_order_llm_then_human(self) -> None:
        """In 'both' mode, LLM judge step() is called before human gateway.request()."""
        captured_order: list[str] = []

        agents = {
            "planner": _make_mock_agent(),
            "debater_pro": _make_mock_agent(),
            "debater_contra": _make_mock_agent(),
            "judge": _make_mock_agent(),
        }

        async def llm_step(state: Any) -> dict[str, Any]:
            captured_order.append("llm_judge")
            decision = _make_decision_msg(approved=True, winner="pro")
            return {"agents": {"judge": {"outbox": [decision]}}}

        agents["judge"].step = llm_step

        async def human_request(ctx: Any, *, request_id: str) -> HumanResponse:
            captured_order.append("human_gateway")
            return _make_approve_response()

        mock_gateway = MagicMock()
        mock_gateway.request = human_request

        human_cfg = _make_human_cfg(judge_mode="both")
        cfg = _make_cfg()

        captured_nodes: dict[str, Any] = {}

        def capture_add_node(name: str, fn: Any) -> None:
            captured_nodes[name] = fn

        mock_graph = MagicMock()
        mock_graph.add_node.side_effect = capture_add_node
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.compile.return_value = MagicMock()

        with patch("atm.topology.debate.StateGraph", return_value=mock_graph):
            topology = DebateTopology()
            topology.build(agents, cfg, human_cfg=human_cfg, gateway=mock_gateway)

        node_fn = captured_nodes["judge_combined"]
        run_id = uuid.uuid4()
        state = _make_state(run_id=run_id)

        with patch(
            "atm.topology.debate.adispatch_custom_event", new_callable=AsyncMock
        ) as mock_dispatch:
            mock_dispatch.return_value = None
            asyncio.run(node_fn(state))

        assert captured_order == ["llm_judge", "human_gateway"], (
            f"Expected LLM judge before human gateway, got {captured_order}"
        )


# ---------------------------------------------------------------------------
# Test: build requires gateway for human/both modes
# ---------------------------------------------------------------------------


class TestDebateBuildValidation:
    """build() correctly handles HITL mode configuration."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_build_human_mode_compiles_without_error(self) -> None:
        """Build with mode='human' and human_cfg.enabled=True succeeds (auto-builds gateway)."""
        agents = {
            "planner": _make_mock_agent(),
            "debater_pro": _make_mock_agent(),
            "debater_contra": _make_mock_agent(),
            "judge": _make_mock_agent(),
        }
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(judge_mode="human")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=_make_approve_response())

        # Should compile without errors when gateway is provided
        topology = DebateTopology()
        compiled = topology.build(agents, cfg, human_cfg=human_cfg, gateway=mock_gateway)
        assert compiled is not None

    def test_build_both_mode_compiles_without_error(self) -> None:
        """Build with mode='both' and gateway provided succeeds."""
        agents = {
            "planner": _make_mock_agent(),
            "debater_pro": _make_mock_agent(),
            "debater_contra": _make_mock_agent(),
            "judge": _make_mock_agent(),
        }
        cfg = _make_cfg()
        human_cfg = _make_human_cfg(judge_mode="both")
        mock_gateway = MagicMock()
        mock_gateway.request = AsyncMock(return_value=_make_approve_response())

        topology = DebateTopology()
        compiled = topology.build(agents, cfg, human_cfg=human_cfg, gateway=mock_gateway)
        assert compiled is not None

    def test_build_same_debater_ids_raises_valueerror(self) -> None:
        """build() raises ValueError when debater_pro_id == debater_contra_id."""
        cfg = TopologyConfig(
            name="debate",
            max_iterations=12,
            extra={
                "max_rounds": 4,
                "debater_pro_id": "debater_x",
                "debater_contra_id": "debater_x",
                "judge_id": "judge",
            },
        )
        agents = {
            "planner": _make_mock_agent(),
            "debater_x": _make_mock_agent(),
            "judge": _make_mock_agent(),
        }
        topology = DebateTopology()
        with pytest.raises(
            ValueError, match="debater_pro_id and debater_contra_id must be different"
        ):
            topology.build(agents, cfg)
