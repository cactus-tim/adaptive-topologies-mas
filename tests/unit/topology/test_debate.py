"""Unit tests for atm.topology.debate — DebateTopology, _judge_postprocess, _route_from_judge.

Tests cover the 10 required cases:
  1. test_planner_fans_out_to_both_debaters
  2. test_judge_postprocess_writes_signals_on_approve
  3. test_judge_postprocess_writes_final_answer_from_winner
  4. test_route_from_judge_returns_end_on_approved
  5. test_route_from_judge_returns_loop_on_not_approved_within_max_rounds
  6. test_route_from_judge_returns_end_on_max_rounds_exceeded
  7. test_global_max_iterations_overrides_topology_max
  8. test_register_under_name_debate
  9. test_judge_postprocess_malformed_decision_treated_as_rejected
  10. test_build_rejects_same_debater_id_for_pro_and_contra
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Side-effect import: triggers @TopologyRegistry.register("debate")
import atm.topology.debate  # noqa: F401
from atm.core.types import Message, MessageKind, ToolCall, ToolResult
from atm.topology.base import TopologyConfig, TopologyRegistry
from atm.topology.debate import (
    DebateTopology,
    _extract_winner_artifact,
    _judge_postprocess,
    _route_from_judge,
)

# ---------------------------------------------------------------------------
# Helpers — minimal state factories
# ---------------------------------------------------------------------------


def _make_shared(
    *,
    iter_total: int = 0,
    debate_round: int = 0,
    judge_decided: bool = False,
    debate_winner: str = "",
) -> dict[str, Any]:
    """Build a minimal SharedState-like dict."""
    signals: dict[str, Any] = {}
    if judge_decided:
        signals["judge_decided"] = True
    if debate_winner:
        signals["debate_winner"] = debate_winner
    return {
        "iter_total": iter_total,
        "debate_round": debate_round,
        "signals": signals,
        "final_answer": None,
    }


def _make_state(
    *,
    iter_total: int = 0,
    debate_round: int = 0,
    judge_decided: bool = False,
    judge_outbox: list[Message] | None = None,
    debater_pro_outbox: list[Message] | None = None,
    debater_contra_outbox: list[Message] | None = None,
) -> dict[str, Any]:
    """Build a minimal GraphState-like dict."""
    agents: dict[str, Any] = {}
    if judge_outbox is not None:
        agents["judge"] = {"outbox": list(judge_outbox)}
    if debater_pro_outbox is not None:
        agents["debater_pro"] = {"outbox": list(debater_pro_outbox)}
    if debater_contra_outbox is not None:
        agents["debater_contra"] = {"outbox": list(debater_contra_outbox)}
    return {
        "shared": _make_shared(
            iter_total=iter_total,
            debate_round=debate_round,
            judge_decided=judge_decided,
        ),
        "agents": agents,
    }


def _make_decision_msg(
    *,
    approved: bool,
    winner: str = "pro",
    sender: str = "judge",
) -> Message:
    """Build a DECISION-kind message from judge."""
    return Message(
        sender=sender,
        kind=MessageKind.DECISION,
        content="APPROVE" if approved else "REJECT",
        payload={"approved": approved, "winner": winner},
    )


def _make_malformed_decision_msg() -> Message:
    """Build a DECISION-kind message without 'approved' key."""
    return Message(
        sender="judge",
        kind=MessageKind.DECISION,
        content="SOME DECISION",
        payload={"comment": "missing approved key"},
    )


def _make_draft_msg(*, content: str = "debater draft") -> Message:
    """Build a DRAFT-kind message from a debater."""
    return Message(
        sender="debater_pro",
        kind=MessageKind.DRAFT,
        content=content,
        payload={},
    )


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


# ---------------------------------------------------------------------------
# Ensure DebateTopology stays registered across test isolation
# ---------------------------------------------------------------------------


def _ensure_debate_registered() -> None:
    """Re-register DebateTopology if the registry was cleared by another test."""
    if "debate" not in TopologyRegistry.list_names():
        TopologyRegistry.register("debate")(DebateTopology)


# ---------------------------------------------------------------------------
# Test 1: test_register_under_name_debate
# ---------------------------------------------------------------------------


class TestDebateRegistration:
    """DebateTopology registers itself under name 'debate' as a side-effect import."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_register_under_name_debate(self) -> None:
        """TopologyRegistry['debate'] returns DebateTopology class after import."""
        cls = TopologyRegistry.get("debate")
        assert cls is DebateTopology

    def test_debate_name_attribute(self) -> None:
        """DebateTopology.name == 'debate'."""
        assert DebateTopology.name == "debate"

    def test_debate_has_build_callable(self) -> None:
        """DebateTopology has callable 'build'."""
        assert callable(DebateTopology.build)


# ---------------------------------------------------------------------------
# Test 10: test_build_rejects_same_debater_id_for_pro_and_contra
# ---------------------------------------------------------------------------


class TestDebateBuildValidation:
    """build() raises ValueError when debater_pro_id == debater_contra_id."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_build_rejects_same_debater_id_for_pro_and_contra(self) -> None:
        """ValueError raised when both debater IDs are the same."""
        cfg = TopologyConfig(
            name="debate",
            max_iterations=12,
            extra={
                "max_rounds": 4,
                "debater_pro_id": "debater_x",
                "debater_contra_id": "debater_x",  # Same as pro!
                "judge_id": "judge",
            },
        )
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "debater_x": mock_agent,
            "judge": mock_agent,
        }
        topology = DebateTopology()
        import pytest

        with pytest.raises(
            ValueError, match="debater_pro_id and debater_contra_id must be different"
        ):
            topology.build(agents, cfg)


# ---------------------------------------------------------------------------
# Test 1: test_planner_fans_out_to_both_debaters
# ---------------------------------------------------------------------------


class TestDebateFanOut:
    """build() creates a graph with parallel fan-out from planner to both debaters."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_planner_fans_out_to_both_debaters(self) -> None:
        """build() returns a compiled graph; both debater nodes are present."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "debater_pro": mock_agent,
            "debater_contra": mock_agent,
            "judge": mock_agent,
        }
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.add_node.return_value = None
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.debate.StateGraph", return_value=mock_graph):
            topology = DebateTopology()
            compiled = topology.build(agents, cfg)

        # Verify both debater nodes were added
        node_calls = [call[0][0] for call in mock_graph.add_node.call_args_list]
        assert "debater_pro" in node_calls
        assert "debater_contra" in node_calls
        assert "planner" in node_calls
        assert "judge" in node_calls
        assert "judge_postprocess" in node_calls
        assert "debate_round_start" in node_calls

        # Verify compiled graph returned
        assert compiled is mock_compiled

    def test_planner_has_two_edges_to_debaters(self) -> None:
        """build() adds edges from planner to both debater_pro and debater_contra."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "debater_pro": mock_agent,
            "debater_contra": mock_agent,
            "judge": mock_agent,
        }
        cfg = _make_cfg()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.add_node.return_value = None
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.debate.StateGraph", return_value=mock_graph):
            topology = DebateTopology()
            topology.build(agents, cfg)

        # Verify edges planner → debater_pro and planner → debater_contra
        edge_calls = [call[0] for call in mock_graph.add_edge.call_args_list]
        assert ("planner", "debater_pro") in edge_calls
        assert ("planner", "debater_contra") in edge_calls


# ---------------------------------------------------------------------------
# Tests 2, 3, 9: _judge_postprocess
# ---------------------------------------------------------------------------


class TestJudgePostprocess:
    """_judge_postprocess correctly parses DECISION messages and updates state."""

    def test_judge_postprocess_writes_signals_on_approve(self) -> None:
        """When approved=True, signals['judge_decided']=True is set."""
        state = _make_state(
            judge_outbox=[_make_decision_msg(approved=True, winner="pro")],
            debater_pro_outbox=[_make_draft_msg(content="pro argument")],
        )

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["signals"]["judge_decided"] is True

    def test_judge_postprocess_writes_final_answer_from_winner(self) -> None:
        """When approved=True, final_answer is set from winning debater's DRAFT."""
        state = _make_state(
            judge_outbox=[_make_decision_msg(approved=True, winner="pro")],
            debater_pro_outbox=[_make_draft_msg(content="winning argument")],
        )

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["final_answer"] == "winning argument"

    def test_judge_postprocess_writes_final_answer_from_contra_winner(self) -> None:
        """When approved=True and winner='contra', final_answer from contra debater."""
        state = _make_state(
            judge_outbox=[_make_decision_msg(approved=True, winner="contra")],
            debater_contra_outbox=[_make_draft_msg(content="contra wins")],
        )

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["final_answer"] == "contra wins"

    def test_judge_postprocess_malformed_decision_treated_as_rejected(self) -> None:
        """Malformed DECISION (no 'approved' key) → judge_decided not set, not approved."""
        state = _make_state(
            judge_outbox=[_make_malformed_decision_msg()],
        )

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        signals = delta["shared"]["signals"]
        assert signals.get("judge_decided") is not True
        # New contract (post-safety-net): on rejection, we still extract the
        # best DRAFT artifact so the run reports SOMETHING instead of
        # quality_score=0 on otherwise-correct debates. With no DRAFT data
        # in this test fixture, the fallback returns "<incomplete>".
        assert delta["shared"].get("final_answer") == "<incomplete>"

    def test_judge_postprocess_no_decision_message_treated_as_rejected(self) -> None:
        """Empty outbox → no DECISION found → treated as rejected."""
        state = _make_state(judge_outbox=[])

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["signals"].get("judge_decided") is not True

    def test_judge_postprocess_increments_debate_round(self) -> None:
        """debate_round is incremented by 1 on each call."""
        state = _make_state(debate_round=2, judge_outbox=[])

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["debate_round"] == 3

    def test_judge_postprocess_increments_iter_total(self) -> None:
        """iter_total grows by 1 per judge_postprocess call (global max_iter precedence).

        Each debate round is one logical iteration — iter_total must advance
        inside the debate loop so _should_stop can enforce cfg.max_iterations.
        """
        state = _make_state(iter_total=3, debate_round=0, judge_outbox=[])

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["iter_total"] == 4

    def test_judge_postprocess_iter_total_grows_per_round(self) -> None:
        """Simulate 3 sequential rounds: iter_total increments 1 per round."""
        state = _make_state(iter_total=0, debate_round=0, judge_outbox=[])

        async def one_round(s: dict[str, Any]) -> dict[str, Any]:
            delta = await _judge_postprocess(
                s,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )
            # Merge delta back into state for next round
            merged = dict(s)
            merged["shared"] = delta["shared"]
            return merged

        async def run_three_rounds() -> dict[str, Any]:
            s = state
            for _ in range(3):
                s = await one_round(s)
            return s

        final_state = asyncio.run(run_three_rounds())
        assert final_state["shared"]["iter_total"] == 3
        assert final_state["shared"]["debate_round"] == 3


# ---------------------------------------------------------------------------
# Tests for _extract_winner_artifact — the in-topology code-artifact extractor
# that removes reliance on the runner-level workspace fallback for code tasks.
# ---------------------------------------------------------------------------


class TestExtractWinnerArtifact:
    """_extract_winner_artifact prefers file_write artifacts over DRAFT text."""

    @staticmethod
    def _file_write_call(path: str, content: str) -> ToolCall:
        return ToolCall(
            tool_name="file_write",
            args={"path": path, "content": content},
            issued_by="debater_pro",
        )

    @staticmethod
    def _ok_result(call: ToolCall) -> ToolResult:
        return ToolResult(call_id=call.id, ok=True, output=None, latency_ms=1)

    @staticmethod
    def _bad_result(call: ToolCall) -> ToolResult:
        return ToolResult(call_id=call.id, ok=False, output=None, latency_ms=1, error="overwrite")

    def test_prefers_solution_py_over_draft(self) -> None:
        """When debater wrote solution.py via file_write, that wins over DRAFT text."""
        call = self._file_write_call("solution.py", "def add(a,b): return a+b")
        agents = {
            "debater_pro": {
                "outbox": [_make_draft_msg(content="my pro argument")],
                "tool_calls": [call],
                "tool_results": [self._ok_result(call)],
            }
        }
        assert _extract_winner_artifact(agents, "debater_pro") == "def add(a,b): return a+b"

    def test_skips_rejected_overwrite(self) -> None:
        """A rejected (ok=False) file_write is ignored; later valid one wins."""
        good = self._file_write_call("solution.py", "GOOD")
        bad = self._file_write_call("solution.py", "BAD")
        agents = {
            "debater_pro": {
                "outbox": [_make_draft_msg(content="argument")],
                "tool_calls": [good, bad],  # bad is later in list
                "tool_results": [self._ok_result(good), self._bad_result(bad)],
            }
        }
        # bad is rejected → should fall through to good
        assert _extract_winner_artifact(agents, "debater_pro") == "GOOD"

    def test_falls_back_to_draft_when_no_file_write(self) -> None:
        """No tool_calls → returns DRAFT text (legacy / non-tool path)."""
        agents = {"debater_pro": {"outbox": [_make_draft_msg(content="just argument text")]}}
        assert _extract_winner_artifact(agents, "debater_pro") == "just argument text"

    def test_returns_incomplete_when_nothing_present(self) -> None:
        """Empty outbox + no tool_calls → '<incomplete>'."""
        agents = {"debater_pro": {"outbox": []}}
        assert _extract_winner_artifact(agents, "debater_pro") == "<incomplete>"

    def test_judge_postprocess_extracts_file_write_artifact(self) -> None:
        """End-to-end: _judge_postprocess pulls from winner's tool_calls when present."""
        call = ToolCall(
            tool_name="file_write",
            args={"path": "solution.py", "content": "def f(): return 42"},
            issued_by="debater_pro",
        )
        state: dict[str, Any] = {
            "shared": _make_shared(),
            "agents": {
                "judge": {"outbox": [_make_decision_msg(approved=True, winner="pro")]},
                "debater_pro": {
                    "outbox": [_make_draft_msg(content="my pro argument")],
                    "tool_calls": [call],
                    "tool_results": [
                        ToolResult(call_id=call.id, ok=True, output=None, latency_ms=1)
                    ],
                },
            },
        }

        async def run() -> dict[str, Any]:
            return await _judge_postprocess(
                state,  # type: ignore[arg-type]
                judge_id="judge",
                debater_pro_id="debater_pro",
                debater_contra_id="debater_contra",
                cfg=_make_cfg(),
            )

        delta = asyncio.run(run())
        assert delta["shared"]["final_answer"] == "def f(): return 42"


# ---------------------------------------------------------------------------
# Tests 4, 5, 6, 7: _route_from_judge
# ---------------------------------------------------------------------------


class TestRouteFromJudge:
    """_route_from_judge returns correct routing string based on state."""

    def test_route_from_judge_returns_end_on_approved(self) -> None:
        """Returns '__end__' when judge_decided=True (topology_success)."""
        state = _make_state(iter_total=2, debate_round=1, judge_decided=True)
        # Manually set judge_decided in signals
        state["shared"]["signals"]["judge_decided"] = True
        cfg = _make_cfg(max_iterations=12, max_rounds=4)

        result = _route_from_judge(state, cfg, max_rounds=4)  # type: ignore[arg-type]
        assert result == "__end__"

    def test_route_from_judge_returns_loop_on_not_approved_within_max_rounds(self) -> None:
        """Returns 'debate_round_start' when not approved and debate_round < max_rounds."""
        state = _make_state(iter_total=2, debate_round=1)
        cfg = _make_cfg(max_iterations=12, max_rounds=4)

        result = _route_from_judge(state, cfg, max_rounds=4)  # type: ignore[arg-type]
        assert result == "debate_round_start"

    def test_route_from_judge_returns_end_on_max_rounds_exceeded(self) -> None:
        """Returns '__end__' when debate_round >= max_rounds (topology_max)."""
        state = _make_state(iter_total=2, debate_round=4)
        cfg = _make_cfg(max_iterations=12, max_rounds=4)

        result = _route_from_judge(state, cfg, max_rounds=4)  # type: ignore[arg-type]
        assert result == "__end__"

    def test_global_max_iterations_overrides_topology_max(self) -> None:
        """Returns '__end__' when iter_total >= max_iterations, overriding debate_round < max_rounds."""
        # iter_total=12 >= max_iterations=12, debate_round=1 (within max_rounds=4)
        state = _make_state(iter_total=12, debate_round=1)
        cfg = _make_cfg(max_iterations=12, max_rounds=4)

        result = _route_from_judge(state, cfg, max_rounds=4)  # type: ignore[arg-type]
        assert result == "__end__"

    def test_route_returns_loop_when_at_round_2_of_4(self) -> None:
        """Returns loop when debate_round=2 < max_rounds=4."""
        state = _make_state(iter_total=3, debate_round=2)
        cfg = _make_cfg(max_iterations=12, max_rounds=4)

        result = _route_from_judge(state, cfg, max_rounds=4)  # type: ignore[arg-type]
        assert result == "debate_round_start"


# ---------------------------------------------------------------------------
# Test: test_build_passes_checkpointer_to_compile
# ---------------------------------------------------------------------------


class TestDebateBuildCheckpointer:
    """build() forwards checkpointer kwarg to graph.compile()."""

    def setup_method(self) -> None:
        _ensure_debate_registered()

    def test_build_passes_checkpointer_to_compile(self) -> None:
        """build() calls graph.compile(checkpointer=...) when checkpointer kwarg is passed."""
        mock_agent = MagicMock()
        mock_agent.step = AsyncMock(return_value={})
        agents = {
            "planner": mock_agent,
            "debater_pro": mock_agent,
            "debater_contra": mock_agent,
            "judge": mock_agent,
        }
        cfg = _make_cfg()
        mock_cp = MagicMock()

        mock_compiled = MagicMock()
        mock_graph = MagicMock()
        mock_graph.add_node.return_value = None
        mock_graph.add_edge.return_value = None
        mock_graph.add_conditional_edges.return_value = None
        mock_graph.set_entry_point.return_value = None
        mock_graph.compile.return_value = mock_compiled

        with patch("atm.topology.debate.StateGraph", return_value=mock_graph):
            topology = DebateTopology()
            compiled = topology.build(agents, cfg, checkpointer=mock_cp)

        assert compiled is mock_compiled
        mock_graph.compile.assert_called_once_with(checkpointer=mock_cp)
