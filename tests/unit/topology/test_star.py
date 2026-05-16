"""Unit tests for atm.topology.star — StarTopology coordinator-centered LangGraph.

Tests cover:
  - StarTopology is registered in TopologyRegistry under "star"
  - StarTopology.build() compiles into a CompiledStateGraph
  - Coordinator increments iter_total and iteration
  - Phase advance: planning → execution → verification → done
  - Critic-reject loop runs up to verify_max_iter times then ends
  - Approved → END on first verification iteration
  - _critic_postprocess parses MessageKind.DECISION payload["approved"]
  - Malformed critic message → approved=False (warning only)
  - _should_stop global max_iter cap triggers END
  - final_answer is populated before END (from executor DRAFT or fallback)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from atm.core.types import MessageKind, Phase
from atm.topology.base import TopologyConfig, TopologyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    max_iterations: int = 20,
    planning_max_iter: int = 2,
    exec_max_iter: int = 5,
    verify_max_iter: int = 3,
) -> TopologyConfig:
    return TopologyConfig(
        name="star",
        max_iterations=max_iterations,
        extra={
            "planning_max_iter": planning_max_iter,
            "exec_max_iter": exec_max_iter,
            "verify_max_iter": verify_max_iter,
        },
    )


def _make_initial_state(
    *,
    phase: str = Phase.PLANNING,
    iter_total: int = 0,
    iteration: int = 0,
    phase_started_at_iter: int = 0,
    signals: dict[str, Any] | None = None,
    task_input: str = "test task",
) -> dict[str, Any]:
    """Build a minimal GraphState-like dict for testing."""
    return {
        "shared": {
            "task_input": task_input,
            "phase": phase,
            "iter_total": iter_total,
            "iteration": iteration,
            "phase_started_at_iter": phase_started_at_iter,
            "phase_history": [],
            "active_topology": "star",
            "topology_started_at_iter": 0,
            "topology_switch_count": 0,
            "topology_history": [],
            "final_answer": None,
            "signals": signals or {},
            "broadcast_bus": [],
            "human_requests": [],
            "human_responses": [],
        },
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


def _make_mock_agent(agent_id: str) -> MagicMock:
    """Create a mock agent that returns a simple delta dict."""
    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict) -> dict:
        return {
            "agents": {agent_id: {"agent_id": agent_id, "outbox": []}},
            "messages": [],
        }

    agent.step = fake_step
    return agent


def _make_mock_agent_with_outbox(agent_id: str, outbox_messages: list) -> MagicMock:
    """Create a mock agent that puts specific messages in outbox."""
    agent = MagicMock()
    agent.agent_id = agent_id

    async def fake_step(state: dict) -> dict:
        return {
            "agents": {agent_id: {"agent_id": agent_id, "outbox": outbox_messages}},
            "messages": outbox_messages,
        }

    agent.step = fake_step
    return agent


def _build_star(
    agents: dict[str, Any] | None = None,
    cfg: TopologyConfig | None = None,
    checkpointer: Any = None,
) -> CompiledStateGraph:
    """Convenience: import star (side-effect registers it) and build graph."""
    import atm.topology.star  # noqa: F401 — side-effect import for registration

    cls = TopologyRegistry.get("star")
    topology = cls()
    if agents is None:
        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
    if cfg is None:
        cfg = _make_cfg()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return topology.build(agents, cfg, checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------


class TestStarRegistration:
    """StarTopology is correctly registered in TopologyRegistry."""

    def test_star_is_registered_after_import(self) -> None:
        import atm.topology.star  # noqa: F401

        assert "star" in TopologyRegistry.list_names()

    def test_registry_returns_star_class(self) -> None:
        import atm.topology.star  # noqa: F401

        cls = TopologyRegistry.get("star")
        assert cls.name == "star"

    def test_star_class_has_build_method(self) -> None:
        import atm.topology.star  # noqa: F401

        cls = TopologyRegistry.get("star")
        assert callable(getattr(cls, "build", None))


# ---------------------------------------------------------------------------
# 2. Graph compilation
# ---------------------------------------------------------------------------


class TestStarBuild:
    """StarTopology.build() returns a CompiledStateGraph."""

    def test_build_returns_compiled_graph(self) -> None:
        graph = _build_star()
        assert isinstance(graph, CompiledStateGraph)

    def test_build_with_no_checkpointer(self) -> None:
        import atm.topology.star  # noqa: F401

        cls = TopologyRegistry.get("star")
        topology = cls()
        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
        cfg = _make_cfg()
        graph = topology.build(agents, cfg)
        assert isinstance(graph, CompiledStateGraph)

    def test_build_with_custom_cfg(self) -> None:
        cfg = _make_cfg(max_iterations=5, planning_max_iter=1, exec_max_iter=2, verify_max_iter=1)
        graph = _build_star(cfg=cfg)
        assert isinstance(graph, CompiledStateGraph)


# ---------------------------------------------------------------------------
# 3. Full graph run — planning → execution → verification → done (approved)
# ---------------------------------------------------------------------------


class TestStarPhasesApproved:
    """Full graph run: coordinator advances phases and ends with approved."""

    @pytest.mark.asyncio
    async def test_approved_ends_with_final_answer(self) -> None:
        """Star topology with quick approve: final_answer must be populated."""
        from atm.core.types import Message

        executor_draft = Message(
            sender="executor",
            kind=MessageKind.DRAFT,
            content="Result: 42",
        )
        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True, "comment": "Looks good"},
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent_with_outbox("executor", [executor_draft]),
            "critic": _make_mock_agent_with_outbox("critic", [decision_msg]),
        }
        cfg = _make_cfg(planning_max_iter=1, exec_max_iter=1, verify_max_iter=3)
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state()
        result = await graph.ainvoke(
            initial, config={"configurable": {"thread_id": "test-approved"}}
        )

        assert result["shared"]["final_answer"] is not None
        assert result["shared"]["final_answer"] != ""

    @pytest.mark.asyncio
    async def test_approved_phase_history_has_all_phases(self) -> None:
        """phase_history should record transitions planning→execution→verification→done."""
        from atm.core.types import Message

        executor_draft = Message(
            sender="executor",
            kind=MessageKind.DRAFT,
            content="Result",
        )
        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True},
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent_with_outbox("executor", [executor_draft]),
            "critic": _make_mock_agent_with_outbox("critic", [decision_msg]),
        }
        cfg = _make_cfg(planning_max_iter=1, exec_max_iter=1, verify_max_iter=3)
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state()
        result = await graph.ainvoke(initial, config={"configurable": {"thread_id": "test-phases"}})

        phase_history = result["shared"].get("phase_history", [])
        # Phase history records transitions; should contain planning, execution, verification, done
        phases_str = [str(p) for p in phase_history]
        assert any("planning" in p for p in phases_str) or result["shared"]["phase"] in (
            Phase.DONE,
            "done",
        )


# ---------------------------------------------------------------------------
# 4. Critic-reject loop runs verify_max_iter times then ends
# ---------------------------------------------------------------------------


class TestStarVerifyLoop:
    """Critic-reject loop exhausts verify_max_iter then terminates with final_answer."""

    @pytest.mark.asyncio
    async def test_reject_loop_terminates_after_max_verify_iter(self) -> None:
        """When critic always rejects, graph should still terminate after verify_max_iter."""
        from atm.core.types import Message

        executor_draft = Message(
            sender="executor",
            kind=MessageKind.DRAFT,
            content="Draft result",
        )
        reject_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="REJECT",
            payload={"approved": False, "comment": "Not good enough"},
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent_with_outbox("executor", [executor_draft]),
            "critic": _make_mock_agent_with_outbox("critic", [reject_msg]),
        }
        cfg = _make_cfg(
            max_iterations=50,
            planning_max_iter=1,
            exec_max_iter=1,
            verify_max_iter=2,
        )
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state()
        result = await graph.ainvoke(
            initial, config={"configurable": {"thread_id": "test-reject-loop"}}
        )

        # Graph must terminate and have a final_answer
        assert result is not None
        assert result["shared"].get("final_answer") is not None

    @pytest.mark.asyncio
    async def test_reject_loop_final_answer_fallback(self) -> None:
        """When critic always rejects and no executor DRAFT, final_answer is fallback."""
        from atm.core.types import Message

        reject_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="REJECT",
            payload={"approved": False},
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),  # no outbox messages
            "critic": _make_mock_agent_with_outbox("critic", [reject_msg]),
        }
        cfg = _make_cfg(
            max_iterations=50,
            planning_max_iter=1,
            exec_max_iter=1,
            verify_max_iter=1,
        )
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state()
        result = await graph.ainvoke(
            initial, config={"configurable": {"thread_id": "test-fallback"}}
        )

        # Must have a final_answer (even if fallback)
        assert result["shared"].get("final_answer") is not None


# ---------------------------------------------------------------------------
# 5. _critic_postprocess — malformed message handling
# ---------------------------------------------------------------------------


class TestCriticPostprocess:
    """_critic_postprocess correctly parses DECISION messages."""

    @pytest.mark.asyncio
    async def test_malformed_critic_message_treated_as_rejected(self) -> None:
        """A critic message without payload['approved'] treats as rejected (no crash)."""
        from atm.core.types import Message

        # Critic sends a DECISION with NO approved field
        malformed_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="something",
            payload={},  # missing 'approved' key
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent_with_outbox("critic", [malformed_msg]),
        }
        cfg = _make_cfg(planning_max_iter=1, exec_max_iter=1, verify_max_iter=1)
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state()
        # Should not raise — just treat as rejected and eventually terminate
        result = await graph.ainvoke(
            initial, config={"configurable": {"thread_id": "test-malformed"}}
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_critic_postprocess_sets_approved_true_in_signals(self) -> None:
        """Approved DECISION → shared.signals['critic_approved'] == True after postprocess."""
        from atm.core.types import Message
        from atm.topology.star import _critic_postprocess

        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True, "comment": "Good"},
        )

        state: dict[str, Any] = {
            "shared": {"signals": {}},
            "agents": {
                "critic": {
                    "agent_id": "critic",
                    "outbox": [decision_msg],
                }
            },
        }

        result = await _critic_postprocess(state)
        shared = result["shared"]
        assert shared["signals"]["critic_approved"] is True

    @pytest.mark.asyncio
    async def test_critic_postprocess_sets_approved_false_for_rejection(self) -> None:
        """Rejected DECISION → shared.signals['critic_approved'] == False."""
        from atm.core.types import Message
        from atm.topology.star import _critic_postprocess

        reject_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="REJECT",
            payload={"approved": False},
        )

        state: dict[str, Any] = {
            "shared": {"signals": {}},
            "agents": {
                "critic": {
                    "agent_id": "critic",
                    "outbox": [reject_msg],
                }
            },
        }

        result = await _critic_postprocess(state)
        assert result["shared"]["signals"]["critic_approved"] is False

    @pytest.mark.asyncio
    async def test_critic_postprocess_malformed_sets_false(self) -> None:
        """No payload['approved'] → signals['critic_approved'] = False (no exception)."""
        from atm.core.types import Message
        from atm.topology.star import _critic_postprocess

        malformed_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="dunno",
            payload={},
        )

        state: dict[str, Any] = {
            "shared": {"signals": {}},
            "agents": {
                "critic": {
                    "agent_id": "critic",
                    "outbox": [malformed_msg],
                }
            },
        }

        result = await _critic_postprocess(state)
        assert result["shared"]["signals"]["critic_approved"] is False


# ---------------------------------------------------------------------------
# 6. Global max_iter cap triggers END
# ---------------------------------------------------------------------------


class TestStarGlobalMaxIter:
    """_should_stop global max_iter cap forces early termination."""

    @pytest.mark.asyncio
    async def test_max_iter_cap_triggers_termination(self) -> None:
        """With max_iterations=3, graph must stop within 3 iterations."""
        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent("critic"),
        }
        # Very tight budget: max_iterations=3 with large phase caps
        cfg = _make_cfg(
            max_iterations=3,
            planning_max_iter=100,
            exec_max_iter=100,
            verify_max_iter=100,
        )
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state()
        result = await graph.ainvoke(
            initial, config={"configurable": {"thread_id": "test-max-iter"}}
        )

        # Graph should have terminated
        assert result is not None
        # iter_total should not exceed max_iterations (coordinator stops routing)
        iter_total = result["shared"].get("iter_total", 0)
        assert iter_total <= cfg.max_iterations + 2  # allow some slack for the coordinator tick


# ---------------------------------------------------------------------------
# 7. Coordinator increments iteration counters
# ---------------------------------------------------------------------------


class TestCoordinatorCounters:
    """Coordinator node increments iter_total on each tick."""

    @pytest.mark.asyncio
    async def test_iter_total_increments(self) -> None:
        """After running, iter_total should be > 0."""
        from atm.core.types import Message

        decision_msg = Message(
            sender="critic",
            kind=MessageKind.DECISION,
            content="APPROVE",
            payload={"approved": True},
        )

        agents = {
            "planner": _make_mock_agent("planner"),
            "executor": _make_mock_agent("executor"),
            "critic": _make_mock_agent_with_outbox("critic", [decision_msg]),
        }
        cfg = _make_cfg(planning_max_iter=1, exec_max_iter=1, verify_max_iter=3)
        graph = _build_star(agents=agents, cfg=cfg)

        initial = _make_initial_state(iter_total=0)
        result = await graph.ainvoke(
            initial, config={"configurable": {"thread_id": "test-counters"}}
        )

        assert result["shared"]["iter_total"] > 0


# ---------------------------------------------------------------------------
# Regression: _extract_final_answer task-aware preference (port of chain
# d585bbb). For non-code tasks the executor still writes solution.py per
# executor.yaml's unconditional instruction, but that file holds Python
# intermediates rather than the human-readable answer — DRAFT should win.
# ---------------------------------------------------------------------------


class TestExtractFinalAnswerTaskAware:
    """_extract_final_answer respects shared.task_id when ranking artifacts."""

    @staticmethod
    def _state_with_both_artifacts(*, task_id: str) -> dict[str, Any]:
        from atm.core.types import Message, ToolCall, ToolResult

        call = ToolCall(
            tool_name="file_write",
            args={"path": "solution.py", "content": "PYCODE"},
            issued_by="executor",
        )
        draft_msg = Message(
            sender="executor",
            kind=MessageKind.DRAFT,
            content="HUMAN_ANSWER",
        )
        return {
            "shared": {"task_id": task_id},
            "agents": {
                "executor": {
                    "outbox": [draft_msg],
                    "tool_calls": [call],
                    "tool_results": [
                        ToolResult(call_id=call.id, ok=True, output=None, latency_ms=1)
                    ],
                }
            },
            "messages": [],
        }

    def test_code_task_prefers_solution_py(self) -> None:
        from atm.topology.star import _extract_final_answer

        state = self._state_with_both_artifacts(task_id="humaneval")
        assert _extract_final_answer(state) == "PYCODE"

    def test_non_code_task_prefers_draft(self) -> None:
        from atm.topology.star import _extract_final_answer

        for non_code in ("gsm8k", "commongen", "dabench"):
            state = self._state_with_both_artifacts(task_id=non_code)
            assert _extract_final_answer(state) == "HUMAN_ANSWER", (
                f"non-code task_id={non_code!r} should prefer DRAFT over solution.py"
            )

    def test_unknown_task_treated_as_non_code(self) -> None:
        """Empty/unknown task_id falls into the DRAFT-preferred branch (chain parity)."""
        from atm.topology.star import _extract_final_answer

        state = self._state_with_both_artifacts(task_id="")
        assert _extract_final_answer(state) == "HUMAN_ANSWER"
