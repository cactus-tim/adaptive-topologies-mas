"""Unit tests for atm.core.types — 17 Pydantic classes and enums (Step 2.2 / M1)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

import pydantic
import pytest

# ---------------------------------------------------------------------------
# 1. Import test — all 17 symbols must be importable
# ---------------------------------------------------------------------------


def test_import_all_symbols() -> None:
    from atm.core.types import (  # noqa: F401
        AgentRole,
        BudgetEvent,
        HumanContext,
        HumanResponse,
        HumanRole,
        LLMResponse,
        Message,
        MessageKind,
        Phase,
        PhaseTransition,
        RunResult,
        TaskResult,
        TaskSpec,
        TokenUsage,
        ToolCall,
        ToolResult,
        TopologyTransition,
    )


# ---------------------------------------------------------------------------
# 2. Enum membership tests
# ---------------------------------------------------------------------------


def test_agent_role_members() -> None:
    from atm.core.types import AgentRole

    assert AgentRole.PLANNER.value == "planner"
    assert AgentRole.RESEARCHER.value == "researcher"
    assert AgentRole.EXECUTOR.value == "executor"
    assert AgentRole.CRITIC.value == "critic"
    assert AgentRole.DEBATER.value == "debater"
    assert AgentRole.COORDINATOR.value == "coordinator"


def test_human_role_members() -> None:
    from atm.core.types import HumanRole

    assert HumanRole.COORDINATOR.value == "coordinator"
    assert HumanRole.REVIEWER.value == "reviewer"
    assert HumanRole.JUDGE.value == "judge"
    assert HumanRole.PEER.value == "peer"
    assert HumanRole.MONITOR.value == "monitor"


def test_phase_members() -> None:
    from atm.core.types import Phase

    assert Phase.PLANNING.value == "planning"
    assert Phase.EXECUTION.value == "execution"
    assert Phase.VERIFICATION.value == "verification"
    assert Phase.DONE.value == "done"


def test_message_kind_members() -> None:
    from atm.core.types import MessageKind

    assert MessageKind.REQUEST.value == "request"
    assert MessageKind.BROADCAST.value == "broadcast"
    assert MessageKind.DRAFT.value == "draft"
    assert MessageKind.CRITIQUE.value == "critique"
    assert MessageKind.DECISION.value == "decision"
    assert MessageKind.PHASE_EMIT.value == "phase_emit"


# ---------------------------------------------------------------------------
# 3. Message instantiation and defaults
# ---------------------------------------------------------------------------


def test_message_creation_minimal() -> None:
    from atm.core.types import Message, MessageKind

    msg = Message(sender="agent_a", kind=MessageKind.REQUEST, content="hello")
    assert msg.sender == "agent_a"
    assert msg.content == "hello"
    assert isinstance(msg.id, UUID)
    assert msg.recipients == ()
    assert msg.payload == {}
    assert msg.refs == ()
    assert isinstance(msg.created_at, datetime)
    assert msg.created_at.tzinfo is not None  # timezone-aware


def test_message_creation_with_string_kind() -> None:
    """MessageKind should accept string value for convenience."""
    from atm.core.types import Message

    msg = Message(sender="a", kind="request", content="hi")  # type: ignore[arg-type]
    assert msg.kind.value == "request"


def test_message_is_frozen() -> None:
    from atm.core.types import Message, MessageKind

    msg = Message(sender="a", kind=MessageKind.REQUEST, content="x")
    with pytest.raises(pydantic.ValidationError):
        msg.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. ToolCall instantiation
# ---------------------------------------------------------------------------


def test_tool_call_defaults() -> None:
    from atm.core.types import ToolCall

    tc = ToolCall(tool_name="search", args={"query": "foo"}, issued_by="agent_b")
    assert isinstance(tc.id, UUID)
    assert tc.tool_name == "search"
    assert tc.args == {"query": "foo"}
    assert isinstance(tc.issued_at, datetime)
    assert tc.issued_at.tzinfo is not None


def test_tool_call_is_frozen() -> None:
    from atm.core.types import ToolCall

    tc = ToolCall(tool_name="run", args={}, issued_by="a")
    with pytest.raises(pydantic.ValidationError):
        tc.tool_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. TokenUsage and LLMResponse
# ---------------------------------------------------------------------------


def test_token_usage_creation() -> None:
    from atm.core.types import TokenUsage

    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.cached_input_tokens == 0
    assert usage.total_tokens == 150


def test_llm_response_creation() -> None:
    from atm.core.types import LLMResponse, TokenUsage

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp = LLMResponse(
        model="openai:gpt-4o",
        text="Hello",
        usage=usage,
        cost_usd=0.001,
        latency_ms=200,
        finish_reason="stop",
    )
    assert isinstance(resp.id, UUID)
    assert resp.model == "openai:gpt-4o"
    assert resp.text == "Hello"
    assert resp.tool_calls == ()
    assert resp.raw == {}


def test_llm_response_started_at_auto_populated() -> None:
    """M2 step 1.2: LLMResponse must auto-populate started_at via _utcnow default factory."""
    from atm.core.types import LLMResponse, TokenUsage

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp = LLMResponse(
        model="openai:gpt-4o",
        text="Hello",
        usage=usage,
        cost_usd=0.001,
        latency_ms=200,
        finish_reason="stop",
    )
    assert isinstance(resp.started_at, datetime)
    assert resp.started_at.tzinfo is not None


def test_llm_response_started_at_in_model_fields() -> None:
    """M2 step 1.2: started_at must be present as a model field."""
    from atm.core.types import LLMResponse

    assert "started_at" in LLMResponse.model_fields


def test_llm_response_started_at_explicit() -> None:
    """M2 step 1.2: started_at can be set explicitly without breaking other fields."""
    from atm.core.types import LLMResponse, TokenUsage

    usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    resp = LLMResponse(
        model="fake:deterministic",
        text=None,
        usage=usage,
        cost_usd=0.0,
        latency_ms=0,
        finish_reason="stop",
        started_at=ts,
    )
    assert resp.started_at == ts


# ---------------------------------------------------------------------------
# 6. ToolResult
# ---------------------------------------------------------------------------


def test_tool_result_creation() -> None:
    from uuid import uuid4

    from atm.core.types import ToolResult

    call_id = uuid4()
    result = ToolResult(call_id=call_id, ok=True, output="done", latency_ms=50)
    assert result.call_id == call_id
    assert result.ok is True
    assert result.error is None
    assert isinstance(result.finished_at, datetime)
    assert result.finished_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 7. HumanContext and HumanResponse
# ---------------------------------------------------------------------------


def test_human_context_creation() -> None:
    from uuid import uuid4

    from atm.core.types import HumanContext, HumanRole, Message, MessageKind

    run_id = uuid4()
    msg = Message(sender="a", kind=MessageKind.REQUEST, content="review this")
    ctx = HumanContext(
        run_id=run_id,
        role=HumanRole.REVIEWER,
        question="Approve?",
        recent_messages=(msg,),
        allowed_actions=("approve", "reject"),
    )
    assert ctx.role == HumanRole.REVIEWER
    assert ctx.deadline_s is None
    assert ctx.artifacts == {}


def test_human_response_defaults() -> None:
    from atm.core.types import HumanResponse

    resp = HumanResponse(action="approve")
    assert resp.comment is None
    assert resp.payload == {}
    assert resp.timed_out is False
    assert resp.source == "human"
    assert isinstance(resp.answered_at, datetime)
    assert resp.answered_at.tzinfo is not None


# ---------------------------------------------------------------------------
# 8. TaskSpec and TaskResult
# ---------------------------------------------------------------------------


def test_task_spec_creation() -> None:
    from atm.core.types import TaskSpec

    spec = TaskSpec(
        id="humaneval/0",
        type="programming",
        input="def add(a, b):",
        evaluator_key="humaneval",
    )
    assert spec.expected is None
    assert spec.metadata == {}


def test_task_result_creation() -> None:
    from atm.core.types import TaskResult

    result = TaskResult(
        task_id="humaneval/0",
        final_answer="return a + b",
        iterations_used=3,
        budget_spent_usd=0.05,
        wall_time_s=12.5,
    )
    assert result.artifacts == {}


# ---------------------------------------------------------------------------
# 9. RunResult
# ---------------------------------------------------------------------------


def test_run_result_creation() -> None:
    from uuid import uuid4

    from atm.core.types import RunResult

    run_id = uuid4()
    rr = RunResult(run_id=run_id, status="completed", task_result=None)
    assert rr.error is None
    assert rr.metrics == {}


# ---------------------------------------------------------------------------
# 10. PhaseTransition and TopologyTransition
# ---------------------------------------------------------------------------


def test_phase_transition_creation() -> None:
    from uuid import uuid4

    from atm.core.types import Phase, PhaseTransition

    run_id = uuid4()
    pt = PhaseTransition(
        run_id=run_id,
        from_phase=None,
        to_phase=Phase.PLANNING,
        entry_reason="init",
        iter_total=0,
        decided_by="initial",
    )
    assert isinstance(pt.id, UUID)
    assert pt.from_phase is None
    assert isinstance(pt.at, datetime)
    assert pt.at.tzinfo is not None


def test_topology_transition_creation() -> None:
    from uuid import uuid4

    from atm.core.types import Phase, TopologyTransition

    run_id = uuid4()
    tt = TopologyTransition(
        run_id=run_id,
        from_topology=None,
        to_topology="star",
        phase_at_decision=Phase.PLANNING,
        iter_within_phase=0,
        iter_within_topology=0,
        decided_by="initial",
        reason="start",
    )
    assert isinstance(tt.id, UUID)
    assert tt.considered_alternatives == ()
    assert tt.guards_applied == ()
    assert tt.signals_snapshot == {}
    assert tt.router_cost_usd == 0.0
    assert tt.at.tzinfo is not None


# ---------------------------------------------------------------------------
# 11. BudgetEvent
# ---------------------------------------------------------------------------


def test_budget_event_creation() -> None:
    from uuid import uuid4

    from atm.core.types import BudgetEvent

    run_id = uuid4()
    ev = BudgetEvent(
        run_id=run_id,
        level="run",
        event="warn",
        limit_usd=0.5,
        current_usd=0.4,
    )
    assert isinstance(ev.at, datetime)
    assert ev.at.tzinfo is not None


# ---------------------------------------------------------------------------
# 12. No DeprecationWarning on model creation (filterwarnings=["error"])
# ---------------------------------------------------------------------------


def test_no_deprecation_warning_on_creation() -> None:
    """datetime.now(timezone.utc) must be used — not deprecated utcnow()."""
    import warnings
    from uuid import uuid4

    from atm.core.types import BudgetEvent, Message, MessageKind, ToolCall

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Message(sender="a", kind=MessageKind.REQUEST, content="x")
        ToolCall(tool_name="t", args={}, issued_by="a")
        BudgetEvent(run_id=uuid4(), level="call", event="warn", limit_usd=1.0, current_usd=0.5)


# ---------------------------------------------------------------------------
# 13. Frozen models raise TypeError on mutation attempt
# ---------------------------------------------------------------------------


def test_all_models_are_frozen() -> None:
    """Verify that all models have frozen=True in their model_config."""
    from atm.core.types import (
        BudgetEvent,
        HumanContext,
        HumanResponse,
        LLMResponse,
        Message,
        PhaseTransition,
        RunResult,
        TaskResult,
        TaskSpec,
        TokenUsage,
        ToolCall,
        ToolResult,
        TopologyTransition,
    )

    frozen_classes = [
        Message,
        ToolCall,
        ToolResult,
        TokenUsage,
        LLMResponse,
        HumanContext,
        HumanResponse,
        TaskSpec,
        TaskResult,
        RunResult,
        PhaseTransition,
        TopologyTransition,
        BudgetEvent,
    ]
    for cls in frozen_classes:
        assert cls.model_config.get("frozen") is True, (
            f"{cls.__name__} should have frozen=True in model_config"
        )


# ---------------------------------------------------------------------------
# 14. CI-2: to_lc / from_lc stubs must raise NotImplementedError with "M2"
# ---------------------------------------------------------------------------


def test_to_lc_raises_not_implemented_with_m2() -> None:
    """CI-2: Message.to_lc() must raise NotImplementedError containing 'M2'."""
    from atm.core.types import Message, MessageKind

    msg = Message(sender="a", kind=MessageKind.REQUEST, content="hi")
    with pytest.raises(NotImplementedError, match="M2"):
        msg.to_lc()


def test_from_lc_raises_not_implemented_with_m2() -> None:
    """CI-2: Message.from_lc(None, sender=..., kind=...) must raise NotImplementedError containing 'M2'."""
    from atm.core.types import Message, MessageKind

    with pytest.raises(NotImplementedError, match="M2"):
        Message.from_lc(None, sender="a", kind=MessageKind.REQUEST)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 15. IR-3: Enum JSON roundtrip
# ---------------------------------------------------------------------------


def test_enum_json_roundtrip_raw() -> None:
    """IR-3a: json.dumps(enum.value) roundtrip works correctly."""
    from atm.core.types import AgentRole, MessageKind, Phase

    for enum_val in [Phase.EXECUTION, MessageKind.BROADCAST, AgentRole.CRITIC]:
        raw = json.dumps(enum_val.value)
        decoded = json.loads(raw)
        assert decoded == enum_val.value


def test_message_model_dump_json_roundtrip() -> None:
    """IR-3b: Message.model_dump_json() and model_validate_json() roundtrip."""
    from atm.core.types import Message, MessageKind

    msg = Message(
        sender="agent_x",
        kind=MessageKind.DECISION,
        content="final answer",
        recipients=("agent_y",),
    )
    json_str = msg.model_dump_json()
    restored = Message.model_validate_json(json_str)
    assert restored.id == msg.id
    assert restored.sender == msg.sender
    assert restored.kind == msg.kind
    assert restored.content == msg.content
    assert restored.recipients == msg.recipients


def test_budget_event_model_dump_json_roundtrip() -> None:
    """IR-3c: BudgetEvent JSON roundtrip."""
    from uuid import uuid4

    from atm.core.types import BudgetEvent

    ev = BudgetEvent(
        run_id=uuid4(),
        level="experiment",
        event="exceed",
        limit_usd=50.0,
        current_usd=51.2,
    )
    json_str = ev.model_dump_json()
    restored = BudgetEvent.model_validate_json(json_str)
    assert restored.run_id == ev.run_id
    assert restored.level == ev.level
    assert restored.event == ev.event
    assert restored.at == ev.at


# ---------------------------------------------------------------------------
# 16. Enums are str subclasses (for Postgres/JSON compatibility)
# ---------------------------------------------------------------------------


def test_enums_are_str_subclass() -> None:
    from atm.core.types import AgentRole, HumanRole, MessageKind, Phase

    for enum_class in [Phase, MessageKind, AgentRole, HumanRole]:
        for member in enum_class:
            assert isinstance(member, str), (
                f"{enum_class.__name__}.{member.name} should be a str subclass"
            )
            assert member == member.value, "str enum member should compare equal to its value"


# ---------------------------------------------------------------------------
# 17. Datetime fields are timezone-aware UTC
# ---------------------------------------------------------------------------


def test_datetime_fields_are_utc_aware() -> None:

    from atm.core.types import HumanResponse, Message, MessageKind, ToolCall

    msg = Message(sender="a", kind=MessageKind.REQUEST, content="test")
    tc = ToolCall(tool_name="t", args={}, issued_by="a")
    hr = HumanResponse(action="ok")

    for dt_val in [msg.created_at, tc.issued_at, hr.answered_at]:
        assert dt_val.tzinfo is not None, "datetime must be timezone-aware"
        assert dt_val.tzinfo == UTC or str(dt_val.tzinfo) in ("UTC", "utc")
