"""Core Pydantic v2 data models and enums for the ATM framework."""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from langchain_core.messages import (
        BaseMessage,
    )


def _utcnow() -> _dt.datetime:
    """Return current UTC time as a timezone-aware datetime (not deprecated)."""
    return _dt.datetime.now(_dt.UTC)


class AgentRole(StrEnum):
    """Role of an LLM agent within the topology."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    CRITIC = "critic"
    DEBATER = "debater"
    COORDINATOR = "coordinator"


class HumanRole(StrEnum):
    """Role of a human participant in the HITL loop."""

    COORDINATOR = "coordinator"
    REVIEWER = "reviewer"
    JUDGE = "judge"
    PEER = "peer"
    MONITOR = "monitor"


class Phase(StrEnum):
    """Monotonic phase of a run (planning → execution → verification → done)."""

    PLANNING = "planning"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    DONE = "done"


class MessageKind(StrEnum):
    """Semantic type of an inter-agent message."""

    REQUEST = "request"
    BROADCAST = "broadcast"
    DRAFT = "draft"
    CRITIQUE = "critique"
    DECISION = "decision"
    PHASE_EMIT = "phase_emit"


class Message(BaseModel):
    """Inter-agent message. Immutable. Domain model parallel to LangChain BaseMessage.

    The LangChain boundary is strictly at LLMWrapper.ainvoke / from_lc:
    topology and agents work only with Message; the adapter is called
    at the LLM invocation point.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    sender: str
    recipients: tuple[str, ...] = ()
    kind: MessageKind
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    refs: tuple[UUID, ...] = ()
    created_at: _dt.datetime = Field(default_factory=_utcnow)

    def to_lc(self) -> BaseMessage:
        """Adapter to LangChain BaseMessage for LLM invocation."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        if self.kind in (
            MessageKind.REQUEST,
            MessageKind.DRAFT,
            MessageKind.CRITIQUE,
        ):
            return HumanMessage(content=self.content)

        if self.kind == MessageKind.DECISION:
            return AIMessage(content=self.content)

        if self.kind == MessageKind.BROADCAST:
            return HumanMessage(
                content=self.content,
                additional_kwargs={"channel": "broadcast"},
            )

        if self.kind == MessageKind.PHASE_EMIT:
            return SystemMessage(content=self.content)

        return HumanMessage(content=self.content)  # pragma: no cover

    @classmethod
    def from_lc(
        cls,
        lc_msg: BaseMessage,
        *,
        sender: str,
        kind: MessageKind,
    ) -> Message:
        """Reverse adapter from LangChain BaseMessage.

        LC-specific fields go into ``payload['_lc']``.
        """
        if isinstance(lc_msg.content, str):
            content = lc_msg.content
        else:
            parts: list[str] = []
            for block in lc_msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            content = "".join(parts)

        lc_extras: dict[str, Any] = {}
        if lc_msg.additional_kwargs:
            lc_extras["additional_kwargs"] = dict(lc_msg.additional_kwargs)

        tool_calls = getattr(lc_msg, "tool_calls", None)
        if tool_calls:
            lc_extras["tool_calls"] = list(tool_calls)

        payload: dict[str, Any] = {"_lc": lc_extras} if lc_extras else {"_lc": {}}

        return cls(
            sender=sender,
            kind=kind,
            content=content,
            payload=payload,
        )


class ToolCall(BaseModel):
    """A single tool invocation requested by an agent."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    args: dict[str, Any]
    issued_by: str
    issued_at: _dt.datetime = Field(default_factory=_utcnow)


class ToolResult(BaseModel):
    """Result of a tool invocation."""

    model_config = ConfigDict(frozen=True)

    call_id: UUID
    ok: bool
    output: Any
    error: str | None = None
    latency_ms: int
    finished_at: _dt.datetime = Field(default_factory=_utcnow)


class TokenUsage(BaseModel):
    """Token usage statistics for a single LLM call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int
    completion_tokens: int
    cached_input_tokens: int = 0
    total_tokens: int


class LLMResponse(BaseModel):
    """Result of a single LLM ainvoke. Contains text and/or tool-calls if requested."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    model: str
    text: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage
    cost_usd: float
    latency_ms: int
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter", "error"]
    started_at: _dt.datetime = Field(default_factory=_utcnow)
    raw: dict[str, Any] = Field(default_factory=dict)


class HumanContext(BaseModel):
    """What the topology passes to the human/gateway for decision-making."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    role: HumanRole
    question: str
    recent_messages: tuple[Message, ...]
    artifacts: dict[str, Any] = Field(default_factory=dict)
    allowed_actions: tuple[str, ...]
    deadline_s: int | None = None


class HumanResponse(BaseModel):
    """Response from a human participant or gateway."""

    model_config = ConfigDict(frozen=True)

    action: str
    comment: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    answered_at: _dt.datetime = Field(default_factory=_utcnow)
    tlx_scores: dict[str, int] | None = None
    timed_out: bool = False
    source: Literal["human", "llm_sim", "fallback", "timeout"] = "human"


class TaskSpec(BaseModel):
    """Description of a single benchmark task."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["programming", "reasoning", "creative", "decision"]
    input: str
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluator_key: str


class TaskResult(BaseModel):
    """Result of a single run against a single TaskSpec. Immutable post-run."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    final_answer: str
    artifacts: dict[str, Any] = Field(default_factory=dict)
    iterations_used: int
    budget_spent_usd: float
    wall_time_s: float


class RunResult(BaseModel):
    """Aggregate outcome of a complete graph run."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    status: Literal["completed", "failed", "budget_exceeded", "cancelled"]
    task_result: TaskResult | None
    error: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class PhaseTransition(BaseModel):
    """Phases are monotonic (planning → execution → verification → done).
    A record is created only when phase advances; intra-phase topology events
    are captured in TopologyTransition.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    from_phase: Phase | None
    to_phase: Phase
    entry_reason: str
    iter_total: int
    decided_by: Literal["rule", "llm_router", "agent_emit", "initial"]
    at: _dt.datetime = Field(default_factory=_utcnow)


class TopologyTransition(BaseModel):
    """TopologyRouter decision (recorded BOTH on actual switch AND 'no change'
    — for complete RQ2 analysis).
    Source of truth for all runtime topology adaptation metrics.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    from_topology: str | None
    to_topology: str
    phase_at_decision: Phase
    iter_within_phase: int
    iter_within_topology: int
    decided_by: Literal[
        "rule", "llm_router", "oracle", "guard_override", "initial", "human_override"
    ]
    reason: str
    considered_alternatives: tuple[str, ...] = ()
    guards_applied: tuple[str, ...] = ()
    signals_snapshot: dict[str, Any] = Field(default_factory=dict)
    router_cost_usd: float = 0.0
    at: _dt.datetime = Field(default_factory=_utcnow)


class BudgetEvent(BaseModel):
    """A budget warning or exceed event during a run."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    level: Literal["call", "run", "experiment"]
    event: Literal["warn", "exceed"]
    limit_usd: float
    current_usd: float
    at: _dt.datetime = Field(default_factory=_utcnow)


class TopologyDecision(BaseModel):
    """Decision produced by TopologyRouter on each meta-graph tick."""

    model_config = ConfigDict(frozen=True)

    topology: str
    reason: str
    decided_by: Literal[
        "rule", "llm_router", "oracle", "guard_override", "initial", "human_override"
    ]
    considered_alternatives: tuple[str, ...] = ()
    router_cost_usd: float = 0.0

    @field_validator("router_cost_usd")
    @classmethod
    def _router_cost_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("router_cost_usd must be >= 0")
        return v


class PhaseDecision(BaseModel):
    """Decision produced by PhaseRouter on each meta-graph tick.

    ``next_phase`` must be monotonic (>= current phase), enforced in TransitionGate.
    """

    model_config = ConfigDict(frozen=True)

    next_phase: Phase
    reason: str
    decided_by: Literal["rule", "llm_router", "agent_emit", "initial"]
