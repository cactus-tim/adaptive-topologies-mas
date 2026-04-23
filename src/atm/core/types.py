"""Core Pydantic v2 data models and enums for the ATM framework.

All 17 classes/enums defined here are used throughout the framework.
LangChain integration (Message.to_lc / from_lc) is stubbed — implemented in M2.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from langchain_core.messages import (  # type: ignore[import-not-found]  # not installed until M2
        BaseMessage,
    )


# ---------------------------------------------------------------------------
# Helper — use this in default_factory to avoid deprecated datetime.utcnow()
# ---------------------------------------------------------------------------


def _utcnow() -> _dt.datetime:
    """Return current UTC time as a timezone-aware datetime (not deprecated)."""
    return _dt.datetime.now(_dt.UTC)


# ---------------------------------------------------------------------------
# Enums — StrEnum subclasses for Postgres/JSON compatibility
# ---------------------------------------------------------------------------


class AgentRole(StrEnum):
    """Role of an LLM agent within the topology."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    CRITIC = "critic"
    DEBATER = "debater"
    COORDINATOR = "coordinator"  # for Hierarchical topology


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

    REQUEST = "request"  # addressed: agent → agent
    BROADCAST = "broadcast"  # to shared bus (Mesh)
    DRAFT = "draft"
    CRITIQUE = "critique"
    DECISION = "decision"  # Coordinator → final answer
    PHASE_EMIT = "phase_emit"  # agent requests phase transition


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """Inter-agent message. Immutable. Domain model parallel to LangChain BaseMessage.

    The LangChain boundary is strictly at LLMWrapper.ainvoke / from_lc:
    topology and agents work only with Message; the adapter is called
    at the LLM invocation point.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    sender: str  # agent_id
    recipients: tuple[str, ...] = ()  # empty = broadcast
    kind: MessageKind
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    refs: tuple[UUID, ...] = ()  # reply-to chain
    created_at: _dt.datetime = Field(default_factory=_utcnow)

    def to_lc(self) -> BaseMessage:
        """Adapter to LangChain BaseMessage for LLM invocation.

        Mapping kind → LC-type:
          request/draft/critique → HumanMessage (role = sender)
          decision               → AIMessage
          broadcast              → HumanMessage with metadata={'channel':'broadcast'}
          phase_emit             → SystemMessage (meta-event)

        NOTE: Implemented in M2.
        """
        raise NotImplementedError("Message.to_lc is not implemented until M2")

    @classmethod
    def from_lc(
        cls,
        lc_msg: BaseMessage,
        *,
        sender: str,
        kind: MessageKind,
    ) -> Message:
        """Reverse adapter from LangChain BaseMessage.

        Contract: LC-specific fields (tool_calls, additional_kwargs) go into
        `payload` under key '_lc'. Callers are responsible for extracting them
        if needed (they are not first-class Message attributes).

        NOTE: Implemented in M2.
        """
        raise NotImplementedError("Message.from_lc is not implemented until M2")


class ToolCall(BaseModel):
    """A single tool invocation requested by an agent."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    args: dict[str, Any]
    issued_by: str  # agent_id
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
    total_tokens: int  # invariant: prompt + completion (cached is a subset of prompt)


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
    raw: dict[str, Any] = Field(default_factory=dict)  # for debug; not fully written to Parquet


class HumanContext(BaseModel):
    """What the topology passes to the human/gateway for decision-making."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    role: HumanRole
    question: str
    recent_messages: tuple[Message, ...]
    artifacts: dict[str, Any] = Field(default_factory=dict)  # draft, tests, diffs, etc.
    allowed_actions: tuple[str, ...]  # e.g. ("approve","reject","revise")
    deadline_s: int | None = None


class HumanResponse(BaseModel):
    """Response from a human participant or gateway."""

    model_config = ConfigDict(frozen=True)

    action: str  # one of allowed_actions OR 'timeout'/'cancelled'
    comment: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)  # structured edits
    answered_at: _dt.datetime = Field(default_factory=_utcnow)
    tlx_scores: dict[str, int] | None = None  # 6 NASA-TLX scales, 0..100
    timed_out: bool = False  # True if gateway returned timeout-response
    source: Literal["human", "llm_sim", "fallback", "timeout"] = "human"


class TaskSpec(BaseModel):
    """Description of a single benchmark task."""

    model_config = ConfigDict(frozen=True)

    id: str  # "humaneval/HumanEval/0"
    type: Literal["programming", "qa", "creative", "analysis"]
    input: str
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluator_key: str  # key in EvaluatorRegistry


class TaskResult(BaseModel):
    """Result of a single run against a single TaskSpec. Immutable post-run."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    final_answer: str
    artifacts: dict[str, Any] = Field(default_factory=dict)  # code, tests, diffs
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
    metrics: dict[str, float] = Field(default_factory=dict)  # filled by Evaluator


class PhaseTransition(BaseModel):
    """Phases are monotonic (planning → execution → verification → done).
    A record is created only when phase advances; intra-phase topology events
    are captured in TopologyTransition.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    from_phase: Phase | None  # None only on initial init
    to_phase: Phase
    entry_reason: str
    iter_total: int  # absolute meta-graph tick counter
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
    from_topology: str | None  # None only on the initial decision
    to_topology: str  # == from_topology if no-change
    phase_at_decision: Phase
    iter_within_phase: int
    iter_within_topology: int  # 0 if this is a switch (new topology)
    decided_by: Literal["rule", "llm_router", "oracle", "guard_override", "initial"]
    reason: str
    considered_alternatives: tuple[str, ...] = ()
    guards_applied: tuple[str, ...] = ()  # names of guards that fired
    signals_snapshot: dict[str, Any] = Field(default_factory=dict)
    router_cost_usd: float = 0.0  # >0 only for llm_router
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
