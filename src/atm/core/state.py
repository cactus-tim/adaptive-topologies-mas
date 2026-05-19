"""LangGraph-compatible TypedDict state containers (AgentState, SharedState, GraphState)."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from atm.core.reducers import dedup_by_id_reducer, merge_agent_states
from atm.core.types import (
    BudgetEvent,
    HumanResponse,
    LLMResponse,
    Message,
    Phase,
    ToolCall,
    ToolResult,
    TopologyTransition,
)


class AgentState(TypedDict, total=False):
    """Per-agent sub-state. Stored in GraphState under agent_id key."""

    agent_id: str
    role: str
    inbox: list[Message]
    outbox: list[Message]
    scratchpad: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    summary_before_window: str
    step_count: int
    tokens_spent: int
    cost_spent_usd: float


class SharedState(TypedDict, total=False):
    """Global graph space: everything visible to all agents."""

    task_id: str
    task_input: str
    final_answer: str | None

    phase: Phase
    phase_started_at_iter: int
    phase_history: list[Phase]

    active_topology: str | None
    topology_started_at_iter: int
    topology_switch_count: int
    topology_history: list[str]

    iteration: int
    iter_total: int
    meta_ticks: int

    broadcast_bus: list[Message]

    human_requests: list[dict[str, Any]]
    human_responses: list[HumanResponse]

    signals: dict[str, Any]


class GraphState(TypedDict, total=False):
    """Full graph state. Top-level keys:
    - 'shared'               — SharedState
    - 'agents'               — dict[agent_id, AgentState] with merge reducer
    - 'messages'             — flat history (dedup-by-id on fan-in from subgraphs)
    - 'llm_calls'            — LLMResponse history (for observability and replay)
    - 'budget_events'        — intra-run budget signals
    - 'topology_transitions' — TopologyRouter decisions (L2); source for RQ2 analysis
    """

    shared: SharedState
    agents: Annotated[dict[str, AgentState], merge_agent_states]
    messages: Annotated[list[Message], dedup_by_id_reducer("id", sort_by="created_at")]
    llm_calls: Annotated[list[LLMResponse], dedup_by_id_reducer("id", sort_by="started_at")]
    budget_events: Annotated[list[BudgetEvent], dedup_by_id_reducer("id", sort_by="at")]
    topology_transitions: Annotated[
        list[TopologyTransition], dedup_by_id_reducer("id", sort_by="at")
    ]
