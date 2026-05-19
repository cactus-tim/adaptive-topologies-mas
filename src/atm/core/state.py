"""LangGraph-compatible TypedDict state containers.

Three TypedDicts:
  - AgentState  — per-agent mutable state
  - SharedState — cross-agent shared state (phase, topology L2, signals, counters)
  - GraphState  — top-level LangGraph state with Annotated reducer fields

Reference: arch.md §3.2
"""

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
    role: str  # AgentRole.value
    inbox: list[Message]  # messages addressed to this agent
    outbox: list[Message]  # for routing by topology router
    scratchpad: list[dict[str, Any]]  # append-only journal
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    summary_before_window: str  # summarizer output (scratchpad policy C)
    step_count: int
    tokens_spent: int
    cost_spent_usd: float


class SharedState(TypedDict, total=False):
    """Global graph space: everything visible to all agents."""

    task_id: str
    task_input: str
    final_answer: str | None

    # --- Phase axis (monotonic: planning → execution → verification → done) ---
    phase: Phase
    phase_started_at_iter: int  # abs meta-graph tick when phase was entered
    phase_history: list[Phase]

    # --- Topology axis (can change at runtime within a phase) ---
    active_topology: str | None  # current active topology name (arch.md §3.2 line 419)
    topology_started_at_iter: int  # abs tick when current topology was activated
    topology_switch_count: int  # count of actual switches (for max_switches guard)
    topology_history: list[str]  # short tail (last K) for cooldown-check

    # --- Adaptive iteration counters ---
    iteration: int  # legacy/general (iterations within current subgraph)
    iter_total: int  # cumulative count of sub-topology work ticks (chain retry,
    # star coordinator cycle, mesh dispatch round, etc.).  Sub-topologies are
    # the sole writers; adaptive's dispatch node does NOT increment it (would
    # double-count vs static baselines).
    meta_ticks: int  # number of adaptive meta-graph cycles
    # (phase_router → topology_router → dispatch → transition_gate).
    # Adaptive-only; static topologies never write this.  Used as a safety
    # cap to prevent infinite meta-graph spin.

    # --- Communication/Mesh ---
    broadcast_bus: list[Message]  # for Mesh; cleared on any transition

    # --- HITL ---
    human_requests: list[dict[str, Any]]  # pending HITL requests (for debug)
    human_responses: list[HumanResponse]

    # --- Agent → router channel (L2 quasi-preemption) ---
    # Agents emit signals (stuck, rejected_count, needs_debate, ready_for_*).
    # TransitionGate passes them to TopologyRouter, then clears consumed keys.
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
