"""Public API for atm.core — re-exports all public types, enums, reducers, and errors."""

from __future__ import annotations

from atm.core.errors import AtmError, BudgetExceededError, LLMError, PhaseError, ToolError
from atm.core.reducers import dedup_by_id_reducer, merge_agent_states
from atm.core.seed import seed_all
from atm.core.state import AgentState, GraphState, SharedState
from atm.core.types import (
    AgentRole,
    BudgetEvent,
    HumanContext,
    HumanResponse,
    HumanRole,
    LLMResponse,
    Message,
    MessageKind,
    Phase,
    PhaseDecision,
    PhaseTransition,
    RunResult,
    TaskResult,
    TaskSpec,
    TokenUsage,
    ToolCall,
    ToolResult,
    TopologyDecision,
    TopologyTransition,
)

__all__ = [
    "AgentRole",
    "AgentState",
    "AtmError",
    "BudgetEvent",
    "BudgetExceededError",
    "GraphState",
    "HumanContext",
    "HumanResponse",
    "HumanRole",
    "LLMError",
    "LLMResponse",
    "Message",
    "MessageKind",
    "Phase",
    "PhaseDecision",
    "PhaseError",
    "PhaseTransition",
    "RunResult",
    "SharedState",
    "TaskResult",
    "TaskSpec",
    "TokenUsage",
    "ToolCall",
    "ToolError",
    "ToolResult",
    "TopologyDecision",
    "TopologyTransition",
    "dedup_by_id_reducer",
    "merge_agent_states",
    "seed_all",
]
