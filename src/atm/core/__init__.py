"""Public API for atm.core — M1 scope.

This module re-exports all public types, enums, reducers, and errors
defined in the atm.core submodules. Downstream modules (M2+) should
import exclusively from here rather than from submodules directly.

M1 scope includes:
  - Enums: AgentRole, HumanRole, Phase, MessageKind
  - Pydantic models: Message, ToolCall, ToolResult, TokenUsage, LLMResponse,
      HumanContext, HumanResponse, TaskSpec, TaskResult, RunResult,
      PhaseTransition, TopologyTransition, BudgetEvent
  - TypedDicts: AgentState, SharedState, GraphState
  - Reducers: merge_agent_states, dedup_by_id_reducer
  - Errors: AtmError, BudgetExceededError, PhaseError, ToolError
"""

from __future__ import annotations

from atm.core.errors import AtmError, BudgetExceededError, PhaseError, ToolError
from atm.core.reducers import dedup_by_id_reducer, merge_agent_states
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
    PhaseTransition,
    RunResult,
    TaskResult,
    TaskSpec,
    TokenUsage,
    ToolCall,
    ToolResult,
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
    "LLMResponse",
    "Message",
    "MessageKind",
    "Phase",
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
    "TopologyTransition",
    "dedup_by_id_reducer",
    "merge_agent_states",
]
