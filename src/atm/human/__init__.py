"""Human-in-the-Loop (HITL) gateway package for the ATM framework (M9 + M9.1).

Public API
----------
- HumanGateway              — Protocol every gateway implementation must satisfy.
- HumanContext              — Context passed to a human/gateway at decision points.
- HumanResponse             — Response returned by a human/gateway.
- HumanRole                 — Enum of 5 human participant roles.
- LLMSimulatedGateway       — LLM-driven gateway for tests and offline experiments.
- CLIGateway                — stdin/stdout gateway for interactive debugging.
- request_with_timeout      — Async wrapper with three timeout/fallback policies.
- run_with_human            — Resume-loop orchestrator for interrupt-capable graphs.
- MaxInteractionsExceededError — Raised when max_interactions is exceeded.
- build_role_prompt         — Build (system_prompt, user_prompt) for a role + context.
- ROLE_SYSTEM_PROMPTS       — Mapping of HumanRole → per-role system prompt string.
- build_human_node_factory  — DRY factory for LangGraph HITL nodes (M9.1).
- HumanRoleRouter           — Protocol for dynamic HumanRole selection (M9.2).
- FixedRoleRouter           — Constant-role implementation (back-compat default).
- LLMRoleRouter             — LLM-based router with Pydantic validation and fallback (M9.2).
- RuleBasedRoleRouter       — Table-driven Phase → HumanRole implementation (M9.2).
"""

from atm.human._node_factory import build_human_node_factory
from atm.human._timeout import request_with_timeout
from atm.human.cli_gateway import CLIGateway
from atm.human.gateway import HumanContext, HumanGateway, HumanResponse, HumanRole
from atm.human.llm_simulated import LLMSimulatedGateway
from atm.human.prompts import ROLE_SYSTEM_PROMPTS, build_role_prompt
from atm.human.role_router import (
    FixedRoleRouter,
    HumanRoleRouter,
    LLMRoleRouter,
    RuleBasedRoleRouter,
)
from atm.human.runner import MaxInteractionsExceededError, run_with_human

__all__ = [
    "ROLE_SYSTEM_PROMPTS",
    "CLIGateway",
    "FixedRoleRouter",
    "HumanContext",
    "HumanGateway",
    "HumanResponse",
    "HumanRole",
    "HumanRoleRouter",
    "LLMRoleRouter",
    "LLMSimulatedGateway",
    "MaxInteractionsExceededError",
    "RuleBasedRoleRouter",
    "build_human_node_factory",
    "build_role_prompt",
    "request_with_timeout",
    "run_with_human",
]
