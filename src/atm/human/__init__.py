"""Human-in-the-Loop (HITL) gateway package for the ATM framework (M9).

Public API
----------
- HumanGateway         — Protocol every gateway implementation must satisfy.
- HumanContext         — Context passed to a human/gateway at decision points.
- HumanResponse        — Response returned by a human/gateway.
- HumanRole            — Enum of 5 human participant roles.
- LLMSimulatedGateway  — LLM-driven gateway for tests and offline experiments.
- CLIGateway           — stdin/stdout gateway for interactive debugging.
- request_with_timeout — Async wrapper with three timeout/fallback policies.
- run_with_human       — Resume-loop orchestrator for interrupt-capable graphs.
- MaxInteractionsExceededError — Raised when max_interactions is exceeded.
- build_role_prompt    — Build (system_prompt, user_prompt) for a role + context.
- ROLE_SYSTEM_PROMPTS  — Mapping of HumanRole → per-role system prompt string.
"""

from atm.human.gateway import HumanContext, HumanGateway, HumanResponse, HumanRole
from atm.human.llm_simulated import LLMSimulatedGateway
from atm.human.cli_gateway import CLIGateway
from atm.human._timeout import request_with_timeout
from atm.human.runner import run_with_human, MaxInteractionsExceededError
from atm.human.prompts import build_role_prompt, ROLE_SYSTEM_PROMPTS

__all__ = [
    "HumanGateway",
    "HumanContext",
    "HumanResponse",
    "HumanRole",
    "LLMSimulatedGateway",
    "CLIGateway",
    "request_with_timeout",
    "run_with_human",
    "MaxInteractionsExceededError",
    "build_role_prompt",
    "ROLE_SYSTEM_PROMPTS",
]
