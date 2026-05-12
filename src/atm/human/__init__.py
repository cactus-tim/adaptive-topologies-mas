"""Human-in-the-Loop (HITL) gateway package for the ATM framework (M9).

Exports:
- HumanGateway    — Protocol that every gateway implementation must satisfy.
- HumanContext    — Context passed to a human/gateway at decision points.
- HumanResponse   — Response returned by a human/gateway.
- HumanRole       — Enum of 5 human participant roles.
- build_role_prompt — Build (system_prompt, user_prompt) for a given role + context.
"""

from atm.human.gateway import HumanContext, HumanGateway, HumanResponse, HumanRole
from atm.human.prompts import build_role_prompt

__all__ = [
    "HumanGateway",
    "HumanContext",
    "HumanResponse",
    "HumanRole",
    "build_role_prompt",
]
