"""HumanGateway Protocol — entry point for all HITL interactions.

Idempotency contract: same (run_id, request_id) pair MUST yield the same
HumanResponse and MUST NOT create duplicate DB rows or side-effects.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from atm.core.types import HumanContext, HumanResponse, HumanRole

__all__ = [
    "HumanContext",
    "HumanGateway",
    "HumanResponse",
    "HumanRole",
]


@runtime_checkable
class HumanGateway(Protocol):
    """Protocol for human-in-the-loop decision gateways (structural subtyping).

    Idempotency contract: the same (run_id, request_id) pair MUST yield the same
    HumanResponse and MUST NOT cause duplicate DB writes.
    """

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        """Request a human decision; idempotent on (run_id, request_id)."""
        ...
