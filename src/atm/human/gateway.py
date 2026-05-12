"""HumanGateway Protocol — entry point for all HITL interactions (M9 Step 1.1).

Idempotency contract
--------------------
Every HumanGateway implementation MUST be idempotent with respect to the pair
(run_id, request_id).  That is:

  * If called twice with the same run_id (embedded in ``ctx``) and the same
    ``request_id``, the gateway MUST return the same HumanResponse.
  * The second call MUST NOT produce a duplicate DB row or trigger side-effects
    (e.g., send a duplicate notification).

Database-level enforcement (UNIQUE constraint on ``(run_id, request_id)`` +
``ON CONFLICT DO NOTHING / RETURNING``) is added in Step 5 (migration).
Gateway implementations are responsible for honouring the contract at the
application level in the meantime.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Re-export from core so callers can do:
#   from atm.human.gateway import HumanContext, HumanResponse, HumanRole
# without importing directly from atm.core.types.
from atm.core.types import HumanContext, HumanResponse, HumanRole

__all__ = [
    "HumanContext",
    "HumanGateway",
    "HumanResponse",
    "HumanRole",
]


@runtime_checkable
class HumanGateway(Protocol):
    """Protocol for human-in-the-loop decision gateways.

    Any class that exposes an ``async def request(self, ctx, *, request_id)``
    method with the correct signature satisfies this Protocol (structural
    subtyping — no explicit inheritance required).

    Idempotency contract: same (run_id, request_id) pair MUST yield the same
    HumanResponse and MUST NOT result in duplicate DB writes.  See module-level
    docstring for details.
    """

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        """Request a human decision for the given context.

        Parameters
        ----------
        ctx:
            Full context describing the decision point (run_id, role, question,
            recent messages, allowed actions, optional deadline).
        request_id:
            Caller-assigned stable identifier for this interaction.  Used to
            enforce idempotency: calling ``request`` twice with the same
            ``request_id`` within the same run MUST return the same response
            without creating duplicate side-effects.

        Returns
        -------
        HumanResponse
            The human participant's (or simulated) decision.
        """
        ...
