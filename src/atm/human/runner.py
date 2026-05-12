"""run_with_human — resume-loop orchestrator for HITL-enabled LangGraph runs (M9 Step 5.1).

This helper wraps a compiled LangGraph and drives the full interrupt/resume cycle.
The graph may interrupt zero or more times; the helper resolves each interrupt by
calling the appropriate gateway and supplying a ``Command(resume=...)`` continuation.

Current M9 scope
----------------
In M9, the Chain topology's ``human_reviewer`` node calls gateways **synchronously
inside the node** (no LangGraph ``interrupt()`` call is made from that node).  The
helper is therefore "no-op" for existing Chain HITL runs — it simply calls ``ainvoke``
once and returns the state.

The interrupt/resume path IS exercised by the unit tests via a
``FakeInterruptingGraph`` stub, validating the forward-compatible design for M14+,
where real ``interrupt()`` calls will be introduced.

Interrupt payload contract (M9 Step 4.1 / future)
-------------------------------------------------
When a node calls ``interrupt(payload)``, the payload is expected to be a dict with
at minimum:

    {
        "ctx":        <HumanContext.model_dump(mode='json')>,
        "request_id": <str>,
    }

Optionally it may contain ``"gateway"`` (a gateway name string) for future
dispatch routing.  Currently the helper uses the single ``gateway`` kwarg passed at
call time.

Idempotency
-----------
To handle crash-recovery scenarios where the same ``(thread_id, request_id)`` pair
is interrupted again on re-execution (the graph is replayed from the checkpoint),
the helper caches resolved responses in a ``dict[(thread_id, request_id), dict]``
(the JSON-serialised model dump, ready to pass to ``Command(resume=...)``) and
short-circuits gateway calls for already-seen pairs.

Guards
------
``max_interactions`` (default 10) caps the number of resume cycles to prevent
runaway loops.  Exceeding the limit raises ``MaxInteractionsExceededError``.

The helper also detects the pathological case where a single invocation result
contains more than one interrupt simultaneously (which the current LangGraph version
does not produce, but could in principle arise from future multi-node parallelism
bugs) and raises ``RuntimeError``.
"""

from __future__ import annotations

import logging
from typing import Any

from atm.core.types import HumanContext, HumanResponse
from atm.human.gateway import HumanGateway

logger = logging.getLogger(__name__)

__all__ = ["run_with_human", "MaxInteractionsExceededError"]


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------


class MaxInteractionsExceededError(RuntimeError):
    """Raised when ``run_with_human`` exceeds ``max_interactions`` resume cycles."""

    def __init__(self, max_interactions: int, thread_id: str) -> None:
        super().__init__(
            f"run_with_human exceeded max_interactions={max_interactions} "
            f"for thread_id={thread_id!r}. "
            "This may indicate an infinite interrupt loop. "
            "Increase max_interactions or inspect the graph logic."
        )
        self.max_interactions = max_interactions
        self.thread_id = thread_id


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


async def run_with_human(
    compiled_graph: Any,
    initial_state: Any,
    *,
    thread_id: str,
    gateway: HumanGateway | None = None,
    checkpointer: Any = None,
    max_interactions: int = 10,
) -> Any:
    """Drive a compiled LangGraph through its full lifecycle, handling interrupts.

    Executes the graph via ``ainvoke`` and loops until the graph reaches END
    (i.e., the result dict no longer contains ``__interrupt__``).

    Parameters
    ----------
    compiled_graph:
        A compiled LangGraph (``CompiledStateGraph``) that may or may not
        contain nodes that call ``interrupt()``.
    initial_state:
        The initial state dict to pass to the first ``ainvoke`` call.
    thread_id:
        LangGraph thread ID for checkpointing.  Required for interrupt/resume
        to work correctly with a stateful checkpointer.
    gateway:
        The :class:`~atm.human.gateway.HumanGateway` to call when an interrupt
        is encountered.  Required if the graph ever calls ``interrupt()``.
        May be ``None`` for graphs that never interrupt.
    checkpointer:
        Optional LangGraph checkpointer.  When provided, it is passed to the
        graph's ``ainvoke`` config (some graph variants accept it here).
        Note: the checkpointer is typically already baked into the compiled
        graph at ``compile()`` time; this parameter is available for callers
        that need to override it at invocation time.
    max_interactions:
        Maximum number of interrupt/resume cycles to allow before raising
        :class:`MaxInteractionsExceededError`.  Defaults to 10.

    Returns
    -------
    Any
        The final state dict from the graph (the last ``ainvoke`` result with
        no ``__interrupt__`` key, or with an empty interrupt list).

    Raises
    ------
    MaxInteractionsExceededError
        If more than ``max_interactions`` interrupts occur.
    RuntimeError
        If a single invocation result contains more than one interrupt
        simultaneously.
    ValueError
        If an interrupt is encountered but ``gateway`` is ``None``.
    """
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    # In-process idempotency cache: (thread_id, request_id) → response payload dict
    _resolved: dict[tuple[str, str], dict[str, Any]] = {}

    interaction_count = 0

    # First invocation — pass the initial state
    result: Any = await compiled_graph.ainvoke(initial_state, config=config)

    while True:
        interrupts = _extract_interrupts(result)

        if not interrupts:
            # Graph reached END normally — return final state.
            return result

        # Pathological guard: more than 1 interrupt in a single result.
        if len(interrupts) > 1:
            raise RuntimeError(
                f"run_with_human: received {len(interrupts)} simultaneous interrupts "
                f"in a single invocation result for thread_id={thread_id!r}. "
                "Only one interrupt per invocation is expected. "
                "This may indicate a bug in the graph or a future LangGraph feature "
                "that requires updated handling."
            )

        if interaction_count >= max_interactions:
            raise MaxInteractionsExceededError(max_interactions, thread_id)

        interrupt = interrupts[0]
        payload: dict[str, Any] = interrupt.value if hasattr(interrupt, "value") else {}

        # --- Resolve request_id from interrupt payload ---
        request_id: str = str(payload.get("request_id", ""))

        # --- Idempotency check ---
        cache_key = (thread_id, request_id)
        if cache_key in _resolved:
            logger.debug(
                "run_with_human: idempotency cache hit for thread_id=%r request_id=%r",
                thread_id,
                request_id,
            )
            resume_payload = _resolved[cache_key]
        else:
            # Need a gateway to resolve the interrupt.
            if gateway is None:
                raise ValueError(
                    f"run_with_human: received an interrupt (request_id={request_id!r}) "
                    f"but gateway=None. Provide a HumanGateway instance to handle interrupts."
                )

            # --- Reconstruct HumanContext from payload ---
            ctx_data = payload.get("ctx")
            if ctx_data is None:
                raise ValueError(
                    f"run_with_human: interrupt payload missing 'ctx' key. "
                    f"Full payload: {payload!r}"
                )
            ctx = HumanContext.model_validate(ctx_data)

            logger.debug(
                "run_with_human: calling gateway for thread_id=%r request_id=%r",
                thread_id,
                request_id,
            )

            response: HumanResponse = await gateway.request(ctx, request_id=request_id)
            resume_payload = response.model_dump(mode="json")

            # Store in idempotency cache
            _resolved[cache_key] = resume_payload

        interaction_count += 1

        # --- Resume the graph ---
        from langgraph.types import Command

        result = await compiled_graph.ainvoke(
            Command(resume=resume_payload),
            config=config,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_interrupts(result: Any) -> list[Any]:
    """Extract the list of Interrupt objects from an ainvoke result dict.

    LangGraph places interrupts under the ``__interrupt__`` key in the returned
    state dict.  The value is a sequence of ``Interrupt`` objects.

    Returns an empty list if the result is not a dict, the key is absent,
    or the value is empty/falsy.
    """
    if not isinstance(result, dict):
        return []
    raw = result.get("__interrupt__")
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    # Single interrupt object (defensive, LangGraph always returns a list)
    return [raw]
