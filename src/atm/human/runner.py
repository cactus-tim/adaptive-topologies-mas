"""run_with_human — resume-loop orchestrator for HITL-enabled LangGraph runs."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event

from atm.core.types import HumanContext, HumanResponse
from atm.human._timeout import request_with_timeout
from atm.human.gateway import HumanGateway

logger = logging.getLogger(__name__)

__all__ = ["MaxInteractionsExceededError", "run_with_human"]


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


async def run_with_human(
    compiled_graph: Any,
    initial_state: Any,
    *,
    thread_id: str,
    gateway: HumanGateway | None = None,
    checkpointer: Any = None,
    max_interactions: int = 10,
    timeout_s: float | None = None,
    timeout_policy: str = "skip",
    fallback_gateway: HumanGateway | None = None,
) -> Any:
    """Drive a compiled LangGraph through its full lifecycle, resolving interrupts.

    Raises:
        MaxInteractionsExceededError: exceeds ``max_interactions`` cycles.
        RuntimeError:                 multiple simultaneous interrupts in one result.
        ValueError:                   interrupt encountered but ``gateway=None``.
    """
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    _resolved: dict[tuple[str, str], dict[str, Any]] = {}

    interaction_count = 0

    result: Any = await compiled_graph.ainvoke(initial_state, config=config)

    while True:
        interrupts = _extract_interrupts(result)

        if not interrupts:
            return result

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

        request_id: str = str(payload.get("request_id", ""))

        cache_key = (thread_id, request_id)
        if cache_key in _resolved:
            logger.debug(
                "run_with_human: idempotency cache hit for thread_id=%r request_id=%r",
                thread_id,
                request_id,
            )
            resume_payload = _resolved[cache_key]
        else:
            if gateway is None:
                raise ValueError(
                    f"run_with_human: received an interrupt (request_id={request_id!r}) "
                    f"but gateway=None. Provide a HumanGateway instance to handle interrupts."
                )

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

            _run_id = ctx.run_id
            try:
                await adispatch_custom_event(
                    "human_request",
                    {
                        "run_id": _run_id,
                        "request_id": request_id,
                        "role": str(ctx.role.value if hasattr(ctx.role, "value") else ctx.role),
                        "context_json": ctx.model_dump(mode="json"),
                        "requested_at": datetime.now(UTC),
                    },
                )
            except Exception:
                logger.debug(
                    "run_with_human: adispatch human_request skipped (no callback ctx)",
                    exc_info=True,
                )

            _t0 = time.monotonic()
            if timeout_s is not None:
                response: HumanResponse = await request_with_timeout(
                    gateway,
                    ctx,
                    request_id=request_id,
                    timeout_s=timeout_s,
                    policy=timeout_policy,  # type: ignore[arg-type]
                    llm_fallback_gateway=fallback_gateway,
                )
            else:
                response = await gateway.request(ctx, request_id=request_id)
            _latency_s = time.monotonic() - _t0

            resume_payload = response.model_dump(mode="json")

            try:
                await adispatch_custom_event(
                    "human_response",
                    {
                        "run_id": _run_id,
                        "request_id": request_id,
                        "answered_at": datetime.now(UTC),
                        "response_json": resume_payload,
                        "source": response.source,
                        "timed_out": response.timed_out,
                        "latency_s": _latency_s,
                    },
                )
            except Exception:
                logger.debug(
                    "run_with_human: adispatch human_response skipped (no callback ctx)",
                    exc_info=True,
                )

            _resolved[cache_key] = resume_payload

        interaction_count += 1

        from langgraph.types import Command

        result = await compiled_graph.ainvoke(
            Command(resume=resume_payload),
            config=config,
        )


def _extract_interrupts(result: Any) -> list[Any]:
    """Extract Interrupt objects from an ainvoke result dict (``__interrupt__`` key)."""
    if not isinstance(result, dict):
        return []
    raw = result.get("__interrupt__")
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]
