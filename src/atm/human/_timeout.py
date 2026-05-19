"""Timeout wrapper for HumanGateway calls with fail/llm_fallback/skip policies."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from atm.human.gateway import HumanContext, HumanGateway, HumanResponse

if TYPE_CHECKING:
    pass

__all__ = ["request_with_timeout"]

_FALLBACK_TIMEOUT_S: float = 60.0

TimeoutPolicy = Literal["fail", "llm_fallback", "skip"]


async def request_with_timeout(
    gateway: HumanGateway,
    ctx: HumanContext,
    *,
    request_id: str,
    timeout_s: float | None,
    policy: TimeoutPolicy,
    llm_fallback_gateway: HumanGateway | None = None,
) -> HumanResponse:
    """Call ``gateway.request`` with an optional timeout and fallback policy.

    Raises:
        ValueError:    ``policy="llm_fallback"`` but ``llm_fallback_gateway`` is None.
        TimeoutError:  ``policy="fail"`` and timeout exceeded.
    """
    if policy == "llm_fallback" and llm_fallback_gateway is None:
        raise ValueError(
            "policy='llm_fallback' requires llm_fallback_gateway to be provided, "
            "but llm_fallback_gateway=None was given.  "
            "Pass a HumanGateway instance or switch policy to 'fail' or 'skip'."
        )

    if timeout_s is None:
        return await gateway.request(ctx, request_id=request_id)

    try:
        return await asyncio.wait_for(
            gateway.request(ctx, request_id=request_id),
            timeout=timeout_s,
        )
    except TimeoutError:
        return await _handle_timeout(
            ctx=ctx,
            request_id=request_id,
            policy=policy,
            llm_fallback_gateway=llm_fallback_gateway,
        )


async def _handle_timeout(
    *,
    ctx: HumanContext,
    request_id: str,
    policy: TimeoutPolicy,
    llm_fallback_gateway: HumanGateway | None,
) -> HumanResponse:
    """Dispatch timeout handling according to ``policy``."""

    if policy == "fail":
        raise TimeoutError(f"gateway.request timed out for request_id={request_id!r}")

    if policy == "llm_fallback":
        assert llm_fallback_gateway is not None  # for mypy
        fallback_response = await asyncio.wait_for(
            llm_fallback_gateway.request(ctx, request_id=request_id),
            timeout=_FALLBACK_TIMEOUT_S,
        )
        return fallback_response.model_copy(update={"source": "fallback"})

    if policy == "skip":
        return HumanResponse(
            action="timeout",
            source="timeout",
            timed_out=True,
            comment=f"Timed out waiting for human response (request_id={request_id!r})",
        )

    raise ValueError(f"Unknown timeout policy: {policy!r}")  # pragma: no cover
