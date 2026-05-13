"""Timeout wrapper + fallback policy helper for HumanGateway calls (M9 Step 2.5).

Three timeout policies
----------------------
- **fail**:        Propagate TimeoutError to the caller — hard failure.
- **llm_fallback**: On timeout, call ``llm_fallback_gateway`` (fixed 60 s inner
                   timeout) and return its response with ``source='fallback'``.
                   Requires ``llm_fallback_gateway`` to be supplied; raises
                   ``ValueError`` at call-time if it is ``None``.
- **skip**:        On timeout, return a synthetic ``HumanResponse`` with
                   ``timed_out=True``, ``source='timeout'``, ``action='timeout'``
                   and a comment containing the ``request_id`` for traceability.
                   No error is raised.

When ``timeout_s is None``, the call is forwarded directly to the gateway
without any timeout logic applied (equivalent to no deadline).

Design decisions
----------------
- ``asyncio.wait_for`` is used as the timeout primitive.  Python 3.11 raises
  ``TimeoutError`` (the built-in, which is an alias for ``asyncio.TimeoutError``
  since 3.11); earlier versions raise ``asyncio.TimeoutError``.  We catch
  ``TimeoutError`` (base class covers both) and re-raise or handle per policy.
- The fallback call uses a fixed 60 s inner timeout so it cannot itself hang
  indefinitely.  If the fallback also times out, the resulting ``TimeoutError``
  propagates to the caller (fail-fast — do not silently lose data).
- ``asyncio.to_thread(input, ...)`` used by ``CLIGateway`` is NOT cancelled
  reliably; this is documented at the CLIGateway level.  ``_timeout.py`` makes
  no special provision for it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Literal

from atm.human.gateway import HumanContext, HumanGateway, HumanResponse

if TYPE_CHECKING:
    pass

__all__ = ["request_with_timeout"]

# Fixed inner timeout for the fallback gateway call (seconds).
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
    """Call ``gateway.request`` with an optional timeout and a fallback policy.

    Parameters
    ----------
    gateway:
        Primary ``HumanGateway`` to call.
    ctx:
        Human interaction context (run_id, role, question, …).
    request_id:
        Caller-assigned stable identifier passed through to the gateway.
    timeout_s:
        Seconds to wait before triggering the policy.  ``None`` means no
        timeout — the call is forwarded directly (no ``asyncio.wait_for``).
    policy:
        How to handle a ``TimeoutError``:
        - ``"fail"``         — re-raise the error.
        - ``"llm_fallback"`` — call ``llm_fallback_gateway`` and return its
                               result with ``source='fallback'``.
        - ``"skip"``         — return a synthetic timeout response, no error.
    llm_fallback_gateway:
        Required when ``policy="llm_fallback"``.  Raises ``ValueError``
        immediately if ``None`` and policy is ``"llm_fallback"``.

    Returns
    -------
    HumanResponse
        Either the gateway's real response, the fallback response (with
        ``source='fallback'``), or a synthetic timeout response.

    Raises
    ------
    ValueError
        When ``policy="llm_fallback"`` but ``llm_fallback_gateway`` is ``None``.
    TimeoutError
        When ``policy="fail"`` and the timeout is exceeded.
    """
    # Validate configuration eagerly — before making any async calls.
    if policy == "llm_fallback" and llm_fallback_gateway is None:
        raise ValueError(
            "policy='llm_fallback' requires llm_fallback_gateway to be provided, "
            "but llm_fallback_gateway=None was given.  "
            "Pass a HumanGateway instance or switch policy to 'fail' or 'skip'."
        )

    # --- No timeout path ---
    if timeout_s is None:
        return await gateway.request(ctx, request_id=request_id)

    # --- Timeout path ---
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
        # Re-raise as the built-in TimeoutError (Python 3.11+ standard).
        raise TimeoutError(f"gateway.request timed out for request_id={request_id!r}")

    if policy == "llm_fallback":
        # llm_fallback_gateway is guaranteed non-None here (validated at entry).
        assert llm_fallback_gateway is not None  # for mypy
        fallback_response = await asyncio.wait_for(
            llm_fallback_gateway.request(ctx, request_id=request_id),
            timeout=_FALLBACK_TIMEOUT_S,
        )
        # Always override source to 'fallback' regardless of what the fallback
        # gateway returned, so callers can detect this path unambiguously.
        return fallback_response.model_copy(update={"source": "fallback"})

    if policy == "skip":
        return HumanResponse(
            action="timeout",
            source="timeout",
            timed_out=True,
            comment=f"Timed out waiting for human response (request_id={request_id!r})",
        )

    # Unreachable in practice (Literal type guards above), but keeps mypy happy.
    raise ValueError(f"Unknown timeout policy: {policy!r}")  # pragma: no cover
