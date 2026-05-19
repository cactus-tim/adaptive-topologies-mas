"""Async exponential-backoff retry with transient-error duck-typing.

Public API
----------
RetryPolicy   -- frozen dataclass: max_retries, base_delay_s, max_delay_s, jitter, retry_on
is_transient  -- predicate: True if exc is in retry_on OR has status_code in {429} or >=500
with_retry    -- async wrapper: calls fn under policy; raises LLMError on exhaustion
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from atm.core.errors import LLMError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for exponential-backoff retry behaviour.

    Attributes:
        max_retries:  Maximum number of *retry* attempts after the initial call.
                      Total calls = max_retries + 1.
        base_delay_s: Base delay in seconds before the first retry.
        max_delay_s:  Upper cap on the computed delay (before jitter).
        jitter:       When True, apply full-jitter: delay = uniform(0, cap).
        retry_on:     Tuple of exception classes that should always be retried
                      (regardless of status_code duck-typing).
    """

    max_retries: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: bool = True
    retry_on: tuple[type[BaseException], ...] = field(default_factory=tuple)


def is_transient(exc: BaseException, policy: RetryPolicy) -> bool:
    """Return True if *exc* should trigger a retry under *policy*.

    Matching rules (OR):
    1. ``isinstance(exc, policy.retry_on)`` — explicit class registration.
    2. ``getattr(exc, "status_code", None) == 429`` — duck-typed rate limit.
    3. ``getattr(exc, "status_code", None) >= 500`` — duck-typed server error.
    """
    if policy.retry_on and isinstance(exc, policy.retry_on):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        if status_code == 429:
            return True
        if status_code >= 500:
            return True

    return False


def _compute_delay(attempt: int, policy: RetryPolicy) -> float:
    """Return the sleep duration for *attempt* (1-indexed, first retry = 1)."""
    cap: float = min(policy.base_delay_s * float(2 ** (attempt - 1)), policy.max_delay_s)
    if policy.jitter:
        return random.uniform(0.0, cap)
    return cap


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    provider: str,
    model: str,
) -> T:
    """Call *fn* under *policy*, retrying on transient errors.

    Parameters
    ----------
    fn:       Zero-argument async callable to invoke.
    policy:   ``RetryPolicy`` governing retry behaviour.
    provider: Provider name forwarded to ``LLMError`` on exhaustion.
    model:    Model identifier forwarded to ``LLMError`` on exhaustion.

    Returns
    -------
    The return value of *fn* on success.

    Raises
    ------
    LLMError
        When all attempts (1 initial + policy.max_retries) are exhausted.
        The ``__cause__`` is the last transient exception.
    Any non-transient exception from *fn* is re-raised immediately.
    """
    last_exc: BaseException | None = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await fn()
        except BaseException as exc:
            if not is_transient(exc, policy):
                raise

            last_exc = exc

            if attempt < policy.max_retries:
                delay = _compute_delay(attempt + 1, policy)
                await asyncio.sleep(delay)

    total_attempts = policy.max_retries + 1
    raise LLMError(
        provider=provider,
        model=model,
        attempts=total_attempts,
        message=f"Transient error after {total_attempts} attempt(s): {last_exc}",
    ) from last_exc
