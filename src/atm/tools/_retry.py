"""Retry adapter for ATM tools layer.

Wraps ``atm.llm.retry.with_retry`` to catch ``LLMError`` and re-raise as
``ToolError``, keeping tool code decoupled from LLM internals.

Public API
----------
with_tool_retry -- decorator factory: wraps an async function with retry + error conversion
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from atm.core.errors import LLMError, ToolError
from atm.llm.retry import RetryPolicy, with_retry

T = TypeVar("T")


def with_tool_retry(
    policy: RetryPolicy,
    tool_name: str,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator factory that wraps an async function with retry and error conversion.

    Behaviour
    ---------
    - On success: returns the function's return value unchanged.
    - On transient error (retried, then exhausted): ``LLMError`` is caught and
      re-raised as ``ToolError(tool_name=tool_name, message=str(llm_error))``.
    - On non-transient error: the original exception propagates unchanged.

    Parameters
    ----------
    policy:    ``RetryPolicy`` controlling retry behaviour.
    tool_name: Tool name embedded in any raised ``ToolError``.

    Returns
    -------
    Decorator that wraps an async callable.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await with_retry(
                    lambda: fn(*args, **kwargs),
                    policy=policy,
                    provider="tool",
                    model=tool_name,
                )
            except LLMError as exc:
                raise ToolError(
                    tool_name=tool_name,
                    message=str(exc),
                ) from exc

        return wrapper

    return decorator
