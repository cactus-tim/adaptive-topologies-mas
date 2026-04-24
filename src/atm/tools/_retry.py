"""Tool retry adapter: with_tool_retry wraps with_retry and converts LLMError to ToolError.

Public API:
    with_tool_retry(policy, tool_name) -- decorator that wraps an async function with retry
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from atm.core.errors import LLMError, ToolError
from atm.llm.retry import RetryPolicy, with_retry


def with_tool_retry(policy: RetryPolicy, tool_name: str) -> Any:
    """Decorator factory that wraps an async method with retry logic.

    Uses the LLM retry infrastructure (with_retry + RetryPolicy) adapted for tools.
    Converts LLMError (raised when retries are exhausted) into ToolError.

    Non-transient exceptions from the decorated function propagate immediately
    without conversion.

    Args:
        policy:    RetryPolicy governing retry behaviour.
        tool_name: Tool name forwarded to ToolError on exhaustion.

    Returns:
        A decorator that wraps an async callable with retry.

    Example::

        class MyTool:
            @with_tool_retry(RetryPolicy(max_retries=2), "my_tool")
            async def _core(self, args):
                ...
    """

    def decorator(fn: Any) -> Any:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await with_retry(
                    lambda: fn(*args, **kwargs),
                    policy=policy,
                    provider="tool",
                    model=tool_name,
                )
            except LLMError as exc:
                raise ToolError(tool_name=tool_name, message=str(exc)) from exc

        return wrapper

    return decorator
