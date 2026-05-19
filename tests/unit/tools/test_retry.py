"""Tests for atm.tools._retry — with_tool_retry decorator."""

from __future__ import annotations

import pytest

from atm.core.errors import LLMError, ToolError
from atm.llm.retry import RetryPolicy
from atm.tools._retry import with_tool_retry


def _make_policy(max_retries: int = 0) -> RetryPolicy:
    return RetryPolicy(max_retries=max_retries, base_delay_s=0.0, jitter=False)


@pytest.mark.asyncio
async def test_with_tool_retry_preserves_ok_path() -> None:
    """When the wrapped function succeeds, the result is returned unchanged."""

    @with_tool_retry(_make_policy(), tool_name="my_tool")
    async def my_fn() -> str:
        return "success"

    result = await my_fn()
    assert result == "success"


@pytest.mark.asyncio
async def test_with_tool_retry_passes_args_through() -> None:
    """Positional and keyword args are forwarded to the wrapped function."""

    @with_tool_retry(_make_policy(), tool_name="my_tool")
    async def add(a: int, b: int = 0) -> int:
        return a + b

    assert await add(2, b=3) == 5


@pytest.mark.asyncio
async def test_with_tool_retry_converts_llmerror_to_toolerror() -> None:
    """LLMError exhausted from with_retry must be re-raised as ToolError."""
    call_count = 0

    class TransientErr(Exception):
        status_code = 429

    @with_tool_retry(_make_policy(max_retries=1), tool_name="failing_tool")
    async def flaky() -> None:
        nonlocal call_count
        call_count += 1
        raise TransientErr("rate limited")

    with pytest.raises(ToolError) as exc_info:
        await flaky()

    err = exc_info.value
    assert err.tool_name == "failing_tool"
    assert isinstance(err.__cause__, LLMError)
    assert call_count == 2


@pytest.mark.asyncio
async def test_with_tool_retry_toolerror_message_contains_llmerror_text() -> None:
    """The ToolError message should reflect the underlying LLMError."""

    class TransientErr(Exception):
        status_code = 500

    @with_tool_retry(_make_policy(max_retries=0), tool_name="tool_x")
    async def fail_once() -> None:
        raise TransientErr("internal server error")

    with pytest.raises(ToolError) as exc_info:
        await fail_once()

    assert exc_info.value.tool_name == "tool_x"


@pytest.mark.asyncio
async def test_with_tool_retry_non_transient_raises_directly() -> None:
    """Non-transient exceptions are NOT converted — they propagate as-is."""

    @with_tool_retry(_make_policy(max_retries=3), tool_name="my_tool")
    async def raise_value_error() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await raise_value_error()
