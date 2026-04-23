"""Unit tests for LLMWrapper retry behaviour (Step 3.2 / M2).

Tests:
- Transient error (RateLimitError with status_code=429) on first call: wrapper retries and succeeds
- All retries exhausted: wrapper raises LLMError
- Non-transient error: raised immediately (no retry)
- retry count visible in LLMError.attempts
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from atm.core.errors import LLMError
from atm.llm.budget import BudgetTracker
from atm.llm.pricing import ModelPricing, Pricing
from atm.llm.retry import RetryPolicy
from atm.llm.wrapper import LLMWrapper

# ---------------------------------------------------------------------------
# Custom rate-limit exception (avoids coupling to openai SDK)
# ---------------------------------------------------------------------------


class FakeRateLimitError(Exception):
    """Fake 429 rate-limit error for testing retry logic."""

    status_code: int = 429

    def __init__(self, message: str = "rate limited") -> None:
        super().__init__(message)
        self.status_code = 429


class FakeServerError(Exception):
    """Fake 500 server error for testing retry logic."""

    status_code: int = 500

    def __init__(self, message: str = "server error") -> None:
        super().__init__(message)
        self.status_code = 500


class FakeNonTransientError(Exception):
    """Non-transient error (no status_code) — should NOT be retried."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ai_message(content: str = "ok") -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 0},
        },
        response_metadata={"finish_reason": "stop"},
    )


def make_pricing() -> Pricing:
    return Pricing(
        version=1,
        models={
            "openai:gpt-4o-mini": ModelPricing(
                input_per_1k=0.00015,
                output_per_1k=0.0006,
                cached_input_per_1k=0.000075,
            ),
        },
    )


def make_budget() -> BudgetTracker:
    return BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=10.0,
        per_experiment_usd=100.0,
    )


def make_retry_policy(max_retries: int = 3, base_delay_s: float = 0.001) -> RetryPolicy:
    """Return a retry policy with very short delays for fast tests."""
    return RetryPolicy(
        max_retries=max_retries,
        base_delay_s=base_delay_s,
        max_delay_s=0.01,
        jitter=False,
    )


def make_wrapper(
    llm: object | None = None,
    retry_policy: RetryPolicy | None = None,
) -> LLMWrapper:
    return LLMWrapper(
        model_id="openai:gpt-4o-mini",
        pricing=make_pricing(),
        budget=make_budget(),
        retry_policy=retry_policy or make_retry_policy(),
        llm=llm,
    )


def make_messages() -> list:
    from atm.core.types import Message, MessageKind

    return [Message(sender="user", kind=MessageKind.REQUEST, content="hello")]


# ---------------------------------------------------------------------------
# 1. Rate-limit (429) retry: fails once then succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_retry_succeeds_on_second_attempt() -> None:
    """Wrapper retries on FakeRateLimitError (status_code=429) and succeeds on 2nd attempt."""
    success_response = make_ai_message("retry worked")

    call_count = 0

    async def flaky_invoke(*args: object, **kwargs: object) -> AIMessage:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FakeRateLimitError("rate limited")
        return success_response

    fake_llm = AsyncMock()
    fake_llm.ainvoke = flaky_invoke

    wrapper = make_wrapper(llm=fake_llm, retry_policy=make_retry_policy(max_retries=3))
    result = await wrapper.ainvoke(make_messages())

    assert result.text == "retry worked"
    assert call_count == 2


@pytest.mark.asyncio
async def test_server_error_retry_succeeds_on_second_attempt() -> None:
    """Wrapper retries on FakeServerError (status_code=500) and succeeds on 2nd attempt."""
    success_response = make_ai_message("server recovered")

    call_count = 0

    async def flaky_invoke(*args: object, **kwargs: object) -> AIMessage:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FakeServerError("server error")
        return success_response

    fake_llm = AsyncMock()
    fake_llm.ainvoke = flaky_invoke

    wrapper = make_wrapper(llm=fake_llm, retry_policy=make_retry_policy(max_retries=3))
    result = await wrapper.ainvoke(make_messages())

    assert result.text == "server recovered"
    assert call_count == 2


# ---------------------------------------------------------------------------
# 2. All retries exhausted: LLMError raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_retries_exhausted_raises_llm_error() -> None:
    """When all retries are exhausted, LLMError is raised with correct attempts count."""
    call_count = 0

    async def always_fail(*args: object, **kwargs: object) -> AIMessage:
        nonlocal call_count
        call_count += 1
        raise FakeRateLimitError("always rate limited")

    fake_llm = AsyncMock()
    fake_llm.ainvoke = always_fail

    policy = make_retry_policy(max_retries=2)
    wrapper = make_wrapper(llm=fake_llm, retry_policy=policy)

    with pytest.raises(LLMError) as exc_info:
        await wrapper.ainvoke(make_messages())

    err = exc_info.value
    assert err.provider == "openai"
    assert err.model == "gpt-4o-mini"
    assert err.attempts == 3  # 1 initial + 2 retries
    assert call_count == 3


# ---------------------------------------------------------------------------
# 3. Non-transient error: raised immediately, no retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_transient_error_not_retried() -> None:
    """Non-transient errors (no status_code) are raised immediately without retry."""
    call_count = 0

    async def always_fail_non_transient(*args: object, **kwargs: object) -> AIMessage:
        nonlocal call_count
        call_count += 1
        raise FakeNonTransientError("value error in tool")

    fake_llm = AsyncMock()
    fake_llm.ainvoke = always_fail_non_transient

    policy = make_retry_policy(max_retries=3)
    wrapper = make_wrapper(llm=fake_llm, retry_policy=policy)

    with pytest.raises(FakeNonTransientError):
        await wrapper.ainvoke(make_messages())

    # Should have been called exactly once (no retry)
    assert call_count == 1


# ---------------------------------------------------------------------------
# 4. Retry with no policy uses defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_retry_policy_used_when_none_given() -> None:
    """When no retry_policy is given, wrapper uses a default policy."""
    success_response = make_ai_message("default policy ok")

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=success_response)

    wrapper = LLMWrapper(
        model_id="openai:gpt-4o-mini",
        pricing=make_pricing(),
        budget=make_budget(),
        retry_policy=None,
        llm=fake_llm,
    )

    result = await wrapper.ainvoke(make_messages())
    assert result.text == "default policy ok"
