"""Unit tests for LLMWrapper usage parsing (Step 3.2 / M2).

Tests:
- OpenAI shape usage_metadata → correct TokenUsage + cost_usd
- Anthropic shape response_metadata → correct TokenUsage + cost_usd (with cache tokens)
- astream raises NotImplementedError
- tool_calls passthrough from AIMessage to LLMResponse
- LLMResponse fields populated correctly (id, model, text, latency_ms, finish_reason)
- budget-exceed-before-call: BudgetExceededError raised without calling _llm.ainvoke
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from atm.core.errors import BudgetExceededError
from atm.llm.budget import BudgetTracker
from atm.llm.pricing import ModelPricing, Pricing
from atm.llm.wrapper import LLMWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_openai_ai_message(
    content: str = "hello",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
) -> AIMessage:
    """Build an AIMessage with OpenAI-style usage_metadata."""
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cache_read},
        },
        response_metadata={"finish_reason": "stop"},
    )


def make_anthropic_ai_message(
    content: str = "hello",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 10,
    cache_write: int = 5,
) -> AIMessage:
    """Build an AIMessage with Anthropic-style response_metadata."""
    return AIMessage(
        content=content,
        response_metadata={
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
            "stop_reason": "end_turn",
        },
    )


def make_pricing() -> Pricing:
    """Return a pricing table for tests."""
    return Pricing(
        version=1,
        models={
            "openai:gpt-4o-mini": ModelPricing(
                input_per_1k=0.00015,
                output_per_1k=0.0006,
                cached_input_per_1k=0.000075,
            ),
            "anthropic:claude-3-5-haiku-latest": ModelPricing(
                input_per_1k=0.0008,
                output_per_1k=0.004,
                cache_read_per_1k=0.00008,
                cache_write_per_1k=0.001,
            ),
            "fake:deterministic": ModelPricing(
                input_per_1k=0.0,
                output_per_1k=0.0,
            ),
        },
    )


def make_budget(
    per_call: float = 1.0,
    per_run: float = 10.0,
    per_exp: float = 100.0,
) -> BudgetTracker:
    return BudgetTracker(
        per_call_usd=per_call,
        per_run_usd=per_run,
        per_experiment_usd=per_exp,
    )


def make_wrapper(
    model_id: str = "openai:gpt-4o-mini",
    llm: object | None = None,
    pricing: Pricing | None = None,
    budget: BudgetTracker | None = None,
) -> LLMWrapper:
    return LLMWrapper(
        model_id=model_id,
        pricing=pricing or make_pricing(),
        budget=budget or make_budget(),
        llm=llm,
    )


def make_messages() -> list:
    from atm.core.types import Message, MessageKind

    return [Message(sender="user", kind=MessageKind.REQUEST, content="hello")]


# ---------------------------------------------------------------------------
# 1. OpenAI usage_metadata shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_usage_parsed_correctly() -> None:
    """Wrapper parses OpenAI usage_metadata into correct TokenUsage + cost."""
    ai_msg = make_openai_ai_message(
        content="response text",
        input_tokens=100,
        output_tokens=50,
        cache_read=0,
    )

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(model_id="openai:gpt-4o-mini", llm=fake_llm)
    result = await wrapper.ainvoke(make_messages())

    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.cached_input_tokens == 0
    assert result.text == "response text"
    assert result.model == "openai:gpt-4o-mini"

    # Cost: 100 input @ 0.00015/1k + 50 output @ 0.0006/1k
    expected_cost = (100 * 0.00015 / 1000) + (50 * 0.0006 / 1000)
    assert abs(result.cost_usd - expected_cost) < 1e-10


@pytest.mark.asyncio
async def test_openai_usage_with_cache_read() -> None:
    """Wrapper applies OpenAI cache discount for cached_input_tokens."""
    ai_msg = make_openai_ai_message(
        content="cached response",
        input_tokens=100,
        output_tokens=50,
        cache_read=40,
    )

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(model_id="openai:gpt-4o-mini", llm=fake_llm)
    result = await wrapper.ainvoke(make_messages())

    assert result.usage.cached_input_tokens == 40
    # Cost: 60 non-cached @ 0.00015/1k + 40 cached @ 0.000075/1k + 50 output @ 0.0006/1k
    expected_cost = (60 * 0.00015 / 1000) + (40 * 0.000075 / 1000) + (50 * 0.0006 / 1000)
    assert abs(result.cost_usd - expected_cost) < 1e-10


# ---------------------------------------------------------------------------
# 2. Anthropic response_metadata shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_usage_parsed_correctly() -> None:
    """Wrapper parses Anthropic response_metadata into correct TokenUsage + cost."""
    ai_msg = make_anthropic_ai_message(
        content="anthropic response",
        input_tokens=100,
        output_tokens=50,
        cache_read=10,
        cache_write=5,
    )

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(
        model_id="anthropic:claude-3-5-haiku-latest",
        llm=fake_llm,
    )
    result = await wrapper.ainvoke(make_messages())

    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.cached_input_tokens == 10

    # Cost: 85 plain @ 0.0008/1k + 10 read @ 0.00008/1k + 5 write @ 0.001/1k + 50 out @ 0.004/1k
    expected_cost = (
        (85 * 0.0008 / 1000) + (10 * 0.00008 / 1000) + (5 * 0.001 / 1000) + (50 * 0.004 / 1000)
    )
    assert abs(result.cost_usd - expected_cost) < 1e-10


# ---------------------------------------------------------------------------
# 3. LLMResponse fields populated correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_response_fields_populated() -> None:
    """LLMResponse has id, model, text, latency_ms, finish_reason, started_at."""
    ai_msg = make_openai_ai_message(content="test text")
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(llm=fake_llm)
    result = await wrapper.ainvoke(make_messages())

    assert isinstance(result.id, UUID)
    assert result.model == "openai:gpt-4o-mini"
    assert result.text == "test text"
    assert result.latency_ms >= 0
    assert isinstance(result.started_at, datetime.datetime)
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_finish_reason_from_response_metadata() -> None:
    """finish_reason is extracted from response_metadata."""
    ai_msg = AIMessage(
        content="done",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 0},
        },
        response_metadata={"finish_reason": "length"},
    )
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(llm=fake_llm)
    result = await wrapper.ainvoke(make_messages())
    assert result.finish_reason == "length"


# ---------------------------------------------------------------------------
# 4. tool_calls passthrough
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_calls_parsed_from_ai_message() -> None:
    """LLMResponse.tool_calls populated from AIMessage.tool_calls."""
    tc = {"name": "bash", "args": {"cmd": "ls -la"}, "id": "call_001", "type": "tool_call"}
    ai_msg = AIMessage(
        content="",
        tool_calls=[tc],
        usage_metadata={
            "input_tokens": 50,
            "output_tokens": 10,
            "total_tokens": 60,
            "input_token_details": {"cache_read": 0},
        },
        response_metadata={"finish_reason": "tool_calls"},
    )
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(llm=fake_llm)
    result = await wrapper.ainvoke(make_messages())

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "bash"
    assert result.tool_calls[0].args == {"cmd": "ls -la"}
    assert result.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# 5. astream raises NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_raises_not_implemented() -> None:
    """astream must raise NotImplementedError in M2."""
    fake_llm = AsyncMock()
    wrapper = make_wrapper(llm=fake_llm)
    with pytest.raises(NotImplementedError):
        await wrapper.astream()


# ---------------------------------------------------------------------------
# 6. Budget-exceed-before-call: BudgetExceededError without calling _llm.ainvoke
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceed_before_call_does_not_invoke_llm() -> None:
    """When budget is exceeded pre-call, _llm.ainvoke must NOT be called."""
    budget = BudgetTracker(
        per_call_usd=0.000001,  # essentially zero
        per_run_usd=10.0,
        per_experiment_usd=100.0,
    )

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=make_openai_ai_message())

    wrapper = LLMWrapper(
        model_id="openai:gpt-4o-mini",
        pricing=make_pricing(),
        budget=budget,
        llm=fake_llm,
    )

    from atm.core.types import Message, MessageKind

    messages = [
        Message(
            sender="user",
            kind=MessageKind.REQUEST,
            content="x" * 10000,  # ~2500 tokens estimate — will exceed $0.000001
        )
    ]

    with pytest.raises(BudgetExceededError):
        await wrapper.ainvoke(messages)

    # CRITICAL: the underlying LLM must NOT have been invoked
    fake_llm.ainvoke.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Message list input types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ainvoke_accepts_list_of_base_messages() -> None:
    """Wrapper accepts list[BaseMessage] directly (no conversion needed)."""
    ai_msg = make_openai_ai_message(content="hi")
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = make_wrapper(llm=fake_llm)
    lc_messages = [HumanMessage(content="hello")]
    result = await wrapper.ainvoke(lc_messages)
    assert result.text == "hi"
