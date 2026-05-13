"""Unit tests for LLMWrapper.last_model_version — Step 3.2 (M11 resync).

Tests:
- Anthropic response_metadata['model'] is captured in last_model_version.
- OpenAI response_metadata['system_fingerprint'] is captured (first priority).
- OpenAI fallback chain: system_fingerprint > model_name > model.
- Empty / missing response_metadata leaves last_model_version as None.
- Multiple sequential calls update to the latest non-None value.
- _extract_model_version helper behaves correctly for all branches.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from atm.llm.budget import BudgetTracker
from atm.llm.pricing import ModelPricing, Pricing
from atm.llm.wrapper import LLMWrapper, _extract_model_version

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pricing() -> Pricing:
    return Pricing(
        version=1,
        models={
            "anthropic:claude-3-5-sonnet": ModelPricing(
                input_per_1k=0.003,
                output_per_1k=0.015,
            ),
            "openai:gpt-4o": ModelPricing(
                input_per_1k=0.005,
                output_per_1k=0.015,
            ),
            "fake:echo": ModelPricing(
                input_per_1k=0.0,
                output_per_1k=0.0,
            ),
        },
    )


def _make_budget() -> BudgetTracker:
    return BudgetTracker(
        per_call_usd=10.0,
        per_run_usd=100.0,
        per_experiment_usd=1000.0,
    )


def _make_wrapper(
    model_id: str = "anthropic:claude-3-5-sonnet", llm: object | None = None
) -> LLMWrapper:
    return LLMWrapper(
        model_id=model_id,
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=llm,
    )


def _make_messages() -> list:
    from atm.core.types import Message, MessageKind

    return [Message(sender="user", kind=MessageKind.REQUEST, content="hello")]


def _anthropic_ai_msg(model: str = "claude-3-5-sonnet-20241022") -> AIMessage:
    """AIMessage with Anthropic-style response_metadata including 'model' key."""
    return AIMessage(
        content="response",
        response_metadata={
            "model": model,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
            },
            "stop_reason": "end_turn",
        },
    )


def _openai_ai_msg(
    system_fingerprint: str | None = "fp_abc123",
    model_name: str | None = None,
    model: str | None = None,
) -> AIMessage:
    """AIMessage with OpenAI-style response_metadata."""
    resp_meta: dict = {"finish_reason": "stop"}
    if system_fingerprint is not None:
        resp_meta["system_fingerprint"] = system_fingerprint
    if model_name is not None:
        resp_meta["model_name"] = model_name
    if model is not None:
        resp_meta["model"] = model
    return AIMessage(
        content="response",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_token_details": {"cache_read": 0},
        },
        response_metadata=resp_meta,
    )


# ---------------------------------------------------------------------------
# 1. _extract_model_version helper
# ---------------------------------------------------------------------------


class TestExtractModelVersion:
    def test_anthropic_model_key(self) -> None:
        """Anthropic uses 'model' key — extracted directly."""
        result = _extract_model_version({"model": "claude-3-5-sonnet-20241022"})
        assert result == "claude-3-5-sonnet-20241022"

    def test_openai_system_fingerprint_first(self) -> None:
        """system_fingerprint takes priority over model_name and model."""
        result = _extract_model_version(
            {
                "system_fingerprint": "fp_abc123",
                "model_name": "gpt-4o",
                "model": "gpt-4o-2024-08-06",
            }
        )
        assert result == "fp_abc123"

    def test_openai_model_name_fallback(self) -> None:
        """model_name is used when system_fingerprint is absent."""
        result = _extract_model_version({"model_name": "gpt-4o"})
        assert result == "gpt-4o"

    def test_openai_model_key_last_fallback(self) -> None:
        """'model' key is used as final fallback."""
        result = _extract_model_version({"model": "gpt-4o-2024-08-06"})
        assert result == "gpt-4o-2024-08-06"

    def test_empty_metadata_returns_none(self) -> None:
        """Empty dict returns None."""
        assert _extract_model_version({}) is None

    def test_none_values_ignored(self) -> None:
        """Keys with None values are skipped."""
        assert _extract_model_version({"system_fingerprint": None, "model_name": None}) is None

    def test_empty_string_values_ignored(self) -> None:
        """Keys with empty-string values are skipped."""
        assert _extract_model_version({"system_fingerprint": "", "model": ""}) is None

    def test_whitespace_only_values_ignored(self) -> None:
        """Keys with whitespace-only values are skipped."""
        assert _extract_model_version({"model": "   "}) is None


# ---------------------------------------------------------------------------
# 2. Wrapper attribute starts None
# ---------------------------------------------------------------------------


def test_last_model_version_starts_none() -> None:
    """last_model_version attribute is None on a freshly created wrapper."""
    wrapper = _make_wrapper()
    assert wrapper.last_model_version is None


# ---------------------------------------------------------------------------
# 3. Anthropic response captures model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_model_captured() -> None:
    """Anthropic response_metadata['model'] is stored in last_model_version."""
    ai_msg = _anthropic_ai_msg(model="claude-3-5-sonnet-20241022")
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = _make_wrapper(model_id="anthropic:claude-3-5-sonnet", llm=fake_llm)
    assert wrapper.last_model_version is None

    await wrapper.ainvoke(_make_messages())

    assert wrapper.last_model_version == "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# 4. OpenAI system_fingerprint captured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_system_fingerprint_captured() -> None:
    """OpenAI response_metadata['system_fingerprint'] is stored."""
    ai_msg = _openai_ai_msg(system_fingerprint="fp_abc123")
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = _make_wrapper(model_id="openai:gpt-4o", llm=fake_llm)
    await wrapper.ainvoke(_make_messages())

    assert wrapper.last_model_version == "fp_abc123"


# ---------------------------------------------------------------------------
# 5. OpenAI fallback: no fingerprint → model_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_model_name_fallback_captured() -> None:
    """When system_fingerprint absent, model_name is stored."""
    ai_msg = _openai_ai_msg(system_fingerprint=None, model_name="gpt-4o")
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = _make_wrapper(model_id="openai:gpt-4o", llm=fake_llm)
    await wrapper.ainvoke(_make_messages())

    assert wrapper.last_model_version == "gpt-4o"


# ---------------------------------------------------------------------------
# 6. OpenAI fallback: no fingerprint, no model_name → model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_model_key_fallback_captured() -> None:
    """When system_fingerprint and model_name absent, 'model' key is stored."""
    ai_msg = _openai_ai_msg(system_fingerprint=None, model_name=None, model="gpt-4o-2024-08-06")
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = _make_wrapper(model_id="openai:gpt-4o", llm=fake_llm)
    await wrapper.ainvoke(_make_messages())

    assert wrapper.last_model_version == "gpt-4o-2024-08-06"


# ---------------------------------------------------------------------------
# 7. Empty / missing response_metadata leaves last_model_version as None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_response_metadata_leaves_version_none() -> None:
    """Missing/empty response_metadata does not update last_model_version."""
    # AIMessage with usage_metadata (OpenAI shape) but no version info in response_metadata
    ai_msg = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "input_token_details": {"cache_read": 0},
        },
        response_metadata={"finish_reason": "stop"},
        # No 'model', 'system_fingerprint', or 'model_name' keys
    )
    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(return_value=ai_msg)

    wrapper = _make_wrapper(model_id="openai:gpt-4o", llm=fake_llm)
    await wrapper.ainvoke(_make_messages())

    assert wrapper.last_model_version is None


# ---------------------------------------------------------------------------
# 8. Multiple sequential calls — updates to latest non-None value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_calls_update_to_latest() -> None:
    """Successive calls with different model values update last_model_version."""
    ai_msg_v1 = _anthropic_ai_msg(model="claude-3-5-sonnet-20241022")
    ai_msg_v2 = _anthropic_ai_msg(model="claude-3-5-sonnet-20250101")

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(side_effect=[ai_msg_v1, ai_msg_v2])

    wrapper = _make_wrapper(model_id="anthropic:claude-3-5-sonnet", llm=fake_llm)

    await wrapper.ainvoke(_make_messages())
    assert wrapper.last_model_version == "claude-3-5-sonnet-20241022"

    await wrapper.ainvoke(_make_messages())
    assert wrapper.last_model_version == "claude-3-5-sonnet-20250101"


# ---------------------------------------------------------------------------
# 9. None response_metadata on second call does NOT overwrite existing value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_result_does_not_overwrite_existing() -> None:
    """If a second call yields no extractable version, the previous value is kept."""
    ai_msg_v1 = _anthropic_ai_msg(model="claude-3-5-sonnet-20241022")
    # Second message has no model version info
    ai_msg_empty = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
            "input_token_details": {"cache_read": 0},
        },
        response_metadata={"finish_reason": "stop"},
    )

    fake_llm = AsyncMock()
    fake_llm.ainvoke = AsyncMock(side_effect=[ai_msg_v1, ai_msg_empty])

    wrapper = _make_wrapper(model_id="anthropic:claude-3-5-sonnet", llm=fake_llm)

    await wrapper.ainvoke(_make_messages())
    assert wrapper.last_model_version == "claude-3-5-sonnet-20241022"

    await wrapper.ainvoke(_make_messages())
    # Should retain the previous value, not reset to None
    assert wrapper.last_model_version == "claude-3-5-sonnet-20241022"
