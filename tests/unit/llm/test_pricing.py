"""Unit tests for llm/pricing.py — Pricing loader + cost calculation.

Covers:
  - YAML loader (Pricing.from_yaml returns Pricing with models keyed by "provider:model_id")
  - Unknown model raises LLMError
  - OpenAI no-cache cost calculation
  - OpenAI cache-hit cost calculation (cached_input_per_1k for cached tokens)
  - Anthropic cache cost calculation (cache_read_per_1k + cache_write_tokens kwarg)
  - Pre-call estimate helper
"""

from __future__ import annotations

import pytest

from atm.core.errors import LLMError
from atm.core.types import TokenUsage
from atm.llm.pricing import ModelPricing, Pricing

PRICING_YAML = "conf/pricing.yaml"


@pytest.fixture
def pricing() -> Pricing:
    """Load pricing from the real conf/pricing.yaml file."""
    return Pricing.from_yaml(PRICING_YAML)


def test_from_yaml_returns_pricing_instance(pricing: Pricing) -> None:
    """from_yaml returns a Pricing instance with version and models dict."""
    assert isinstance(pricing, Pricing)
    assert pricing.version == 1
    assert isinstance(pricing.models, dict)


def test_from_yaml_has_expected_model_keys(pricing: Pricing) -> None:
    """Pricing.models has all expected keys from conf/pricing.yaml.

    fake:scripted and fake:echo were added in M6 so runner tests can use
    those model_ids without triggering "Unknown model" pricing errors.
    """
    expected_keys = {
        "openai:gpt-4o",
        "openai:gpt-4o-mini",
        "openai:gpt-4.1-mini",
        "anthropic:claude-3-5-sonnet-latest",
        "anthropic:claude-3-5-haiku-latest",
        "cerebras:llama3.1-8b",
        "cerebras:gpt-oss-120b",
        "cerebras:zai-glm-4.7",
        "cerebras:qwen-3-235b-a22b-instruct-2507",
        "fake:deterministic",
        "fake:scripted",
        "fake:echo",
    }
    assert expected_keys == set(pricing.models.keys())


def test_from_yaml_model_pricing_is_model_pricing_instance(pricing: Pricing) -> None:
    """Each entry in models is a ModelPricing instance."""
    for key, model_pricing in pricing.models.items():
        assert isinstance(model_pricing, ModelPricing), f"Expected ModelPricing for {key!r}"


def test_from_yaml_openai_rates_loaded_correctly(pricing: Pricing) -> None:
    """OpenAI gpt-4o-mini rates are parsed exactly from YAML."""
    mp = pricing.models["openai:gpt-4o-mini"]
    assert mp.input_per_1k == pytest.approx(0.00015)
    assert mp.output_per_1k == pytest.approx(0.00060)
    assert mp.cached_input_per_1k == pytest.approx(0.000075)


def test_cost_unknown_model_raises_llm_error(pricing: Pricing) -> None:
    """cost() raises LLMError for an unknown model_id."""
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    with pytest.raises(LLMError, match="openai:unknown-model"):
        pricing.cost("openai:unknown-model", usage)


def test_cost_unknown_model_error_has_model_info(pricing: Pricing) -> None:
    """LLMError raised for unknown model contains provider and model fields."""
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    with pytest.raises(LLMError) as exc_info:
        pricing.cost("anthropic:nonexistent", usage)
    err = exc_info.value
    assert hasattr(err, "provider")
    assert hasattr(err, "model")


def test_cost_openai_no_cache(pricing: Pricing) -> None:
    """OpenAI gpt-4o-mini cost matches plan acceptance criterion exactly.

    TokenUsage(1000, 500, 0, 1500) should cost:
      1000 * 0.00015 / 1000 + 500 * 0.00060 / 1000 = 0.00015 + 0.00030 = 0.00045
    """
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        cached_input_tokens=0,
        total_tokens=1500,
    )
    cost = pricing.cost("openai:gpt-4o-mini", usage)
    assert cost == pytest.approx(0.00045, rel=1e-9)


def test_cost_openai_gpt4o_no_cache(pricing: Pricing) -> None:
    """OpenAI gpt-4o no-cache with known rates.

    100 input, 50 output, 0 cached:
      100 * 0.0025 / 1000 + 50 * 0.010 / 1000 = 0.00025 + 0.00050 = 0.00075
    """
    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    cost = pricing.cost("openai:gpt-4o", usage)
    assert cost == pytest.approx(0.00075, rel=1e-9)


def test_cost_openai_cache_hit(pricing: Pricing) -> None:
    """Cached prompt tokens billed at cached_input_per_1k; non-cached at input_per_1k.

    prompt=1000, completion=200, cached=800:
      non_cached_input = 200 tokens at input_per_1k = 0.00015
      cached_input     = 800 tokens at cached_input_per_1k = 0.000075
      completion       = 200 tokens at output_per_1k = 0.00060
      cost = 200 * 0.00015/1000 + 800 * 0.000075/1000 + 200 * 0.00060/1000
           = 0.00003 + 0.00006 + 0.00012 = 0.00021
    """
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=200,
        cached_input_tokens=800,
        total_tokens=1200,
    )
    cost = pricing.cost("openai:gpt-4o-mini", usage)
    assert cost == pytest.approx(0.00021, rel=1e-9)


def test_cost_openai_all_cached(pricing: Pricing) -> None:
    """Entire prompt cached — all prompt tokens at cached rate."""
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=0,
        cached_input_tokens=1000,
        total_tokens=1000,
    )
    cost = pricing.cost("openai:gpt-4o-mini", usage)
    assert cost == pytest.approx(0.000075, rel=1e-9)


def test_cost_anthropic_cache(pricing: Pricing) -> None:
    """Anthropic: cache_read_tokens billed at cache_read_per_1k; cache_write kwarg at cache_write_per_1k.

    model: anthropic:claude-3-5-sonnet-latest
    input_per_1k=0.003, output_per_1k=0.015, cache_read_per_1k=0.0003, cache_write_per_1k=0.00375

    prompt=1000, completion=300, cached_input_tokens=600, cache_write_tokens=200:
      plain_input   = 1000 - 600 - 200 = 200 tokens at input_per_1k
      cache_read    = 600 tokens at cache_read_per_1k
      cache_write   = 200 tokens at cache_write_per_1k
      completion    = 300 tokens at output_per_1k
      cost = 200*0.003/1000 + 600*0.0003/1000 + 200*0.00375/1000 + 300*0.015/1000
           = 0.0006 + 0.00018 + 0.00075 + 0.0045 = 0.00603
    """
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=300,
        cached_input_tokens=600,
        total_tokens=1300,
    )
    cost = pricing.cost(
        "anthropic:claude-3-5-sonnet-latest",
        usage,
        cache_write_tokens=200,
    )
    assert cost == pytest.approx(0.00603, rel=1e-9)


def test_cost_anthropic_no_cache(pricing: Pricing) -> None:
    """Anthropic with no caching falls back to plain input cost."""
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        cached_input_tokens=0,
        total_tokens=1500,
    )
    cost = pricing.cost("anthropic:claude-3-5-sonnet-latest", usage)
    assert cost == pytest.approx(0.0105, rel=1e-9)


def test_cost_fake_is_zero(pricing: Pricing) -> None:
    """fake:deterministic always returns 0.0 cost."""
    usage = TokenUsage(
        prompt_tokens=10000,
        completion_tokens=5000,
        cached_input_tokens=0,
        total_tokens=15000,
    )
    cost = pricing.cost("fake:deterministic", usage)
    assert cost == 0.0


def test_estimate_returns_float(pricing: Pricing) -> None:
    """estimate() returns a float cost prediction for token counts."""
    cost = pricing.estimate(
        "openai:gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert isinstance(cost, float)


def test_estimate_no_cache_matches_direct(pricing: Pricing) -> None:
    """estimate() without cache args matches cost() with no-cache TokenUsage."""
    estimated = pricing.estimate(
        "openai:gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    usage = TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        cached_input_tokens=0,
        total_tokens=1500,
    )
    direct = pricing.cost("openai:gpt-4o-mini", usage)
    assert estimated == pytest.approx(direct, rel=1e-9)


def test_estimate_unknown_model_raises_llm_error(pricing: Pricing) -> None:
    """estimate() raises LLMError for unknown model."""
    with pytest.raises(LLMError):
        pricing.estimate("nonexistent:model", prompt_tokens=100, completion_tokens=50)
