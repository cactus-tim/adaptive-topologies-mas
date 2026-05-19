"""Tests for estimate_prompt_tokens helper in _tokens.py.

Test cases (5 required per plan):
1. test_estimate_empty_returns_one — empty message list returns 1 (non-zero sentinel)
2. test_estimate_monotonic — longer text => higher or equal estimate
3. test_estimate_fallback_for_unknown_provider — unknown provider uses len//4 heuristic
4. test_estimate_openai_uses_tiktoken — openai provider diverges from heuristic by >15%
5. test_estimate_with_lrucache_hits — repeated calls hit cache (no crash + cache_info works)
"""

from __future__ import annotations

from atm.agents._tokens import estimate_prompt_tokens


def _msg(content: str) -> dict:
    return {"role": "user", "content": content}


class TestEstimateEmptyReturnsOne:
    """Empty message list must return 1 (non-zero sentinel)."""

    def test_estimate_empty_returns_one(self) -> None:
        result = estimate_prompt_tokens([], model_id="openai:gpt-4o-mini")
        assert result == 1

    def test_estimate_empty_returns_one_heuristic(self) -> None:
        result = estimate_prompt_tokens([], model_id="fake:any")
        assert result == 1


class TestEstimateMonotonic:
    """Longer text must produce higher or equal estimate."""

    def test_estimate_monotonic(self) -> None:
        short = [_msg("Hi")]
        long = [_msg("Hi " * 200)]
        short_est = estimate_prompt_tokens(short, model_id="fake:deterministic")
        long_est = estimate_prompt_tokens(long, model_id="fake:deterministic")
        assert long_est >= short_est

    def test_estimate_monotonic_openai(self) -> None:
        short = [_msg("Hello world")]
        long = [_msg("Hello world " * 100)]
        short_est = estimate_prompt_tokens(short, model_id="openai:gpt-4o-mini")
        long_est = estimate_prompt_tokens(long, model_id="openai:gpt-4o-mini")
        assert long_est >= short_est


class TestEstimateFallbackForUnknownProvider:
    """Unknown or fake providers use len(text)//4 heuristic."""

    def test_estimate_fallback_for_unknown_provider(self) -> None:
        text = "a" * 100
        messages = [_msg(text)]
        result = estimate_prompt_tokens(messages, model_id="fake:deterministic")
        assert 20 <= result <= 50

    def test_estimate_fallback_no_provider_prefix(self) -> None:
        text = "x" * 80
        messages = [_msg(text)]
        result = estimate_prompt_tokens(messages, model_id="unknown-model")
        assert result > 0

    def test_estimate_fallback_deterministic_value(self) -> None:
        messages = [_msg("a" * 40)]
        result = estimate_prompt_tokens(messages, model_id="fake:x")
        assert result == 14


class TestEstimateOpenAIUsesTiktoken:
    """For openai:* models, tiktoken is used, which diverges from len//4 heuristic by >15%."""

    def test_estimate_openai_uses_tiktoken(self) -> None:
        prose = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
            "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
            "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
            "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in "
            "culpa qui officia deserunt mollit anim id est laborum."
        )
        messages = [_msg(prose)]
        model_id = "openai:gpt-4o-mini"

        tiktoken_est = estimate_prompt_tokens(messages, model_id=model_id)

        heuristic_est = len(prose) // 4 + 4

        divergence = abs(tiktoken_est - heuristic_est) / heuristic_est
        assert divergence > 0.15, (
            f"Expected >15% divergence between tiktoken ({tiktoken_est}) "
            f"and heuristic ({heuristic_est}), got {divergence:.2%}"
        )

    def test_estimate_openai_strips_prefix(self) -> None:
        messages = [_msg("Test message for gpt-4o")]
        result = estimate_prompt_tokens(messages, model_id="openai:gpt-4o")
        assert result > 0


class TestEstimateWithLRUCacheHits:
    """Repeated calls with the same model use the LRU-cached encoder."""

    def test_estimate_with_lrucache_hits(self) -> None:
        from atm.agents._tokens import _get_encoder

        _get_encoder.cache_clear()

        model_id = "openai:gpt-4o-mini"
        messages = [_msg("Cache test message")]

        result1 = estimate_prompt_tokens(messages, model_id=model_id)
        info_after_first = _get_encoder.cache_info()
        assert info_after_first.misses >= 1

        result2 = estimate_prompt_tokens(messages, model_id=model_id)
        info_after_second = _get_encoder.cache_info()
        assert info_after_second.hits >= 1

        assert result1 == result2

    def test_estimate_lrucache_no_crash_on_repeat(self) -> None:
        model_id = "openai:gpt-4o-mini"
        messages = [_msg("Repeated message")]
        results = [estimate_prompt_tokens(messages, model_id=model_id) for _ in range(10)]
        assert len(set(results)) == 1
        assert results[0] > 0
