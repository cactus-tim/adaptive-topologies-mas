"""Unit tests for atm.llm.retry — RetryPolicy, is_transient, with_retry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from atm.core.errors import LLMError
from atm.llm.retry import RetryPolicy, is_transient, with_retry

# ---------------------------------------------------------------------------
# Helpers / fake exception classes
# ---------------------------------------------------------------------------


class FakeRateLimitError(Exception):
    """Simulates a 429 rate-limit error via duck-typed status_code."""

    status_code = 429


class FakeServerError(Exception):
    """Simulates a 500 server error via duck-typed status_code."""

    status_code = 500


class FakeTransientExplicitError(Exception):
    """Exception registered in RetryPolicy.retry_on tuple."""


class FakeNonTransientError(Exception):
    """Plain error with no status_code — should NOT be retried by default."""


# ---------------------------------------------------------------------------
# is_transient tests
# ---------------------------------------------------------------------------


class TestIsTransient:
    def _policy(self, retry_on: tuple = ()) -> RetryPolicy:
        return RetryPolicy(retry_on=retry_on)

    def test_explicit_class_in_retry_on_is_transient(self) -> None:
        policy = self._policy(retry_on=(FakeTransientExplicitError,))
        exc = FakeTransientExplicitError("boom")
        assert is_transient(exc, policy) is True

    def test_non_transient_class_not_in_retry_on(self) -> None:
        policy = self._policy(retry_on=())
        exc = FakeNonTransientError("nope")
        assert is_transient(exc, policy) is False

    def test_duck_type_status_code_429(self) -> None:
        """Exception not in retry_on tuple but status_code==429 → transient."""
        policy = self._policy(retry_on=())
        exc = FakeRateLimitError("rate limited")
        assert is_transient(exc, policy) is True

    def test_duck_type_status_code_500(self) -> None:
        """Exception not in retry_on tuple but status_code>=500 → transient."""
        policy = self._policy(retry_on=())
        exc = FakeServerError("internal server error")
        assert is_transient(exc, policy) is True

    def test_duck_type_status_code_400_not_transient(self) -> None:
        """Client error 400 with status_code should NOT be transient."""

        class ClientRequestError(Exception):
            status_code = 400

        policy = self._policy(retry_on=())
        exc = ClientRequestError("bad request")
        assert is_transient(exc, policy) is False

    def test_duck_type_status_code_503_transient(self) -> None:
        """Any status_code >= 500 is transient."""

        class ServiceUnavailableError(Exception):
            status_code = 503

        policy = self._policy(retry_on=())
        exc = ServiceUnavailableError("unavailable")
        assert is_transient(exc, policy) is True

    def test_subclass_in_retry_on_detected(self) -> None:
        """Subclass of a listed retry_on type should also be detected."""

        class SubTransientError(FakeTransientExplicitError):
            pass

        policy = self._policy(retry_on=(FakeTransientExplicitError,))
        exc = SubTransientError("sub")
        assert is_transient(exc, policy) is True


# ---------------------------------------------------------------------------
# with_retry — success cases
# ---------------------------------------------------------------------------


class TestWithRetrySuccess:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        """Callable called exactly once on first-attempt success."""
        fn = AsyncMock(return_value="ok")
        policy = RetryPolicy(max_retries=3, base_delay_s=0.0, jitter=False)

        result = await with_retry(fn, policy=policy, provider="test", model="m")

        assert result == "ok"
        fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_transient_then_success_called_twice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callable attempted twice: first raises transient, second succeeds."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FakeRateLimitError("rate limited")
            return "success"

        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        policy = RetryPolicy(max_retries=3, base_delay_s=1.0, jitter=False)
        result = await with_retry(fn, policy=policy, provider="test", model="m")

        assert result == "success"
        assert call_count == 2
        assert len(slept) == 1
        # First retry: delay = base_delay_s * 2**0 = 1.0 (no jitter)
        assert slept[0] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay_doubles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify exponential backoff: delays double each attempt (no jitter)."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise FakeServerError("server error")
            return "done"

        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        policy = RetryPolicy(max_retries=5, base_delay_s=1.0, max_delay_s=30.0, jitter=False)
        result = await with_retry(fn, policy=policy, provider="test", model="m")

        assert result == "done"
        assert call_count == 4
        assert len(slept) == 3
        # attempt 1→2: 1.0 * 2**0 = 1.0
        # attempt 2→3: 1.0 * 2**1 = 2.0
        # attempt 3→4: 1.0 * 2**2 = 4.0
        assert slept[0] == pytest.approx(1.0)
        assert slept[1] == pytest.approx(2.0)
        assert slept[2] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# with_retry — non-transient raises immediately
# ---------------------------------------------------------------------------


class TestWithRetryNonTransient:
    @pytest.mark.asyncio
    async def test_non_transient_raises_immediately(self) -> None:
        """Non-transient error must propagate immediately, no retries."""
        original = FakeNonTransientError("fatal error")
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise original

        policy = RetryPolicy(max_retries=5, base_delay_s=0.0, jitter=False)

        with pytest.raises(FakeNonTransientError) as exc_info:
            await with_retry(fn, policy=policy, provider="test", model="m")

        assert exc_info.value is original
        assert call_count == 1  # called exactly once, not retried

    @pytest.mark.asyncio
    async def test_non_transient_cause_not_wrapped_in_llm_error(self) -> None:
        """Non-transient error is re-raised directly, NOT wrapped in LLMError."""
        original = FakeNonTransientError("fatal")

        async def fn() -> str:
            raise original

        policy = RetryPolicy(max_retries=3, base_delay_s=0.0, jitter=False)

        with pytest.raises(FakeNonTransientError):
            await with_retry(fn, policy=policy, provider="p", model="m")


# ---------------------------------------------------------------------------
# with_retry — exhaustion wraps into LLMError
# ---------------------------------------------------------------------------


class TestWithRetryExhaustion:
    @pytest.mark.asyncio
    async def test_exhaustion_raises_llm_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After max_retries transient failures, raises LLMError."""
        original = FakeRateLimitError("always rate limited")

        async def fn() -> str:
            raise original

        async def fake_sleep(_: float) -> None:
            pass

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        policy = RetryPolicy(max_retries=3, base_delay_s=0.0, jitter=False)

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, policy=policy, provider="openai", model="gpt-4o")

        err = exc_info.value
        assert err.provider == "openai"
        assert err.model == "gpt-4o"
        assert err.attempts == 3 + 1  # initial attempt + max_retries
        assert err.__cause__ is original  # chain preserved

    @pytest.mark.asyncio
    async def test_exhaustion_cause_is_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit __cause__ check: exc.__cause__ is original_rate_limit_error."""
        original_rate_limit_error = FakeRateLimitError("429 always")

        async def fn() -> str:
            raise original_rate_limit_error

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        policy = RetryPolicy(max_retries=2, base_delay_s=0.0, jitter=False)

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, policy=policy, provider="openai", model="gpt-4o-mini")

        exc = exc_info.value
        assert exc.__cause__ is original_rate_limit_error

    @pytest.mark.asyncio
    async def test_exhaustion_attempt_count_correct(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """attempts field = max_retries + 1 (initial call + retries)."""
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeServerError("always 500")

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        max_retries = 4
        policy = RetryPolicy(max_retries=max_retries, base_delay_s=0.0, jitter=False)

        with pytest.raises(LLMError) as exc_info:
            await with_retry(fn, policy=policy, provider="vllm", model="llama")

        assert exc_info.value.attempts == max_retries + 1
        assert call_count == max_retries + 1


# ---------------------------------------------------------------------------
# with_retry — jitter behaviour
# ---------------------------------------------------------------------------


class TestWithRetryJitter:
    @pytest.mark.asyncio
    async def test_jitter_delays_within_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With jitter=True, all observed delays are in [0, cap] where cap = base * 2**(attempt-1)."""
        observed_delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            observed_delays.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        base = 1.0
        max_delay = 30.0
        policy = RetryPolicy(
            max_retries=5,
            base_delay_s=base,
            max_delay_s=max_delay,
            jitter=True,
            retry_on=(FakeRateLimitError,),
        )

        # Run 10 repetitions to gather enough jitter samples
        for _ in range(10):
            observed_delays.clear()
            call_count = 0

            async def fn() -> str:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise FakeRateLimitError("429")
                return "ok"

            await with_retry(fn, policy=policy, provider="p", model="m")

        # After 10 runs we have many delay samples; all must be >= 0
        # Each run produces 2 delays (attempt 1→2, attempt 2→3)
        assert len(observed_delays) > 0
        for d in observed_delays:
            assert d >= 0.0, f"Delay {d} is negative"
            assert d <= max_delay, f"Delay {d} exceeds max_delay_s {max_delay}"

    @pytest.mark.asyncio
    async def test_jitter_produces_nonzero_delay_eventually(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With jitter=True across 10 samples, at least one delay > 0."""
        all_delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            all_delays.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        base = 2.0
        policy = RetryPolicy(
            max_retries=10,
            base_delay_s=base,
            max_delay_s=60.0,
            jitter=True,
            retry_on=(FakeRateLimitError,),
        )

        for _ in range(10):
            call_count = 0

            async def fn() -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise FakeRateLimitError("429")
                return "done"

            await with_retry(fn, policy=policy, provider="p", model="m")

        assert any(d > 0.0 for d in all_delays), (
            f"Expected at least one positive delay, got: {all_delays}"
        )

    @pytest.mark.asyncio
    async def test_no_jitter_produces_exact_delay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without jitter, delay is exactly base * 2**(attempt-1) (capped at max_delay_s)."""
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        policy = RetryPolicy(max_retries=3, base_delay_s=2.0, max_delay_s=100.0, jitter=False)
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeRateLimitError("429")
            return "done"

        await with_retry(fn, policy=policy, provider="p", model="m")

        # attempt 1→2: 2.0 * 2**0 = 2.0
        # attempt 2→3: 2.0 * 2**1 = 4.0
        assert slept == pytest.approx([2.0, 4.0])

    @pytest.mark.asyncio
    async def test_max_delay_caps_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Delay is capped at max_delay_s even when exponential formula exceeds it."""
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        policy = RetryPolicy(
            max_retries=10,
            base_delay_s=1.0,
            max_delay_s=5.0,
            jitter=False,
        )
        call_count = 0

        async def fn() -> str:
            nonlocal call_count
            call_count += 1
            raise FakeRateLimitError("429 always")

        with pytest.raises(LLMError):
            await with_retry(fn, policy=policy, provider="p", model="m")

        for d in slept:
            assert d <= 5.0, f"Delay {d} exceeds max_delay_s 5.0"
