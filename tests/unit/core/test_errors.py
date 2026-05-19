"""Unit tests for atm.core.errors — exception hierarchy."""

from __future__ import annotations

import pytest

from atm.core.errors import AtmError, BudgetExceededError, LLMError, PhaseError, ToolError


class TestAtmErrorHierarchy:
    """All custom exceptions must inherit from AtmError (which inherits from Exception)."""

    def test_atm_error_is_exception(self) -> None:
        assert issubclass(AtmError, Exception)

    def test_budget_exceeded_inherits_atm_error(self) -> None:
        assert issubclass(BudgetExceededError, AtmError)

    def test_phase_error_inherits_atm_error(self) -> None:
        assert issubclass(PhaseError, AtmError)

    def test_tool_error_inherits_atm_error(self) -> None:
        assert issubclass(ToolError, AtmError)


class TestBudgetExceededError:
    """BudgetExceededError must carry level, limit_usd, spent_usd."""

    def test_fields_stored(self) -> None:
        exc = BudgetExceededError(level="call", limit_usd=0.10, spent_usd=0.15)
        assert exc.level == "call"
        assert exc.limit_usd == 0.10
        assert exc.spent_usd == 0.15

    def test_is_raiseable(self) -> None:
        with pytest.raises(BudgetExceededError) as exc_info:
            raise BudgetExceededError(level="run", limit_usd=1.0, spent_usd=1.5)
        assert exc_info.value.level == "run"

    def test_str_contains_useful_info(self) -> None:
        exc = BudgetExceededError(level="experiment", limit_usd=5.0, spent_usd=6.0)
        text = str(exc)
        assert len(text) > 0

    def test_level_values(self) -> None:
        """level can be call, run, or experiment."""
        for level in ("call", "run", "experiment"):
            exc = BudgetExceededError(level=level, limit_usd=1.0, spent_usd=2.0)
            assert exc.level == level


class TestPhaseError:
    """PhaseError must carry attempted and current fields."""

    def test_fields_stored(self) -> None:
        exc = PhaseError(attempted="execution", current="planning")
        assert exc.attempted == "execution"
        assert exc.current == "planning"

    def test_is_raiseable(self) -> None:
        with pytest.raises(PhaseError) as exc_info:
            raise PhaseError(attempted="done", current="execution")
        assert exc_info.value.attempted == "done"
        assert exc_info.value.current == "execution"

    def test_str_non_empty(self) -> None:
        exc = PhaseError(attempted="verification", current="planning")
        assert len(str(exc)) > 0


class TestToolError:
    """ToolError.cause must reflect __cause__ set via `raise ... from`."""

    def test_tool_name_stored(self) -> None:
        exc = ToolError(tool_name="bash", message="failed")
        assert exc.tool_name == "bash"

    def test_cause_set_via_chaining(self) -> None:
        """Test #4: chaining via `from` sets .cause and .__cause__ equivalently."""
        original = ValueError("underlying problem")
        try:
            raise ToolError(tool_name="python_repl", message="execution error") from original
        except ToolError as exc:
            assert exc.cause is original
            assert exc.__cause__ is original
            assert exc.cause is exc.__cause__

    def test_cause_is_none_without_chaining(self) -> None:
        """Test #5: no chaining → .cause is None."""
        exc = ToolError(tool_name="search", message="not found")
        assert exc.cause is None
        assert exc.__cause__ is None

    def test_cause_property_returns_cause(self) -> None:
        """The .cause property must return self.__cause__."""
        original = RuntimeError("root cause")
        try:
            raise ToolError(tool_name="docker", message="container failed") from original
        except ToolError as exc:
            assert exc.cause is exc.__cause__

    def test_no_cause_constructor_arg(self) -> None:
        """ToolError must NOT accept a cause= keyword argument in constructor."""
        with pytest.raises(TypeError):
            ToolError(tool_name="bash", message="fail", cause=ValueError("x"))  # type: ignore[call-arg]

    def test_is_raiseable(self) -> None:
        with pytest.raises(ToolError) as exc_info:
            raise ToolError(tool_name="grep", message="pattern error")
        assert exc_info.value.tool_name == "grep"

    def test_standard_chaining_pattern(self) -> None:
        """Standard pattern: raise ToolError(tool_name, '...') from original_exc."""
        original = OSError("file not found")
        try:
            try:
                raise original
            except OSError as e:
                raise ToolError(tool_name="file_reader", message="cannot read file") from e
        except ToolError as exc:
            assert exc.cause is original
            assert isinstance(exc.__cause__, OSError)


class TestAtmErrorDirect:
    """AtmError is a valid base exception that can be raised directly."""

    def test_can_raise_atm_error(self) -> None:
        with pytest.raises(AtmError):
            raise AtmError("base error")

    def test_all_errors_caught_by_atm_error(self) -> None:
        for exc in [
            BudgetExceededError(level="call", limit_usd=1.0, spent_usd=2.0),
            PhaseError(attempted="done", current="planning"),
            ToolError(tool_name="x", message="y"),
            LLMError(provider="openai", model="gpt-4o", attempts=3, message="exhausted"),
        ]:
            assert isinstance(exc, AtmError)


class TestLLMError:
    """LLMError must inherit AtmError and carry provider, model, attempts attrs."""

    def test_inherits_atm_error(self) -> None:
        assert issubclass(LLMError, AtmError)

    def test_fields_stored(self) -> None:
        exc = LLMError(provider="openai", model="gpt-4o", attempts=3, message="retry exhausted")
        assert exc.provider == "openai"
        assert exc.model == "gpt-4o"
        assert exc.attempts == 3

    def test_str_non_empty(self) -> None:
        exc = LLMError(
            provider="anthropic", model="claude-3-5-sonnet-latest", attempts=1, message="rate limit"
        )
        assert len(str(exc)) > 0

    def test_is_raiseable(self) -> None:
        with pytest.raises(LLMError) as exc_info:
            raise LLMError(provider="openai", model="gpt-4o-mini", attempts=2, message="timeout")
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "gpt-4o-mini"
        assert exc_info.value.attempts == 2

    def test_cause_set_via_chaining(self) -> None:
        """LLMError.cause reflects __cause__ set via `raise ... from`."""
        original = TimeoutError("network timeout")
        try:
            raise LLMError(
                provider="openai", model="gpt-4o", attempts=3, message="retry exhausted"
            ) from original
        except LLMError as exc:
            assert exc.cause is original
            assert exc.__cause__ is original

    def test_cause_is_none_without_chaining(self) -> None:
        exc = LLMError(provider="fake", model="deterministic", attempts=0, message="no cause")
        assert exc.cause is None

    def test_cause_property_returns_cause(self) -> None:
        original = RuntimeError("root")
        try:
            raise LLMError(provider="vllm", model="llama", attempts=1, message="fail") from original
        except LLMError as exc:
            assert exc.cause is exc.__cause__

    def test_no_cause_constructor_arg(self) -> None:
        """LLMError must NOT accept a cause= keyword argument in constructor."""
        with pytest.raises(TypeError):
            LLMError(provider="x", model="y", attempts=1, message="z", cause=ValueError("x"))  # type: ignore[call-arg]
