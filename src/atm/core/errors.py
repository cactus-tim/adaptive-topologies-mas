"""Exception hierarchy for the ATM framework.

All custom exceptions inherit from AtmError. Designed for use with standard
Python exception chaining: `raise ToolError(tool_name, "...") from original_exc`.
"""

from __future__ import annotations


class AtmError(Exception):
    """Base exception for all ATM framework errors."""


class BudgetExceededError(AtmError):
    """Raised when a budget threshold (call / run / experiment) is exceeded.

    Attributes:
        level:     One of "call", "run", or "experiment".
        limit_usd: The configured budget ceiling in USD.
        spent_usd: The actual amount spent (exceeds limit_usd).
    """

    def __init__(self, *, level: str, limit_usd: float, spent_usd: float) -> None:
        self.level = level
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        super().__init__(
            f"Budget exceeded at level={level!r}: spent ${spent_usd:.4f} > limit ${limit_usd:.4f}"
        )


class PhaseError(AtmError):
    """Raised when an invalid phase transition is attempted.

    Attributes:
        attempted: The phase that was requested.
        current:   The phase the system is currently in.
    """

    def __init__(self, *, attempted: str, current: str) -> None:
        self.attempted = attempted
        self.current = current
        super().__init__(
            f"Invalid phase transition: attempted={attempted!r} while current={current!r}"
        )


class ToolError(AtmError):
    """Raised when a tool invocation fails.

    Uses standard Python exception chaining to attach the original cause:
        raise ToolError(tool_name="bash", message="...") from original_exc

    Attributes:
        tool_name: Name of the tool that failed.
        message:   Human-readable description of the failure.

    Properties:
        cause: Returns ``self.__cause__`` (set via ``raise ... from``).
               Is ``None`` when no chaining was used.
    """

    def __init__(self, *, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool {tool_name!r} failed: {message}")

    @property
    def cause(self) -> BaseException | None:
        """The chained exception set via ``raise ToolError(...) from exc``."""
        return self.__cause__


class LLMError(AtmError):
    """Raised when an LLM call fails after all retry attempts are exhausted.

    Uses standard Python exception chaining to attach the original cause:
        raise LLMError(provider="openai", model="gpt-4o", attempts=3, message="...") from original_exc

    Attributes:
        provider: LLM provider name (e.g. "openai", "anthropic", "vllm", "fake").
        model:    Model identifier (e.g. "gpt-4o", "claude-3-5-sonnet-latest").
        attempts: Number of attempts made before giving up.
        message:  Human-readable description of the failure.

    Properties:
        cause: Returns ``self.__cause__`` (set via ``raise ... from``).
               Is ``None`` when no chaining was used.
    """

    def __init__(self, *, provider: str, model: str, attempts: int, message: str) -> None:
        self.provider = provider
        self.model = model
        self.attempts = attempts
        super().__init__(
            f"LLM call failed after {attempts} attempt(s) [{provider}:{model}]: {message}"
        )

    @property
    def cause(self) -> BaseException | None:
        """The chained exception set via ``raise LLMError(...) from exc``."""
        return self.__cause__
