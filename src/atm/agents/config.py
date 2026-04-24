"""AgentConfig pydantic v2 model and YAML loader for ATM agent configurations.

AgentConfig is frozen (immutable) — use model_copy(update=...) to create
modified copies without mutating the original instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentConfig(BaseModel):
    """Configuration for an ATM agent.

    All fields except `role` and `system_prompt` have sensible defaults
    so minimal YAML files only need to specify those two fields.

    Frozen — direct attribute assignment raises TypeError.
    Use `model_copy(update={...})` to create modified copies.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    """Agent role identifier (e.g. 'planner', 'researcher', 'debater')."""

    system_prompt: str
    """System prompt template. May contain '{{stance}}' for Debater."""

    window_size: int = Field(default=10, ge=1)
    """Number of scratchpad events to include in the prompt window.

    Note: this is *events*, not logical steps. One logical step can produce
    up to 3 events (reasoning + tool_call + observation).
    """

    max_tool_iters: int = Field(default=5, ge=0)
    """Maximum number of tool-calling iterations per step. 0 = no tool calls."""

    temperature: float = Field(default=0.0)
    """LLM sampling temperature. 0.0 = deterministic."""

    summarizer_model_id: str | None = Field(default=None)
    """Model ID for the optional summarizer LLM (e.g. 'openai:gpt-4o-mini').
    When None, summarization is disabled.
    """

    context_token_budget: int = Field(default=8000, ge=1)
    """Token budget for the prompt context. When estimated prompt tokens exceed
    this value AND summarizer_model_id is set, summarization is triggered.
    """

    tools: list[str] = Field(default_factory=list)
    """List of tool names available to this agent."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary extra parameters (e.g. {'stance': 'pro'} for Debater)."""

    @field_validator("window_size")
    @classmethod
    def _validate_window_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"window_size must be >= 1, got {v}")
        return v

    @field_validator("max_tool_iters")
    @classmethod
    def _validate_max_tool_iters(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_tool_iters must be >= 0, got {v}")
        return v

    @field_validator("context_token_budget")
    @classmethod
    def _validate_context_token_budget(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"context_token_budget must be >= 1, got {v}")
        return v


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load an AgentConfig from a YAML file.

    Args:
        path: Path to the YAML file (str or pathlib.Path).

    Returns:
        A fully-validated, frozen AgentConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValidationError: If the YAML content fails AgentConfig validation.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Agent config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}

    return AgentConfig(**data)
