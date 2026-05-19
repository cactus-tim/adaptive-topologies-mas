"""Tests for BudgetExceededError propagation from Agent.step().

BudgetExceededError must NOT be caught by Agent — it must propagate
unchanged from LLMWrapper.ainvoke through the tool loop.

Test cases:
1. test_budget_exceeded_propagates — BudgetExceededError bubbles out of step()
2. test_budget_exceeded_level_preserved — error level attribute is preserved
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.core.errors import BudgetExceededError
from atm.llm.pricing import Pricing
from atm.tools.base import ToolRegistry

PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_pricing() -> Pricing:
    return Pricing.from_yaml(PRICING_PATH)


class _RaisingLLM:
    """Fake LLM that raises BudgetExceededError on every ainvoke call.

    Used as a drop-in replacement for FakeLLM to test budget propagation.
    Does NOT inherit from LLMWrapper — it is injected directly as the
    underlying ``llm`` of LLMWrapper. However for simplicity in these tests
    we create a mock LLMWrapper whose ainvoke raises directly.
    """

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        raise BudgetExceededError(level="run", limit_usd=1.0, spent_usd=2.0)


class TestBudgetExceededPropagates:
    """BudgetExceededError must propagate out of Agent.step() unchanged."""

    async def test_budget_exceeded_propagates(self) -> None:
        """BudgetExceededError raised by LLMWrapper must propagate through step()."""
        from unittest.mock import AsyncMock

        mock_llm = AsyncMock()
        mock_llm.model_id = "fake:deterministic"
        mock_llm.ainvoke = AsyncMock(
            side_effect=BudgetExceededError(level="run", limit_usd=1.0, spent_usd=2.0)
        )

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=mock_llm, tools=registry)

        state: dict[str, Any] = {
            "shared": {"task_input": "Test budget propagation."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }

        with pytest.raises(BudgetExceededError):
            await agent.step(state)

    async def test_budget_exceeded_level_preserved(self) -> None:
        """BudgetExceededError.level attribute is preserved when propagated."""
        mock_llm = AsyncMock()
        mock_llm.model_id = "fake:deterministic"
        mock_llm.ainvoke = AsyncMock(
            side_effect=BudgetExceededError(level="run", limit_usd=1.0, spent_usd=2.0)
        )

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
        )
        registry = ToolRegistry()

        agent = Agent(agent_id="p1", cfg=cfg, llm=mock_llm, tools=registry)

        state: dict[str, Any] = {
            "shared": {"task_input": "Test."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }

        with pytest.raises(BudgetExceededError) as exc_info:
            await agent.step(state)

        assert exc_info.value.level == "run"
        assert exc_info.value.limit_usd == 1.0
        assert exc_info.value.spent_usd == 2.0

    async def test_budget_exceeded_not_swallowed_in_tool_loop(self) -> None:
        """BudgetExceededError propagates even from within _run_tool_loop."""
        mock_llm = AsyncMock()
        mock_llm.model_id = "fake:deterministic"
        mock_llm.ainvoke = AsyncMock(
            side_effect=BudgetExceededError(level="experiment", limit_usd=5.0, spent_usd=6.0)
        )

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=5,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=mock_llm, tools=registry)

        state: dict[str, Any] = {
            "shared": {"task_input": "Test."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }

        with pytest.raises(BudgetExceededError) as exc_info:
            await agent.step(state)

        assert exc_info.value.level == "experiment"
