"""Tests for Agent summarizer integration (scratchpad policy C).

Test cases:
1. test_summarizer_invoked_when_over_budget — summary text is "SUMMARY_TEXT"
2. test_summarizer_not_invoked_when_under_budget — summarizer LLM not called
3. test_summarizer_noop_when_summarizer_llm_is_none — no summarization without summarizer_llm
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from atm.agents.base import Agent, AgentView
from atm.agents.config import AgentConfig
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_loose_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0)


def _make_pricing() -> Pricing:
    return Pricing.from_yaml(PRICING_PATH)


def _make_primary_llm(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:deterministic",
        pricing=_make_pricing(),
        budget=_make_loose_budget(),
        llm=fake,
    )


def _make_summarizer_llm(fixture_name: str) -> LLMWrapper:
    """Create summarizer LLMWrapper with its own loose BudgetTracker."""
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    budget = BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0)
    return LLMWrapper(
        model_id="fake:deterministic",
        pricing=_make_pricing(),
        budget=budget,
        llm=fake,
    )


def _make_state_with_long_scratchpad(n_events: int = 20) -> dict[str, Any]:
    """Build a state with many scratchpad events to trigger summarization."""
    scratchpad = [
        {
            "kind": "reasoning",
            "content": f"Long reasoning content to fill tokens. Event {i}. " + "x" * 200,
            "step": i,
        }
        for i in range(n_events)
    ]
    return {
        "shared": {"task_input": "Test summarization trigger."},
        "agents": {
            "p1": {
                "agent_id": "p1",
                "role": "planner",
                "scratchpad": scratchpad,
                "inbox": [],
                "outbox": [],
                "tool_calls": [],
                "tool_results": [],
                "summary_before_window": "",
                "step_count": n_events,
                "tokens_spent": 0,
                "cost_spent_usd": 0.0,
            }
        },
        "messages": [],
        "llm_calls": [],
    }


class TestSummarizerInvokedWhenOverBudget:
    """Summarizer is called when prompt token estimate exceeds context_token_budget."""

    async def test_summarizer_invoked_when_over_budget(self) -> None:
        """When over budget, summary_before_window should be 'SUMMARY_TEXT'."""
        primary_llm = _make_primary_llm("m5_agent_summarizer_primary.yaml")
        summarizer_llm = _make_summarizer_llm("m5_agent_summarizer_secondary.yaml")

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=3,
            context_token_budget=1,
        )
        registry = ToolRegistry()
        agent = Agent(
            agent_id="p1",
            cfg=cfg,
            llm=primary_llm,
            tools=registry,
            summarizer_llm=summarizer_llm,
        )

        state = _make_state_with_long_scratchpad(n_events=10)
        delta = await agent.step(state)

        agent_state = delta["agents"]["p1"]
        summary = agent_state["summary_before_window"]
        assert summary == "SUMMARY_TEXT", f"Expected 'SUMMARY_TEXT' as summary, got: {summary!r}"


class TestSummarizerNotInvokedWhenUnderBudget:
    """Summarizer is NOT called when prompt is within token budget."""

    async def test_summarizer_not_invoked_when_under_budget(self) -> None:
        """When under budget, summary_before_window should remain unchanged."""
        primary_llm = _make_primary_llm("m5_agent_summarizer_primary.yaml")

        mock_summarizer = MagicMock()
        mock_summarizer.ainvoke = AsyncMock(
            side_effect=AssertionError("Summarizer should NOT be called")
        )

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=100,
            context_token_budget=999999,
        )
        registry = ToolRegistry()
        agent = Agent(
            agent_id="p1",
            cfg=cfg,
            llm=primary_llm,
            tools=registry,
            summarizer_llm=mock_summarizer,
        )

        state: dict[str, Any] = {
            "shared": {"task_input": "Short task."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }
        delta = await agent.step(state)
        assert delta is not None


class TestSummarizerNoopWhenSummarizerLlmIsNone:
    """No summarization occurs when summarizer_llm is None."""

    async def test_summarizer_noop_when_summarizer_llm_is_none(self) -> None:
        """With no summarizer_llm, summary_before_window is empty/unchanged."""
        primary_llm = _make_primary_llm("m5_agent_summarizer_primary.yaml")

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=3,
            context_token_budget=1,
        )
        registry = ToolRegistry()
        agent = Agent(
            agent_id="p1",
            cfg=cfg,
            llm=primary_llm,
            tools=registry,
            summarizer_llm=None,
        )

        state = _make_state_with_long_scratchpad(n_events=10)
        delta = await agent.step(state)

        summary = delta["agents"]["p1"]["summary_before_window"]
        assert summary == "", (
            f"Expected empty summary when summarizer_llm is None, got: {summary!r}"
        )

    async def test_summarizer_noop_view_unchanged(self) -> None:
        """_maybe_summarize returns view unchanged when summarizer_llm is None."""
        primary_llm = _make_primary_llm("m5_agent_summarizer_primary.yaml")

        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=3,
            context_token_budget=1,
        )
        registry = ToolRegistry()
        agent = Agent(
            agent_id="p1",
            cfg=cfg,
            llm=primary_llm,
            tools=registry,
            summarizer_llm=None,
        )

        original_scratchpad = tuple(
            {"kind": "reasoning", "content": f"event {i}", "step": i} for i in range(5)
        )
        view = AgentView(
            agent_id="p1",
            self_state={},
            shared={"task_input": "test"},
            inbox=(),
            scratchpad=original_scratchpad,
            summary_before_window=None,
        )
        returned_view = await agent._maybe_summarize(view)
        assert returned_view.scratchpad == original_scratchpad
        assert returned_view.summary_before_window is None
