"""Tests for Agent basic happy-path and LLMWrapper.model_id property.

Test cases:
1. test_happy_path_no_tools  — step_count==1, outbox DRAFT created, tool_calls==[]
2. test_llm_wrapper_exposes_model_id — LLMWrapper.model_id returns the stored model id
"""

from __future__ import annotations

from pathlib import Path

from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.core.types import MessageKind
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_loose_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing.from_yaml(Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml")


def _make_llm(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    pricing = _make_pricing()
    budget = _make_loose_budget()
    return LLMWrapper(
        model_id="fake:deterministic",
        pricing=pricing,
        budget=budget,
        llm=fake,
    )


def _make_empty_state() -> dict:
    return {
        "shared": {"task_input": "Write a plan for testing."},
        "agents": {},
        "messages": [],
        "llm_calls": [],
    }


class TestHappyPathNoTools:
    """Agent.step() with no tools configured."""

    async def test_happy_path_no_tools(self) -> None:
        llm = _make_llm("m5_agent_basic.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state = _make_empty_state()
        delta = await agent.step(state)

        # step_count must be absolute (1 for first step)
        agents_delta = delta["agents"]
        agent_state = agents_delta["p1"]
        assert agent_state["step_count"] == 1

        # outbox must contain exactly one DRAFT message
        outbox = agent_state["outbox"]
        assert len(outbox) == 1
        assert outbox[0].kind == MessageKind.DRAFT
        assert outbox[0].sender == "p1"

        # No tool calls on a no-tools agent
        assert agent_state["tool_calls"] == []

        # delta has required keys
        assert "messages" in delta
        assert "llm_calls" in delta

    async def test_step_count_increments_on_second_step(self) -> None:
        """step_count starts from current state + 1."""
        llm = _make_llm("m5_agent_basic.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        # Simulate existing state with step_count=3
        state = {
            "shared": {"task_input": "Test"},
            "agents": {"p1": {"step_count": 3, "agent_id": "p1", "role": "planner"}},
            "messages": [],
            "llm_calls": [],
        }
        delta = await agent.step(state)
        assert delta["agents"]["p1"]["step_count"] == 4


class TestLLMWrapperExposesModelId:
    """LLMWrapper must expose model_id as a property."""

    def test_llm_wrapper_exposes_model_id(self) -> None:
        llm = _make_llm("m5_agent_basic.yaml")
        assert llm.model_id == "fake:deterministic"

    def test_llm_wrapper_model_id_matches_init(self) -> None:
        fake = FakeLLM(mode="echo")
        pricing = _make_pricing()
        budget = _make_loose_budget()
        wrapper = LLMWrapper(
            model_id="openai:gpt-4o-mini",
            pricing=pricing,
            budget=budget,
            llm=fake,
        )
        assert wrapper.model_id == "openai:gpt-4o-mini"
