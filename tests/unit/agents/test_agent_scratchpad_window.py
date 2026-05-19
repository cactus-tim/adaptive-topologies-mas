"""Tests for Agent scratchpad windowing (policy C).

Test cases:
1. test_window_limits_events_in_prompt — window_size=3, after 5 steps prompt uses only last 3 events
2. test_original_scratchpad_not_mutated — full scratchpad preserved, not truncated
3. test_prompt_contains_expected_events — after 5 steps, prompt includes R2/R3/R4, not R0/R1

Note on accumulation:
``Agent.step()`` returns a per-step *delta* for ``scratchpad`` (only the
events produced in that step). The state-bloat fix made the LangGraph
reducer pick the longer of the two scratchpad lists rather than concat,
so the full append-only scratchpad is reconstructed by the caller, not by
``merge_agent_states``. These tests therefore accumulate the per-step
deltas explicitly (``_accumulate``) — exercising the windowing logic of
``_build_prompt`` independently of the reducer's merge policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atm.agents.base import Agent, AgentView
from atm.agents.config import AgentConfig
from atm.core.reducers import merge_agent_states
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
    return Pricing.from_yaml(PRICING_PATH)


def _make_llm(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:deterministic",
        pricing=_make_pricing(),
        budget=_make_loose_budget(),
        llm=fake,
    )


async def _run_steps_accumulating(
    agent: Agent, state: dict[str, Any], n_steps: int
) -> list[dict[str, Any]]:
    """Run ``n_steps`` agent steps, returning the full append-only scratchpad.

    ``agent.step()`` returns a per-step delta for ``scratchpad``; the LangGraph
    reducer keeps the longer list rather than concatenating, so the full
    scratchpad must be reconstructed by the caller. This helper does exactly
    that: it concatenates each step's scratchpad delta and feeds the running
    scratchpad back into ``state`` so the next step sees prior context.
    """
    full_scratchpad: list[dict[str, Any]] = []
    for _ in range(n_steps):
        delta = await agent.step(state)
        agent_delta = delta["agents"][agent.agent_id]
        full_scratchpad += list(agent_delta.get("scratchpad") or [])
        existing_agents = state.get("agents") or {}
        merged = merge_agent_states(existing_agents, delta["agents"])
        merged[agent.agent_id]["scratchpad"] = list(full_scratchpad)
        state["agents"] = merged
    return full_scratchpad


class TestScratchpadWindow:
    """Scratchpad windowing semantics for policy C."""

    async def test_window_limits_events_in_prompt(self) -> None:
        """After 5 steps, _build_prompt uses only last window_size=3 events."""
        llm = _make_llm("m5_agent_scratchpad.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=3,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state: dict[str, Any] = {
            "shared": {"task_input": "Test windowing."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }

        accumulated_scratchpad = await _run_steps_accumulating(agent, state, 5)
        assert len(accumulated_scratchpad) == 5, (
            f"Expected 5 scratchpad events after 5 steps, got {len(accumulated_scratchpad)}"
        )

        view = AgentView(
            agent_id="p1",
            self_state=state["agents"]["p1"],
            shared=state.get("shared") or {},
            inbox=(),
            scratchpad=tuple(accumulated_scratchpad),
            summary_before_window=None,
        )
        prompt_messages = agent._build_prompt(view)

        scratchpad_msgs = [
            m
            for m in prompt_messages
            if m.content.startswith("[Reasoning step")
            or m.content.startswith("[Tool")
            or m.content.startswith("[Observation")
        ]

        assert len(scratchpad_msgs) == 3, (
            f"Expected 3 scratchpad events in prompt (window_size=3), got {len(scratchpad_msgs)}"
        )

    async def test_original_scratchpad_not_mutated(self) -> None:
        """Scratchpad in state is append-only — not truncated by windowing."""
        llm = _make_llm("m5_agent_scratchpad.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=2,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state: dict[str, Any] = {
            "shared": {"task_input": "Test."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }

        full_scratchpad = await _run_steps_accumulating(agent, state, 5)
        assert len(full_scratchpad) == 5, (
            f"Original scratchpad must have all 5 events, got {len(full_scratchpad)}"
        )

    async def test_prompt_contains_r2_r3_r4_not_r0_r1(self) -> None:
        """After 5 steps with window_size=3, prompt contains R2/R3/R4, not R0/R1."""
        llm = _make_llm("m5_agent_scratchpad.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=[],
            max_tool_iters=0,
            window_size=3,
        )
        registry = ToolRegistry()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state: dict[str, Any] = {
            "shared": {"task_input": "Test windowing R2/R3/R4."},
            "agents": {},
            "messages": [],
            "llm_calls": [],
        }

        full_scratchpad = await _run_steps_accumulating(agent, state, 5)
        view = AgentView(
            agent_id="p1",
            self_state=state["agents"]["p1"],
            shared=state.get("shared") or {},
            inbox=(),
            scratchpad=tuple(full_scratchpad),
            summary_before_window=None,
        )
        prompt_messages = agent._build_prompt(view)
        prompt_text = " ".join(m.content for m in prompt_messages)

        assert "R2:" in prompt_text, f"R2 should be in prompt window. Prompt: {prompt_text[:500]}"
        assert "R3:" in prompt_text, f"R3 should be in prompt window. Prompt: {prompt_text[:500]}"
        assert "R4:" in prompt_text, f"R4 should be in prompt window. Prompt: {prompt_text[:500]}"

        assert "R0:" not in prompt_text, f"R0 should NOT be in window. Prompt: {prompt_text[:500]}"
        assert "R1:" not in prompt_text, f"R1 should NOT be in window. Prompt: {prompt_text[:500]}"
