"""Tests for Agent tool-calling loop and C1 accumulation invariant.

Test cases:
1. test_tool_call_then_final — 4 scratchpad events (reasoning, tool_call, observation, reasoning)
2. test_tool_calls_accumulate_across_iters — C1 assertion: len(tool_calls)==2 for 2-iter fixture
3. test_tool_error_converted_to_failed_result — ToolError -> ToolResult(ok=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from atm.agents.base import Agent
from atm.agents.config import AgentConfig
from atm.core.errors import ToolError
from atm.core.types import ToolResult
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry, ToolSchema

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


def _make_empty_state() -> dict:
    return {
        "shared": {"task_input": "Use the dummy tool."},
        "agents": {},
        "messages": [],
        "llm_calls": [],
    }


class FakeDummyTool:
    """Minimal tool stub for tool_loop tests."""

    name: ClassVar[str] = "dummy_tool"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="dummy_tool",
        description="A dummy tool for testing.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        from uuid import uuid4

        return ToolResult(call_id=uuid4(), ok=True, output={"result": "dummy_output"}, latency_ms=0)


class FakeRaisingTool:
    """Tool that raises ToolError on invocation."""

    name: ClassVar[str] = "raising_tool"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="raising_tool",
        description="A tool that always fails.",
        parameters={"type": "object", "properties": {}},
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        raise ToolError(tool_name="raising_tool", message="forced failure")


def _make_registry_with_dummy() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(FakeDummyTool())
    return registry


class TestToolCallThenFinal:
    """One tool call iteration followed by final stop response."""

    async def test_tool_call_then_final(self) -> None:
        """Expect 4 scratchpad events: reasoning(0), tool_call(0), observation(0), reasoning(1)."""
        llm = _make_llm("m5_agent_tool_loop.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=["dummy_tool"],
            max_tool_iters=5,
        )
        registry = _make_registry_with_dummy()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state = _make_empty_state()
        delta = await agent.step(state)

        scratchpad = delta["agents"]["p1"]["scratchpad"]
        assert len(scratchpad) == 4, f"Expected 4 events, got {len(scratchpad)}: {scratchpad}"

        kinds = [e["kind"] for e in scratchpad]
        assert kinds == ["reasoning", "tool_call", "observation", "reasoning"]

    async def test_tool_call_produces_tool_calls_in_delta(self) -> None:
        llm = _make_llm("m5_agent_tool_loop.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=["dummy_tool"],
            max_tool_iters=5,
        )
        registry = _make_registry_with_dummy()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state = _make_empty_state()
        delta = await agent.step(state)

        tool_calls = delta["agents"]["p1"]["tool_calls"]
        tool_results = delta["agents"]["p1"]["tool_results"]
        assert len(tool_calls) == 1
        assert len(tool_results) == 1
        assert tool_results[0].ok is True


class TestToolCallsAccumulateAcrossIters:
    """C1 assertion: tool_calls accumulate across ALL iterations of the tool loop."""

    async def test_tool_calls_accumulate_across_iters(self) -> None:
        """Two iterations each with one tool call -> len(tool_calls) == 2."""
        llm = _make_llm("m5_agent_tool_loop_two_iters.yaml")
        cfg = AgentConfig(
            role="planner",
            system_prompt="You are a planner.",
            tools=["dummy_tool"],
            max_tool_iters=5,
        )
        registry = _make_registry_with_dummy()
        agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

        state = _make_empty_state()
        delta = await agent.step(state)

        agent_delta = delta["agents"]["p1"]
        tool_calls = agent_delta["tool_calls"]

        assert len(tool_calls) == 2, (
            f"C1 violation: expected 2 accumulated tool_calls, got {len(tool_calls)}"
        )
        assert len(agent_delta["tool_results"]) == 2


class TestToolErrorConverted:
    """ToolError is converted to failed ToolResult — not propagated."""

    async def test_tool_error_converted_to_failed_result(self) -> None:
        """ToolError from tool -> ToolResult(ok=False) in tool_results."""
        import yaml as _yaml

        fixture_data = {
            "version": 1,
            "mode": "scripted",
            "entries": [
                {
                    "agent_id": "p1",
                    "role": "planner",
                    "step_idx": 0,
                    "content": "Calling the raising tool.",
                    "tool_calls": [{"name": "raising_tool", "args": {}}],
                    "finish_reason": "tool_calls",
                },
                {
                    "agent_id": "p1",
                    "role": "planner",
                    "step_idx": 1,
                    "content": "Tool failed but I recovered.",
                    "tool_calls": [],
                    "finish_reason": "stop",
                },
            ],
        }
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(fixture_data, f)
            tmp_path = f.name

        try:
            fake = FakeLLM(mode="scripted", fixture=tmp_path)
            llm = LLMWrapper(
                model_id="fake:deterministic",
                pricing=_make_pricing(),
                budget=_make_loose_budget(),
                llm=fake,
            )
            cfg = AgentConfig(
                role="planner",
                system_prompt="You are a planner.",
                tools=["raising_tool"],
                max_tool_iters=5,
            )
            registry = ToolRegistry()
            registry.register(FakeRaisingTool())
            agent = Agent(agent_id="p1", cfg=cfg, llm=llm, tools=registry)

            state = _make_empty_state()
            delta = await agent.step(state)

            tool_results = delta["agents"]["p1"]["tool_results"]
            assert len(tool_results) >= 1
            assert tool_results[0].ok is False
        finally:
            os.unlink(tmp_path)
