"""Tests for signal emission in Planner.step() — M8.5 requirements.

Verifies:
  1. test_planner_emits_ready_for_execution — after any step(), signals["ready_for_execution"] == True
  2. test_planner_signals_additive         — pre-existing signals are preserved
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atm.agents.config import AgentConfig
from atm.agents.planner import Planner
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.phases.signals import READY_FOR_EXECUTION
from atm.tools.base import ToolRegistry

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


def _make_pricing() -> Pricing:
    if PRICING_PATH.exists():
        return Pricing.from_yaml(PRICING_PATH)
    return Pricing(version=1, models={})


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_cfg() -> AgentConfig:
    return AgentConfig(
        role="planner",
        system_prompt="You are a planner.",
        tools=[],
        max_tool_iters=0,
    )


def _make_state(signals: dict[str, Any] | None = None) -> dict[str, Any]:
    shared: dict[str, Any] = {"task_input": "Write a plan.", "iter_total": 0}
    if signals is not None:
        shared["signals"] = signals
    return {
        "shared": shared,
        "agents": {},
        "messages": [],
        "llm_calls": [],
    }


class TestPlannerSignalEmission:
    """Planner.step() must emit ready_for_execution signal."""

    async def test_planner_emits_ready_for_execution(self) -> None:
        """After Planner.step(), signals['ready_for_execution'] == True."""
        llm = _make_llm("m6_chain_planner.yaml")
        planner = Planner(agent_id="planner", cfg=_make_cfg(), llm=llm, tools=ToolRegistry())

        delta = await planner.step(_make_state())

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        assert signals.get(READY_FOR_EXECUTION) is True, (
            f"Expected signals['{READY_FOR_EXECUTION}'] == True after Planner.step(), "
            f"got signals={signals}"
        )

    async def test_planner_signals_additive(self) -> None:
        """Pre-existing signals in shared state are preserved after emission."""
        llm = _make_llm("m6_chain_planner.yaml")
        planner = Planner(agent_id="planner", cfg=_make_cfg(), llm=llm, tools=ToolRegistry())

        existing = {"stuck": False, "rejected_count": 1}
        delta = await planner.step(_make_state(signals=existing))

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        assert signals.get("stuck") is False, "stuck should be preserved (False)"
        assert signals.get("rejected_count") == 1, "rejected_count should be preserved (1)"
        assert signals.get(READY_FOR_EXECUTION) is True
