"""Tests for signal emission in Critic.step() — M8.5 requirements.

Verifies that Critic emits the correct signals into shared.signals:
  1. test_critic_approve_sets_critic_approved   — approve → signals["critic_approved"] == True
  2. test_critic_reject_increments_rejected_count — reject → signals["rejected_count"] increments
  3. test_critic_three_rejects_sets_needs_debate  — 3 rejects → signals["needs_debate"] == True
  4. test_critic_signals_additive               — existing signals preserved after emit
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atm.agents.config import AgentConfig
from atm.agents.critic import Critic
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.phases.signals import (
    CRITIC_APPROVED,
    NEEDS_DEBATE,
    REJECTED_COUNT,
)
from atm.tools.base import ToolRegistry

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        role="critic",
        system_prompt="You are a critic.",
        tools=[],
        max_tool_iters=0,
    )


def _make_state(signals: dict[str, Any] | None = None) -> dict[str, Any]:
    shared: dict[str, Any] = {"task_input": "Review the output.", "iter_total": 1}
    if signals is not None:
        shared["signals"] = signals
    return {
        "shared": shared,
        "agents": {},
        "messages": [],
        "llm_calls": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCriticSignalEmission:
    """Critic.step() must emit correct signals into delta['shared']['signals']."""

    async def test_critic_approve_sets_critic_approved(self) -> None:
        """Approve response → signals['critic_approved'] == True."""
        # m6_chain_critic.yaml produces "APPROVE fib"
        llm = _make_llm("m6_chain_critic.yaml")
        critic = Critic(agent_id="critic", cfg=_make_cfg(), llm=llm, tools=ToolRegistry())

        delta = await critic.step(_make_state())

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        assert signals.get(CRITIC_APPROVED) is True, (
            f"Expected signals['{CRITIC_APPROVED}'] == True, got {signals}"
        )
        # rejected_count should NOT be set on approve
        assert REJECTED_COUNT not in signals or signals.get(REJECTED_COUNT) == 0, (
            f"rejected_count should not be incremented on approve, signals={signals}"
        )

    async def test_critic_reject_increments_rejected_count(self) -> None:
        """Reject response → signals['rejected_count'] increments by 1."""
        llm = _make_llm("m8_critic_reject.yaml")
        critic = Critic(agent_id="critic", cfg=_make_cfg(), llm=llm, tools=ToolRegistry())

        state = _make_state(signals={REJECTED_COUNT: 1})
        delta = await critic.step(state)

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        count = signals.get(REJECTED_COUNT, 0)
        assert count == 2, f"Expected rejected_count == 2 (was 1, incremented by 1), got {count}"

    async def test_critic_three_rejects_sets_needs_debate(self) -> None:
        """After 3 rejects, signals['needs_debate'] == True."""
        # Use reject fixture — start at rejected_count=2 so this step makes it 3
        llm = _make_llm("m8_critic_reject.yaml")
        critic = Critic(agent_id="critic", cfg=_make_cfg(), llm=llm, tools=ToolRegistry())

        # Start with count=2 — after this step it becomes 3 → needs_debate
        state = _make_state(signals={REJECTED_COUNT: 2})
        delta = await critic.step(state)

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        assert signals.get(NEEDS_DEBATE) is True, (
            f"Expected signals['{NEEDS_DEBATE}'] == True at count=3, got signals={signals}"
        )
        assert signals.get(REJECTED_COUNT) == 3, (
            f"Expected rejected_count == 3, got {signals.get(REJECTED_COUNT)}"
        )

    async def test_critic_signals_additive(self) -> None:
        """Existing signals in shared.signals are preserved after emission."""
        llm = _make_llm("m6_chain_critic.yaml")  # APPROVE
        critic = Critic(agent_id="critic", cfg=_make_cfg(), llm=llm, tools=ToolRegistry())

        # Existing unrelated signal
        existing_signals: dict[str, Any] = {"ready_for_execution": True}
        state = _make_state(signals=existing_signals)
        delta = await critic.step(state)

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        # Existing signal preserved
        assert signals.get("ready_for_execution") is True, (
            "Pre-existing signals should be preserved in the output"
        )
        # New signal added
        assert signals.get(CRITIC_APPROVED) is True
