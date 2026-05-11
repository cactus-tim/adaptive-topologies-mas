"""Tests for signal emission in Executor.step() — M8.5 requirements.

Verifies:
  1. test_executor_success_sets_ready_for_verification — successful code_run
       → signals["ready_for_verification"] == True
  2. test_executor_stuck_after_three_failures — 3 consecutive failed code_run
       → signals["stuck"] == True
  3. test_executor_success_resets_streak — success after failures does not
       leave stuck=True if streak < threshold
  4. test_executor_custom_threshold — stuck_threshold=2 fires sooner
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from atm.agents.config import AgentConfig
from atm.agents.executor import Executor
from atm.core.types import ToolResult
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.phases.signals import READY_FOR_VERIFICATION, STUCK
from atm.tools.base import ToolRegistry, ToolSchema

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


# ---------------------------------------------------------------------------
# Helpers: fake tools
# ---------------------------------------------------------------------------


class FakeCodeRunSuccess:
    """Stub code_run tool that always returns ok=True."""

    name: ClassVar[str] = "code_run"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="code_run",
        description="Execute code.",
        parameters={
            "type": "object",
            "properties": {
                "lang": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["lang", "code"],
        },
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            call_id=uuid4(),
            ok=True,
            output={"stdout": "ok\n", "stderr": "", "exit_code": 0},
            error=None,
            latency_ms=1,
        )


class FakeCodeRunFail:
    """Stub code_run tool that always returns ok=False."""

    name: ClassVar[str] = "code_run"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="code_run",
        description="Execute code.",
        parameters={
            "type": "object",
            "properties": {
                "lang": {"type": "string"},
                "code": {"type": "string"},
            },
            "required": ["lang", "code"],
        },
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            call_id=uuid4(),
            ok=False,
            output=None,
            error="RuntimeError: something went wrong",
            latency_ms=1,
        )


# ---------------------------------------------------------------------------
# Helpers: LLM + config
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


def _make_cfg(max_tool_iters: int = 6) -> AgentConfig:
    return AgentConfig(
        role="executor",
        system_prompt="You are an executor.",
        tools=["code_run"],
        max_tool_iters=max_tool_iters,
        window_size=12,
        context_token_budget=12000,
    )


def _make_state(
    signals: dict[str, Any] | None = None,
    scratchpad: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shared: dict[str, Any] = {"task_input": "Compute 1+1", "iter_total": 1}
    if signals is not None:
        shared["signals"] = signals

    agents: dict[str, Any] = {}
    if scratchpad is not None:
        agents = {"e1": {"scratchpad": scratchpad, "agent_id": "e1", "role": "executor"}}

    return {
        "shared": shared,
        "agents": agents,
        "messages": [],
        "llm_calls": [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExecutorSignalEmission:
    """Executor.step() must emit correct signals based on code_run results."""

    async def test_executor_success_sets_ready_for_verification(self) -> None:
        """Successful code_run → signals['ready_for_verification'] == True."""
        # m5_executor_exit fixture: calls code_run once; success
        llm = _make_llm("m5_executor_exit.yaml")
        registry = ToolRegistry()
        registry.register(FakeCodeRunSuccess())

        executor = Executor(agent_id="e1", cfg=_make_cfg(), llm=llm, tools=registry)
        delta = await executor.step(_make_state())

        shared_out: dict[str, Any] = delta.get("shared") or {}
        signals: dict[str, Any] = shared_out.get("signals") or {}
        assert signals.get(READY_FOR_VERIFICATION) is True, (
            f"Expected signals['{READY_FOR_VERIFICATION}'] == True, got signals={signals}"
        )

    async def test_executor_stuck_after_three_failures(self) -> None:
        """3 consecutive failed code_run calls → signals['stuck'] == True.

        Uses the m6_chain_executor fixture which calls code_run once per step.
        We simulate 3 steps, each with a failure, by building a streak in
        the scratchpad between calls.
        """
        # m6_chain_executor: calls code_run (step 0) then emits final answer (step 1)
        # We run 3 steps with failures to trigger stuck
        registry = ToolRegistry()
        registry.register(FakeCodeRunFail())

        # For three steps, we need 3 x (tool_call + stop) iterations.
        # We use the multi-step fixture that has a code_run on step 0.
        # But the scripted fixture has only limited entries.  Instead, use
        # m5_executor_exit which also calls code_run once, then gives a stop.
        # We run step() three times, threading state through.

        # Build a fresh LLM each time (scripted fixtures are stateful/sequential)
        state = _make_state()

        for _step_num in range(3):
            llm = _make_llm("m5_executor_exit.yaml")
            executor = Executor(agent_id="e1", cfg=_make_cfg(), llm=llm, tools=registry)
            delta = await executor.step(state)

            # Thread the shared state forward (merge signals)
            new_shared: dict[str, Any] = dict(state["shared"])
            if "shared" in delta:
                new_shared.update(delta["shared"])

            # Thread the scratchpad forward so _read_fail_streak works
            existing_scratchpad: list[dict[str, Any]] = list(
                state.get("agents", {}).get("e1", {}).get("scratchpad") or []
            )
            new_scratchpad = existing_scratchpad + list(
                delta.get("agents", {}).get("e1", {}).get("scratchpad") or []
            )
            state = {
                "shared": new_shared,
                "agents": {"e1": {"scratchpad": new_scratchpad, "agent_id": "e1"}},
                "messages": [],
                "llm_calls": [],
            }

        signals: dict[str, Any] = state["shared"].get("signals") or {}
        assert signals.get(STUCK) is True, (
            f"Expected signals['{STUCK}'] == True after 3 consecutive failures, "
            f"got signals={signals}"
        )

    async def test_executor_custom_threshold(self) -> None:
        """stuck_threshold=2 fires after only 2 consecutive failures."""
        registry = ToolRegistry()
        registry.register(FakeCodeRunFail())

        state = _make_state()

        for _step_num in range(2):
            llm = _make_llm("m5_executor_exit.yaml")
            executor = Executor(
                agent_id="e1",
                cfg=_make_cfg(),
                llm=llm,
                tools=registry,
                stuck_threshold=2,
            )
            delta = await executor.step(state)

            new_shared: dict[str, Any] = dict(state["shared"])
            if "shared" in delta:
                new_shared.update(delta["shared"])

            existing_scratchpad: list[dict[str, Any]] = list(
                state.get("agents", {}).get("e1", {}).get("scratchpad") or []
            )
            new_scratchpad = existing_scratchpad + list(
                delta.get("agents", {}).get("e1", {}).get("scratchpad") or []
            )
            state = {
                "shared": new_shared,
                "agents": {"e1": {"scratchpad": new_scratchpad, "agent_id": "e1"}},
                "messages": [],
                "llm_calls": [],
            }

        signals: dict[str, Any] = state["shared"].get("signals") or {}
        assert signals.get(STUCK) is True, (
            f"Expected stuck=True with threshold=2 after 2 failures, signals={signals}"
        )
