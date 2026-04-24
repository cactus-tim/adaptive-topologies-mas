"""Exit-criterion test for Executor agent: proves end-to-end correctness of the
tool-loop on a realistic code_run scenario.

Verifies:
- Executor.step() completes without error.
- scratchpad accumulates >= 3 events (reasoning + tool_call + observation
  from the first iteration, plus reasoning from the final stop response).
- tool_calls has exactly 1 entry (C1 accumulation — name == "code_run").
- tool_results has exactly 1 entry; ok is True; output["stdout"] == "2\\n".
- delta["messages"] includes a DRAFT message from "e1" whose content
  mentions the result ("output was 2" or contains "2").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from atm.agents.config import AgentConfig
from atm.agents.executor import Executor
from atm.core.types import MessageKind, ToolResult
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper
from atm.tools.base import ToolRegistry, ToolSchema

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


# ---------------------------------------------------------------------------
# FakeCodeRunTool — inline deterministic stub (no Docker dependency)
# ---------------------------------------------------------------------------


class FakeCodeRunTool:
    """Minimal stub that satisfies the Tool Protocol for code_run.

    Always returns stdout="2\\n" regardless of the code argument, simulating
    successful execution of print(1+1).

    Note: ToolRegistry.ainvoke_by_name stamps call_id from the ToolCall, so
    the call_id returned by ainvoke() itself is overridden — we return a
    fresh UUID4 here merely to satisfy the ToolResult contract; the registry
    replaces it.
    """

    name: ClassVar[str] = "code_run"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="code_run",
        description="Execute code in a sandboxed environment and return stdout/stderr.",
        parameters={
            "type": "object",
            "properties": {
                "lang": {"type": "string", "description": "Programming language (e.g. 'python')."},
                "code": {"type": "string", "description": "Source code to execute."},
            },
            "required": ["lang", "code"],
        },
        returns={
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "exit_code": {"type": "integer"},
                "timed_out": {"type": "boolean"},
                "duration_ms": {"type": "integer"},
            },
        },
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        from uuid import uuid4

        return ToolResult(
            call_id=uuid4(),  # registry will override this with the ToolCall's id
            ok=True,
            output={
                "stdout": "2\n",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "duration_ms": 1,
            },
            error=None,
            latency_ms=1,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=1.0, per_run_usd=10.0, per_experiment_usd=100.0)


def _make_pricing() -> Pricing:
    return Pricing.from_yaml(PRICING_PATH)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:deterministic",
        pricing=_make_pricing(),
        budget=_make_budget(),
        llm=fake,
    )


def _make_initial_state() -> dict[str, Any]:
    return {
        "shared": {"task_input": "Compute 1+1 and report the result"},
        "agents": {},
        "messages": [],
        "llm_calls": [],
        "budget_events": [],
        "topology_transitions": [],
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestExecutorExitCriterion:
    """End-to-end exit-criterion test for the Executor agent."""

    async def test_executor_runs_code_via_tool_loop(self) -> None:
        """Executor calls code_run through the tool loop; scratchpad grows; result captured."""

        # Build LLMWrapper with the scripted fixture
        llm = _make_llm_wrapper("m5_executor_exit.yaml")

        # Build ToolRegistry and register the fake code_run tool
        registry = ToolRegistry()
        registry.register(FakeCodeRunTool())

        # Build AgentConfig for executor role
        cfg = AgentConfig(
            role="executor",
            system_prompt="You are an executor.",
            window_size=12,
            max_tool_iters=6,
            context_token_budget=12000,
            tools=["code_run"],
        )

        # Build the Executor agent
        executor = Executor(agent_id="e1", cfg=cfg, llm=llm, tools=registry)

        # Initial state
        state = _make_initial_state()

        # Execute one step
        delta = await executor.step(state)

        # ------------------------------------------------------------------
        # 1. Delta structure
        # ------------------------------------------------------------------
        assert isinstance(delta, dict), "step() must return a dict"
        assert "agents" in delta, "delta must contain 'agents' key"
        assert "messages" in delta, "delta must contain 'messages' key"
        assert "e1" in delta["agents"], "delta['agents'] must contain entry for 'e1'"

        agent_delta = delta["agents"]["e1"]

        # ------------------------------------------------------------------
        # 2. Scratchpad has >= 3 events
        #    Minimum expected layout:
        #      step 0: reasoning ("I will run print(1+1)…")
        #      step 0: tool_call (code_run)
        #      step 0: observation (result of code_run)
        #      step 1: reasoning ("Successfully executed…")
        #    = 4 events; relaxed to >= 3 to avoid brittleness
        # ------------------------------------------------------------------
        scratchpad = agent_delta["scratchpad"]
        assert len(scratchpad) >= 3, (
            f"scratchpad must have >= 3 events (reasoning + tool_call + observation + …), "
            f"got {len(scratchpad)}: {scratchpad}"
        )

        # Verify the scratchpad contains the expected kinds in order
        kinds = [ev["kind"] for ev in scratchpad]
        assert "reasoning" in kinds, "scratchpad must contain a reasoning event"
        assert "tool_call" in kinds, "scratchpad must contain a tool_call event"
        assert "observation" in kinds, "scratchpad must contain an observation event"

        # ------------------------------------------------------------------
        # 3. tool_calls has exactly 1 entry (C1 accumulation check)
        # ------------------------------------------------------------------
        tool_calls = agent_delta["tool_calls"]
        assert len(tool_calls) == 1, (
            f"C1: exactly 1 tool_call expected (code_run), got {len(tool_calls)}: {tool_calls}"
        )
        assert tool_calls[0].tool_name == "code_run", (
            f"Expected tool_name='code_run', got {tool_calls[0].tool_name!r}"
        )

        # ------------------------------------------------------------------
        # 4. tool_results has exactly 1 entry; ok is True; stdout == "2\n"
        # ------------------------------------------------------------------
        tool_results = agent_delta["tool_results"]
        assert len(tool_results) == 1, (
            f"Exactly 1 tool_result expected, got {len(tool_results)}: {tool_results}"
        )
        tr = tool_results[0]
        assert tr.ok is True, f"tool_results[0].ok must be True, got {tr.ok}"
        assert isinstance(tr.output, dict), f"output must be a dict, got {type(tr.output)}"
        assert tr.output.get("stdout") == "2\n", (
            f"output['stdout'] must be '2\\n', got {tr.output.get('stdout')!r}"
        )

        # Verify the registry stamped the correct call_id from the ToolCall
        assert tr.call_id == tool_calls[0].id, (
            "tool_results[0].call_id must match the issued ToolCall's id"
        )

        # ------------------------------------------------------------------
        # 5. delta["messages"] includes a DRAFT from "e1" mentioning "2"
        # ------------------------------------------------------------------
        messages = delta["messages"]
        assert len(messages) >= 1, "delta['messages'] must contain at least one message"

        draft_messages = [m for m in messages if m.kind == MessageKind.DRAFT and m.sender == "e1"]
        assert len(draft_messages) >= 1, (
            f"Must have at least one DRAFT message from 'e1', got: {messages}"
        )

        # The final response content is "Successfully executed; the output was 2."
        draft_content = draft_messages[0].content
        assert "2" in draft_content, (
            f"DRAFT message content must mention '2' (the result), got: {draft_content!r}"
        )
