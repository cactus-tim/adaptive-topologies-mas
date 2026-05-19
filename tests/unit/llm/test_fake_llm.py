"""Unit tests for FakeLLM — scripted / echo / replay modes.

Test cases:
1. Scripted basic: returns LLMResponse with correct text, latency_ms=0.
2. Scripted step advancement: step_idx increments per ainvoke call.
3. Scripted tool_call: entry with tool_calls returns ToolCall objects in LLMResponse.
4. Echo mode: mirrors last user-kind message content.
5. Replay mode: returns LLMResponse with id matching fixture call_id.
6. Determinism: two FakeLLM instances produce identical LLMResponse (Pydantic ==).
7. astream raises NotImplementedError.
8. Concurrent ainvoke: step counter increments correctly under asyncio.gather.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pyarrow as pa
import pytest

from atm.core.types import LLMResponse, Message, MessageKind
from atm.llm.fake import REPLAY_SCHEMA, FakeLLM

FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "llm"
PLANNER_SIMPLE = FIXTURE_DIR / "planner_simple.yaml"
EXECUTOR_CODE_RUN = FIXTURE_DIR / "executor_code_run.yaml"
DETERMINISM_SEED = FIXTURE_DIR / "determinism_seed.yaml"


def _make_replay_table(call_id: str, content: str, model: str = "fake:deterministic") -> pa.Table:
    """Build a single-row pyarrow Table matching REPLAY_SCHEMA for replay tests."""
    import datetime

    now_str = datetime.datetime.now(datetime.UTC).isoformat()
    data = {
        "call_id": pa.array([call_id], type=pa.string()),
        "model": pa.array([model], type=pa.string()),
        "content": pa.array([content], type=pa.string()),
        "usage_input": pa.array([10], type=pa.int64()),
        "usage_output": pa.array([5], type=pa.int64()),
        "usage_total": pa.array([15], type=pa.int64()),
        "usage_cached": pa.array([0], type=pa.int64()),
        "cost_usd": pa.array([0.0], type=pa.float64()),
        "latency_ms": pa.array([0], type=pa.int64()),
        "finish_reason": pa.array(["stop"], type=pa.string()),
        "started_at": pa.array([now_str], type=pa.string()),
        "tool_calls_json": pa.array(["[]"], type=pa.string()),
    }
    return pa.table(data, schema=REPLAY_SCHEMA)


@pytest.mark.asyncio
async def test_scripted_basic_returns_llm_response() -> None:
    """FakeLLM(mode='scripted') returns LLMResponse with correct text and latency_ms=0."""
    llm = FakeLLM(mode="scripted", fixture=str(PLANNER_SIMPLE))
    response = await llm.ainvoke(messages=[], agent_id="planner_1")

    assert isinstance(response, LLMResponse)
    assert (
        response.text
        == "Plan: Step 1 — gather requirements. Step 2 — design solution. Step 3 — implement."
    )
    assert response.latency_ms == 0
    assert response.model == "fake:deterministic"
    assert response.finish_reason == "stop"
    assert response.usage.prompt_tokens == 50
    assert response.usage.completion_tokens == 20


@pytest.mark.asyncio
async def test_scripted_step_advancement() -> None:
    """Two ainvoke calls for same agent_id increment step_idx from 0 to 1."""
    llm = FakeLLM(mode="scripted", fixture=str(EXECUTOR_CODE_RUN))

    resp0 = await llm.ainvoke(messages=[], agent_id="executor_1")
    resp1 = await llm.ainvoke(messages=[], agent_id="executor_1")

    assert resp0.finish_reason == "tool_calls"
    assert len(resp0.tool_calls) == 1

    assert resp1.text == "Done. The result of print(1+1) is 2."
    assert resp1.finish_reason == "stop"


@pytest.mark.asyncio
async def test_scripted_tool_call_entry() -> None:
    """entry[0] of executor_code_run returns LLMResponse with ToolCall(tool_name='code_run')."""
    llm = FakeLLM(mode="scripted", fixture=str(EXECUTOR_CODE_RUN))
    response = await llm.ainvoke(messages=[], agent_id="executor_1")

    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert tc.tool_name == "code_run"
    assert tc.args == {"code": "print(1+1)", "lang": "python"}


@pytest.mark.asyncio
async def test_echo_mode_returns_last_user_content() -> None:
    """FakeLLM(mode='echo') returns LLMResponse(text=last_user_message, latency_ms=0)."""
    llm = FakeLLM(mode="echo")
    msg = Message(sender="user", kind=MessageKind.REQUEST, content="hi")
    response = await llm.ainvoke(messages=[msg])

    assert isinstance(response, LLMResponse)
    assert response.text == "hi"
    assert response.latency_ms == 0


@pytest.mark.asyncio
async def test_replay_mode_returns_matching_id() -> None:
    """FakeLLM(mode='replay') returns LLMResponse whose id matches fixture call_id."""
    fixed_call_id = "550e8400-e29b-41d4-a716-446655440000"
    table = _make_replay_table(call_id=fixed_call_id, content="replayed response")

    llm = FakeLLM(mode="replay", replay_table=table)
    response = await llm.ainvoke(messages=[], agent_id="any_agent")

    assert str(response.id) == fixed_call_id
    assert response.text == "replayed response"
    assert response.latency_ms == 0


@pytest.mark.asyncio
async def test_determinism_two_instances_equal() -> None:
    """Two separate FakeLLM instances with same fixture produce identical LLMResponse."""
    llm1 = FakeLLM(mode="scripted", fixture=str(DETERMINISM_SEED))
    llm2 = FakeLLM(mode="scripted", fixture=str(DETERMINISM_SEED))

    r1 = await llm1.ainvoke(messages=[], agent_id="planner_1")
    r2 = await llm2.ainvoke(messages=[], agent_id="planner_1")

    assert r1.text == r2.text
    assert r1.model == r2.model
    assert r1.finish_reason == r2.finish_reason
    assert r1.usage == r2.usage
    assert r1.latency_ms == r2.latency_ms == 0


@pytest.mark.asyncio
async def test_astream_raises_not_implemented() -> None:
    """FakeLLM.astream raises NotImplementedError explicitly."""
    llm = FakeLLM(mode="echo")
    with pytest.raises(NotImplementedError, match="streaming"):
        async for _ in llm.astream():
            pass


@pytest.mark.asyncio
async def test_concurrent_step_counter_no_races() -> None:
    """Concurrent ainvoke calls increment step_idx correctly under asyncio.Lock."""
    llm = FakeLLM(mode="scripted", fixture=str(EXECUTOR_CODE_RUN))

    results = await asyncio.gather(
        llm.ainvoke(messages=[], agent_id="executor_1"),
        llm.ainvoke(messages=[], agent_id="executor_1"),
    )

    finish_reasons = {r.finish_reason for r in results}
    assert finish_reasons == {"tool_calls", "stop"}
