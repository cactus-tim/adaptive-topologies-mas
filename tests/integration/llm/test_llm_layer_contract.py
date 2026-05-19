"""Integration acceptance tests for the LLM layer public contract (M2, Step 4.1).

Three scenarios:
1. Scripted FakeLLM + LLMWrapper end-to-end — exercises the public contract.
2. Budget-exceed path — pre-call estimate triggers BudgetExceededError without invoking LLM.
3. Replay round-trip — FakeLLM(mode='replay') reproduces a saved call from a pyarrow Table.
"""

from __future__ import annotations

import datetime
import uuid
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest

from atm.core.errors import BudgetExceededError
from atm.core.types import LLMResponse, Message, MessageKind
from atm.llm.budget import BudgetLevel, BudgetTracker
from atm.llm.fake import REPLAY_SCHEMA, FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper

_PRICING_YAML = "conf/pricing.yaml"
_PLANNER_FIXTURE = "tests/fixtures/llm/planner_simple.yaml"
_PLANNER_EXPECTED_TEXT = (
    "Plan: Step 1 — gather requirements. Step 2 — design solution. Step 3 — implement."
)


@pytest.mark.asyncio
async def test_scripted_fake_llm_wrapper_end_to_end() -> None:
    """LLMWrapper with scripted FakeLLM returns a well-formed LLMResponse.

    Verifies:
    - Return type is LLMResponse.
    - resp.text matches fixture content.
    - resp.cost_usd == 0.0 (fake:deterministic has zero rates).
    - resp.latency_ms >= 0.
    - resp.model == "fake:deterministic" (wrapper's model_id).
    - resp.started_at is a timezone-aware datetime.
    - budget.totals[BudgetLevel.RUN] reflects the recorded cost.
    """
    pricing = Pricing.from_yaml(_PRICING_YAML)
    budget = BudgetTracker(
        per_call_usd=1.0,
        per_run_usd=10.0,
        per_experiment_usd=100.0,
    )
    fake = FakeLLM(mode="scripted", fixture=_PLANNER_FIXTURE)
    wrapper = LLMWrapper("fake:deterministic", pricing=pricing, budget=budget, llm=fake)

    resp = await wrapper.ainvoke(
        messages=[Message(sender="user", kind=MessageKind.REQUEST, content="plan hello")],
        agent_id="planner_1",
    )

    assert isinstance(resp, LLMResponse)
    assert resp.text == _PLANNER_EXPECTED_TEXT
    assert resp.cost_usd == 0.0
    assert resp.latency_ms >= 0
    assert resp.model == "fake:deterministic"
    assert isinstance(resp.started_at, datetime.datetime)
    assert resp.started_at.tzinfo is not None

    assert budget.totals[BudgetLevel.RUN] == 0.0


@pytest.mark.asyncio
async def test_budget_exceed_raises_before_llm_call() -> None:
    """BudgetExceededError is raised on pre-call check; underlying LLM is NOT called.

    Uses a per_call_usd ceiling of 0.0000001 (effectively zero) so that any
    non-zero prompt estimate triggers BudgetExceededError.  The injected LLM is
    wrapped in a MagicMock with an AsyncMock .ainvoke to verify it is never called.
    """
    pricing = Pricing.from_yaml(_PRICING_YAML)
    budget = BudgetTracker(
        per_call_usd=0.0000001,
        per_run_usd=1.0,
        per_experiment_usd=10.0,
    )

    inner_fake = FakeLLM(mode="echo")
    llm_spy = MagicMock(wraps=inner_fake)
    llm_spy.ainvoke = AsyncMock(wraps=inner_fake.ainvoke)

    wrapper = LLMWrapper(
        "openai:gpt-4o",
        pricing=pricing,
        budget=budget,
        llm=llm_spy,
    )

    with pytest.raises(BudgetExceededError) as exc_info:
        await wrapper.ainvoke(
            messages=[Message(sender="a", kind=MessageKind.REQUEST, content="x")],
            agent_id="a",
        )

    assert isinstance(exc_info.value, BudgetExceededError)

    llm_spy.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_replay_round_trip() -> None:
    """FakeLLM(mode='replay') reconstructs LLMResponse from a pyarrow Table row.

    Verifies:
    - resp.id matches the call_id stored in the table.
    - resp.text matches the content column.
    """
    call_id = str(uuid.uuid4())
    started_at_iso = datetime.datetime.now(datetime.UTC).isoformat()

    table = pa.table(
        {
            "call_id": pa.array([call_id], type=pa.string()),
            "model": pa.array(["fake:deterministic"], type=pa.string()),
            "content": pa.array(["replayed text"], type=pa.string()),
            "usage_input": pa.array([10], type=pa.int64()),
            "usage_output": pa.array([5], type=pa.int64()),
            "usage_total": pa.array([15], type=pa.int64()),
            "usage_cached": pa.array([0], type=pa.int64()),
            "cost_usd": pa.array([0.0], type=pa.float64()),
            "latency_ms": pa.array([0], type=pa.int64()),
            "finish_reason": pa.array(["stop"], type=pa.string()),
            "started_at": pa.array([started_at_iso], type=pa.string()),
            "tool_calls_json": pa.array([None], type=pa.string()),
        },
        schema=REPLAY_SCHEMA,
    )

    fake = FakeLLM(mode="replay", replay_table=table)
    resp = await fake.ainvoke([], agent_id="a")

    assert str(resp.id) == table.column("call_id")[0].as_py()
    assert resp.text == "replayed text"
