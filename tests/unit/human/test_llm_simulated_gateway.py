"""Unit tests for LLMSimulatedGateway (M9 Step 2.1).

Each scenario is driven by a FakeLLM with a YAML fixture so that no real LLM
calls are made.  Tests cover: approve, reject, invalid-JSON retry, Protocol
check, source/timed_out fields, and in-process idempotency cache.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from atm.core.types import HumanContext, HumanResponse, HumanRole, Message, MessageKind
from atm.human.gateway import HumanGateway
from atm.llm.budget import BudgetTracker
from atm.llm.fake import FakeLLM
from atm.llm.pricing import Pricing
from atm.llm.wrapper import LLMWrapper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
PRICING_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "pricing.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pricing() -> Pricing:
    """Load the shared pricing table which includes fake:scripted / fake:echo."""
    return Pricing.from_yaml(PRICING_PATH)


def _make_loose_budget() -> BudgetTracker:
    return BudgetTracker(per_call_usd=10.0, per_run_usd=100.0, per_experiment_usd=1000.0)


def _make_llm_wrapper(fixture_name: str) -> LLMWrapper:
    """Build an LLMWrapper backed by a scripted FakeLLM fixture."""
    fake = FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)
    return LLMWrapper(
        model_id="fake:scripted",
        pricing=_make_pricing(),
        budget=_make_loose_budget(),
        llm=fake,
    )


def _make_context(
    role: HumanRole = HumanRole.REVIEWER,
    allowed_actions: tuple[str, ...] = ("approve", "reject"),
    run_id: uuid.UUID | None = None,
) -> HumanContext:
    """Build a minimal HumanContext for testing."""
    msg = Message(
        sender="executor",
        kind=MessageKind.DRAFT,
        content="Here is the proposed solution.",
    )
    return HumanContext(
        run_id=run_id or uuid.uuid4(),
        role=role,
        question="Does the proposed solution meet the requirements?",
        recent_messages=(msg,),
        allowed_actions=allowed_actions,
    )


# ---------------------------------------------------------------------------
# 1. Protocol check — LLMSimulatedGateway satisfies HumanGateway
# ---------------------------------------------------------------------------


def test_llm_simulated_gateway_satisfies_protocol() -> None:
    """LLMSimulatedGateway must be an instance of HumanGateway Protocol."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    gw = LLMSimulatedGateway(wrapper)
    assert isinstance(gw, HumanGateway), (
        "LLMSimulatedGateway must satisfy the HumanGateway Protocol"
    )


# ---------------------------------------------------------------------------
# 2. Approve scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approve_action_returned() -> None:
    """Reviewer LLM returns approve JSON → HumanResponse.action == 'approve'."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()
    response = await gw.request(ctx, request_id="req-001")

    assert response.action == "approve"
    assert response.comment == "LGTM"
    assert isinstance(response.payload, dict)
    assert response.source == "llm_sim"
    assert response.timed_out is False


# ---------------------------------------------------------------------------
# 3. Reject scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reject_action_returned() -> None:
    """Reviewer LLM returns reject JSON → HumanResponse.action == 'reject'."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_reviewer_reject.yaml")
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()
    response = await gw.request(ctx, request_id="req-002")

    assert response.action == "reject"
    assert response.comment == "Add tests"
    assert response.source == "llm_sim"
    assert response.timed_out is False


# ---------------------------------------------------------------------------
# 4. HumanResponse fields — source and timed_out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_source_is_llm_sim() -> None:
    """source field must be 'llm_sim' for all LLMSimulatedGateway responses."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()
    response = await gw.request(ctx, request_id="req-003")
    assert response.source == "llm_sim"


@pytest.mark.asyncio
async def test_response_timed_out_is_false() -> None:
    """timed_out must always be False for LLMSimulatedGateway."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()
    response = await gw.request(ctx, request_id="req-004")
    assert response.timed_out is False


# ---------------------------------------------------------------------------
# 5. Invalid JSON retry path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_retry_succeeds() -> None:
    """First LLM response is garbage; second (retry) is valid JSON → action returned."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_invalid_json.yaml")
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()
    response = await gw.request(ctx, request_id="req-005")

    # After retry, valid JSON is parsed
    assert response.action == "approve"
    assert response.comment == "Recovered after retry"
    assert response.source == "llm_sim"
    assert response.timed_out is False


@pytest.mark.asyncio
async def test_invalid_json_both_attempts_raises_value_error() -> None:
    """If both LLM attempts return invalid JSON, ValueError is raised."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    # Use echo FakeLLM — returns the last message content which is the prompt text,
    # not valid JSON.
    echo_llm = FakeLLM(mode="echo")
    wrapper = LLMWrapper(
        model_id="fake:echo",
        pricing=_make_pricing(),
        budget=_make_loose_budget(),
        llm=echo_llm,
    )
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()

    with pytest.raises(ValueError, match="invalid JSON after retry"):
        await gw.request(ctx, request_id="req-006")


# ---------------------------------------------------------------------------
# 6. In-process idempotency cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotency_same_request_id_returns_cached() -> None:
    """Second call with same (run_id, request_id) returns cached response; no second LLM call."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    # Fixture only has 1 entry — second LLM call would raise LookupError.
    wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    gw = LLMSimulatedGateway(wrapper)
    run_id = uuid.uuid4()
    ctx = _make_context(run_id=run_id)
    request_id = "req-idem-001"

    first = await gw.request(ctx, request_id=request_id)
    second = await gw.request(ctx, request_id=request_id)

    # Must return the exact same object from cache
    assert first is second


@pytest.mark.asyncio
async def test_idempotency_different_request_id_makes_new_call() -> None:
    """Different request_id on the same run → cache miss → new LLM call."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    # Fixture has only 1 entry — second call with different request_id hits step_idx=1,
    # which would raise LookupError. We use an echo LLM for this test instead.
    echo_llm = FakeLLM(mode="echo")
    wrapper = LLMWrapper(
        model_id="fake:echo",
        pricing=_make_pricing(),
        budget=_make_loose_budget(),
        llm=echo_llm,
    )
    gw = LLMSimulatedGateway(wrapper)
    run_id = uuid.uuid4()
    ctx = _make_context(run_id=run_id)

    # Echo mode returns the user prompt, not JSON — both calls will fail with ValueError.
    with pytest.raises(ValueError, match="invalid JSON after retry"):
        await gw.request(ctx, request_id="req-a")

    # With request_id="req-b" (different key) the cache should not be populated from req-a.
    # The gateway makes a new LLM call (no cache hit).
    with pytest.raises(ValueError, match="invalid JSON after retry"):
        await gw.request(ctx, request_id="req-b")


@pytest.mark.asyncio
async def test_idempotency_cache_key_is_run_id_plus_request_id() -> None:
    """Different run_ids with same request_id are distinct cache entries."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    # We need 2 entries in the fixture since each run_id-keyed ctx will hit a new step.
    # Use scripted fixture with 2 entries — one per call (different run_ids, same request_id).
    import io
    import yaml as _yaml

    fixture_data = {
        "version": 1,
        "mode": "scripted",
        "entries": [
            {
                "agent_id": "llm_simulator",
                "step_idx": 0,
                "content": '{"action": "approve", "comment": "run-1"}',
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
                "model": "fake:scripted",
            },
            {
                "agent_id": "llm_simulator",
                "step_idx": 1,
                "content": '{"action": "reject", "comment": "run-2"}',
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
                "model": "fake:scripted",
            },
        ],
    }

    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        _yaml.dump(fixture_data, f)
        tmp_path = f.name

    try:
        fake = FakeLLM(mode="scripted", fixture=tmp_path)
        wrapper = LLMWrapper(
            model_id="fake:scripted",
            pricing=_make_pricing(),
            budget=_make_loose_budget(),
            llm=fake,
        )
        gw = LLMSimulatedGateway(wrapper)

        run_id_1 = uuid.uuid4()
        run_id_2 = uuid.uuid4()
        ctx1 = _make_context(run_id=run_id_1)
        ctx2 = _make_context(run_id=run_id_2)

        resp1 = await gw.request(ctx1, request_id="same-req-id")
        resp2 = await gw.request(ctx2, request_id="same-req-id")

        # Different run_ids → different cache keys → independent responses
        assert resp1.comment == "run-1"
        assert resp2.comment == "run-2"
        assert resp1 is not resp2
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 7. HumanResponse is a proper Pydantic model instance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_response_is_human_response_instance() -> None:
    """Gateway must return a HumanResponse instance."""
    from atm.human.llm_simulated import LLMSimulatedGateway

    wrapper = _make_llm_wrapper("m9_human_reviewer_approve.yaml")
    gw = LLMSimulatedGateway(wrapper)
    ctx = _make_context()
    response = await gw.request(ctx, request_id="req-007")
    assert isinstance(response, HumanResponse)
