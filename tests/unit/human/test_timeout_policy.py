"""Unit tests for atm.human._timeout — request_with_timeout with three policies.

Coverage:
- fast gateway under any policy → normal response returned, no timeout path
- fail policy + slow gateway → TimeoutError propagated
- llm_fallback + slow primary + fast fallback → response from fallback with source='fallback'
- skip + slow gateway → HumanResponse with timed_out=True, source='timeout', action='timeout'
- llm_fallback without fallback_gateway → ValueError raised
- timeout_s=None → direct await (no timeout logic applied)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from atm.core.types import HumanContext, HumanResponse, HumanRole, Message, MessageKind
from atm.human._timeout import request_with_timeout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(run_id: uuid.UUID | None = None) -> HumanContext:
    run_id = run_id or uuid.uuid4()
    msg = Message(
        id=uuid.uuid4(),
        run_id=run_id,
        kind=MessageKind.REQUEST,
        sender=HumanRole.REVIEWER,
        content="Do you approve?",
    )
    return HumanContext(
        run_id=run_id,
        role=HumanRole.REVIEWER,
        question="Do you approve?",
        recent_messages=(msg,),
        allowed_actions=("approve", "reject"),
    )


def _make_response(action: str = "approve", **kw: Any) -> HumanResponse:
    return HumanResponse(action=action, **kw)


class SlowGateway:
    """Gateway that sleeps `delay_s` seconds before responding."""

    def __init__(self, delay_s: float, response: HumanResponse) -> None:
        self._delay = delay_s
        self._response = response

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        await asyncio.sleep(self._delay)
        return self._response


class FastGateway:
    """Gateway that responds immediately."""

    def __init__(self, response: HumanResponse) -> None:
        self._response = response

    async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse:
        return self._response


# ---------------------------------------------------------------------------
# Tests: fast gateway — no timeout reached under any policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_gateway_fail_policy_returns_response() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="approve", source="human")
    gw = FastGateway(resp)

    result = await request_with_timeout(
        gw, ctx, request_id="req-001", timeout_s=5.0, policy="fail"
    )

    assert result.action == "approve"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_fast_gateway_skip_policy_returns_response() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="reject", source="human")
    gw = FastGateway(resp)

    result = await request_with_timeout(
        gw, ctx, request_id="req-002", timeout_s=5.0, policy="skip"
    )

    assert result.action == "reject"
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_fast_gateway_llm_fallback_policy_returns_response() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="approve", source="human")
    gw = FastGateway(resp)
    fallback = FastGateway(_make_response(action="reject", source="llm_sim"))

    result = await request_with_timeout(
        gw,
        ctx,
        request_id="req-003",
        timeout_s=5.0,
        policy="llm_fallback",
        llm_fallback_gateway=fallback,
    )

    # Primary responded fast → fallback not invoked
    assert result.action == "approve"
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Tests: slow gateway — timeout reached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_policy_slow_gateway_raises_timeout() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="approve")
    gw = SlowGateway(delay_s=10.0, response=resp)

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await request_with_timeout(
            gw, ctx, request_id="req-fail", timeout_s=0.05, policy="fail"
        )


@pytest.mark.asyncio
async def test_skip_policy_slow_gateway_returns_timeout_response() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="approve")
    gw = SlowGateway(delay_s=10.0, response=resp)

    result = await request_with_timeout(
        gw, ctx, request_id="req-skip", timeout_s=0.05, policy="skip"
    )

    assert result.timed_out is True
    assert result.source == "timeout"
    assert result.action == "timeout"


@pytest.mark.asyncio
async def test_llm_fallback_policy_slow_primary_uses_fallback() -> None:
    ctx = _make_ctx()
    primary_resp = _make_response(action="approve")
    fallback_resp = _make_response(action="reject", source="llm_sim")

    primary = SlowGateway(delay_s=10.0, response=primary_resp)
    fallback = FastGateway(fallback_resp)

    result = await request_with_timeout(
        primary,
        ctx,
        request_id="req-fallback",
        timeout_s=0.05,
        policy="llm_fallback",
        llm_fallback_gateway=fallback,
    )

    assert result.action == "reject"
    assert result.source == "fallback"
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Tests: configuration errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_fallback_without_fallback_gateway_raises_value_error() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="approve")
    gw = FastGateway(resp)

    with pytest.raises(ValueError, match="llm_fallback_gateway"):
        await request_with_timeout(
            gw,
            ctx,
            request_id="req-err",
            timeout_s=5.0,
            policy="llm_fallback",
            llm_fallback_gateway=None,
        )


# ---------------------------------------------------------------------------
# Tests: timeout_s=None → no timeout, direct await
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_none_timeout_passes_directly_to_gateway() -> None:
    ctx = _make_ctx()
    resp = _make_response(action="approve")
    gw = FastGateway(resp)

    result = await request_with_timeout(
        gw, ctx, request_id="req-none", timeout_s=None, policy="fail"
    )

    assert result.action == "approve"
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Tests: fallback response always has source='fallback' overridden
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fallback_source_overridden_to_fallback() -> None:
    """Even if fallback gateway returns source='llm_sim', the wrapper overrides to 'fallback'."""
    ctx = _make_ctx()
    primary = SlowGateway(delay_s=10.0, response=_make_response(action="approve"))
    # Fallback returns source='human' — should be overridden to 'fallback'
    fallback_resp = HumanResponse(action="revise", source="human")
    fallback = FastGateway(fallback_resp)

    result = await request_with_timeout(
        primary,
        ctx,
        request_id="req-src-override",
        timeout_s=0.05,
        policy="llm_fallback",
        llm_fallback_gateway=fallback,
    )

    assert result.source == "fallback"
    assert result.action == "revise"


# ---------------------------------------------------------------------------
# Tests: skip response has correct comment containing request_id context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_timeout_response_has_timeout_comment() -> None:
    ctx = _make_ctx()
    gw = SlowGateway(delay_s=10.0, response=_make_response("approve"))

    result = await request_with_timeout(
        gw,
        ctx,
        request_id="my-req-id",
        timeout_s=0.05,
        policy="skip",
    )

    # HumanResponse does not carry run_id/request_id; verify key timeout fields
    assert result.timed_out is True
    assert result.source == "timeout"
    assert result.action == "timeout"
    # comment should contain the request_id for traceability
    assert result.comment is not None
    assert "my-req-id" in result.comment
