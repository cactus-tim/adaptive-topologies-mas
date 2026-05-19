"""Unit tests for atm.human.cli_gateway.CLIGateway (M9 Step 2.2).

All tests patch ``asyncio.to_thread`` so no real stdin/input is used.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from atm.core.types import HumanContext, HumanResponse, HumanRole, Message, MessageKind
from atm.human.cli_gateway import CLIGateway
from atm.human.gateway import HumanGateway


def _make_ctx(
    *,
    run_id: uuid.UUID | None = None,
    question: str = "Approve the draft?",
    allowed_actions: tuple[str, ...] = ("approve", "reject", "revise"),
    recent_messages: tuple[Message, ...] = (),
) -> HumanContext:
    return HumanContext(
        run_id=run_id or uuid.uuid4(),
        role=HumanRole.REVIEWER,
        question=question,
        recent_messages=recent_messages,
        allowed_actions=allowed_actions,
    )


def _make_msg(content: str = "hello") -> Message:
    return Message(
        sender="executor",
        kind=MessageKind.DRAFT,
        content=content,
    )


async def _call_gateway(
    gw: CLIGateway,
    ctx: HumanContext,
    request_id: str,
    simulated_input: str,
) -> HumanResponse:
    """Helper: calls gw.request with asyncio.to_thread patched to return simulated_input."""
    with patch("atm.human.cli_gateway.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = simulated_input
        return await gw.request(ctx, request_id=request_id)


def test_cli_gateway_satisfies_human_gateway_protocol() -> None:
    gw = CLIGateway()
    assert isinstance(gw, HumanGateway)


@pytest.mark.asyncio
async def test_plain_action_no_comment() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-001", "approve")

    assert resp.action == "approve"
    assert resp.comment is None
    assert resp.source == "human"
    assert resp.timed_out is False


@pytest.mark.asyncio
async def test_action_with_comment() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-002", "reject looks wrong")

    assert resp.action == "reject"
    assert resp.comment == "looks wrong"
    assert resp.source == "human"


@pytest.mark.asyncio
async def test_json_input_full() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()
    raw = '{"action": "revise", "comment": "fix indent", "payload": {"line": 42}}'
    resp = await _call_gateway(gw, ctx, "req-003", raw)

    assert resp.action == "revise"
    assert resp.comment == "fix indent"
    assert resp.payload == {"line": 42}
    assert resp.source == "human"


@pytest.mark.asyncio
async def test_json_input_action_only() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()
    raw = '{"action": "approve"}'
    resp = await _call_gateway(gw, ctx, "req-004", raw)

    assert resp.action == "approve"
    assert resp.comment is None
    assert resp.payload == {}


@pytest.mark.asyncio
async def test_invalid_action_falls_back_to_first_allowed() -> None:
    ctx = _make_ctx(allowed_actions=("approve", "reject"))
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-005", "UNKNOWN_CMD")

    assert resp.action == "approve"
    assert resp.comment is not None
    assert "UNKNOWN_CMD" in resp.comment
    assert resp.source == "human"


@pytest.mark.asyncio
async def test_empty_input_falls_back() -> None:
    ctx = _make_ctx(allowed_actions=("approve", "reject"))
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-006", "   ")

    assert resp.action == "approve"
    assert resp.source == "human"


@pytest.mark.asyncio
async def test_invalid_json_falls_back() -> None:
    """Malformed JSON (starts with '{') should fallback gracefully."""
    ctx = _make_ctx(allowed_actions=("approve", "reject"))
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-007", '{"broken json')

    assert resp.action == "approve"
    assert resp.source == "human"


@pytest.mark.asyncio
async def test_idempotency_same_request_id_cached() -> None:
    run_id = uuid.uuid4()
    ctx = _make_ctx(run_id=run_id)
    gw = CLIGateway()

    resp1 = await _call_gateway(gw, ctx, "req-idem", "approve")
    assert resp1.action == "approve"

    resp2 = await _call_gateway(gw, ctx, "req-idem", "reject")
    assert resp2.action == "approve"
    assert resp1 is resp2


@pytest.mark.asyncio
async def test_different_request_ids_not_cached_together() -> None:
    run_id = uuid.uuid4()
    ctx = _make_ctx(run_id=run_id)
    gw = CLIGateway()

    resp1 = await _call_gateway(gw, ctx, "req-A", "approve")
    resp2 = await _call_gateway(gw, ctx, "req-B", "reject")

    assert resp1.action == "approve"
    assert resp2.action == "reject"
    assert resp1 is not resp2


@pytest.mark.asyncio
async def test_response_source_is_human() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-src", "approve")
    assert resp.source == "human"


@pytest.mark.asyncio
async def test_response_timed_out_is_false() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-to", "approve")
    assert resp.timed_out is False


@pytest.mark.asyncio
async def test_with_recent_messages() -> None:
    msgs = (_make_msg("first msg"), _make_msg("second msg"))
    ctx = _make_ctx(recent_messages=msgs)
    gw = CLIGateway()
    resp = await _call_gateway(gw, ctx, "req-msgs", "approve")
    assert resp.action == "approve"


@pytest.mark.asyncio
async def test_to_thread_receives_input_builtin() -> None:
    ctx = _make_ctx()
    gw = CLIGateway()

    with patch("atm.human.cli_gateway.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = "approve"
        await gw.request(ctx, request_id="req-builtin")

    assert mock_thread.called
    call_args = mock_thread.call_args
    assert call_args is not None
    positional = call_args.args
    assert len(positional) >= 1
    assert positional[0] is input, "asyncio.to_thread must be called with builtins.input"
