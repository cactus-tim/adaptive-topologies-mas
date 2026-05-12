"""Unit tests for build_human_node_factory (M9.1 Step 1.5).

Tests (6+):
  1.  build_human_node_factory returns a callable (async function)
  2.  Default apply_decision: approve action sets shared["human_approved"]=True
  3.  Default apply_decision: reject action sets shared["needs_rerun"]=True
  4.  Gateway is invoked with correct request_id from request_id_template
  5.  Idempotency-key passthrough via request_id_template with iter_total
  6.  Timed-out response (action="timeout") triggers needs_rerun via default handler
  7.  Custom apply_decision callback is called instead of default
  8.  Lazy imports: LLMSimulatedGateway/CLIGateway/request_with_timeout patchable
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atm.core.types import HumanResponse, HumanRole
from atm.human._node_factory import build_human_node_factory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_human_cfg(
    role: HumanRole = HumanRole.REVIEWER,
    timeout_s: float | None = None,
    timeout_policy: str = "skip",
) -> Any:
    """Build a minimal duck-typed HumanCfg for testing."""
    cfg = MagicMock()
    cfg.role = role
    cfg.timeout_s = timeout_s
    cfg.timeout_policy = timeout_policy
    return cfg


def _make_approve_response() -> HumanResponse:
    return HumanResponse(
        action="approve",
        comment="LGTM",
        source="llm_sim",
        timed_out=False,
    )


def _make_reject_response() -> HumanResponse:
    return HumanResponse(
        action="reject",
        comment="Needs revision",
        source="llm_sim",
        timed_out=False,
    )


def _make_timeout_response() -> HumanResponse:
    return HumanResponse(
        action="timeout",
        comment="Timed out",
        source="timeout",
        timed_out=True,
    )


def _make_state(run_id: Any = None, iter_total: int = 3) -> dict[str, Any]:
    """Build a minimal LangGraph state dict for testing."""
    if run_id is None:
        run_id = uuid.uuid4()
    return {
        "shared": {
            "run_id": run_id,
            "iter_total": iter_total,
            "signals": {},
        },
        "agents": {},
        "messages": [],
    }


def _make_gateway(response: HumanResponse) -> Any:
    """Build a mock gateway that returns the given response."""
    gw = AsyncMock()
    gw.request = AsyncMock(return_value=response)
    return gw


# ---------------------------------------------------------------------------
# 1. Factory returns an async callable
# ---------------------------------------------------------------------------


def test_build_human_node_factory_returns_callable() -> None:
    """build_human_node_factory must return an async callable."""
    cfg = _make_human_cfg()
    gw = _make_gateway(_make_approve_response())

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda s: "Should we proceed?",
    )

    assert callable(node)
    assert asyncio.iscoroutinefunction(node)


# ---------------------------------------------------------------------------
# 2. Default apply_decision: approve → human_approved=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_apply_decision_approve() -> None:
    """Approve action sets shared['human_approved']=True."""
    cfg = _make_human_cfg()
    gw = _make_gateway(_make_approve_response())
    state = _make_state()

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda s: "Approve?",
    )

    with patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock):
        result = await node(state)

    assert result["shared"]["human_approved"] is True
    assert result["shared"].get("needs_rerun") is not True


# ---------------------------------------------------------------------------
# 3. Default apply_decision: reject → needs_rerun=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_apply_decision_reject() -> None:
    """Reject action sets shared['needs_rerun']=True."""
    cfg = _make_human_cfg()
    gw = _make_gateway(_make_reject_response())
    state = _make_state()

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda s: "Approve?",
    )

    with patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock):
        result = await node(state)

    assert result["shared"].get("needs_rerun") is True
    assert result["shared"].get("human_approved") is not True


# ---------------------------------------------------------------------------
# 4. Gateway is invoked with correct request_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_invoked_with_correct_request_id() -> None:
    """request_id is built from the template with correct substitutions."""
    run_id = uuid.uuid4()
    cfg = _make_human_cfg()
    gw = _make_gateway(_make_approve_response())
    state = _make_state(run_id=run_id, iter_total=7)

    template = "star:{run_id}:{iter_total}:reviewer"
    expected_request_id = f"star:{run_id}:7:reviewer"

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template=template,
        question_extractor=lambda s: "Q?",
    )

    with patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock):
        await node(state)

    gw.request.assert_called_once()
    call_kwargs = gw.request.call_args
    assert call_kwargs.kwargs.get("request_id") == expected_request_id


# ---------------------------------------------------------------------------
# 5. Idempotency-key passthrough via request_id_template with iter_total
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_id_template_iter_total_passthrough() -> None:
    """iter_total is correctly embedded in the request_id."""
    run_id = uuid.uuid4()
    cfg = _make_human_cfg()
    gw = _make_gateway(_make_approve_response())

    for iter_val in (0, 3, 10):
        gw.request = AsyncMock(return_value=_make_approve_response())
        state = _make_state(run_id=run_id, iter_total=iter_val)

        node = build_human_node_factory(
            topology_name="mesh",
            human_cfg=cfg,
            gateway=gw,
            request_id_template="mesh:{run_id}:{iter_total}:peer",
            question_extractor=lambda s: "Vote?",
        )

        with patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock):
            await node(state)

        call_kwargs = gw.request.call_args
        expected = f"mesh:{run_id}:{iter_val}:peer"
        assert call_kwargs.kwargs.get("request_id") == expected


# ---------------------------------------------------------------------------
# 6. Timed-out response triggers needs_rerun via default handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timed_out_response_triggers_needs_rerun() -> None:
    """Timeout action (timed_out=True) is treated as reject → needs_rerun=True."""
    cfg = _make_human_cfg()
    gw = _make_gateway(_make_timeout_response())
    state = _make_state()

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda s: "Approve?",
    )

    with patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock):
        result = await node(state)

    # Timeout action != "approve" → needs_rerun=True
    assert result["shared"].get("needs_rerun") is True


# ---------------------------------------------------------------------------
# 7. Custom apply_decision callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_apply_decision_callback() -> None:
    """Custom apply_decision overrides default behavior."""
    custom_called: list[Any] = []

    def _custom_apply(response: Any, state: dict[str, Any], shared: dict[str, Any]) -> None:
        custom_called.append(response.action)
        shared["custom_flag"] = True

    cfg = _make_human_cfg()
    gw = _make_gateway(_make_approve_response())
    state = _make_state()

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda s: "Q?",
        apply_decision=_custom_apply,
    )

    with patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock):
        result = await node(state)

    # Custom callback was invoked
    assert custom_called == ["approve"]
    # Custom side effect applied
    assert result["shared"].get("custom_flag") is True
    # Default behavior NOT applied (human_approved not set by default handler)
    assert "human_approved" not in result["shared"]


# ---------------------------------------------------------------------------
# 8. Lazy imports: module-level names are patchable
# ---------------------------------------------------------------------------


def test_lazy_imports_are_patchable() -> None:
    """LLMSimulatedGateway, CLIGateway, request_with_timeout are module-level names."""
    import atm.human._node_factory as _nf

    # The module exposes these at module level (may be None if import failed,
    # or the actual class if imports succeeded).
    assert hasattr(_nf, "LLMSimulatedGateway")
    assert hasattr(_nf, "CLIGateway")
    assert hasattr(_nf, "request_with_timeout")


@pytest.mark.asyncio
async def test_node_factory_works_when_request_with_timeout_is_none() -> None:
    """When request_with_timeout is None (import failed), direct gateway.request() is used."""
    cfg = _make_human_cfg(timeout_s=None)
    gw = _make_gateway(_make_approve_response())
    state = _make_state()

    node = build_human_node_factory(
        topology_name="star",
        human_cfg=cfg,
        gateway=gw,
        request_id_template="star:{run_id}:{iter_total}:reviewer",
        question_extractor=lambda s: "Q?",
    )

    # Patch request_with_timeout to None to simulate failed import
    with (
        patch("atm.human._node_factory.request_with_timeout", None),
        patch("atm.human._node_factory.adispatch_custom_event", new_callable=AsyncMock),
    ):
        result = await node(state)

    # Should fall back to direct gateway.request()
    gw.request.assert_called_once()
    assert result["shared"]["human_approved"] is True
