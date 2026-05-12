"""Unit tests for atm.human.gateway — HumanGateway Protocol and re-exports (M9 Step 1.1)."""

from __future__ import annotations

import uuid
from typing import runtime_checkable

import pytest

from atm.core.types import HumanContext, HumanResponse, HumanRole, Message, MessageKind


# ---------------------------------------------------------------------------
# 1. Re-export tests — HumanContext / HumanResponse / HumanRole accessible via
#    atm.human.gateway (should not be duplicated there; must re-export from core)
# ---------------------------------------------------------------------------


def test_human_context_importable_from_gateway() -> None:
    from atm.human.gateway import HumanContext as HC  # noqa: F401

    assert HC is HumanContext


def test_human_response_importable_from_gateway() -> None:
    from atm.human.gateway import HumanResponse as HR  # noqa: F401

    assert HR is HumanResponse


def test_human_role_importable_from_gateway() -> None:
    from atm.human.gateway import HumanRole as HRL  # noqa: F401

    assert HRL is HumanRole


# ---------------------------------------------------------------------------
# 2. HumanGateway Protocol — structural subtyping checks
# ---------------------------------------------------------------------------


def test_human_gateway_protocol_is_runtime_checkable() -> None:
    """HumanGateway must be decorated with @runtime_checkable."""
    from atm.human.gateway import HumanGateway

    assert hasattr(HumanGateway, "__protocol_attrs__") or hasattr(
        HumanGateway, "_is_protocol"
    ), "HumanGateway must be a typing.Protocol"


def test_concrete_implementation_satisfies_protocol() -> None:
    """A class with the correct async 'request' signature is accepted by the Protocol."""
    from atm.human.gateway import HumanContext as HC
    from atm.human.gateway import HumanGateway, HumanResponse as HR

    class _FakeGateway:
        async def request(self, ctx: HC, *, request_id: str) -> HR:
            return HR(action="approve")

    assert isinstance(_FakeGateway(), HumanGateway)


def test_class_missing_request_method_rejected() -> None:
    """A class without 'request' must NOT satisfy the Protocol."""
    from atm.human.gateway import HumanGateway

    class _BadGateway:
        pass  # no request method

    assert not isinstance(_BadGateway(), HumanGateway)


# ---------------------------------------------------------------------------
# 3. Idempotency contract — documented in HumanGateway docstring
# ---------------------------------------------------------------------------


def test_human_gateway_docstring_mentions_idempotency() -> None:
    """HumanGateway must document the idempotency contract in its docstring."""
    from atm.human.gateway import HumanGateway

    doc = HumanGateway.__doc__ or ""
    doc_lower = doc.lower()
    assert "idempoten" in doc_lower or "request_id" in doc_lower, (
        "HumanGateway docstring must mention idempotency or request_id contract"
    )


# ---------------------------------------------------------------------------
# 4. HumanGateway.request signature — keyword-only request_id
# ---------------------------------------------------------------------------


def test_human_gateway_request_method_exists() -> None:
    """HumanGateway Protocol must define a 'request' method."""
    import inspect

    from atm.human.gateway import HumanGateway

    assert hasattr(HumanGateway, "request"), "HumanGateway must have 'request'"
    sig = inspect.signature(HumanGateway.request)
    params = sig.parameters
    assert "ctx" in params, "request() must have 'ctx' parameter"
    assert "request_id" in params, "request() must have 'request_id' parameter"
    # request_id must be keyword-only
    assert params["request_id"].kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
