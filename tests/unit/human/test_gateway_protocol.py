"""Unit tests for atm.human.gateway — HumanGateway Protocol and re-exports (M9 Step 1.1)."""

from __future__ import annotations

from atm.core.types import HumanContext, HumanResponse, HumanRole


def test_human_context_importable_from_gateway() -> None:
    from atm.human.gateway import HumanContext as ImportedHumanContext

    assert ImportedHumanContext is HumanContext


def test_human_response_importable_from_gateway() -> None:
    from atm.human.gateway import HumanResponse as ImportedHumanResponse

    assert ImportedHumanResponse is HumanResponse


def test_human_role_importable_from_gateway() -> None:
    from atm.human.gateway import HumanRole as ImportedHumanRole

    assert ImportedHumanRole is HumanRole


def test_human_gateway_protocol_is_runtime_checkable() -> None:
    """HumanGateway must be decorated with @runtime_checkable."""
    from atm.human.gateway import HumanGateway

    assert hasattr(HumanGateway, "__protocol_attrs__") or hasattr(HumanGateway, "_is_protocol"), (
        "HumanGateway must be a typing.Protocol"
    )


def test_concrete_implementation_satisfies_protocol() -> None:
    """A class with the correct async 'request' signature is accepted by the Protocol."""
    from atm.human.gateway import HumanContext as ImportedHumanContext
    from atm.human.gateway import HumanGateway
    from atm.human.gateway import HumanResponse as ImportedHumanResponse

    class _FakeGateway:
        async def request(
            self, ctx: ImportedHumanContext, *, request_id: str
        ) -> ImportedHumanResponse:
            return ImportedHumanResponse(action="approve")

    assert isinstance(_FakeGateway(), HumanGateway)


def test_class_missing_request_method_rejected() -> None:
    """A class without 'request' must NOT satisfy the Protocol."""
    from atm.human.gateway import HumanGateway

    class _BadGateway:
        pass

    assert not isinstance(_BadGateway(), HumanGateway)


def test_human_gateway_docstring_mentions_idempotency() -> None:
    """HumanGateway must document the idempotency contract in its docstring."""
    from atm.human.gateway import HumanGateway

    doc = HumanGateway.__doc__ or ""
    doc_lower = doc.lower()
    assert "idempoten" in doc_lower or "request_id" in doc_lower, (
        "HumanGateway docstring must mention idempotency or request_id contract"
    )


def test_human_gateway_request_method_exists() -> None:
    """HumanGateway Protocol must define a 'request' method."""
    import inspect

    from atm.human.gateway import HumanGateway

    assert hasattr(HumanGateway, "request"), "HumanGateway must have 'request'"
    sig = inspect.signature(HumanGateway.request)
    params = sig.parameters
    assert "ctx" in params, "request() must have 'ctx' parameter"
    assert "request_id" in params, "request() must have 'request_id' parameter"
    assert params["request_id"].kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
