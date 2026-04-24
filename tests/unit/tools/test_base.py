"""Tests for atm.tools.base — Tool Protocol, ToolSchema, ToolRegistry."""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar
from uuid import uuid4

import pytest
from pydantic import ValidationError

from atm.core.errors import ToolError
from atm.core.types import ToolCall, ToolResult
from atm.tools.base import Tool, ToolRegistry, ToolSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(tool_name: str = "test_tool") -> ToolCall:
    return ToolCall(tool_name=tool_name, args={}, issued_by="test_agent")


class GoodTool:
    """A minimal valid Tool implementation."""

    name: ClassVar[str] = "good_tool"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="good_tool",
        description="A good test tool",
        parameters={"type": "object", "properties": {}},
        returns={"type": "object", "properties": {"value": {"type": "integer"}}},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(call_id=uuid4(), ok=True, output={"value": 42}, latency_ms=1)


class FailingTool:
    """A Tool that always raises."""

    name: ClassVar[str] = "failing_tool"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="failing_tool",
        description="A failing test tool",
        parameters={},
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("boom")


class SlowTool:
    """A Tool that sleeps briefly to test latency measurement."""

    name: ClassVar[str] = "slow_tool"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="slow_tool",
        description="A slow test tool",
        parameters={},
        returns={"type": "object"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(0.05)
        return ToolResult(call_id=uuid4(), ok=True, output={}, latency_ms=50)


# ---------------------------------------------------------------------------
# ToolSchema tests
# ---------------------------------------------------------------------------


def test_tool_schema_frozen_has_returns_field() -> None:
    schema = ToolSchema(
        name="my_tool",
        description="desc",
        parameters={"type": "object"},
        returns={"type": "string"},
    )
    assert schema.name == "my_tool"
    assert schema.returns == {"type": "string"}

    # Must be frozen — assignment should raise (Pydantic v2 raises ValidationError)
    with pytest.raises((TypeError, AttributeError, ValidationError)):
        schema.name = "other"  # type: ignore[misc]


def test_tool_schema_requires_returns_field() -> None:
    """returns field is required (no default)."""
    with pytest.raises(Exception):
        ToolSchema(name="x", description="d", parameters={})  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ToolRegistry — basic register/get/names
# ---------------------------------------------------------------------------


def test_registry_register_get_names() -> None:
    reg = ToolRegistry()
    tool = GoodTool()
    reg.register(tool)

    assert "good_tool" in reg.names()
    retrieved = reg.get("good_tool")
    assert retrieved is tool


def test_registry_duplicate_raises() -> None:
    reg = ToolRegistry()
    reg.register(GoodTool())
    with pytest.raises(ToolError):
        reg.register(GoodTool())


def test_registry_unknown_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.get("nonexistent")


# ---------------------------------------------------------------------------
# ToolRegistry — ainvoke_by_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_ainvoke_wraps_exception_in_toolresult() -> None:
    reg = ToolRegistry()
    reg.register(FailingTool())
    call = _make_call("failing_tool")
    result = await reg.ainvoke_by_name("failing_tool", call)

    assert result.ok is False
    assert result.error is not None
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_registry_ainvoke_measures_latency() -> None:
    reg = ToolRegistry()
    reg.register(SlowTool())
    call = _make_call("slow_tool")
    t0 = time.monotonic()
    result = await reg.ainvoke_by_name("slow_tool", call)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert result.ok is True
    # latency_ms must be positive and at least ~40ms (80% of the 50ms sleep)
    assert result.latency_ms >= 40
    assert result.latency_ms <= elapsed_ms + 50  # reasonable upper bound


@pytest.mark.asyncio
async def test_registry_ainvoke_returns_toolresult_on_success() -> None:
    reg = ToolRegistry()
    reg.register(GoodTool())
    call = _make_call("good_tool")
    result = await reg.ainvoke_by_name("good_tool", call)

    assert result.ok is True
    assert result.output == {"value": 42}
