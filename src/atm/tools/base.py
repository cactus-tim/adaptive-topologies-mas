"""Tool Protocol, ToolSchema, and ToolRegistry for the ATM tools layer.

This module defines the contracts that all tools must implement.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from atm.core.errors import ToolError
from atm.core.types import ToolCall, ToolResult


class ToolSchema(BaseModel):
    """Declarative schema for a tool — name, description, parameters, returns."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Protocol that every tool must satisfy."""

    name: ClassVar[str]
    schema: ClassVar[ToolSchema]

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the tool with the given arguments."""
        ...


class ToolRegistry:
    """Registry of tools, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises ToolError if name already registered."""
        if tool.name in self._tools:
            raise ToolError(
                tool_name=tool.name,
                message=f"tool {tool.name!r} is already registered",
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name. Raises ToolError if not found."""
        if name not in self._tools:
            raise ToolError(
                tool_name=name,
                message=f"tool {name!r} is not registered",
            )
        return self._tools[name]

    def names(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    async def ainvoke(self, name: str, args: dict[str, Any], call_id: Any = None) -> ToolResult:
        """Invoke a tool by name, wrapping exceptions into ToolResult(ok=False).

        Measures latency via time.monotonic.
        """
        import uuid

        tool = self.get(name)
        call_id_val = call_id if call_id is not None else uuid.uuid4()

        start = time.monotonic()
        try:
            result = await tool.ainvoke(args)
            return result
        except ToolError:
            raise
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                call_id=call_id_val,
                ok=False,
                output=None,
                error=str(exc),
                latency_ms=elapsed_ms,
            )
