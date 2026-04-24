"""Tool Protocol, ToolSchema, and ToolRegistry for the ATM framework.

Public API:
    ToolSchema   -- frozen Pydantic model: name, description, parameters, returns
    Tool         -- runtime-checkable Protocol with ClassVar name/schema and ainvoke
    ToolRegistry -- register/get/names/ainvoke_by_name with latency tracking
"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from atm.core.errors import ToolError
from atm.core.types import ToolResult


class ToolSchema(BaseModel):
    """Schema description for a single tool.

    Fields:
        name:        Machine-readable tool identifier.
        description: Human-readable explanation of what the tool does.
        parameters:  JSON-schema-style dict of accepted input parameters.
        returns:     JSON-schema-style dict describing the output shape.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = {}
    returns: dict[str, Any] = {}


@runtime_checkable
class Tool(Protocol):
    """Protocol for all ATM tools.

    Every tool must expose:
        name   -- ClassVar[str] tool identifier (matches ToolSchema.name)
        schema -- ClassVar[ToolSchema] describing the tool interface
        ainvoke(args) -- async invocation returning ToolResult
    """

    name: ClassVar[str]
    schema: ClassVar[ToolSchema]

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the tool with the given arguments.

        Args:
            args: Tool-specific input dict (as described in schema.parameters).

        Returns:
            ToolResult with ok=True and populated output on success,
            or ok=False with error message on failure.
        """
        ...


class ToolRegistry:
    """Registry for Tool instances. Supports register/get/names/ainvoke_by_name.

    Wraps all ainvoke exceptions into ToolResult(ok=False, ...) so callers
    never receive raw exceptions from tool dispatch.

    Latency is measured via time.monotonic() and recorded in ToolResult.latency_ms.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance.

        Raises:
            ToolError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ToolError(
                tool_name=tool.name,
                message=f"tool {tool.name!r} is already registered",
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the registered tool by name.

        Raises:
            ToolError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise ToolError(
                tool_name=name,
                message=f"tool {name!r} is not registered",
            )
        return self._tools[name]

    def names(self) -> list[str]:
        """Return a sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    async def ainvoke_by_name(
        self,
        name: str,
        args: dict[str, Any],
        *,
        call_id: uuid.UUID | None = None,
    ) -> ToolResult:
        """Invoke a registered tool by name, returning ToolResult.

        Any exception raised by the tool (including ToolError) is caught and
        converted to ToolResult(ok=False, error=...). Latency is always recorded.

        Args:
            name:    Tool name to invoke.
            args:    Arguments to pass to the tool.
            call_id: Optional UUID for the ToolResult call_id (auto-generated if None).

        Returns:
            ToolResult with ok=True on success or ok=False on any failure.
        """
        resolved_call_id = call_id or uuid.uuid4()
        start = time.monotonic()

        try:
            tool = self.get(name)
        except ToolError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                call_id=resolved_call_id,
                ok=False,
                output=None,
                error=str(exc),
                latency_ms=latency_ms,
            )

        try:
            result = await tool.ainvoke(args)
            latency_ms = int((time.monotonic() - start) * 1000)
            # Re-wrap with measured latency (ToolResult is frozen, so we rebuild)
            return ToolResult(
                call_id=result.call_id,
                ok=result.ok,
                output=result.output,
                error=result.error,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                call_id=resolved_call_id,
                ok=False,
                output=None,
                error=str(exc),
                latency_ms=latency_ms,
            )
