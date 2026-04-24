"""Tool Protocol, ToolSchema, and ToolRegistry for the ATM framework.

Public API
----------
ToolSchema    -- frozen Pydantic model describing a tool's interface
Tool          -- runtime-checkable Protocol that all tools must satisfy
ToolRegistry  -- registers, retrieves, and dispatches tool invocations
"""

from __future__ import annotations

import time
import traceback
from typing import Any, ClassVar, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from atm.core.errors import ToolError
from atm.core.types import ToolCall, ToolResult


class ToolSchema(BaseModel):
    """Frozen Pydantic model describing a tool's interface.

    Attributes:
        name:        Unique tool identifier (must match Tool.name ClassVar).
        description: Human-readable description for LLM tool selection.
        parameters:  JSON-Schema dict describing accepted input arguments.
        returns:     JSON-Schema dict describing the output shape.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


@runtime_checkable
class Tool(Protocol):
    """Protocol that all tool implementations must satisfy.

    Class variables
    ---------------
    name:   Unique identifier string (used as registry key).
    schema: ToolSchema instance describing the tool's interface.

    Methods
    -------
    ainvoke(args) -> ToolResult
        Asynchronously execute the tool with the given arguments.
        Must never raise — errors should be encoded in ToolResult.ok=False.
    """

    name: ClassVar[str]
    schema: ClassVar[ToolSchema]

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool asynchronously."""
        ...


class ToolRegistry:
    """Registry that maps tool names to Tool instances.

    Usage
    -----
    registry = ToolRegistry()
    registry.register(my_tool_instance)
    result = await registry.ainvoke_by_name("my_tool", tool_call)

    Thread safety: not guaranteed — construct once before any async use.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance.

        Raises
        ------
        ToolError
            If a tool with the same name is already registered.
        """
        name = tool.name
        if name in self._tools:
            raise ToolError(
                tool_name=name,
                message=f"tool '{name}' is already registered",
            )
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """Retrieve a registered tool by name.

        Raises
        ------
        ToolError
            If no tool with the given name is registered.
        """
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(
                tool_name=name,
                message=f"no tool named '{name}' is registered",
            ) from exc

    def names(self) -> list[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    async def ainvoke_by_name(self, name: str, call: ToolCall) -> ToolResult:
        """Invoke a registered tool by name, wrapping any exception in ToolResult.

        Measures wall-clock latency via ``time.monotonic``.

        If the tool raises, the exception is caught, and a ``ToolResult`` with
        ``ok=False`` and ``error`` containing the traceback is returned.

        Parameters
        ----------
        name:
            Name of the tool to invoke.
        call:
            The ``ToolCall`` whose ``args`` are forwarded to ``ainvoke``.

        Returns
        -------
        ToolResult
            Always returns a result; never raises.
        """
        tool = self.get(name)  # may raise ToolError — let it propagate (registry bug)
        t0 = time.monotonic()
        call_id: UUID = call.id

        try:
            result = await tool.ainvoke(call.args)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            # Return a new ToolResult with accurate latency from registry perspective
            return ToolResult(
                call_id=call_id,
                ok=result.ok,
                output=result.output,
                error=result.error,
                latency_ms=elapsed_ms,
            )
        except Exception:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            tb = traceback.format_exc()
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=tb,
                latency_ms=elapsed_ms,
            )
