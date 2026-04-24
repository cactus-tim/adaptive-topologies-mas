"""TodoWriteTool — writes todos into shared.todos (NOT shared.signals).

Output contains 'state_update' dict; ToolNode in M5 merges it into GraphState.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class Todo(BaseModel):
    """A single todo item."""

    model_config = ConfigDict(frozen=True)

    id: str
    content: str
    status: Literal["open", "done", "in_progress"]


class TodoWriteTool:
    """Write a list of todos into shared state.

    Args (ainvoke):
        todos: list of todo dicts, each with ``id``, ``content``, ``status``.

    Output: ``{"state_update": {"shared": {"todos": <todos_list>}}}``

    IMPORTANT: emits to ``shared.todos``, NOT to ``shared.signals``.
    This is a plan-locked contract (M4 Step 9).
    """

    name: ClassVar[str] = "todo_write"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="todo_write",
        description=(
            "Write a list of todos into shared state. "
            "Emits to shared.todos, not shared.signals. "
            "M5 reducer is responsible for merging (append semantics)."
        ),
        parameters={
            "todos": (
                "list of {id: str, content: str, status: 'open'|'done'|'in_progress'}"
            ),
        },
        returns={"state_update": "object (reducer-consumed)"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Validate todos and produce the state_update payload.

        Returns ToolResult with ok=False if any todo fails validation.
        """
        raw_todos: list[Any] = args.get("todos", [])
        validated: list[dict[str, Any]] = []

        for item in raw_todos:
            try:
                todo = Todo.model_validate(item)
                validated.append(todo.model_dump())
            except ValidationError as exc:
                return ToolResult(
                    call_id=uuid.uuid4(),
                    ok=False,
                    output=None,
                    error=f"Invalid todo item: {exc}",
                    latency_ms=0,
                )

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=True,
            output={"state_update": {"shared": {"todos": validated}}},
            error=None,
            latency_ms=0,
        )
