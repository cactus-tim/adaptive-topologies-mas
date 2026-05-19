"""Tests for TodoWriteTool — writes to shared.todos (NOT signals)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_todo_write_output_shape() -> None:
    """Valid todos list produces state_update with shared.todos key."""
    from atm.tools.local_.todo_write import TodoWriteTool

    tool = TodoWriteTool()
    todos = [
        {"id": "1", "content": "Write tests", "status": "open"},
        {"id": "2", "content": "Implement feature", "status": "done"},
    ]
    result = await tool.ainvoke({"todos": todos})
    assert result.ok is True
    assert "state_update" in result.output
    assert "shared" in result.output["state_update"]
    assert "todos" in result.output["state_update"]["shared"]


@pytest.mark.asyncio
async def test_todo_write_todos_value_matches_input() -> None:
    """The todos in output match the input list."""
    from atm.tools.local_.todo_write import TodoWriteTool

    tool = TodoWriteTool()
    todos = [{"id": "t1", "content": "Task one", "status": "open"}]
    result = await tool.ainvoke({"todos": todos})
    assert result.output["state_update"]["shared"]["todos"] == todos


@pytest.mark.asyncio
async def test_todo_write_no_signals_in_shared() -> None:
    """Output must NOT write to signals — 'signals' must not appear in shared."""
    from atm.tools.local_.todo_write import TodoWriteTool

    tool = TodoWriteTool()
    result = await tool.ainvoke({"todos": [{"id": "1", "content": "x", "status": "open"}]})
    shared = result.output["state_update"]["shared"]
    assert "signals" not in shared


@pytest.mark.asyncio
async def test_todo_write_invalid_todo_missing_required_field() -> None:
    """Todo missing a required field (e.g. 'id') returns ok=False."""
    from atm.tools.local_.todo_write import TodoWriteTool

    tool = TodoWriteTool()
    result = await tool.ainvoke({"todos": [{"content": "no id here", "status": "open"}]})
    assert result.ok is False


@pytest.mark.asyncio
async def test_todo_write_invalid_status_returns_error() -> None:
    """Todo with invalid status value returns ok=False."""
    from atm.tools.local_.todo_write import TodoWriteTool

    tool = TodoWriteTool()
    result = await tool.ainvoke(
        {"todos": [{"id": "1", "content": "bad status", "status": "invalid_value"}]}
    )
    assert result.ok is False


def test_todo_write_tool_name() -> None:
    """TodoWriteTool.name must equal 'todo_write'."""
    from atm.tools.local_.todo_write import TodoWriteTool

    assert TodoWriteTool.name == "todo_write"


def test_todo_write_schema_returns() -> None:
    """TodoWriteTool.schema.returns must contain 'state_update' key."""
    from atm.tools.local_.todo_write import TodoWriteTool

    assert "state_update" in TodoWriteTool.schema.returns
