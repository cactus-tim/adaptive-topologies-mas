"""Tests for PlanUpdateTool — writes to shared.plan (NOT signals)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_plan_update_output_shape() -> None:
    """Valid steps list produces state_update with shared.plan key."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    tool = PlanUpdateTool()
    steps = [
        {"id": "s1", "title": "Setup", "status": "done"},
        {"id": "s2", "title": "Implement", "status": "in_progress"},
    ]
    result = await tool.ainvoke({"steps": steps})
    assert result.ok is True
    assert "state_update" in result.output
    assert "shared" in result.output["state_update"]
    assert "plan" in result.output["state_update"]["shared"]


@pytest.mark.asyncio
async def test_plan_update_steps_value_matches_input() -> None:
    """The plan in output matches the input steps list."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    tool = PlanUpdateTool()
    steps = [{"id": "s1", "title": "Only step", "status": "open"}]
    result = await tool.ainvoke({"steps": steps})
    assert result.output["state_update"]["shared"]["plan"] == steps


@pytest.mark.asyncio
async def test_plan_update_no_signals_in_shared() -> None:
    """Output must NOT write to signals — 'signals' must not appear in shared."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    tool = PlanUpdateTool()
    result = await tool.ainvoke({"steps": [{"id": "1", "title": "x", "status": "open"}]})
    shared = result.output["state_update"]["shared"]
    assert "signals" not in shared


@pytest.mark.asyncio
async def test_plan_update_invalid_step_missing_required_field() -> None:
    """Step missing a required field (e.g. 'id') returns ok=False."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    tool = PlanUpdateTool()
    # Missing 'id' field
    result = await tool.ainvoke({"steps": [{"title": "no id", "status": "open"}]})
    assert result.ok is False


@pytest.mark.asyncio
async def test_plan_update_invalid_status_returns_error() -> None:
    """Step with invalid status returns ok=False."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    tool = PlanUpdateTool()
    result = await tool.ainvoke({"steps": [{"id": "1", "title": "bad", "status": "flying"}]})
    assert result.ok is False


def test_plan_update_tool_name() -> None:
    """PlanUpdateTool.name must equal 'plan_update'."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    assert PlanUpdateTool.name == "plan_update"


def test_plan_update_schema_returns() -> None:
    """PlanUpdateTool.schema.returns must contain 'state_update' key."""
    from atm.tools.local_.plan_update import PlanUpdateTool

    assert "state_update" in PlanUpdateTool.schema.returns
