"""Tests for DiffTool — unified diff via difflib."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_diff_basic_change() -> None:
    """Basic diff shows - and + lines for changed content."""
    from atm.tools.local_.diff import DiffTool

    tool = DiffTool()
    result = await tool.ainvoke({"before": "hello\nworld\n", "after": "hello\npython\n"})
    assert result.ok is True
    diff = result.output["diff"]
    assert "-world" in diff
    assert "+python" in diff


@pytest.mark.asyncio
async def test_diff_identical_strings_empty_diff() -> None:
    """Identical before/after produces an empty diff string."""
    from atm.tools.local_.diff import DiffTool

    tool = DiffTool()
    result = await tool.ainvoke({"before": "same content\n", "after": "same content\n"})
    assert result.ok is True
    assert result.output["diff"] == ""


@pytest.mark.asyncio
async def test_diff_hunk_header_present() -> None:
    """Unified diff contains @@ hunk markers."""
    from atm.tools.local_.diff import DiffTool

    tool = DiffTool()
    result = await tool.ainvoke({"before": "line1\nline2\n", "after": "line1\nline3\n"})
    assert result.ok is True
    assert "@@" in result.output["diff"]


@pytest.mark.asyncio
async def test_diff_default_filenames_in_header() -> None:
    """Default fromfile/tofile appear in the diff header as 'before' and 'after'."""
    from atm.tools.local_.diff import DiffTool

    tool = DiffTool()
    result = await tool.ainvoke({"before": "a\n", "after": "b\n"})
    assert result.ok is True
    diff = result.output["diff"]
    assert "before" in diff
    assert "after" in diff


@pytest.mark.asyncio
async def test_diff_custom_filenames() -> None:
    """Custom filename_before / filename_after appear in the diff header."""
    from atm.tools.local_.diff import DiffTool

    tool = DiffTool()
    result = await tool.ainvoke(
        {
            "before": "old content\n",
            "after": "new content\n",
            "filename_before": "old.py",
            "filename_after": "new.py",
        }
    )
    assert result.ok is True
    diff = result.output["diff"]
    assert "old.py" in diff
    assert "new.py" in diff


@pytest.mark.asyncio
async def test_diff_output_key_is_diff() -> None:
    """Output dict must have exactly the key 'diff'."""
    from atm.tools.local_.diff import DiffTool

    tool = DiffTool()
    result = await tool.ainvoke({"before": "x\n", "after": "y\n"})
    assert "diff" in result.output
    assert isinstance(result.output["diff"], str)


def test_diff_tool_name() -> None:
    """DiffTool.name must equal 'diff'."""
    from atm.tools.local_.diff import DiffTool

    assert DiffTool.name == "diff"


def test_diff_schema_has_returns() -> None:
    """DiffTool.schema.returns must contain 'diff' key."""
    from atm.tools.local_.diff import DiffTool

    assert "diff" in DiffTool.schema.returns
