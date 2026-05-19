"""Tests for LintTool — ruff + optional pylint linter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_lint_clean_code() -> None:
    """Clean Python code produces ruff=[], ok=True."""
    from atm.tools.local_.lint import LintTool

    tool = LintTool()
    result = await tool.ainvoke({"code": "x = 1\n", "filename": "snippet.py"})

    assert result.ok is True
    assert isinstance(result.output["ruff"], list)
    assert result.output["ruff"] == []
    assert result.output["pylint"] is None or result.output["pylint_skipped"] is not None


@pytest.mark.asyncio
async def test_lint_detects_f811() -> None:
    """Duplicate function definition produces a diagnostic with code F811."""
    from atm.tools.local_.lint import LintTool

    tool = LintTool()
    result = await tool.ainvoke(
        {"code": "def f(): pass\ndef f(): pass\n", "filename": "snippet.py"}
    )

    assert result.ok is True
    ruff_diags = result.output["ruff"]
    assert isinstance(ruff_diags, list)
    assert len(ruff_diags) > 0
    codes = [d.get("code") for d in ruff_diags]
    assert "F811" in codes


@pytest.mark.asyncio
async def test_lint_pylint_missing_graceful() -> None:
    """When pylint binary is missing, tool returns ok=True with pylint_skipped='not installed'."""
    from atm.tools.local_.lint import LintTool

    original_create = asyncio.create_subprocess_exec

    async def patched_create_subprocess(*args: Any, **kwargs: Any) -> Any:
        if args and "pylint" in str(args[0]):
            raise FileNotFoundError("pylint not found")
        return await original_create(*args, **kwargs)

    with patch("asyncio.create_subprocess_exec", side_effect=patched_create_subprocess):
        tool = LintTool(include_pylint=True)
        result = await tool.ainvoke({"code": "x = 1\n", "filename": "snippet.py"})

    assert result.ok is True
    assert result.output["pylint"] is None
    assert result.output["pylint_skipped"] == "not installed"


@pytest.mark.asyncio
async def test_lint_output_shape() -> None:
    """Output always contains all three keys: ruff, pylint, pylint_skipped."""
    from atm.tools.local_.lint import LintTool

    tool = LintTool()
    result = await tool.ainvoke({"code": "x = 1\n"})

    assert result.ok is True
    output = result.output
    assert "ruff" in output
    assert "pylint" in output
    assert "pylint_skipped" in output


@pytest.mark.asyncio
async def test_lint_include_pylint_false_default() -> None:
    """When include_pylint=False (default), pylint key is None and pylint_skipped is None."""
    from atm.tools.local_.lint import LintTool

    tool = LintTool()
    assert tool.include_pylint is False

    result = await tool.ainvoke({"code": "x = 1\n", "filename": "test.py"})

    assert result.ok is True
    assert result.output["pylint"] is None
    assert result.output["pylint_skipped"] is None
