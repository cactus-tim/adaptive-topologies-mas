"""Tests for atm.tools.local_.test_run — TestRunTool.

Tests cover:
- Trivial unittest with one passing test → passed=True, summary starts with "OK"
- Unittest with failing assertion → passed=False, summary contains "FAILED"
- Multi-test mix (some failing) → summary parsed from last non-empty line of stdout
"""

from __future__ import annotations

import pytest

from atm.tools.local_.test_run import TestRunTool as _TestRunTool
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox

_SANDBOX = SubprocessSandbox()


@pytest.fixture
def tool() -> _TestRunTool:
    return _TestRunTool(sandbox=_SANDBOX)


@pytest.mark.asyncio
async def test_test_run_passing(tool: _TestRunTool) -> None:
    """Trivial passing test → passed=True, summary starts with 'OK'."""
    test_code = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_ok(self):\n"
        "        self.assertEqual(1 + 1, 2)\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    result = await tool.ainvoke({"test_code": test_code})
    assert result.ok is True
    assert isinstance(result.output, dict)
    assert result.output["passed"] is True
    assert result.output["summary"].startswith("OK")
    assert "stdout" in result.output
    assert "stderr" in result.output
    assert "exit_code" in result.output


@pytest.mark.asyncio
async def test_test_run_failing(tool: _TestRunTool) -> None:
    """Failing assertion → passed=False, summary contains 'FAILED'."""
    test_code = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_bad(self):\n"
        "        self.assertEqual(1 + 1, 3)\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    result = await tool.ainvoke({"test_code": test_code})
    assert result.ok is False
    assert result.output["passed"] is False
    assert "FAILED" in result.output["summary"]
    assert result.output["exit_code"] != 0


@pytest.mark.asyncio
async def test_test_run_summary_parse(tool: _TestRunTool) -> None:
    """Multi-test suite with some failing → summary is last non-empty stdout line."""
    test_code = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_ok(self):\n"
        "        self.assertEqual(1, 1)\n"
        "    def test_fail(self):\n"
        "        self.assertEqual(1, 2)\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    result = await tool.ainvoke({"test_code": test_code})
    assert result.output["passed"] is False
    summary = result.output["summary"]
    assert summary
    stdout: str = result.output["stdout"]
    non_empty_lines = [line for line in stdout.splitlines() if line.strip()]
    if non_empty_lines:
        assert summary == non_empty_lines[-1]
