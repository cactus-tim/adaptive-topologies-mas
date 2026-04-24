"""Tests for atm.tools.local_.code_run — CodeRunTool.

Tests cover:
- Python print("hi") → ok=True, stdout contains "hi"
- Non-zero exit code → ok=False, exit_code matches
- Timeout (infinite loop, timeout=1s) → ok=False, timed_out=True
- Multi-file: main.py imports from lib.py
- Schema returns shape: all 6 keys present
"""

from __future__ import annotations

import pytest

from atm.tools.local_.code_run import CodeRunTool
from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SANDBOX = SubprocessSandbox()


@pytest.fixture
def tool() -> CodeRunTool:
    return CodeRunTool(sandbox=_SANDBOX)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_run_python_hello(tool: CodeRunTool) -> None:
    """print('hi') should produce ok=True and stdout='hi\\n'."""
    result = await tool.ainvoke({"lang": "python", "code": 'print("hi")'})
    assert result.ok is True
    assert isinstance(result.output, dict)
    assert result.output["stdout"] == "hi\n"
    assert result.output["exit_code"] == 0
    assert result.output["timed_out"] is False


@pytest.mark.asyncio
async def test_code_run_nonzero_exit(tool: CodeRunTool) -> None:
    """raise SystemExit(3) should produce ok=False, exit_code=3."""
    result = await tool.ainvoke({"lang": "python", "code": "raise SystemExit(3)"})
    assert result.ok is False
    assert result.output["exit_code"] == 3
    assert result.output["timed_out"] is False


@pytest.mark.asyncio
async def test_code_run_timeout(tool: CodeRunTool) -> None:
    """Infinite loop with timeout=1 should produce ok=False and timed_out=True."""
    result = await tool.ainvoke(
        {"lang": "python", "code": "while True: pass", "timeout_s": 1.0}
    )
    assert result.ok is False
    assert result.output["timed_out"] is True


@pytest.mark.asyncio
async def test_code_run_multi_file(tool: CodeRunTool) -> None:
    """Main code can import from an extra file passed via files dict."""
    main_code = "from lib import greet\nprint(greet())"
    lib_code = "def greet():\n    return 'hello from lib'"
    result = await tool.ainvoke(
        {
            "lang": "python",
            "code": main_code,
            "files": {"lib.py": lib_code},
        }
    )
    assert result.ok is True
    assert "hello from lib" in result.output["stdout"]


def test_code_run_schema_returns_shape(tool: CodeRunTool) -> None:
    """schema.returns must include all 6 required keys."""
    required_keys = {"stdout", "stderr", "exit_code", "timed_out", "duration_ms", "oom_killed"}
    assert required_keys == set(tool.schema.returns.keys())
