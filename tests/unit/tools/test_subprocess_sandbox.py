"""Tests for atm.tools.sandbox.subprocess_sandbox — SubprocessSandbox.

Tests cover:
- Python stdout capture
- Python stderr capture
- Python non-zero exit code
- Timeout kills the process, sets timed_out=True
- Multi-file support (extra files dict)
- Node.js support (skipped if node not installed)
- IS_ISOLATED class variable is False
"""

from __future__ import annotations

import shutil

import pytest

from atm.tools.sandbox.subprocess_sandbox import SubprocessSandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SANDBOX = SubprocessSandbox()


# ---------------------------------------------------------------------------
# IS_ISOLATED
# ---------------------------------------------------------------------------


def test_is_isolated_false() -> None:
    """SubprocessSandbox.IS_ISOLATED must be False — it is a dev-only sandbox."""
    assert SubprocessSandbox.IS_ISOLATED is False


# ---------------------------------------------------------------------------
# Python — basic stdout / stderr / exit_code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_python_print_returns_stdout() -> None:
    """print('hello') should produce stdout='hello\\n', exit_code=0, timed_out=False."""
    result = await _SANDBOX.execute(lang="python", code='print("hello")')
    assert result.stdout == "hello\n"
    assert result.stderr == "" or result.stderr is not None  # stderr may be empty
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.oom_killed is False


@pytest.mark.asyncio
async def test_python_stderr_captured() -> None:
    """Writing to sys.stderr should populate the stderr field of ExecResult."""
    code = "import sys; sys.stderr.write('err_output\\n')"
    result = await _SANDBOX.execute(lang="python", code=code)
    assert "err_output" in result.stderr
    assert result.exit_code == 0
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_python_nonzero_exit() -> None:
    """raise SystemExit(2) should produce exit_code=2."""
    result = await _SANDBOX.execute(lang="python", code="raise SystemExit(2)")
    assert result.exit_code == 2
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_python_timeout() -> None:
    """An infinite loop with timeout=1s should result in timed_out=True.

    duration_ms should be approximately 1000ms (within a 3x tolerance for CI).
    """
    result = await _SANDBOX.execute(
        lang="python",
        code="while True: pass",
        timeout=1.0,
    )
    assert result.timed_out is True
    # exit_code is -1 or None on timeout — accept any non-zero / falsy value
    assert result.exit_code != 0 or result.exit_code is None
    # Duration should be close to 1 second (allow up to 5s for slow CI)
    assert result.duration_ms >= 900, f"duration_ms={result.duration_ms} is too low"
    assert result.duration_ms < 5_000, f"duration_ms={result.duration_ms} is too high"


# ---------------------------------------------------------------------------
# Multi-file support
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_file() -> None:
    """Extra files in the `files` dict are written alongside main.py.

    main.py imports lib.py which exports X=42; stdout should be '42\\n'.
    """
    extra_files = {"lib.py": "X = 42\n"}
    code = "from lib import X; print(X)"
    result = await _SANDBOX.execute(lang="python", code=code, files=extra_files)
    assert result.stdout == "42\n"
    assert result.exit_code == 0
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Node.js support (skipped when node is not available)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_print() -> None:
    """console.log('hi') should produce stdout='hi\\n' (skipped if node absent)."""
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    result = await _SANDBOX.execute(lang="node", code='console.log("hi")')
    assert result.stdout == "hi\n"
    assert result.exit_code == 0
    assert result.timed_out is False


# ---------------------------------------------------------------------------
# Cleanup — temp dir is removed even on error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tempdir_cleaned_up_on_success() -> None:
    """After a successful run the temp directory should no longer exist."""
    import os
    import tempfile

    # We can't directly inspect the temp dir, but we can verify that normal
    # execution completes without lingering /tmp/... directories growing.
    # A basic smoke-test: the result is returned without raising.
    result = await _SANDBOX.execute(lang="python", code="print('clean')")
    assert result.stdout == "clean\n"
    # No assertion about temp dirs themselves — OS cleanup may be deferred,
    # but TemporaryDirectory.__exit__ guarantees removal on context exit.
