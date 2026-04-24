"""TestRunTool — sandbox-backed unittest runner.

Writes test code into the sandbox and runs it via ``python -m unittest -v``,
merging stderr into stdout (``2>&1``) so that unittest's summary line is
captured in stdout and can be parsed.

Public API
----------
TestRunTool -- Tool implementation that runs unittest test suites in a sandbox
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar, Literal

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema
from atm.tools.sandbox.base import CodeSandbox


class TestRunTool:
    """Run a unittest test suite in a sandbox and parse the result summary.

    The test code is written as ``{module_name}.py`` inside the sandbox.
    The runner command is::

        python -m unittest -v {module_name} 2>&1

    stderr is merged into stdout so that Python's unittest summary line
    ("OK" / "FAILED (failures=N, errors=M)") — which unittest writes to
    stderr — is captured and parseable.

    Parameters
    ----------
    sandbox:
        A CodeSandbox implementation.
    default_timeout_s:
        Default execution timeout.  Defaults to 60.0.
    runner:
        Test runner to use.  Currently only ``"unittest"`` is supported.

    Output shape
    ------------
    stdout    : str  — captured output (stderr merged in via 2>&1)
    stderr    : str  — always empty string (merged into stdout)
    exit_code : int  — runner exit code (0 = all tests passed)
    timed_out : bool — True if the runner was killed due to timeout
    summary   : str  — last non-empty line of stdout (the unittest summary)
    passed    : bool — True if summary starts with "OK"
    """

    name: ClassVar[str] = "test_run"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="test_run",
        description=(
            "Run a unittest test suite in a sandbox. "
            "Returns stdout (with stderr merged), exit code, "
            "a parsed summary line, and a passed boolean."
        ),
        parameters={
            "test_code": "string — unittest source code to run",
            "module_name": "string (optional, default 'test_solution') — module filename without .py",
            "files": "dict[str, str] (optional) — extra files to write alongside the test file",
            "timeout_s": "float | null (optional) — timeout override in seconds",
        },
        returns={
            "stdout": "string",
            "stderr": "string",
            "exit_code": "integer",
            "timed_out": "boolean",
            "summary": "string",
            "passed": "boolean",
        },
    )

    def __init__(
        self,
        sandbox: CodeSandbox,
        default_timeout_s: float = 60.0,
        runner: Literal["unittest"] = "unittest",
    ) -> None:
        self._sandbox = sandbox
        self._default_timeout_s = default_timeout_s
        self._runner = runner

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Run test code in the sandbox.

        Parameters
        ----------
        args:
            test_code   : str  — unittest source code
            module_name : str  (optional, default 'test_solution')
            files       : dict[str, str]  (optional) — extra files
            timeout_s   : float | None  (optional) — timeout override

        Returns
        -------
        ToolResult
            ok matches ``passed``.
            Output contains stdout, stderr, exit_code, timed_out, summary, passed.
        """
        test_code: str = args["test_code"]
        module_name: str = args.get("module_name", "test_solution")
        extra_files: dict[str, str] = dict(args.get("files") or {})
        timeout_s: float | None = args.get("timeout_s")

        effective_timeout = timeout_s if timeout_s is not None else self._default_timeout_s

        # The test file is written via the files dict; the main code runs the
        # unittest discovery command with stderr merged into stdout.
        test_filename = f"{module_name}.py"

        # Build the runner harness: a small wrapper that invokes unittest and
        # ensures stderr output (the summary line) reaches stdout via 2>&1.
        # We achieve this by writing the test code as a file, then running:
        #   python -c "import subprocess, sys; ..."
        # However, SubprocessSandbox uses a fixed main.py entry-point.
        # The cleanest approach: make main.py a shell-like harness that
        # spawns unittest as a subprocess with stderr→stdout merge.
        # But SubprocessSandbox only supports python and node, so we use
        # Python's subprocess module inside the main.py to run unittest.

        runner_code = (
            "import subprocess, sys\n"
            f"result = subprocess.run(\n"
            f"    [sys.executable, '-m', 'unittest', '-v', {module_name!r}],\n"
            f"    stdout=subprocess.PIPE,\n"
            f"    stderr=subprocess.STDOUT,  # merge stderr → stdout\n"
            f"    text=True,\n"
            f")\n"
            f"print(result.stdout, end='')\n"
            f"sys.exit(result.returncode)\n"
        )

        # Add the test file to the extra files
        extra_files[test_filename] = test_code

        exec_result = await self._sandbox.execute(
            lang="python",
            code=runner_code,
            files=extra_files,
            timeout=effective_timeout,
        )

        stdout = exec_result.stdout
        stderr = exec_result.stderr
        exit_code = exec_result.exit_code
        timed_out = exec_result.timed_out

        # Parse summary: last non-empty line of stdout
        non_empty_lines = [line for line in stdout.splitlines() if line.strip()]
        summary = non_empty_lines[-1] if non_empty_lines else ""

        passed = summary.startswith("OK")

        output: dict[str, Any] = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "summary": summary,
            "passed": passed,
        }

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=passed,
            output=output,
            error=None,
            latency_ms=exec_result.duration_ms,
        )


__all__ = ["TestRunTool"]
