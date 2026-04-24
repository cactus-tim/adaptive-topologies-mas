"""CodeRunTool — sandbox-backed code execution tool.

Delegates to a CodeSandbox instance and wraps ExecResult into ToolResult.

Public API
----------
CodeRunTool -- Tool implementation that runs arbitrary code in a sandbox
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema
from atm.tools.sandbox.base import CodeSandbox


class CodeRunTool:
    """Execute arbitrary code in a sandbox and return the execution result.

    Parameters
    ----------
    sandbox:
        A CodeSandbox implementation (e.g. SubprocessSandbox or DockerSandbox).
    default_timeout_s:
        Default timeout in seconds used when the caller does not supply
        ``timeout_s`` in args.  Defaults to 30.0.

    Output shape (all 6 keys always present)
    -----------------------------------------
    stdout      : str   — captured standard output
    stderr      : str   — captured standard error
    exit_code   : int   — process exit code
    timed_out   : bool  — True if the process was killed due to timeout
    duration_ms : int   — wall-clock execution time in milliseconds
    oom_killed  : bool  — True if the process was killed due to OOM
    """

    name: ClassVar[str] = "code_run"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="code_run",
        description=(
            "Execute code in a sandboxed environment. "
            "Returns stdout, stderr, exit code, timeout/OOM flags, and duration."
        ),
        parameters={
            "lang": "string — language identifier (e.g. 'python', 'node')",
            "code": "string — source code to execute",
            "files": "dict[str, str] (optional) — extra files to write alongside the main code",
            "timeout_s": "float | null (optional) — timeout override in seconds",
        },
        returns={
            "stdout": "string",
            "stderr": "string",
            "exit_code": "integer",
            "timed_out": "boolean",
            "duration_ms": "integer",
            "oom_killed": "boolean",
        },
    )

    def __init__(
        self,
        sandbox: CodeSandbox,
        default_timeout_s: float = 30.0,
    ) -> None:
        self._sandbox = sandbox
        self._default_timeout_s = default_timeout_s

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Execute code in the sandbox.

        Parameters
        ----------
        args:
            lang      : str  — language identifier
            code      : str  — source code
            files     : dict[str, str] (optional) — extra files
            timeout_s : float | None (optional) — timeout override

        Returns
        -------
        ToolResult
            ok=True when exit_code==0 and timed_out==False.
            Output is the ExecResult serialised as a dict (model_dump).
        """
        lang: str = args["lang"]
        code: str = args["code"]
        files: dict[str, str] = args.get("files") or {}
        timeout_s: float | None = args.get("timeout_s")

        effective_timeout = timeout_s if timeout_s is not None else self._default_timeout_s

        exec_result = await self._sandbox.execute(
            lang=lang,
            code=code,
            files=files or None,
            timeout=effective_timeout,
        )

        ok = exec_result.exit_code == 0 and not exec_result.timed_out

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=ok,
            output=exec_result.model_dump(),
            error=None,
            latency_ms=exec_result.duration_ms,
        )


__all__ = ["CodeRunTool"]
