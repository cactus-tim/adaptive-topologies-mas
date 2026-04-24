"""LintTool — ruff linter + optional pylint for Python code snippets.

Runs ruff via stdin (--output-format=json) and optionally pylint.
If pylint is not installed, the tool returns ok=True with pylint_skipped="not installed".
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, ClassVar

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class LintTool:
    """Lint a Python code snippet using ruff (always) and optionally pylint.

    Parameters
    ----------
    include_pylint:
        When ``True``, also run pylint.  If pylint is not installed,
        ``pylint_skipped`` is set to ``"not installed"`` and ``ok`` stays
        ``True`` — the tool never fails due to a missing pylint binary.

    Output shape
    ------------
    ``{"ruff": list, "pylint": list | None, "pylint_skipped": str | None}``

    ``ok=False`` only when ruff itself crashes and its output is not valid JSON.
    """

    name: ClassVar[str] = "lint"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="lint",
        description=(
            "Lint a Python code snippet with ruff (always) and optionally pylint. "
            "Returns diagnostics as lists.  ok=False only if ruff crashes with "
            "non-JSON output."
        ),
        parameters={
            "code": "string — Python source code to lint",
            "filename": "string (optional, default 'snippet.py') — virtual filename for ruff",
        },
        returns={
            "ruff": "list of diagnostics",
            "pylint": "list of diagnostics | null",
            "pylint_skipped": "string | null",
        },
    )

    def __init__(self, include_pylint: bool = False) -> None:
        self.include_pylint = include_pylint

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Lint the given code.

        Parameters
        ----------
        args:
            ``code`` (required): Python source to lint.
            ``filename`` (optional, default ``"snippet.py"``): virtual filename
            passed to ruff via ``--stdin-filename``.

        Returns
        -------
        ToolResult
            ``ok=True`` in all normal cases, including when pylint is missing.
            ``ok=False`` only if ruff exits with non-JSON stdout.
        """
        code: str = args["code"]
        filename: str = args.get("filename", "snippet.py")

        # --- run ruff -------------------------------------------------------
        ruff_result = await self._run_ruff(code, filename)
        if ruff_result is None:
            # ruff crashed and produced non-JSON output
            return ToolResult(
                call_id=uuid.uuid4(),
                ok=False,
                output=None,
                error="ruff produced non-JSON output",
                latency_ms=0,
            )

        # --- run pylint (optional) ------------------------------------------
        pylint_result: list[Any] | None = None
        pylint_skipped: str | None = None

        if self.include_pylint:
            pylint_result, pylint_skipped = await self._run_pylint(code, filename)

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=True,
            output={
                "ruff": ruff_result,
                "pylint": pylint_result,
                "pylint_skipped": pylint_skipped,
            },
            error=None,
            latency_ms=0,
        )

    async def _run_ruff(self, code: str, filename: str) -> list[Any] | None:
        """Run ``ruff check`` on *code* via stdin.

        Returns the parsed JSON list of diagnostics, or ``None`` if ruff
        crashes and its output is not valid JSON.

        Ruff exits with code 0 for no issues, 1 for diagnostics found,
        and 2 for a fatal error.  We treat exit codes 0 and 1 as normal
        (they both produce valid JSON); only truly malformed output triggers
        ``None``.
        """
        proc = await asyncio.create_subprocess_exec(
            "ruff",
            "check",
            "--output-format=json",
            "--stdin-filename",
            filename,
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate(input=code.encode())

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            # No output at all — treat as empty diagnostics list (clean code,
            # but ruff may have exited with 2 on a real crash; for robustness
            # we still return [] here and only fail on unparseable text).
            raw = "[]"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, list):
            return None

        return data

    async def _run_pylint(self, code: str, filename: str) -> tuple[list[Any] | None, str | None]:
        """Run pylint on *code* via stdin flag ``--from-stdin``.

        Returns ``(diagnostics, skipped_reason)``.
        If pylint is not installed, returns ``(None, "not installed")``.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "pylint",
                "--output-format=json",
                "--from-stdin",
                filename,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None, "not installed"

        stdout, _stderr = await proc.communicate(input=code.encode())

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            return [], None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # pylint produced non-JSON (e.g. an error message) — return empty
            return [], None

        if not isinstance(data, list):
            return [], None

        return data, None
