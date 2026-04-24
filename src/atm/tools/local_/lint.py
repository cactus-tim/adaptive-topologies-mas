"""LintTool — ruff linter with optional pylint backend.

Runs ``ruff check`` on the provided source code via stdin.
Optionally runs pylint if ``include_pylint=True`` and pylint is installed.

Public API
----------
LintTool -- Tool implementation wrapping ruff (+ optional pylint)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, ClassVar

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class LintTool:
    """Run ruff (and optionally pylint) on Python source code.

    Parameters
    ----------
    include_pylint:
        If True, also run pylint. If pylint is not installed, set
        ``pylint_skipped="not installed"`` in output but still return ok=True.

    Output shape
    ------------
    ruff           : list of ruff diagnostic dicts (may be empty)
    pylint         : list of pylint diagnostic dicts | None (when not requested)
    pylint_skipped : str | None — "not installed" if pylint missing, else None
    """

    name: ClassVar[str] = "lint"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="lint",
        description=(
            "Run ruff linter on Python source code. "
            "Optionally also run pylint if available. "
            "Returns lists of diagnostic objects."
        ),
        parameters={
            "code": {
                "type": "string",
                "description": "Python source code to lint.",
            },
            "include_pylint": {
                "type": "boolean",
                "description": "Also run pylint (if installed). Default false.",
            },
        },
        returns={
            "ruff": "array of ruff diagnostic objects",
            "pylint": "array of pylint diagnostic objects | null",
            "pylint_skipped": "string | null",
        },
    )

    def __init__(self, include_pylint: bool = False) -> None:
        self._include_pylint = include_pylint

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Lint the provided code.

        Parameters
        ----------
        args:
            code           : str  — Python source code to lint
            include_pylint : bool (optional) — override instance default

        Returns
        -------
        ToolResult
            ok=True always (even if diagnostics found or pylint missing).
            ok=False only if ruff itself crashes with non-JSON output.
        """
        call_id = uuid.uuid4()
        code: str = args.get("code", "")
        include_pylint: bool = args.get("include_pylint", self._include_pylint)

        # --- Ruff ---
        ruff_diagnostics, ruff_error = await self._run_ruff(code)
        if ruff_error is not None:
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=ruff_error,
                latency_ms=0,
            )

        # --- Pylint (optional) ---
        pylint_diagnostics: list[dict[str, Any]] | None = None
        pylint_skipped: str | None = None

        if include_pylint:
            pylint_diagnostics, pylint_skipped = await self._run_pylint(code)

        return ToolResult(
            call_id=call_id,
            ok=True,
            output={
                "ruff": ruff_diagnostics,
                "pylint": pylint_diagnostics,
                "pylint_skipped": pylint_skipped,
            },
            error=None,
            latency_ms=0,
        )

    async def _run_ruff(
        self, code: str
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Run ruff check on code via stdin.

        Returns (diagnostics, None) on success or ([], error_msg) on crash.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff",
                "check",
                "--output-format=json",
                "--stdin-filename=input.py",
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate(
                input=code.encode("utf-8")
            )
        except FileNotFoundError:
            return [], "ruff not found: please install ruff"

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")

        # ruff exits 0 (no issues) or 1 (issues found) — both are valid
        # ruff exits 2 on fatal error
        if proc.returncode == 2:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            return [], f"ruff error: {stderr_text or stdout_text}"

        if not stdout_text.strip():
            return [], None

        try:
            diagnostics: list[dict[str, Any]] = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            return [], f"ruff output parse error: {exc} — raw: {stdout_text[:200]}"

        return diagnostics, None

    async def _run_pylint(
        self, code: str
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Run pylint on code via stdin (--from-stdin).

        Returns (diagnostics, None) or (None, "not installed") if not found.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "pylint",
                "--output-format=json",
                "--from-stdin",
                "input",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, _ = await proc.communicate(input=code.encode("utf-8"))
        except FileNotFoundError:
            return None, "not installed"

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")

        if not stdout_text.strip():
            return [], None

        try:
            diagnostics: list[dict[str, Any]] = json.loads(stdout_text)
        except json.JSONDecodeError:
            return [], None

        return diagnostics, None


__all__ = ["LintTool"]
