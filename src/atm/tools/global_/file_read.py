"""FileReadTool — workspace-scoped file read with path traversal and symlink protection.

Security guarantees
-------------------
1. Absolute input paths (starting with ``/``) are rejected immediately.
2. The resolved path is checked to be within ``workspace.resolve()`` via
   ``Path.is_relative_to`` — this catches ``../``-based traversal.
3. ``os.stat(follow_symlinks=False)`` is used to detect symlinks; their
   *target* is then checked against the workspace boundary.
4. Files exceeding ``max_bytes`` are rejected — no silent truncation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class FileReadTool:
    """Read a file from the workspace.

    Parameters
    ----------
    workspace:
        Absolute path to the root directory that this tool is allowed to access.
        All ``path`` arguments are interpreted relative to this directory.
    max_bytes:
        Maximum file size in bytes. Files larger than this limit are rejected
        with ``ok=False`` — there is NO silent truncation.
    """

    name: ClassVar[str] = "file_read"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="file_read",
        description=(
            "Read the contents of a file inside the agent workspace. "
            "Paths are relative to the workspace root. "
            "Absolute paths and path traversal (../) are rejected."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": "Relative path to the file within the workspace.",
            }
        },
        returns={
            "content": "string",
            "path": "string",
            "size": "integer",
        },
    )

    def __init__(self, workspace: Path, max_bytes: int = 1_000_000) -> None:
        self._workspace = workspace.resolve()
        self._max_bytes = max_bytes

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Execute the file read.

        Returns a ``ToolResult`` with ``ok=True`` and
        ``output={"content": str, "path": str, "size": int}`` on success,
        or ``ok=False`` with an ``error`` string on any failure.
        This method never raises.
        """
        t0 = time.monotonic()
        call_id = uuid4()

        try:
            result_output, error = self._read(args)
        except Exception as exc:
            elapsed = int((time.monotonic() - t0) * 1000)
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=str(exc),
                latency_ms=elapsed,
            )

        elapsed = int((time.monotonic() - t0) * 1000)
        if error is not None:
            return ToolResult(
                call_id=call_id,
                ok=False,
                output=None,
                error=error,
                latency_ms=elapsed,
            )

        return ToolResult(
            call_id=call_id,
            ok=True,
            output=result_output,
            error=None,
            latency_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Validate the path and read the file.

        Returns
        -------
        (output_dict, None)   on success
        (None, error_message) on failure
        """
        raw_path: str = args.get("path", "")

        # 1. Reject absolute paths in the input
        if raw_path.startswith("/"):
            return None, "absolute paths not allowed"

        # 2. Resolve path relative to workspace
        resolved = (self._workspace / raw_path).resolve()

        # 3. Ensure the resolved path stays within the workspace
        try:
            resolved.relative_to(self._workspace)
        except ValueError:
            return None, "path outside workspace"

        # 4. Symlink detection: stat without following symlinks
        try:
            lstat = os.stat(resolved, follow_symlinks=False)
        except FileNotFoundError:
            return None, f"file not found: {raw_path}"
        except OSError as exc:
            return None, f"cannot stat file: {exc}"

        import stat as _stat

        if _stat.S_ISLNK(lstat.st_mode):
            # Resolve the symlink target and verify it stays within the workspace
            try:
                target = resolved.resolve(strict=True)
            except (OSError, RuntimeError):
                return None, "path outside workspace"

            try:
                target.relative_to(self._workspace)
            except ValueError:
                return None, "path outside workspace"

            # Use the stat of the *target* for the size check
            try:
                fstat = os.stat(resolved, follow_symlinks=True)
            except OSError as exc:
                return None, f"cannot stat symlink target: {exc}"
        else:
            fstat = lstat

        # 5. Size check — no silent truncation
        st_size: int = fstat.st_size
        if st_size > self._max_bytes:
            return None, f"file too large: {st_size} bytes exceeds limit of {self._max_bytes}"

        # 6. Read the file
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return None, f"cannot read file: {exc}"

        # Compute a clean relative path string (no leading ./)
        try:
            rel_path = str(resolved.relative_to(self._workspace))
        except ValueError:
            rel_path = raw_path

        return {"content": content, "path": rel_path, "size": st_size}, None
