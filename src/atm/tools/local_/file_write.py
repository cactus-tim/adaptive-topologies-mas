"""FileWriteTool — atomic workspace-scoped file write.

Security guarantees
-------------------
1. Absolute input paths (starting with ``/``) are rejected immediately.
2. The resolved path is checked to be within ``workspace.resolve()`` via
   ``Path.is_relative_to`` — catches ``../``-based traversal.
3. Write is atomic: data goes to a ``.tmp`` sibling file, then ``os.replace``
   moves it into place.  No partial writes visible to readers.
4. ``overwrite=False`` (default) refuses to overwrite existing files.
5. ``create_parents=False`` (default) refuses to create missing parent dirs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class FileWriteTool:
    """Write content to a file inside the agent workspace.

    Parameters
    ----------
    workspace:
        Absolute path to the root directory that this tool is allowed to write.
        All ``path`` arguments are interpreted relative to this directory.
    """

    name: ClassVar[str] = "file_write"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="file_write",
        description=(
            "Atomically write content to a file inside the agent workspace. "
            "Paths are relative to the workspace root. "
            "Absolute paths and path traversal (../) are rejected."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": "Relative path to the file within the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow overwriting an existing file. Default false.",
            },
            "create_parents": {
                "type": "boolean",
                "description": "Create parent directories if missing. Default false.",
            },
        },
        returns={
            "bytes_written": "integer",
            "path": "string",
        },
    )

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Execute the file write.

        Returns a ``ToolResult`` with ``ok=True`` and
        ``output={"bytes_written": int, "path": str}`` on success,
        or ``ok=False`` with an ``error`` string on any failure.
        This method never raises.
        """
        t0 = time.monotonic()
        call_id = uuid4()

        try:
            result_output, error = self._write(args)
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

    def _write(self, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Validate the path and write the file atomically.

        Returns
        -------
        (output_dict, None)   on success
        (None, error_message) on failure
        """
        raw_path: str = args.get("path", "")
        content: str = args.get("content", "")
        overwrite: bool = bool(args.get("overwrite", False))
        create_parents: bool = bool(args.get("create_parents", False))

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

        # 4. Check parent directory
        parent = resolved.parent
        if not parent.exists():
            if not create_parents:
                return None, f"parent directory does not exist: {parent}"
            parent.mkdir(parents=True, exist_ok=True)

        # 5. Check overwrite guard
        if resolved.exists() and not overwrite:
            return None, f"file already exists (use overwrite=true): {raw_path}"

        # 6. Atomic write: write to .tmp, then os.replace
        tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(str(tmp_path), str(resolved))
        except OSError as exc:
            # Clean up tmp file on failure
            import contextlib

            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            return None, f"cannot write file: {exc}"

        # Compute bytes written (UTF-8 encoded)
        bytes_written = len(content.encode("utf-8"))

        # Compute a clean relative path string (no leading ./)
        try:
            rel_path = str(resolved.relative_to(self._workspace))
        except ValueError:
            rel_path = raw_path

        return {"bytes_written": bytes_written, "path": rel_path}, None


__all__ = ["FileWriteTool"]
