"""FileWriteTool — atomic, workspace-scoped file write.

Security guarantees
-------------------
1. Absolute input paths (starting with ``/``) are rejected immediately.
2. The resolved path is checked to be within ``workspace.resolve()`` via
   ``Path.is_relative_to`` — this catches ``../``-based traversal.
3. Existing files are protected unless ``overwrite=True`` is explicitly set.
4. Missing parent directories are only created when ``create_parents=True``.
5. Writes are atomic: data is written to a temporary file in the same
   directory, then renamed over the target via ``os.replace``. If any
   error occurs the temporary file is removed before returning.
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class FileWriteTool:
    """Write a file atomically inside the agent workspace.

    Parameters
    ----------
    workspace:
        Absolute path to the root directory that this tool is allowed to
        write into. All ``path`` arguments are interpreted relative to this
        directory.
    create_parents:
        When ``True``, missing intermediate parent directories are created
        automatically before writing. Default ``False``.
    """

    name: ClassVar[str] = "file_write"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="file_write",
        description=(
            "Write content to a file inside the agent workspace. "
            "Paths are relative to the workspace root. "
            "Absolute paths and path traversal (../) are rejected. "
            "The write is atomic: content is staged to a temp file then renamed."
        ),
        parameters={
            "path": {
                "type": "string",
                "description": "Relative path to the file within the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write (UTF-8 encoded).",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow overwriting an existing file. Default false.",
            },
            "create_parents": {
                "type": "boolean",
                "description": (
                    "Create missing parent directories automatically. "
                    "Overrides the constructor-level default for this invocation "
                    "when explicitly provided."
                ),
            },
        },
        returns={
            "bytes_written": "integer — number of UTF-8 encoded bytes written",
            "path": "string — relative path within the workspace",
        },
    )

    def __init__(self, workspace: Path, create_parents: bool = False) -> None:
        self._workspace = workspace.resolve()
        self._create_parents = create_parents

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Execute the file write.

        Returns a ``ToolResult`` with ``ok=True`` and
        ``output={"bytes_written": int, "path": str}`` on success,
        or ``ok=False`` with an ``error`` string on any failure.
        This method never raises.
        """
        t0 = time.monotonic()
        call_id = uuid.uuid4()

        try:
            result_output, error = self._write(args)
        except Exception as exc:  # pragma: no cover
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

    def _write(self, args: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Validate arguments, then perform the atomic write.

        Returns
        -------
        (output_dict, None)    on success
        (None, error_message)  on any validation or I/O failure
        """
        raw_path: str = args.get("path", "")
        content: str = args.get("content", "")
        overwrite: bool = bool(args.get("overwrite", False))
        # Per-call create_parents may override the constructor default if
        # the caller passes it explicitly in args.
        create_parents: bool = bool(args.get("create_parents", self._create_parents))

        # 1. Reject absolute paths in the input
        if raw_path.startswith("/"):
            return None, "absolute paths not allowed"

        # 2. Resolve path relative to workspace
        resolved = (self._workspace / raw_path).resolve()

        # 3. Ensure the resolved path stays within the workspace
        if not resolved.is_relative_to(self._workspace):
            return None, "outside workspace"

        # 4. Check existing file
        if resolved.exists() and not overwrite:
            return None, "file exists"

        # 5. Check/create parent directory
        parent = resolved.parent
        if not parent.exists():
            if not create_parents:
                return None, "parent directory missing"
            parent.mkdir(parents=True, exist_ok=True)

        # 6. Atomic write
        encoded: bytes = content.encode("utf-8")
        tmp_path = resolved.parent / f"{resolved.name}.tmp.{uuid.uuid4().hex}"
        try:
            tmp_path.write_bytes(encoded)
            os.replace(str(tmp_path), str(resolved))
        except Exception as exc:
            # Clean up the tmp file if it exists
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            return None, f"write failed: {exc}"

        # 7. Build a clean relative path string for the output
        try:
            rel_path = str(resolved.relative_to(self._workspace))
        except ValueError:  # pragma: no cover
            rel_path = raw_path

        return {"bytes_written": len(encoded), "path": rel_path}, None
