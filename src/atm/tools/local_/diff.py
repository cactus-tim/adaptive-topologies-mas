"""DiffTool — produces a unified diff string between two text blobs."""

from __future__ import annotations

import difflib
import uuid
from typing import Any, ClassVar

from atm.core.types import ToolResult
from atm.tools.base import ToolSchema


class DiffTool:
    """Compute a unified diff between *before* and *after* text.

    Output: ``{"diff": "<unified-diff-string>"}``

    An empty string is returned when before == after.
    """

    name: ClassVar[str] = "diff"
    schema: ClassVar[ToolSchema] = ToolSchema(
        name="diff",
        description=(
            "Produce a unified diff between two text blobs. "
            "Returns an empty string if the content is identical."
        ),
        parameters={
            "before": "string — original text",
            "after": "string — modified text",
            "filename_before": "string (optional, default 'before') — label for the original file",
            "filename_after": "string (optional, default 'after') — label for the modified file",
        },
        returns={"diff": "string (unified diff)"},
    )

    async def ainvoke(self, args: dict[str, Any]) -> ToolResult:
        """Compute the diff.

        Args:
            args: Must contain ``before`` and ``after`` strings.
                  Optionally ``filename_before`` and ``filename_after``.

        Returns:
            ToolResult with ``output["diff"]`` as a unified-diff string.
        """
        before: str = args["before"]
        after: str = args["after"]
        fromfile: str = args.get("filename_before", "before")
        tofile: str = args.get("filename_after", "after")

        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=fromfile,
                tofile=tofile,
            )
        )
        diff_str = "".join(diff_lines)

        return ToolResult(
            call_id=uuid.uuid4(),
            ok=True,
            output={"diff": diff_str},
            error=None,
            latency_ms=0,
        )
