"""build_default_registry — convenience factory for the M4 default tool set.

Placed in a separate module to avoid circular imports between base.py and the
individual tool implementation modules that all import from base.py.

Public API
----------
build_default_registry -- returns a ToolRegistry pre-populated with all 12 tools
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from atm.core.errors import ToolError
from atm.tools.base import ToolRegistry

if TYPE_CHECKING:
    from atm.tools.sandbox.base import CodeSandbox


def build_default_registry(
    *,
    workspace: Path,
    corpus_dir: Path,
    sandbox: CodeSandbox,
    prod_mode: bool = False,
    include_pylint: bool = False,
    retry_policy: Any = None,
) -> ToolRegistry:
    """Return a ToolRegistry pre-populated with the default M4 tool set.

    Parameters
    ----------
    workspace:
        Absolute path to the agent workspace directory. Used by FileReadTool,
        FileWriteTool.
    corpus_dir:
        Directory containing ``*.txt`` corpus files for SemanticSearchTool.
    sandbox:
        CodeSandbox implementation (SubprocessSandbox or DockerSandbox).
    prod_mode:
        If True, raises ToolError when sandbox is not isolated (IS_ISOLATED=False).
        Fail-closed: prevents accidental use of dev sandbox in production.
    include_pylint:
        Forward to LintTool — include pylint if installed (default False).
    retry_policy:
        Optional RetryPolicy forwarded to DuckDuckGoSearchTool and UrlFetchTool.

    Returns
    -------
    ToolRegistry
        Populated with all 12 M4 tools.

    Raises
    ------
    ToolError
        If prod_mode=True and sandbox.IS_ISOLATED is False.
    """
    if prod_mode and not sandbox.IS_ISOLATED:
        raise ToolError(
            tool_name="build_default_registry",
            message=(
                "prod_mode requires an isolated sandbox (DockerSandbox); "
                "got non-isolated sandbox"
            ),
        )

    # Lazy imports to prevent circular imports at module load time
    from atm.tools.global_.calculator import CalculatorTool
    from atm.tools.global_.file_read import FileReadTool
    from atm.tools.global_.search import DuckDuckGoSearchTool
    from atm.tools.global_.url_fetch import UrlFetchTool
    from atm.tools.local_.code_run import CodeRunTool
    from atm.tools.local_.diff import DiffTool
    from atm.tools.local_.file_write import FileWriteTool
    from atm.tools.local_.lint import LintTool
    from atm.tools.local_.plan_update import PlanUpdateTool
    from atm.tools.local_.semantic_search import SemanticSearchTool
    from atm.tools.local_.test_run import TestRunTool
    from atm.tools.local_.todo_write import TodoWriteTool

    registry = ToolRegistry()

    # Global tools
    registry.register(CalculatorTool())
    registry.register(FileReadTool(workspace=workspace))
    registry.register(DuckDuckGoSearchTool(retry_policy=retry_policy))
    registry.register(UrlFetchTool(retry_policy=retry_policy))

    # Local tools
    registry.register(CodeRunTool(sandbox=sandbox))
    registry.register(TestRunTool(sandbox=sandbox))
    registry.register(FileWriteTool(workspace=workspace))
    registry.register(DiffTool())
    registry.register(TodoWriteTool())
    registry.register(PlanUpdateTool())
    registry.register(LintTool(include_pylint=include_pylint))
    registry.register(SemanticSearchTool(corpus_dir=corpus_dir))

    return registry


__all__ = ["build_default_registry"]
