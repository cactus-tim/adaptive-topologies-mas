"""ATM tasks module — public API.

Re-exports all public symbols from ``atm.tasks.base``.

Side-effect imports (guarded):
  - humaneval.py, mmlu.py, creative.py, and analysis.py register their
    loaders and evaluators via ``@TASKS.register`` / ``@EVALUATORS.register``
    when imported. Guards ensure this package loads cleanly even if a
    module does not yet exist.

Registration list finalized at M10.
"""

from __future__ import annotations

import contextlib
import importlib

from atm.tasks.base import (
    EVALUATORS,
    TASKS,
    EvalResult,
    Evaluator,
    EvaluatorRegistry,
    LLMLike,
    TaskLoader,
    TaskRegistry,
    TaskSpec,
)

__all__ = [
    "EVALUATORS",
    "TASKS",
    "EvalResult",
    "Evaluator",
    "EvaluatorRegistry",
    "LLMLike",
    "TaskLoader",
    "TaskRegistry",
    "TaskSpec",
]

# ---------------------------------------------------------------------------
# Side-effect registration — guarded imports so the package loads cleanly
# even when a loader module does not yet exist.
# ---------------------------------------------------------------------------

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.humaneval")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.mmlu")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.creative")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.analysis")
