"""ATM tasks module — public API.

Re-exports all public symbols from ``atm.tasks.base``.

Side-effect imports (guarded):
  - humaneval.py registers its loader and evaluator via
    ``@TASKS.register`` / ``@EVALUATORS.register`` when imported.
    Guards ensure this package loads cleanly even if a module does not
    yet exist.

Registration list: humaneval.py (Wave 2 will add gsm8k, commongen, dabench).
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
