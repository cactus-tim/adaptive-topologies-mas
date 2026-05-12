"""ATM tasks module — public API.

Re-exports all public symbols from ``atm.tasks.base``.

Side-effect imports (guarded):
  - Each loader/evaluator module registers itself via
    ``@TASKS.register`` / ``@EVALUATORS.register`` when imported.
    Guards ensure this package loads cleanly even if a module does not
    yet exist.

Registration list: commongen.py, dabench.py, gsm8k.py, humaneval.py
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
    importlib.import_module("atm.tasks.commongen")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.dabench")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.gsm8k")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.tasks.humaneval")
