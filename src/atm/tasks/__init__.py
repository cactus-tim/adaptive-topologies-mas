"""ATM tasks module — public API.

Re-exports all public symbols from ``atm.tasks.base``.

Side-effect imports (guarded):
  - humaneval.py, mmlu.py, creative.py, and analysis.py register their
    loaders and evaluators via ``@TASKS.register`` / ``@EVALUATORS.register``
    when imported. Guards ensure this package loads cleanly even if a
    module does not yet exist.

Registration list finalized at M10.

M11 addition: ``resolve_spec`` helper for M6 inline-prompt fallback detection.
"""

from __future__ import annotations

import contextlib
import importlib
from typing import Any

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
    "resolve_spec",
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


# ---------------------------------------------------------------------------
# resolve_spec — M11 helper
# ---------------------------------------------------------------------------


def resolve_spec(task_cfg: Any) -> TaskSpec | None:
    """Resolve a TaskCfg to a TaskSpec, or return None for M6 inline-prompt tasks.

    Resolution rules:
    1. If the task name is NOT registered in TASKS (e.g. an M6 inline-prompt
       task like "fib_test"), return None.  The runner should short-circuit
       to ``quality_score = 0.0`` without calling the aggregator.
    2. If the task name IS registered (a dataset-backed task), call
       ``TASKS.sample(name, n=1, seed=shuffle_seed)`` and return the first
       TaskSpec.
    3. If the registered dataset is empty, return None (graceful fallback).

    This function never raises. Unknown task names produce None rather than
    KeyError so that the runner can continue without crashing.

    Args:
        task_cfg: A TaskCfg (or any object with ``name`` and
                  ``shuffle_seed`` attributes).

    Returns:
        A TaskSpec for dataset-backed tasks, or None for inline-prompt /
        unknown tasks.
    """
    name: str = getattr(task_cfg, "name", "")
    shuffle_seed: int = getattr(task_cfg, "shuffle_seed", 0)

    # Check if the name is registered in the task registry
    try:
        TASKS.get(name)
    except KeyError:
        # Not registered — M6 inline-prompt path or unknown task → None
        return None

    # Registered dataset task — sample one spec deterministically
    specs = TASKS.sample(name, n=1, seed=shuffle_seed)
    if not specs:
        # Dataset registered but empty — graceful fallback
        return None  # pragma: no cover
    return specs[0]
