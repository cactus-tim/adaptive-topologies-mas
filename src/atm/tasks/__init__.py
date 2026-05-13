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
    "resolve_spec",
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


# ---------------------------------------------------------------------------
# resolve_spec — M11 helper (re-added after M10 merge)
# ---------------------------------------------------------------------------


def resolve_spec(task_cfg: object) -> TaskSpec | None:
    """Resolve a TaskCfg to a sampled TaskSpec, or None for inline-prompt fallback.

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
    name: str = getattr(task_cfg, "name", "") or ""
    shuffle_seed: int = getattr(task_cfg, "shuffle_seed", 0) or 0

    try:
        TASKS.get(name)
    except KeyError:
        return None

    specs = TASKS.sample(name, n=1, seed=shuffle_seed)
    if not specs:
        return None  # pragma: no cover
    return specs[0]
