"""Base Protocols, Registries, and EvalResult for the ATM tasks module.

Public API:
- ``LLMLike``      — runtime-checkable Protocol for LLM-like objects with ``ainvoke``
- ``Evaluator``    — runtime-checkable Protocol for task evaluators
- ``EvalResult``   — frozen Pydantic model: score ∈ [0, 1], passed, details, error
- ``TaskLoader``   — runtime-checkable Protocol for task loaders
- ``EvaluatorRegistry`` — registry of evaluator classes keyed by name
- ``TaskRegistry`` — registry of loader classes with lazy load + cache + ``sample``
- ``TASKS``        — module-level singleton ``TaskRegistry``
- ``EVALUATORS``   — module-level singleton ``EvaluatorRegistry``
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from atm.core.types import LLMResponse, TaskSpec

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
# LLMLike Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMLike(Protocol):
    """Minimal LLM interface required by judge evaluators.

    Any object with a compatible ``ainvoke`` method (including ``LLMWrapper``
    and ``FakeLLM``) satisfies this Protocol without explicit inheritance.
    """

    async def ainvoke(
        self,
        messages: list[Any],
        *,
        agent_id: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke the LLM and return an LLMResponse."""
        ...


# ---------------------------------------------------------------------------
# Evaluator Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for task evaluators.

    Conforming classes must expose:
    - ``name: str`` — class-level attribute used for registry key
    - ``evaluate(spec, answer, *, artifacts) -> EvalResult`` — async evaluation method
    """

    name: str

    async def evaluate(
        self,
        spec: TaskSpec,
        answer: str,
        *,
        artifacts: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Evaluate ``answer`` against ``spec`` and return an ``EvalResult``."""
        ...


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Immutable result of a single evaluation.

    Attributes:
        score:   Normalised score in [0.0, 1.0].
        passed:  Convenience boolean (threshold set by each evaluator).
        details: Free-form dict with evaluator-specific diagnostics.
        error:   Optional error message if evaluation itself failed.
    """

    model_config = ConfigDict(frozen=True)

    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# TaskLoader Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskLoader(Protocol):
    """Protocol for task loaders.

    Conforming classes must expose:
    - ``name: str`` — class-level attribute used for registry key
    - ``load(cache_dir)`` — returns a list of TaskSpec
    """

    name: str

    def load(
        self,
        cache_dir: Path | None = None,
    ) -> list[TaskSpec]:
        """Load and return all task specs for this dataset."""
        ...


# ---------------------------------------------------------------------------
# EvaluatorRegistry
# ---------------------------------------------------------------------------


class EvaluatorRegistry:
    """Registry of evaluator classes keyed by name.

    Usage::

        @EVALUATORS.register
        class MyEval:
            name = "my_eval"
            async def evaluate(self, spec, answer): ...

        eval_cls = EVALUATORS.get("my_eval")
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Any]] = {}

    def register(self, cls: type[Any]) -> type[Any]:
        """Register ``cls`` under ``cls.name``. Returns ``cls`` unchanged (decorator-safe)."""
        key: str = cls.name
        self._registry[key] = cls
        return cls

    def get(self, name: str) -> type[Any]:
        """Return the evaluator class registered under ``name``.

        Raises:
            KeyError: if ``name`` is not in the registry.
        """
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"{name!r} not registered. Known: {sorted(self._registry.keys())}"
            ) from None

    def names(self) -> list[str]:
        """Return sorted list of registered evaluator names."""
        return sorted(self._registry.keys())


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------


class TaskRegistry:
    """Registry of loader classes with lazy loading, caching, and deterministic sampling.

    Usage::

        @TASKS.register
        class MyLoader:
            name = "my_dataset"
            def load(self, cache_dir=None): ...

        specs = TASKS.sample("my_dataset", n=10, seed=42)

    Lazy loading: ``load()`` is called once on first ``sample()`` call; the result
    is cached in ``self._cache`` and reused on subsequent calls.

    Sampling: the pool is sorted by ``TaskSpec.id`` before sampling using
    ``random.Random(seed)`` for determinism. If ``n >= len(pool)``, all tasks
    are returned in sorted order.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Any]] = {}
        self._cache: dict[str, list[TaskSpec]] = {}

    def register(self, cls: type[Any]) -> type[Any]:
        """Register ``cls`` under ``cls.name``. Returns ``cls`` unchanged (decorator-safe)."""
        key: str = cls.name
        self._registry[key] = cls
        return cls

    def get(self, name: str) -> type[Any]:
        """Return the loader class registered under ``name``.

        Raises:
            KeyError: if ``name`` is not in the registry.
        """
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"{name!r} not registered. Known: {sorted(self._registry.keys())}"
            ) from None

    def _load_and_cache(self, name: str) -> list[TaskSpec]:
        """Load tasks for ``name`` if not already cached, and return sorted pool."""
        if name not in self._cache:
            loader_cls = self.get(name)
            loader = loader_cls()
            raw: list[TaskSpec] = loader.load()
            # Sort by id for deterministic, stable sampling
            self._cache[name] = sorted(raw, key=lambda s: s.id)
        return self._cache[name]

    def sample(self, name: str, *, n: int, seed: int) -> list[TaskSpec]:
        """Return a deterministic sample of up to ``n`` tasks for loader ``name``.

        Args:
            name: The registered loader name.
            n:    Maximum number of tasks to return.
            seed: Random seed for ``random.Random`` — guarantees determinism.

        Returns:
            A list of up to ``n`` TaskSpec instances, chosen from the id-sorted pool.
            If ``n >= len(pool)``, returns all tasks in sorted order.
        """
        pool = self._load_and_cache(name)
        if n >= len(pool):
            return list(pool)
        rng = random.Random(seed)
        return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

TASKS: TaskRegistry = TaskRegistry()
EVALUATORS: EvaluatorRegistry = EvaluatorRegistry()
