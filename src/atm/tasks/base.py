"""Base protocols, registries, and EvalResult for the ATM tasks module."""

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


@runtime_checkable
class LLMLike(Protocol):
    """Minimal LLM interface (ainvoke) required by judge evaluators."""

    async def ainvoke(
        self,
        messages: list[Any],
        *,
        agent_id: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke the LLM and return an LLMResponse."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for task evaluators (name attribute + async evaluate method)."""

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


class EvalResult(BaseModel):
    """Immutable result of a single evaluation; score ∈ [0.0, 1.0]."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class TaskLoader(Protocol):
    """Protocol for task loaders (name attribute + load method)."""

    name: str

    def load(
        self,
        cache_dir: Path | None = None,
    ) -> list[TaskSpec]:
        """Load and return all task specs for this dataset."""
        ...


class EvaluatorRegistry:
    """Registry of evaluator classes keyed by name."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Any]] = {}

    def register(self, cls: type[Any]) -> type[Any]:
        """Register ``cls`` under ``cls.name``. Returns ``cls`` unchanged (decorator-safe)."""
        key: str = cls.name
        self._registry[key] = cls
        return cls

    def get(self, name: str) -> type[Any]:
        """Return the evaluator class registered under ``name``; raises KeyError if absent."""
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"{name!r} not registered. Known: {sorted(self._registry.keys())}"
            ) from None

    def names(self) -> list[str]:
        """Return sorted list of registered evaluator names."""
        return sorted(self._registry.keys())


class TaskRegistry:
    """Registry of loader classes with lazy loading, caching, and deterministic sampling."""

    def __init__(self) -> None:
        self._registry: dict[str, type[Any]] = {}
        self._cache: dict[str, list[TaskSpec]] = {}

    def register(self, cls: type[Any]) -> type[Any]:
        """Register ``cls`` under ``cls.name``. Returns ``cls`` unchanged (decorator-safe)."""
        key: str = cls.name
        self._registry[key] = cls
        return cls

    def get(self, name: str) -> type[Any]:
        """Return the loader class registered under ``name``; raises KeyError if absent."""
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"{name!r} not registered. Known: {sorted(self._registry.keys())}"
            ) from None

    def _load_and_cache(self, name: str) -> list[TaskSpec]:
        """Load tasks for ``name`` if not already cached; return sorted pool."""
        if name not in self._cache:
            loader_cls = self.get(name)
            loader = loader_cls()
            raw: list[TaskSpec] = loader.load()
            self._cache[name] = sorted(raw, key=lambda s: s.id)
        return self._cache[name]

    def sample(self, name: str, *, n: int, seed: int) -> list[TaskSpec]:
        """Return a deterministic sample of up to ``n`` tasks for loader ``name``."""
        pool = self._load_and_cache(name)
        if n >= len(pool):
            return list(pool)
        rng = random.Random(seed)
        return rng.sample(pool, n)


TASKS: TaskRegistry = TaskRegistry()
EVALUATORS: EvaluatorRegistry = EvaluatorRegistry()
