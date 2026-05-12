"""Unit tests for atm.tasks.base — Protocols, Registries, EvalResult.

Tests cover (TDD red phase — 2.1):
1. TaskRegistry register + get via TASKS
2. EvaluatorRegistry register + get via EVALUATORS
3. KeyError raised when getting unknown name from both registries
4. TaskRegistry.sample(n, seed) determinism + sort-by-id stability
5. TaskRegistry.sample() overflow — returns all available when n > len
6. EvalResult score validation: accepts 0.0 and 1.0, rejects <0 and >1
7. LLMLike and Evaluator are runtime-checkable Protocols
8. TaskLoader Protocol has load() method
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atm.core.types import TaskSpec
from atm.tasks.base import (
    EVALUATORS,
    TASKS,
    EvalResult,
    Evaluator,
    TaskLoader,
    TaskRegistry,
)

# ---------------------------------------------------------------------------
# Helpers — minimal conforming implementations for testing Protocols
# ---------------------------------------------------------------------------


def _make_task_spec(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        type="reasoning",
        input="What is 2+2?",
        expected="4",
        evaluator_key="dummy",
    )


# ---------------------------------------------------------------------------
# Test 1: TASKS registry — register + get
# ---------------------------------------------------------------------------


def test_tasks_register_and_get() -> None:
    """Registering a loader class via TASKS and getting it back by name works."""

    class _DummyLoader:
        name = "test_loader_unique_1"

        def load(self, cache_dir: object = None) -> list[TaskSpec]:
            return [_make_task_spec("qa/1"), _make_task_spec("qa/2")]

    # Register
    registered = TASKS.register(_DummyLoader)
    assert registered is _DummyLoader

    # Get back — should be the same class
    loader_cls = TASKS.get("test_loader_unique_1")
    assert loader_cls is _DummyLoader

    # Conforms to TaskLoader Protocol (structural subtyping)
    assert isinstance(_DummyLoader(), TaskLoader)


# ---------------------------------------------------------------------------
# Test 2: EVALUATORS registry — register + get
# ---------------------------------------------------------------------------


def test_evaluators_register_and_get() -> None:
    """Registering an evaluator class via EVALUATORS and getting it by name works."""

    class _DummyEval:
        name = "test_eval_unique_1"

        async def evaluate(self, spec: TaskSpec, answer: str) -> EvalResult:
            return EvalResult(score=1.0, passed=True)

    registered = EVALUATORS.register(_DummyEval)
    assert registered is _DummyEval

    eval_cls = EVALUATORS.get("test_eval_unique_1")
    assert eval_cls is _DummyEval

    # Conforms to Evaluator Protocol
    assert isinstance(_DummyEval(), Evaluator)


# ---------------------------------------------------------------------------
# Test 3: KeyError on unknown name — both registries
# ---------------------------------------------------------------------------


def test_tasks_get_unknown_raises_key_error() -> None:
    """Getting an unregistered loader name raises KeyError."""
    with pytest.raises(KeyError, match="no_such_loader_xyz"):
        TASKS.get("no_such_loader_xyz")


def test_evaluators_get_unknown_raises_key_error() -> None:
    """Getting an unregistered evaluator name raises KeyError."""
    with pytest.raises(KeyError, match="no_such_eval_xyz"):
        EVALUATORS.get("no_such_eval_xyz")


# ---------------------------------------------------------------------------
# Test 4: TaskRegistry.sample(n, seed) determinism + sort-by-id stability
# ---------------------------------------------------------------------------


def test_task_registry_sample_is_deterministic() -> None:
    """sample(n, seed) with same seed always returns the same ordered subset."""
    registry = TaskRegistry()

    class _LoaderForDet:
        name = "det_loader_unique"

        def load(self, cache_dir: object = None) -> list[TaskSpec]:
            # Intentionally out-of-id order to test sort
            return [
                _make_task_spec("qa/c"),
                _make_task_spec("qa/a"),
                _make_task_spec("qa/b"),
                _make_task_spec("qa/d"),
                _make_task_spec("qa/e"),
            ]

    registry.register(_LoaderForDet)

    result1 = registry.sample("det_loader_unique", n=3, seed=42)
    result2 = registry.sample("det_loader_unique", n=3, seed=42)

    assert len(result1) == 3
    assert result1 == result2  # deterministic with same seed

    # seed=99 is also deterministic with itself
    result3 = registry.sample("det_loader_unique", n=3, seed=99)
    result4 = registry.sample("det_loader_unique", n=3, seed=99)
    assert result3 == result4


def test_task_registry_sample_sorted_by_id() -> None:
    """The pool that sample draws from is sorted by TaskSpec.id (lexicographic)."""
    registry = TaskRegistry()

    class _LoaderForSort:
        name = "sort_loader_unique"

        def load(self, cache_dir: object = None) -> list[TaskSpec]:
            # Deliberately shuffle ids
            return [
                _make_task_spec("qa/z"),
                _make_task_spec("qa/a"),
                _make_task_spec("qa/m"),
            ]

    registry.register(_LoaderForSort)

    # Sampling all 3 with a seed; since n == len(pool), all are returned in sorted order
    result = registry.sample("sort_loader_unique", n=3, seed=1)
    assert len(result) == 3
    ids = [s.id for s in result]
    # sorted pool: qa/a, qa/m, qa/z — result should be that order (all selected)
    assert ids == ["qa/a", "qa/m", "qa/z"]


# ---------------------------------------------------------------------------
# Test 5: sample() overflow — returns all available when n > len
# ---------------------------------------------------------------------------


def test_task_registry_sample_overflow_returns_all() -> None:
    """When n > number of tasks available, sample returns all tasks."""
    registry = TaskRegistry()

    class _LoaderForOverflow:
        name = "overflow_loader_unique"

        def load(self, cache_dir: object = None) -> list[TaskSpec]:
            return [_make_task_spec("qa/1"), _make_task_spec("qa/2")]

    registry.register(_LoaderForOverflow)

    result = registry.sample("overflow_loader_unique", n=100, seed=7)
    assert len(result) == 2  # only 2 available


# ---------------------------------------------------------------------------
# Test 6: EvalResult score validation — rejects <0 and >1
# ---------------------------------------------------------------------------


def test_eval_result_valid_boundary_scores() -> None:
    """EvalResult accepts scores 0.0 and 1.0 exactly."""
    r0 = EvalResult(score=0.0, passed=False)
    assert r0.score == 0.0

    r1 = EvalResult(score=1.0, passed=True)
    assert r1.score == 1.0

    r_mid = EvalResult(score=0.75, passed=True)
    assert r_mid.score == 0.75


def test_eval_result_rejects_score_below_zero() -> None:
    """EvalResult raises ValidationError when score < 0."""
    with pytest.raises(ValidationError):
        EvalResult(score=-0.01, passed=False)


def test_eval_result_rejects_score_above_one() -> None:
    """EvalResult raises ValidationError when score > 1."""
    with pytest.raises(ValidationError):
        EvalResult(score=1.01, passed=True)


# ---------------------------------------------------------------------------
# Test 7: EvalResult is frozen and has expected fields
# ---------------------------------------------------------------------------


def test_eval_result_is_frozen() -> None:
    """EvalResult is immutable after creation."""
    r = EvalResult(score=0.5, passed=False, details={"note": "test"})
    with pytest.raises(ValidationError):
        r.score = 0.9  # type: ignore[misc]


def test_eval_result_optional_error_field() -> None:
    """EvalResult accepts optional error string."""
    r = EvalResult(score=0.0, passed=False, error="timeout")
    assert r.error == "timeout"

    r_no_err = EvalResult(score=1.0, passed=True)
    assert r_no_err.error is None


# ---------------------------------------------------------------------------
# Test 8: TaskRegistry lazy load + cache
# ---------------------------------------------------------------------------


def test_task_registry_lazy_load_caches() -> None:
    """TaskRegistry loads tasks only once and caches the result."""
    registry = TaskRegistry()
    call_count = 0

    class _LoaderForCache:
        name = "cache_loader_unique"

        def load(self, cache_dir: object = None) -> list[TaskSpec]:
            nonlocal call_count
            call_count += 1
            return [_make_task_spec("qa/cached")]

    registry.register(_LoaderForCache)

    registry.sample("cache_loader_unique", n=1, seed=1)
    registry.sample("cache_loader_unique", n=1, seed=2)
    registry.sample("cache_loader_unique", n=1, seed=3)

    assert call_count == 1  # load() called only once despite 3 sample calls
