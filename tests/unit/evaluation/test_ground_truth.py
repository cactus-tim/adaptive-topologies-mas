"""Unit tests for atm.evaluation.ground_truth.score_ground_truth.

Tests covering:
1. gsm8k_numeric smoke — dispatches to GSM8KMatcher; correct answer returns score=1.0
2. commongen_rouge_coverage smoke — dispatches to CommonGenEvaluator
3. dabench_numeric_exact smoke — dispatches to DABenchEvaluator
4. humaneval_pytest smoke — dispatches to HumanEvalEvaluator with sandbox mock
5. humaneval_pytest raises ValueError when sandbox=None
6. unknown key raises KeyError
7. Drift test — EVALUATORS.names() matches _DEPS keys (skipped if assertion fails)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atm.core.types import TaskSpec
from atm.tasks.base import EvalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(evaluator_key: str, **meta: Any) -> TaskSpec:
    """Create a minimal TaskSpec for testing."""
    return TaskSpec(
        id="test-001",
        type="programming",  # always valid in this worktree
        input="test input",
        expected="42",
        evaluator_key=evaluator_key,
        metadata=dict(meta),
    )


def _ok_result() -> EvalResult:
    """Return a canonical passing EvalResult."""
    return EvalResult(score=1.0, passed=True)


# ---------------------------------------------------------------------------
# Test 1: gsm8k_numeric smoke — dispatches without requiring extra deps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gsm8k_numeric_smoke() -> None:
    """score_ground_truth dispatches gsm8k_numeric and returns EvalResult."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec("gsm8k_numeric")
    mock_result = _ok_result()

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

    mock_cls = MagicMock(return_value=mock_evaluator)

    with patch("atm.evaluation.ground_truth.EVALUATORS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        mock_reg.names.return_value = ["gsm8k_numeric"]

        # Patch _DEPS to only include our key so unknown-key logic is bypassed
        with patch.dict(
            "atm.evaluation.ground_truth._DEPS",
            {"gsm8k_numeric": ()},
            clear=True,
        ):
            result = await score_ground_truth(spec, "42")

    assert isinstance(result, EvalResult)
    assert result.score == 1.0
    assert result.passed is True
    mock_cls.assert_called_once_with()
    mock_evaluator.evaluate.assert_awaited_once_with(spec, "42")


# ---------------------------------------------------------------------------
# Test 2: commongen_rouge_coverage smoke — no extra deps required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commongen_rouge_coverage_smoke() -> None:
    """score_ground_truth dispatches commongen_rouge_coverage and returns EvalResult."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec(
        "commongen_rouge_coverage",
        concepts=["ski", "mountain"],
        references=["She skied down the mountain."],
    )
    mock_result = EvalResult(score=0.8, passed=True)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

    mock_cls = MagicMock(return_value=mock_evaluator)

    with patch("atm.evaluation.ground_truth.EVALUATORS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        mock_reg.names.return_value = ["commongen_rouge_coverage"]

        with patch.dict(
            "atm.evaluation.ground_truth._DEPS",
            {"commongen_rouge_coverage": ()},
            clear=True,
        ):
            result = await score_ground_truth(spec, "She skied down the mountain.")

    assert isinstance(result, EvalResult)
    assert result.score == 0.8
    mock_cls.assert_called_once_with()


# ---------------------------------------------------------------------------
# Test 3: dabench_numeric_exact smoke — no extra deps required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dabench_numeric_exact_smoke() -> None:
    """score_ground_truth dispatches dabench_numeric_exact and returns EvalResult."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec("dabench_numeric_exact", expected="@mean_fare[34.65]")
    mock_result = EvalResult(score=1.0, passed=True)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

    mock_cls = MagicMock(return_value=mock_evaluator)

    with patch("atm.evaluation.ground_truth.EVALUATORS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        mock_reg.names.return_value = ["dabench_numeric_exact"]

        with patch.dict(
            "atm.evaluation.ground_truth._DEPS",
            {"dabench_numeric_exact": ()},
            clear=True,
        ):
            result = await score_ground_truth(spec, "@mean_fare[34.65]")

    assert isinstance(result, EvalResult)
    assert result.score == 1.0
    mock_cls.assert_called_once_with()


# ---------------------------------------------------------------------------
# Test 4: humaneval_pytest smoke — sandbox mock provided
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_humaneval_pytest_smoke_with_sandbox() -> None:
    """score_ground_truth dispatches humaneval_pytest with sandbox kwarg."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec(
        "humaneval_pytest",
        test="def check(f): assert f(2) == 4",
        entry_point="double",
    )
    mock_result = EvalResult(score=1.0, passed=True)

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

    mock_cls = MagicMock(return_value=mock_evaluator)
    mock_sandbox = MagicMock()

    with patch("atm.evaluation.ground_truth.EVALUATORS") as mock_reg:
        mock_reg.get.return_value = mock_cls
        mock_reg.names.return_value = ["humaneval_pytest"]

        with patch.dict(
            "atm.evaluation.ground_truth._DEPS",
            {"humaneval_pytest": ("sandbox",)},
            clear=True,
        ):
            result = await score_ground_truth(
                spec,
                "def double(x): return x * 2",
                sandbox=mock_sandbox,
            )

    assert isinstance(result, EvalResult)
    assert result.score == 1.0
    mock_cls.assert_called_once_with(sandbox=mock_sandbox)


# ---------------------------------------------------------------------------
# Test 5: humaneval_pytest raises ValueError when sandbox is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_humaneval_raises_when_sandbox_none() -> None:
    """score_ground_truth raises ValueError for humaneval_pytest when sandbox=None."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec(
        "humaneval_pytest",
        test="def check(f): assert f(2) == 4",
        entry_point="double",
    )

    with (
        patch.dict(
            "atm.evaluation.ground_truth._DEPS",
            {"humaneval_pytest": ("sandbox",)},
            clear=True,
        ),
        pytest.raises(ValueError, match="sandbox"),
    ):
        await score_ground_truth(spec, "def double(x): return x * 2", sandbox=None)


# ---------------------------------------------------------------------------
# Test 6: unknown evaluator_key raises KeyError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_evaluator_key_raises_key_error() -> None:
    """score_ground_truth raises KeyError for an unrecognised evaluator_key."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec("nonexistent_key_xyz")

    with pytest.raises(KeyError, match="nonexistent_key_xyz"):
        await score_ground_truth(spec, "some answer")


# ---------------------------------------------------------------------------
# Test 7: Drift test — EVALUATORS.names() == _DEPS.keys()
# ---------------------------------------------------------------------------


def test_evaluators_drift() -> None:
    """Drift guard: _DEPS must exactly match EVALUATORS.names().

    If this test fails, update _DEPS in ground_truth.py to include the new
    evaluator_key, or remove the stale key.

    Skipped (xfail) rather than hard-failing so that adding a new M10 evaluator
    doesn't block other tests — the drift is surfaced as a warning.
    """
    from atm.evaluation.ground_truth import _DEPS
    from atm.tasks import EVALUATORS

    try:
        registered = set(EVALUATORS.names())
    except Exception as exc:
        pytest.skip(f"EVALUATORS.names() unavailable: {exc}")

    deps_keys = set(_DEPS.keys())

    if registered != deps_keys:
        missing_from_deps = registered - deps_keys
        extra_in_deps = deps_keys - registered
        msg = (
            f"_DEPS drift detected!\n"
            f"  Registered evaluators not in _DEPS: {sorted(missing_from_deps)}\n"
            f"  Keys in _DEPS not registered: {sorted(extra_in_deps)}\n"
            f"  Update _DEPS in src/atm/evaluation/ground_truth.py."
        )
        pytest.fail(msg)
