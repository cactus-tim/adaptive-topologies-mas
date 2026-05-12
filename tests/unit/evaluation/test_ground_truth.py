"""Unit tests for atm.evaluation.ground_truth.score_ground_truth.

Five tests covering:
1. mmlu_exact_match — dispatches to MMLUEvaluator; correct answer returns score=1.0
2. humaneval_pytest — raises ValueError when sandbox=None
3. creative_judge  — raises ValueError when judge_llm=None
4. analysis_hybrid — raises ValueError when judge_llm=None
5. unknown key     — raises KeyError with message
"""

from __future__ import annotations

from typing import Any

import pytest

from atm.core.types import TaskSpec
from atm.tasks.base import EvalResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(evaluator_key: str, **meta: Any) -> TaskSpec:
    return TaskSpec(
        id="test-001",
        type="qa",
        input="What is 2+2?",
        expected="A",
        evaluator_key=evaluator_key,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Test 1: mmlu_exact_match dispatches correctly; no sandbox or judge required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mmlu_exact_match_correct_answer() -> None:
    """score_ground_truth with mmlu_exact_match returns score=1.0 for correct answer."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = TaskSpec(
        id="mmlu-001",
        type="qa",
        input="Which is the answer?",
        expected="A",
        evaluator_key="mmlu_exact_match",
        metadata={},
    )

    result = await score_ground_truth(spec, "The answer is (A).")

    assert isinstance(result, EvalResult)
    assert result.score == 1.0
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 2: humaneval_pytest raises ValueError when sandbox is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_humaneval_raises_when_sandbox_none() -> None:
    """score_ground_truth raises ValueError for humaneval_pytest when sandbox=None."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec(
        "humaneval_pytest",
        test="def check(f): assert f(2)==4",
        entry_point="double",
    )

    with pytest.raises(ValueError, match="sandbox"):
        await score_ground_truth(spec, "def double(x): return x * 2", sandbox=None)


# ---------------------------------------------------------------------------
# Test 3: creative_judge raises ValueError when judge_llm is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creative_judge_raises_when_judge_none() -> None:
    """score_ground_truth raises ValueError for creative_judge when judge_llm=None."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = TaskSpec(
        id="creative-001",
        type="creative",
        input="Write a poem.",
        expected=None,
        evaluator_key="creative_judge",
        metadata={"rubric": "Evaluate for creativity."},
    )

    with pytest.raises(ValueError, match="judge_llm"):
        await score_ground_truth(spec, "Roses are red.", judge_llm=None)


# ---------------------------------------------------------------------------
# Test 4: analysis_hybrid raises ValueError when judge_llm is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analysis_hybrid_raises_when_judge_none() -> None:
    """score_ground_truth raises ValueError for analysis_hybrid when judge_llm=None."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = TaskSpec(
        id="analysis-001",
        type="analysis",
        input="Analyse the data.",
        expected="mean",
        evaluator_key="analysis_hybrid",
        metadata={"rubric": "Check quality.", "structural_checks": []},
    )

    with pytest.raises(ValueError, match="judge_llm"):
        await score_ground_truth(spec, "The mean is 5.", judge_llm=None)


# ---------------------------------------------------------------------------
# Test 5: unknown evaluator_key raises KeyError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_evaluator_key_raises_key_error() -> None:
    """score_ground_truth raises KeyError for an unrecognised evaluator_key."""
    from atm.evaluation.ground_truth import score_ground_truth

    spec = _make_spec("nonexistent_key_xyz")

    with pytest.raises(KeyError, match="nonexistent_key_xyz"):
        await score_ground_truth(spec, "some answer")
