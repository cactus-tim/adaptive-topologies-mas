"""Unit tests for atm.evaluation.judges — 10 tests."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from atm.evaluation.judges import (
    PairwiseJudge,
    PairwiseResult,
    RubricJudge,
    SelfConsistentJudge,
    _to_answer_relative,
)
from atm.llm.fake import FakeLLM

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "llm"
_PAIRWISE_AB = _FIXTURES / "m11_judge_pairwise_ab.yaml"
_PAIRWISE_DISAGREE = _FIXTURES / "m11_judge_pairwise_swap_disagree.yaml"
_SELF_CONSISTENCY = _FIXTURES / "m11_judge_self_consistency.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC_INPUT = "Explain the concept of recursion."
_RUBRIC = "Correctness\nClarity\nExamples"
_ANSWER_A = "Recursion is a function that calls itself."
_ANSWER_B = "Recursion is a loop."


# ---------------------------------------------------------------------------
# _to_answer_relative (4 tests)
# ---------------------------------------------------------------------------


def test_to_answer_relative_not_swapped_a() -> None:
    """Slot A win in AB order → answer_a wins."""
    assert _to_answer_relative("A", swapped=False) == "a"


def test_to_answer_relative_not_swapped_b() -> None:
    """Slot B win in AB order → answer_b wins."""
    assert _to_answer_relative("B", swapped=False) == "b"


def test_to_answer_relative_swapped_a() -> None:
    """Slot A win in BA order → slot A holds answer_b → answer_b wins."""
    assert _to_answer_relative("A", swapped=True) == "b"


def test_to_answer_relative_tie() -> None:
    """Tie always produces tie regardless of swap."""
    assert _to_answer_relative("tie", swapped=False) == "tie"
    assert _to_answer_relative("tie", swapped=True) == "tie"


# ---------------------------------------------------------------------------
# RubricJudge (2 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rubric_judge_score_ok() -> None:
    """RubricJudge returns normalised score in [0,1] from scripted LLM."""
    # Use echo mode: returns the prompt itself, which is not valid JSON → score=0.
    # Use scripted mode for a controlled response.
    llm = FakeLLM(
        mode="scripted",
        fixture=str(_SELF_CONSISTENCY),
    )
    judge = RubricJudge()
    score, reasoning, error = await judge.score(_SPEC_INPUT, _RUBRIC, _ANSWER_A, judge_llm=llm)
    # Fixture step 0 returns score=8 → normalised 0.8
    assert error is None
    assert math.isclose(score, 0.8, rel_tol=1e-6)
    assert "Strong answer" in reasoning


@pytest.mark.asyncio
async def test_rubric_judge_error_on_bad_response() -> None:
    """RubricJudge returns non-None error when LLM returns unparseable text."""
    llm = FakeLLM(mode="echo")
    judge = RubricJudge()
    score, _reasoning, error = await judge.score(_SPEC_INPUT, _RUBRIC, _ANSWER_A, judge_llm=llm)
    assert error is not None
    assert score == 0.0


# ---------------------------------------------------------------------------
# PairwiseJudge (3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pairwise_judge_swap_consistent_a_wins() -> None:
    """Swap-consistent case: AB→A, BA→B (answer_a wins both) → winner=a."""
    llm = FakeLLM(mode="scripted", fixture=str(_PAIRWISE_AB))
    judge = PairwiseJudge()
    result = await judge.compare(_SPEC_INPUT, _RUBRIC, _ANSWER_A, _ANSWER_B, judge_llm=llm)
    assert isinstance(result, PairwiseResult)
    assert result.swap_consistent is True
    assert result.winner == "a"
    assert result.reason_ab != ""
    assert result.reason_ba != ""


@pytest.mark.asyncio
async def test_pairwise_judge_swap_disagree_gives_tie() -> None:
    """Swap-disagree scenario: AB→A, BA→A (judge prefers slot A always).

    Answer-relative: AB says 'a' wins, BA says 'b' wins → disagree → tie.
    """
    llm = FakeLLM(mode="scripted", fixture=str(_PAIRWISE_DISAGREE))
    judge = PairwiseJudge()
    result = await judge.compare(_SPEC_INPUT, _RUBRIC, _ANSWER_A, _ANSWER_B, judge_llm=llm)
    assert result.swap_consistent is False
    assert result.winner == "tie"


@pytest.mark.asyncio
async def test_pairwise_result_is_frozen() -> None:
    """PairwiseResult is immutable (frozen Pydantic model)."""
    result = PairwiseResult(
        winner="tie",
        swap_consistent=False,
        reason_ab="ab reason",
        reason_ba="ba reason",
    )
    with pytest.raises(ValidationError):
        result.winner = "a"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SelfConsistentJudge (1 test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_consistent_judge_mean_score() -> None:
    """SelfConsistentJudge returns mean of 3 normalised scores.

    Fixture scores: 8, 6, 7 → normalised 0.8, 0.6, 0.7 → mean = 0.7.
    """
    llm = FakeLLM(mode="scripted", fixture=str(_SELF_CONSISTENCY))
    judge = SelfConsistentJudge()
    mean_score = await judge.score(_SPEC_INPUT, _RUBRIC, _ANSWER_A, judge_llm=llm, run_seed=42, n=3)
    assert math.isclose(mean_score, 0.7, rel_tol=1e-6)
