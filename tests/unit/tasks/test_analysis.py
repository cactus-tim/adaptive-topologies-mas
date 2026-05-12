"""Unit tests for atm.tasks.analysis — AnalysisLoader and AnalysisHybridEvaluator.

Tests (TDD red phase — step 7.3):
1. loader_parses_yaml_minimum_5 — loader returns >= 5 TaskSpecs; raises ValueError if < 5
2. structural_checks_contains    — "contains" check passes/fails correctly
3. structural_checks_regex       — "regex" check passes/fails correctly
4. structural_checks_min_length  — "min_length" check passes/fails correctly
5. evaluator_hybrid_score        — score = 0.5*struct + 0.5*judge; struct=1.0, judge=0.7 → 0.85
6. evaluator_passed_threshold    — struct=0.0, all checks fail, judge=0.8 → score=0.4, passed=False
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from atm.core.types import TaskSpec
from atm.llm.fake import FakeLLM
from atm.tasks.analysis import AnalysisHybridEvaluator, AnalysisLoader

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "llm"
YAML_PATH = Path(__file__).parent.parent.parent.parent / "conf" / "tasks" / "analysis_prompts.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    structural_checks: list[dict[str, str]] | None = None,
    rubric: str = "Evaluate well.",
    input_text: str = "Analyze the data.",
    expected: str = "trend",
) -> TaskSpec:
    """Build a minimal TaskSpec for analysis evaluation tests."""
    return TaskSpec(
        id="analysis/test_spec",
        type="analysis",
        input=input_text,
        expected=expected,
        evaluator_key="analysis_hybrid",
        metadata={
            "rubric": rubric,
            "structural_checks": structural_checks or [],
        },
    )


def _make_fake_llm(fixture_name: str) -> FakeLLM:
    """Create a FakeLLM in scripted mode from a fixture file."""
    return FakeLLM(mode="scripted", fixture=FIXTURES_DIR / fixture_name)


# ---------------------------------------------------------------------------
# Test 1: loader parses YAML, minimum 5 (raise if < 5)
# ---------------------------------------------------------------------------


def test_loader_parses_yaml_returns_at_least_5() -> None:
    """AnalysisLoader.load() returns >= 5 TaskSpec entries from the YAML file."""
    loader = AnalysisLoader(yaml_path=YAML_PATH)
    specs = loader.load()

    assert len(specs) >= 5, f"Expected at least 5 prompts, got {len(specs)}"
    for spec in specs:
        assert spec.type == "analysis"
        assert spec.evaluator_key == "analysis_hybrid"
        assert spec.input  # non-empty
        assert "rubric" in spec.metadata
        assert "structural_checks" in spec.metadata


def test_loader_raises_if_fewer_than_5_prompts(tmp_path: Path) -> None:
    """AnalysisLoader.load() raises ValueError if the YAML has fewer than 5 prompts."""
    small_yaml = tmp_path / "small_analysis.yaml"
    data: dict[str, Any] = {
        "prompts": [
            {
                "id": f"analysis/p{i}",
                "prompt": f"Prompt {i}",
                "data_csv": "x,y\n1,2",
                "expected_pattern": "pattern",
                "rubric": "rubric text",
                "structural_checks": [],
            }
            for i in range(4)  # Only 4 entries — should raise
        ]
    }
    small_yaml.write_text(yaml.dump(data))

    loader = AnalysisLoader(yaml_path=small_yaml)
    with pytest.raises(ValueError, match="analysis: need at least 5 prompts"):
        loader.load()


# ---------------------------------------------------------------------------
# Test 2: structural_checks "contains"
# ---------------------------------------------------------------------------


def test_structural_check_contains_pass() -> None:
    """structural check type='contains' returns 1.0 when value is present in answer."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "contains", "value": "trend"}]
    answer = "The data shows an upward trend over time."

    score = evaluator._run_structural_checks(checks, answer)
    assert score == 1.0


def test_structural_check_contains_fail() -> None:
    """structural check type='contains' returns 0.0 when value is absent."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "contains", "value": "anomaly"}]
    answer = "The data shows a steady trend."

    score = evaluator._run_structural_checks(checks, answer)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Test 3: structural_checks "regex"
# ---------------------------------------------------------------------------


def test_structural_check_regex_pass() -> None:
    """structural check type='regex' returns 1.0 when pattern matches."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "regex", "value": r"(?i)(upward|increasing)"}]
    answer = "Sales are increasing month over month."

    score = evaluator._run_structural_checks(checks, answer)
    assert score == 1.0


def test_structural_check_regex_fail() -> None:
    """structural check type='regex' returns 0.0 when pattern does not match."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "regex", "value": r"\d{4}-Q[1-4]"}]
    answer = "The analysis shows a flat trend."

    score = evaluator._run_structural_checks(checks, answer)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Test 4: structural_checks "min_length"
# ---------------------------------------------------------------------------


def test_structural_check_min_length_pass() -> None:
    """structural check type='min_length' returns 1.0 when answer meets length threshold."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "min_length", "value": "10"}]
    answer = "The data shows an upward trend."  # > 10 chars

    score = evaluator._run_structural_checks(checks, answer)
    assert score == 1.0


def test_structural_check_min_length_fail() -> None:
    """structural check type='min_length' returns 0.0 when answer is too short."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "min_length", "value": "100"}]
    answer = "Short."

    score = evaluator._run_structural_checks(checks, answer)
    assert score == 0.0


def test_structural_check_empty_returns_1_0() -> None:
    """_run_structural_checks with empty list returns 1.0 (no checks = all pass)."""
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))
    score = evaluator._run_structural_checks([], "any answer")
    assert score == 1.0


# ---------------------------------------------------------------------------
# Test 5: evaluator hybrid score — struct=1.0, judge=0.7 → 0.85, passed=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_hybrid_score_pass() -> None:
    """Hybrid score = 0.5*struct + 0.5*judge; struct=1.0, judge_raw=7 → judge=0.7 → 0.85, passed=True.

    The fixture m10_judge_analysis.yaml returns {"score": 7, ...}
    which _invoke_judge normalises to 0.7.
    All structural checks (contains "trend") pass → struct=1.0.
    final = 0.5*1.0 + 0.5*0.7 = 0.85 >= 0.6 → passed=True.
    """
    fake_llm = _make_fake_llm("m10_judge_analysis.yaml")
    evaluator = AnalysisHybridEvaluator(judge_llm=fake_llm)

    spec = _make_spec(
        structural_checks=[{"type": "contains", "value": "trend"}],
        rubric="Identify trend direction.",
        input_text="Analyze the data.\n\nData:\nmonth,sales\nJan,100\nFeb,120",
    )
    answer = "The sales show an upward trend over time."

    result = await evaluator.evaluate(spec, answer)

    assert abs(result.score - 0.85) < 1e-9, f"Expected 0.85, got {result.score}"
    assert result.passed is True
    assert result.details["structural_score"] == 1.0
    assert abs(result.details["judge_score"] - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: evaluator — struct=0.0 (all checks fail), judge=0.8 → 0.4, passed=False
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_hybrid_score_fail_threshold() -> None:
    """struct=0.0 (all structural checks fail), judge=0.7 → final=0.35, passed=False.

    The fixture returns judge_raw=7 → judge=0.7.
    structural checks: contains "MISSING_VALUE" — will fail → struct=0.0.
    final = 0.5*0.0 + 0.5*0.7 = 0.35 < 0.6 → passed=False.

    Note: plan example used judge=0.8 but fixture gives 0.7; we use fixture value.
    The important invariant is passed=False when score < 0.6.
    """
    fake_llm = _make_fake_llm("m10_judge_analysis.yaml")
    evaluator = AnalysisHybridEvaluator(judge_llm=fake_llm)

    spec = _make_spec(
        structural_checks=[
            {"type": "contains", "value": "MISSING_VALUE_THAT_WILL_NEVER_APPEAR_XYZ"},
            {"type": "regex", "value": r"NOMATCH\d{99}"},
            {"type": "min_length", "value": "99999"},
        ],
        rubric="Evaluate thoroughly.",
        input_text="Analyze the data.\n\nData:\nmonth,sales\nJan,100",
    )
    answer = "Short."

    result = await evaluator.evaluate(spec, answer)

    assert result.details["structural_score"] == 0.0
    assert result.passed is False
    # score must be < 0.6 since struct=0.0 and judge<=1.0 → max possible = 0.5 < 0.6
    assert result.score < 0.6


# ---------------------------------------------------------------------------
# Test 7: malformed regex does not raise — treated as failed check
# ---------------------------------------------------------------------------


def test_structural_check_malformed_regex_does_not_raise() -> None:
    """_run_structural_checks with an invalid regex pattern does not raise; check is failed (0.0).

    An unclosed bracket '[' is an invalid regex pattern that would normally
    raise re.error.  The evaluator must catch this and treat the check as
    failed (score contribution = 0.0) rather than propagating the exception.
    """
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "regex", "value": "["}]  # invalid regex: unclosed bracket
    answer = "Some answer text."

    # Must not raise any exception
    score = evaluator._run_structural_checks(checks, answer)

    # Malformed regex check is treated as failed
    assert score == 0.0


def test_structural_check_invalid_min_length_does_not_raise() -> None:
    """_run_structural_checks with a non-integer min_length value does not raise.

    An non-parseable value like "abc" would normally raise ValueError from int().
    The evaluator must catch this and treat the check as failed (score = 0.0).
    """
    evaluator = AnalysisHybridEvaluator(judge_llm=_make_fake_llm("m10_judge_analysis.yaml"))

    checks = [{"type": "min_length", "value": "not_a_number"}]
    answer = "Some answer text."

    # Must not raise any exception
    score = evaluator._run_structural_checks(checks, answer)

    # Invalid min_length check is treated as failed
    assert score == 0.0
