"""Unit tests for atm.tasks.mmlu — MMLULoader and MMLUEvaluator.

Tests (TDD red phase — step 5.2):
1. loader: loads from fixture via mocked datasets.load_dataset with limit=N
2. cache-hit: repeated load() does not call the network (datasets.load_dataset)
3. evaluator: answer="A" expected="A" → passed=True
4. evaluator: answer="The answer is B." expected="B" → passed=True (regex extract)
5. evaluator: answer="C" expected="A" → passed=False
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from atm.core.types import TaskSpec
from atm.tasks.mmlu import MMLUEvaluator, MMLULoader

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "tasks" / "mmlu_sample.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture() -> list[dict[str, Any]]:
    with FIXTURE_PATH.open() as f:
        return json.load(f)


def _make_fake_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a list-like object that simulates a HuggingFace Dataset slice."""
    # HF Dataset supports iteration and indexing; a plain list is sufficient for mocking.
    return rows


# ---------------------------------------------------------------------------
# Test 1: loader loads from fixture via mocked datasets.load_dataset
# ---------------------------------------------------------------------------


def test_mmlu_loader_loads_via_dataset(tmp_path: Path) -> None:
    """MMLULoader.load() calls datasets.load_dataset and converts rows to TaskSpec."""
    rows = _load_fixture()
    fake_dataset = _make_fake_dataset(rows)

    with patch("atm.tasks.mmlu.datasets.load_dataset", return_value=fake_dataset) as mock_ld:
        loader = MMLULoader(limit=3)
        specs = loader.load(cache_dir=tmp_path)

    mock_ld.assert_called_once_with("TIGER-Lab/MMLU-Pro", split="test[:3]")

    assert len(specs) == 3
    # Verify structure of first spec
    spec = specs[0]
    assert isinstance(spec, TaskSpec)
    assert spec.id == "mmlu_pro_0001"
    assert spec.type == "qa"
    assert spec.evaluator_key == "mmlu_exact_match"
    assert spec.expected == "B"
    assert "A." in spec.input  # formatted options
    assert spec.metadata["category"] == "math"
    assert spec.metadata["answer_index"] == 1
    assert len(spec.metadata["options"]) == 10


# ---------------------------------------------------------------------------
# Test 2: cache-hit — repeated load does not call datasets.load_dataset
# ---------------------------------------------------------------------------


def test_mmlu_loader_cache_hit(tmp_path: Path) -> None:
    """Second call to MMLULoader.load() reads from cache; network is NOT called."""
    rows = _load_fixture()
    fake_dataset = _make_fake_dataset(rows)

    with patch("atm.tasks.mmlu.datasets.load_dataset", return_value=fake_dataset) as mock_ld:
        loader = MMLULoader(limit=3)
        specs1 = loader.load(cache_dir=tmp_path)
        # Second load — should read from parquet cache
        specs2 = loader.load(cache_dir=tmp_path)

    # datasets.load_dataset was called exactly once (first load only)
    assert mock_ld.call_count == 1
    assert len(specs1) == len(specs2) == 3
    assert [s.id for s in specs1] == [s.id for s in specs2]


# ---------------------------------------------------------------------------
# Test 3: evaluator — exact match passes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mmlu_evaluator_exact_match_pass() -> None:
    """MMLUEvaluator: answer='A' expected='A' → passed=True, score=1.0."""
    spec = TaskSpec(
        id="mmlu/test_1",
        type="qa",
        input="Question?\n\nA. opt1\nB. opt2",
        expected="A",
        evaluator_key="mmlu_exact_match",
        metadata={"category": "math", "answer_index": 0, "options": ["opt1", "opt2"]},
    )
    evaluator = MMLUEvaluator()
    result = await evaluator.evaluate(spec, "A")

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["normalised"] == "A"
    assert result.error is None


# ---------------------------------------------------------------------------
# Test 4: evaluator — regex extracts letter from prose
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mmlu_evaluator_regex_extraction_pass() -> None:
    """MMLUEvaluator: answer='The answer is B.' expected='B' → passed=True."""
    spec = TaskSpec(
        id="mmlu/test_2",
        type="qa",
        input="Question?\n\nA. opt1\nB. opt2",
        expected="B",
        evaluator_key="mmlu_exact_match",
        metadata={"category": "chemistry", "answer_index": 1, "options": ["opt1", "opt2"]},
    )
    evaluator = MMLUEvaluator()
    result = await evaluator.evaluate(spec, "The answer is B.")

    assert result.passed is True
    assert result.score == 1.0
    assert result.details["normalised"] == "B"
    assert result.error is None


# ---------------------------------------------------------------------------
# Test 5: evaluator — wrong answer fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mmlu_evaluator_wrong_answer_fail() -> None:
    """MMLUEvaluator: answer='C' expected='A' → passed=False, score=0.0."""
    spec = TaskSpec(
        id="mmlu/test_3",
        type="qa",
        input="Question?\n\nA. opt1\nB. opt2\nC. opt3",
        expected="A",
        evaluator_key="mmlu_exact_match",
        metadata={"category": "literature", "answer_index": 0, "options": ["opt1", "opt2", "opt3"]},
    )
    evaluator = MMLUEvaluator()
    result = await evaluator.evaluate(spec, "C")

    assert result.passed is False
    assert result.score == 0.0
    assert result.details["normalised"] == "C"
    assert result.error is None
