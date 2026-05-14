"""Unit tests for atm.tasks.humaneval — HumanEvalLoader and HumanEvalEvaluator.

Tests (TDD red phase — 4.2):
1. loader_from_fixture — loader reads rows from fixture (datasets.load_dataset mocked)
2. cache_hit_no_network — second load() call uses cache, never calls network again
3. evaluator_canonical_solution — canonical answer → passed=True, score=1.0
4. evaluator_empty_answer — empty string → passed=False, score=0.0
5. evaluator_syntax_error — SyntaxError code → passed=False, error contains "SyntaxError"
6. strip_code_fences — removes ```python...```, ```...```, plain text unchanged
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from atm.core.types import TaskSpec
from atm.tasks.humaneval import (
    HumanEvalEvaluator,
    HumanEvalLoader,
    _strip_code_fences,
)

# ---------------------------------------------------------------------------
# Fixture path
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "tasks" / "humaneval_sample.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture_rows() -> list[dict[str, Any]]:
    """Load the 2-row JSONL fixture into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with _FIXTURE_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_mock_hf_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a MagicMock that mimics datasets.load_dataset(...)[\"test\"]."""
    mock_split = MagicMock()
    mock_split.__iter__ = MagicMock(return_value=iter(rows))
    mock_dataset = MagicMock()
    mock_dataset.__getitem__ = MagicMock(side_effect=lambda key: mock_split)
    return mock_dataset


# ---------------------------------------------------------------------------
# Test 1: loader_from_fixture
# ---------------------------------------------------------------------------


def test_loader_from_fixture(tmp_path: Path) -> None:
    """Loader returns TaskSpec list from mocked datasets.load_dataset."""
    rows = _load_fixture_rows()
    mock_dataset = _make_mock_hf_dataset(rows)

    with patch("atm.tasks.humaneval.datasets.load_dataset", return_value=mock_dataset):
        loader = HumanEvalLoader()
        specs = loader.load(cache_dir=tmp_path)

    assert len(specs) == 2
    ids = {s.id for s in specs}
    assert "HumanEval/0" in ids
    assert "HumanEval/1" in ids

    spec0 = next(s for s in specs if s.id == "HumanEval/0")
    assert spec0.type == "programming"
    assert spec0.evaluator_key == "humaneval_pytest"
    assert "has_close_elements" in spec0.input
    assert spec0.metadata["entry_point"] == "has_close_elements"
    assert "def check" in spec0.metadata["test"]


# ---------------------------------------------------------------------------
# Test 2: cache_hit_no_network
# ---------------------------------------------------------------------------


def test_cache_hit_no_network(tmp_path: Path) -> None:
    """Second load() call reads from Parquet cache — datasets.load_dataset not called again."""
    rows = _load_fixture_rows()
    mock_dataset = _make_mock_hf_dataset(rows)

    call_count = 0

    def counting_load_dataset(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return mock_dataset

    with patch("atm.tasks.humaneval.datasets.load_dataset", side_effect=counting_load_dataset):
        loader = HumanEvalLoader()
        specs_first = loader.load(cache_dir=tmp_path)
        specs_second = loader.load(cache_dir=tmp_path)

    assert call_count == 1, f"Expected 1 network call, got {call_count}"
    assert len(specs_first) == len(specs_second)


# ---------------------------------------------------------------------------
# Test 3: evaluator_canonical_solution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_canonical_solution() -> None:
    """Canonical solution yields passed=True and score=1.0."""
    rows = _load_fixture_rows()
    row = rows[0]

    spec = TaskSpec(
        id=row["task_id"],
        type="programming",
        input=row["prompt"],
        expected=row["canonical_solution"],
        evaluator_key="humaneval_pytest",
        metadata={"test": row["test"], "entry_point": row["entry_point"]},
    )

    evaluator = HumanEvalEvaluator()
    result = await evaluator.evaluate(spec, row["canonical_solution"])

    assert result.passed is True
    assert result.score == 1.0
    assert result.error is None


# ---------------------------------------------------------------------------
# Test 4: evaluator_empty_answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_empty_answer() -> None:
    """Empty answer yields passed=False and score=0.0."""
    rows = _load_fixture_rows()
    row = rows[0]

    spec = TaskSpec(
        id=row["task_id"],
        type="programming",
        input=row["prompt"],
        expected=row["canonical_solution"],
        evaluator_key="humaneval_pytest",
        metadata={"test": row["test"], "entry_point": row["entry_point"]},
    )

    evaluator = HumanEvalEvaluator()
    result = await evaluator.evaluate(spec, "")

    assert result.passed is False
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Test 5: evaluator_syntax_error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_syntax_error() -> None:
    """SyntaxError code yields passed=False and error containing 'SyntaxError'."""
    rows = _load_fixture_rows()
    row = rows[0]

    spec = TaskSpec(
        id=row["task_id"],
        type="programming",
        input=row["prompt"],
        expected=row["canonical_solution"],
        evaluator_key="humaneval_pytest",
        metadata={"test": row["test"], "entry_point": row["entry_point"]},
    )

    bad_code = "    def oops(\n    return broken syntax !!!\n"

    evaluator = HumanEvalEvaluator()
    result = await evaluator.evaluate(spec, bad_code)

    assert result.passed is False
    assert result.score == 0.0
    assert result.error is not None
    assert "SyntaxError" in result.error or "SyntaxError" in result.details.get("stderr", "")


# ---------------------------------------------------------------------------
# Test 6: _strip_code_fences
# ---------------------------------------------------------------------------


def test_strip_code_fences() -> None:
    """_strip_code_fences removes markdown fences and leaves plain text unchanged."""
    # ```python ... ```
    fenced_python = "```python\nprint('hello')\n```"
    assert _strip_code_fences(fenced_python) == "print('hello')"

    # ``` ... ```  (no language tag)
    fenced_plain = "```\nsome code\n```"
    assert _strip_code_fences(fenced_plain) == "some code"

    # Plain text — returned unchanged
    plain = "    return x + 1"
    assert _strip_code_fences(plain) == plain

    # Only opening fence with no closing — returned unchanged
    only_open = "```python\nsome code"
    assert _strip_code_fences(only_open) == only_open
