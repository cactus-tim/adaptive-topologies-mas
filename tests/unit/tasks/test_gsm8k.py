"""Unit tests for atm.tasks.gsm8k — GSM8KLoader and GSM8KMatcher.

Tests (TDD — 7 tests):
1. test_gsm8k_loader_via_dataset  — monkeypatch load_dataset, loader returns ≥3 TaskSpec
   with correct type/evaluator_key/id format
2. test_gsm8k_cache_hit           — write parquet to cache_dir first, assert no network call
3. test_gsm8k_numeric_exact_match — "#### 42" in expected + answer ending "42" → passed=True
4. test_gsm8k_comma_formatted     — "#### 1234" vs "1,234" (strip commas) → passed=True
5. test_gsm8k_float_tolerance     — "#### 72" vs "72.0" → passed=True via math.isclose
6. test_gsm8k_wrong_answer        — wrong number → passed=False, score=0.0
7. test_gsm8k_no_number_in_answer — no numeric token → passed=False, error="no number found"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from atm.core.types import TaskSpec
from atm.tasks.gsm8k import GSM8KLoader, GSM8KMatcher

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "tasks" / "gsm8k_sample.json"


def _load_fixture_rows() -> list[dict[str, Any]]:
    """Load the GSM8K JSON fixture into a list of dicts."""
    with _FIXTURE_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _make_mock_hf_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a MagicMock mimicking datasets.load_dataset(...) iterable split."""
    mock_split = MagicMock()
    mock_split.__iter__ = MagicMock(return_value=iter(rows))
    return mock_split


def _make_spec_with_expected(expected_number: str) -> TaskSpec:
    """Build a minimal GSM8K TaskSpec with the given expected number string."""
    return TaskSpec(
        id="gsm8k/0",
        type="reasoning",
        input="What is 2+2?",
        expected=expected_number,
        evaluator_key="gsm8k_numeric",
    )


# ---------------------------------------------------------------------------
# Test 1: loader_via_dataset
# ---------------------------------------------------------------------------


def test_gsm8k_loader_via_dataset(tmp_path: Path) -> None:
    """Loader returns ≥3 TaskSpec from mocked datasets.load_dataset."""
    rows = _load_fixture_rows()
    mock_split = _make_mock_hf_dataset(rows)

    with patch("atm.tasks.gsm8k.datasets.load_dataset", return_value=mock_split):
        loader = GSM8KLoader()
        specs = loader.load(cache_dir=tmp_path)

    assert len(specs) >= 3, f"Expected ≥3 specs, got {len(specs)}"

    for spec in specs:
        assert spec.type == "reasoning", f"Expected type='reasoning', got {spec.type!r}"
        assert spec.evaluator_key == "gsm8k_numeric", (
            f"Expected evaluator_key='gsm8k_numeric', got {spec.evaluator_key!r}"
        )
        assert spec.id.startswith("gsm8k/"), f"id should start with 'gsm8k/', got {spec.id!r}"

    # All ids should be unique and in the format "gsm8k/<idx>"
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids)), "Duplicate ids found"


# ---------------------------------------------------------------------------
# Test 2: cache_hit
# ---------------------------------------------------------------------------


def test_gsm8k_cache_hit(tmp_path: Path) -> None:
    """Second load() call reads from Parquet cache — datasets.load_dataset not called."""
    rows = _load_fixture_rows()
    mock_split = _make_mock_hf_dataset(rows)

    call_count = 0

    def counting_load_dataset(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return mock_split

    with patch("atm.tasks.gsm8k.datasets.load_dataset", side_effect=counting_load_dataset):
        loader = GSM8KLoader()
        specs_first = loader.load(cache_dir=tmp_path)
        # Reset iterator for potential second call
        mock_split.__iter__ = MagicMock(return_value=iter(rows))
        specs_second = loader.load(cache_dir=tmp_path)

    assert call_count == 1, f"Expected 1 network call, got {call_count}"
    assert len(specs_first) == len(specs_second)


# ---------------------------------------------------------------------------
# Test 3: numeric_exact_match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gsm8k_numeric_exact_match() -> None:
    """Answer ending with '42' matches expected '#### 42' → passed=True, score=1.0."""
    spec = _make_spec_with_expected("42")
    matcher = GSM8KMatcher()
    result = await matcher.evaluate(spec, "The answer is 42")

    assert result.passed is True
    assert result.score == 1.0
    assert result.error is None


# ---------------------------------------------------------------------------
# Test 4: comma_formatted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gsm8k_comma_formatted() -> None:
    """Comma-formatted numbers are normalised before comparison: 1,234 == 1234."""
    spec = _make_spec_with_expected("1234")
    matcher = GSM8KMatcher()
    result = await matcher.evaluate(spec, "The total is 1,234 items.")

    assert result.passed is True, (
        f"Expected passed=True for comma-formatted answer, got passed={result.passed}"
    )
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Test 5: float_tolerance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gsm8k_float_tolerance() -> None:
    """Integer expected vs float answer: 72 == 72.0 via math.isclose(abs_tol=1e-6)."""
    spec = _make_spec_with_expected("72")
    matcher = GSM8KMatcher()
    result = await matcher.evaluate(spec, "The answer is 72.0")

    assert result.passed is True, (
        f"Expected passed=True for float-tolerance, got passed={result.passed}"
    )
    assert result.score == 1.0


# ---------------------------------------------------------------------------
# Test 6: wrong_answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gsm8k_wrong_answer() -> None:
    """Wrong numeric answer → passed=False, score=0.0."""
    spec = _make_spec_with_expected("42")
    matcher = GSM8KMatcher()
    result = await matcher.evaluate(spec, "I think the answer is 99")

    assert result.passed is False
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# Test 7: no_number_in_answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gsm8k_no_number_in_answer() -> None:
    """No numeric token in answer → passed=False, error contains 'no number found'."""
    spec = _make_spec_with_expected("42")
    matcher = GSM8KMatcher()
    result = await matcher.evaluate(spec, "I have no idea what the answer is.")

    assert result.passed is False
    assert result.score == 0.0
    assert result.error is not None
    assert "no number found" in result.error.lower()
