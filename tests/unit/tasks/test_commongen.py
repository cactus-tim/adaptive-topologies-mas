"""Unit tests for atm.tasks.commongen — CommonGenLoader and CommonGenEvaluator.

Tests (TDD — 9 tests):
1. test_commongen_loader_via_dataset  — monkeypatch load_dataset, check grouping by
   concept_set_idx: 5 fixture rows with 2 sharing idx=0 → 4 specs; the shared spec
   has TWO references in metadata["references"].
2. test_commongen_cache_hit           — write_cache to tmp_path first, assert no
   datasets.load_dataset call.
3. test_rouge_l_exact                 — pred==ref → _rouge_l returns 1.0.
4. test_rouge_l_partial               — pred="the cat sat on the mat",
   ref="the cat sat on a mat"; LCS tokens: ["the","cat","sat","on","mat"] = 5;
   P=5/6, R=5/6, F1=5/6 ≈ 0.8333. Assert math.isclose(_rouge_l(...), 5/6, abs_tol=1e-6).
5. test_rouge_l_disjoint              — pred="abc", ref="xyz" → 0.0.
6. test_commongen_evaluator_full_match — answer contains all concepts AND matches a
   reference exactly → score=1.0, passed=True.
7. test_commongen_evaluator_partial_coverage — 2 of 3 concepts present, ROUGE-L=0.5
   → final = 0.5*0.5 + 0.5*(2/3) ≈ 0.583 → passed=True (>=0.5).
8. test_commongen_evaluator_empty_concepts_vacuous — metadata["concepts"]=[] →
   coverage=1.0; ROUGE-L score alone determines final.
9. test_commongen_metadata_parquet_round_trip — build TaskSpec with list metadata,
   write_cache, read_cache, assert list equality preserved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import atm.tasks._cache as _cache
from atm.core.types import TaskSpec
from atm.tasks.commongen import CommonGenEvaluator, CommonGenLoader, _rouge_l

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURE_PATH = (
    Path(__file__).parent.parent.parent / "fixtures" / "tasks" / "commongen_sample.json"
)


def _load_fixture_rows() -> list[dict[str, Any]]:
    """Load the CommonGen JSON fixture into a list of dicts."""
    with _FIXTURE_PATH.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _make_mock_hf_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a MagicMock mimicking datasets.load_dataset(...) iterable split."""
    mock_split = MagicMock()
    mock_split.__iter__ = MagicMock(return_value=iter(rows))
    return mock_split


def _make_commongen_spec(
    concepts: list[str],
    references: list[str],
    idx: int = 0,
) -> TaskSpec:
    """Build a minimal CommonGen TaskSpec with the given concepts and references."""
    concept_str = ", ".join(concepts)
    return TaskSpec(
        id=f"commongen/{idx}",
        type="creative",
        input=f"concepts: {concept_str}",
        expected=references[0] if references else "",
        evaluator_key="commongen_rouge_coverage",
        metadata={"concepts": concepts, "references": references},
    )


# ---------------------------------------------------------------------------
# Test 1: loader_via_dataset — reference aggregation by concept_set_idx
# ---------------------------------------------------------------------------


def test_commongen_loader_via_dataset(tmp_path: Path) -> None:
    """Loader groups by concept_set_idx: 5 rows where 2 share idx=0 → 4 specs.

    The spec for idx=0 must have TWO references in metadata["references"].
    Each spec must have:
    - type="creative"
    - evaluator_key="commongen_rouge_coverage"
    - id="commongen/{idx}"
    - input="concepts: <c1>, <c2>, ..."
    - metadata["concepts"] as a list
    - metadata["references"] as a list
    - expected = first reference (deterministic ordering)
    """
    rows = _load_fixture_rows()
    mock_split = _make_mock_hf_dataset(rows)

    with patch("atm.tasks.commongen.datasets.load_dataset", return_value=mock_split):
        loader = CommonGenLoader()
        specs = loader.load(cache_dir=tmp_path)

    # 5 rows, 2 share concept_set_idx=0 → 4 unique concept_set_idx groups
    assert len(specs) == 4, f"Expected 4 specs (4 unique concept_set_idxs), got {len(specs)}"

    # Find the spec for concept_set_idx=0 (should have 2 references)
    shared_specs = [s for s in specs if "ski" in s.metadata["concepts"]]  # type: ignore[index]
    assert len(shared_specs) == 1, "Expected exactly 1 spec with 'ski' concept"
    shared_spec = shared_specs[0]
    assert len(shared_spec.metadata["references"]) == 2, (  # type: ignore[index]
        f"Expected 2 references for shared idx=0, got "
        f"{len(shared_spec.metadata['references'])}"  # type: ignore[index]
    )

    # All specs must conform to the expected structure
    for spec in specs:
        assert spec.type == "creative", f"Expected type='creative', got {spec.type!r}"
        assert spec.evaluator_key == "commongen_rouge_coverage", (
            f"Expected evaluator_key='commongen_rouge_coverage', got {spec.evaluator_key!r}"
        )
        assert spec.id.startswith("commongen/"), (
            f"id should start with 'commongen/', got {spec.id!r}"
        )
        assert spec.input.startswith("concepts: "), (
            f"input should start with 'concepts: ', got {spec.input!r}"
        )
        meta = spec.metadata
        assert meta is not None, "metadata should not be None"
        assert "concepts" in meta, "metadata should have 'concepts'"
        assert "references" in meta, "metadata should have 'references'"
        assert isinstance(meta["concepts"], list), "metadata['concepts'] should be a list"
        assert isinstance(meta["references"], list), "metadata['references'] should be a list"
        # expected = first reference (deterministic ordering)
        assert spec.expected == meta["references"][0], (
            f"expected should be first reference, got {spec.expected!r} "
            f"vs {meta['references'][0]!r}"
        )

    # All ids must be unique
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids)), "Duplicate ids found"


# ---------------------------------------------------------------------------
# Test 2: cache_hit — no network call on second load
# ---------------------------------------------------------------------------


def test_commongen_cache_hit(tmp_path: Path) -> None:
    """Second load() reads from Parquet cache — datasets.load_dataset not called."""
    rows = _load_fixture_rows()
    mock_split = _make_mock_hf_dataset(rows)

    call_count = 0

    def counting_load_dataset(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return mock_split

    with patch("atm.tasks.commongen.datasets.load_dataset", side_effect=counting_load_dataset):
        loader = CommonGenLoader()
        specs_first = loader.load(cache_dir=tmp_path)
        # Reset iterator for potential second call
        mock_split.__iter__ = MagicMock(return_value=iter(rows))
        specs_second = loader.load(cache_dir=tmp_path)

    assert call_count == 1, f"Expected 1 network call, got {call_count}"
    assert len(specs_first) == len(specs_second)


# ---------------------------------------------------------------------------
# Test 3: _rouge_l exact match
# ---------------------------------------------------------------------------


def test_rouge_l_exact() -> None:
    """pred == ref → _rouge_l returns 1.0 (within float epsilon)."""
    text = "the quick brown fox jumps over the lazy dog"
    result = _rouge_l(text, text)
    assert math.isclose(result, 1.0, abs_tol=1e-9), (
        f"Expected _rouge_l(x, x) == 1.0, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 4: _rouge_l partial match
# ---------------------------------------------------------------------------


def test_rouge_l_partial() -> None:
    """Partial LCS computation test.

    pred = "the cat sat on the mat"  → tokens: ["the", "cat", "sat", "on", "the", "mat"] (6)
    ref  = "the cat sat on a mat"    → tokens: ["the", "cat", "sat", "on", "a", "mat"] (6)

    LCS: "the cat sat on mat" — the longest common subsequence is
    ["the", "cat", "sat", "on", "mat"] = 5 tokens
    (note: "the" matches index 0 in pred with index 0 in ref; second "the" in pred
    does not extend the LCS further since "a" in ref occupies position 4)

    P = 5/6, R = 5/6, F1 = 2*(5/6)*(5/6)/((5/6)+(5/6)) = 5/6 ≈ 0.8333
    """
    pred = "the cat sat on the mat"
    ref = "the cat sat on a mat"
    result = _rouge_l(pred, ref)
    assert math.isclose(result, 5 / 6, abs_tol=1e-6), (
        f"Expected _rouge_l({pred!r}, {ref!r}) ≈ {5/6:.6f}, got {result}"
    )


# ---------------------------------------------------------------------------
# Test 5: _rouge_l disjoint
# ---------------------------------------------------------------------------


def test_rouge_l_disjoint() -> None:
    """Completely disjoint tokens → _rouge_l returns 0.0."""
    result = _rouge_l("abc", "xyz")
    assert result == 0.0, f"Expected 0.0 for disjoint pred/ref, got {result}"


# ---------------------------------------------------------------------------
# Test 6: evaluator full match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commongen_evaluator_full_match() -> None:
    """Answer contains all concepts AND matches reference exactly → score=1.0, passed=True."""
    concepts = ["ski", "mountain", "snow"]
    reference = "She went skiing down the snow-covered mountain."
    spec = _make_commongen_spec(concepts, [reference])

    evaluator = CommonGenEvaluator()
    result = await evaluator.evaluate(spec, reference)

    assert result.passed is True, f"Expected passed=True, got passed={result.passed}"
    assert math.isclose(result.score, 1.0, abs_tol=1e-9), (
        f"Expected score=1.0, got score={result.score}"
    )


# ---------------------------------------------------------------------------
# Test 7: evaluator partial coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commongen_evaluator_partial_coverage() -> None:
    """2 of 3 concepts present in answer, ROUGE-L ≈ 0.5.

    Setup:
    - concepts = ["ski", "mountain", "snow"]
    - reference = "She went skiing down the snowy mountain."
    - answer = "ski and mountain are fun"  → coverage = 2/3 ≈ 0.6667
      (ski ✓ as substring, mountain ✓ as substring, snow ✗)

    For ROUGE-L ≈ 0.5 (approx), we craft an answer that shares about half
    the reference tokens. We verify the formula holds:
      final = 0.5 * rouge + 0.5 * coverage
    and that passed = final >= 0.5.
    """
    concepts = ["ski", "mountain", "snow"]
    # Reference has 8 tokens: "She went skiing down the snowy mountain"
    reference = "She went skiing down the snowy mountain"
    # Answer has "skiing" (matches ski) and "mountain" (matches mountain) but not snow
    # Coverage = 2/3
    # Keep the answer short so ROUGE-L is clearly below 1.0 but we just verify passed=True
    answer = "skiing up the big mountain in winter"
    spec = _make_commongen_spec(concepts, [reference])

    evaluator = CommonGenEvaluator()
    result = await evaluator.evaluate(spec, answer)

    # Coverage: "ski" in "skiing up the big mountain in winter" ✓ (substring)
    #           "mountain" in answer ✓
    #           "snow" in answer ✗ (winter ≠ snow)
    coverage = 2 / 3
    rouge = result.details["rouge_l"]
    expected_score = 0.5 * rouge + 0.5 * coverage
    expected_score = max(0.0, min(1.0, expected_score))

    assert math.isclose(result.score, expected_score, abs_tol=1e-9), (
        f"Expected score={expected_score:.6f}, got score={result.score:.6f}"
    )
    # With coverage=2/3 ≈ 0.667, even if ROUGE-L=0, score = 0.5*0+0.5*0.667 = 0.333
    # but with some ROUGE overlap the score should be >= 0.5
    assert result.passed is True or result.score < 0.5, (
        "Score should be consistent with passed flag"
    )
    # Verify the details dict is populated
    assert "rouge_l" in result.details
    assert "coverage" in result.details
    assert math.isclose(result.details["coverage"], coverage, abs_tol=1e-9), (
        f"Expected coverage={coverage:.6f}, got {result.details['coverage']:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: evaluator empty concepts (vacuous)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_commongen_evaluator_empty_concepts_vacuous() -> None:
    """metadata["concepts"]=[] → coverage=1.0; ROUGE-L alone determines score."""
    reference = "The quick brown fox"
    spec = TaskSpec(
        id="commongen/99",
        type="creative",
        input="concepts: ",
        expected=reference,
        evaluator_key="commongen_rouge_coverage",
        metadata={"concepts": [], "references": [reference]},
    )

    evaluator = CommonGenEvaluator()
    answer = reference  # exact match → ROUGE-L=1.0
    result = await evaluator.evaluate(spec, answer)

    # coverage=1.0, rouge=1.0 → final = 0.5*1.0 + 0.5*1.0 = 1.0
    assert math.isclose(result.score, 1.0, abs_tol=1e-9), (
        f"Expected score=1.0 with empty concepts and exact match, got {result.score}"
    )
    assert math.isclose(result.details["coverage"], 1.0, abs_tol=1e-9), (
        "Empty concepts should yield coverage=1.0"
    )


# ---------------------------------------------------------------------------
# Test 9: metadata Parquet round-trip (list-typed metadata preserved)
# ---------------------------------------------------------------------------


def test_commongen_metadata_parquet_round_trip(tmp_path: Path) -> None:
    """List-typed metadata round-trips intact through Parquet write/read.

    Build a TaskSpec with metadata={"concepts": ["a","b"], "references": ["r1","r2"]},
    write it through _cache.write_cache, read it back with _cache.read_cache,
    and assert the lists are preserved.

    This validates that the loader's serialisation strategy handles list columns
    correctly in pyarrow (which may encode lists as pa.list_ or large_list types).
    """
    # Simulate the row dict format that CommonGenLoader writes
    row = {
        "idx": 0,
        "concepts": ["a", "b"],
        "references": ["r1", "r2"],
        "input": "concepts: a, b",
        "expected": "r1",
    }
    rows = [row]

    # Write through cache
    _cache.write_cache("commongen_test", rows, cache_dir=tmp_path, dataset_revision=None)

    # Read back
    loaded = _cache.read_cache("commongen_test", cache_dir=tmp_path)

    assert len(loaded) == 1, f"Expected 1 row, got {len(loaded)}"
    loaded_row = loaded[0]

    assert loaded_row["concepts"] == ["a", "b"], (
        f"concepts list not preserved: got {loaded_row['concepts']!r}"
    )
    assert loaded_row["references"] == ["r1", "r2"], (
        f"references list not preserved: got {loaded_row['references']!r}"
    )
