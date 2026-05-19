"""Unit tests for atm.tasks.dabench — DABenchLoader and DABenchEvaluator.

Tests (TDD — 11 tests):
1.  test_dabench_offline_curated         — ATM_DABENCH_OFFLINE=1, loader returns ≥8 specs;
    at least one curated row has TWO @name[value] pairs in sorted name order (N1).
2.  test_dabench_remote_serialises_common_answers — monkeypatch load_dataset returns synthetic
    data; loader produces spec with expected="@a_metric[1.0] @b_metric[2.5]" (sorted by name).
3.  test_dabench_remote_multi_pair_sorted — same as
4.  test_dabench_falls_back_on_network_error — load_dataset raises OSError; loader returns
    curated ≥8 specs + emits structlog warning.
5.  test_dabench_cache_hit               — pre-write parquet; datasets.load_dataset not called.
6.  test_dabench_evaluator_full_match    — "@mean_fare[34.65]" vs "@mean_fare[34.65]" → 1.0/True.
7.  test_dabench_evaluator_close_within_tol — abs_tol=1e-2: 34.65/34.66 → pass; 34.65/34.67 → fail.
8.  test_dabench_evaluator_multi_pair_partial — "@a[1.0] @b[2.0]": @a correct, @b wrong → 0.5/False.
9.  test_dabench_evaluator_missing_template — "@a[1.0]" expected, plain text answer → 0.0/False.
10. test_dabench_evaluator_categorical_fallback — "@cat[YES]" vs "@cat[yes]" → 1.0/True (N2);
    sub-assertion: float("YES") raises ValueError proving categorical path triggered;
    sub-case: "@n[1.0]" vs "@n[2.0]" → False (numeric mismatch, not categorical string equality).
11. test_dabench_evaluator_vacuous       — expected="" (no pairs) → 1.0/True.

Workspace staging tests (4 tests — step 3.2):
12. test_stage_workspace_for_non_dabench_spec_returns_empty
13. test_stage_workspace_for_downloads_and_caches
14. test_stage_workspace_for_offline_returns_empty
15. test_stage_workspace_for_network_error_returns_empty_and_logs
"""

from __future__ import annotations

import math
import re
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import atm.tasks._cache as _cache
from atm.core.types import TaskSpec
from atm.tasks.dabench import DABenchEvaluator, DABenchLoader, stage_workspace_for

_FIXTURE_PATH = Path(__file__).parent.parent.parent / "fixtures" / "tasks" / "dabench_curated.jsonl"

_PAIR_RE = re.compile(r"@([A-Za-z_][\w]*)\[([^\]]+)\]")


def _make_dabench_spec(expected: str, id_suffix: str = "0") -> TaskSpec:
    """Build a minimal DABench TaskSpec with the given expected string."""
    return TaskSpec(
        id=f"dabench/{id_suffix}",
        type="decision",
        input="Some data analysis question",
        expected=expected,
        evaluator_key="dabench_numeric_exact",
    )


def _make_mock_hf_split(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a MagicMock mimicking datasets.load_dataset(...) split."""
    mock_split = MagicMock()
    mock_split.__iter__ = MagicMock(return_value=iter(rows))
    return mock_split


def _make_synthetic_hf_dataset(
    questions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> MagicMock:
    """Return a MagicMock for load_dataset that returns q and l splits."""
    q_mock = _make_mock_hf_split(questions)
    l_mock = _make_mock_hf_split(labels)
    return MagicMock(side_effect=[q_mock, l_mock])


def test_dabench_offline_curated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ATM_DABENCH_OFFLINE=1 → loader reads curated JSONL without any network call.

    Asserts:
    - Returns ≥8 TaskSpec instances.
    - All specs have type="decision" and evaluator_key="dabench_numeric_exact".
    - All spec ids start with "dabench/".
    - At least one spec's expected contains TWO @name[value] pairs in sorted name order (N1).
    """
    monkeypatch.setenv("ATM_DABENCH_OFFLINE", "1")

    loader = DABenchLoader()
    specs = loader.load(cache_dir=tmp_path)

    assert len(specs) >= 8, f"Expected ≥8 specs from curated JSONL, got {len(specs)}"

    for spec in specs:
        assert spec.type == "decision", f"Expected type='decision', got {spec.type!r}"
        assert spec.evaluator_key == "dabench_numeric_exact", (
            f"Expected evaluator_key='dabench_numeric_exact', got {spec.evaluator_key!r}"
        )
        assert spec.id.startswith("dabench/"), f"id should start with 'dabench/', got {spec.id!r}"

    multi_pair_specs = [s for s in specs if len(_PAIR_RE.findall(s.expected or "")) >= 2]
    assert len(multi_pair_specs) >= 1, (
        "Expected at least one spec with ≥2 @name[value] pairs in curated JSONL (N1)"
    )

    for spec in multi_pair_specs:
        pairs = _PAIR_RE.findall(spec.expected or "")
        names = [p[0] for p in pairs]
        assert names == sorted(names), (
            f"Pairs in '{spec.expected}' are not in sorted name order: {names}"
        )


def test_dabench_remote_serialises_common_answers(tmp_path: Path) -> None:
    """Remote load joins questions+labels; common_answers sorted by name → expected string.

    Input:  common_answers = [["b_metric", "2.5"], ["a_metric", "1.0"]]
    Output: expected = "@a_metric[1.0] @b_metric[2.5]"  (sorted by name)
    """
    questions = [{"id": 1, "question": "What are the metrics?"}]
    labels = [{"id": 1, "common_answers": [["b_metric", "2.5"], ["a_metric", "1.0"]]}]

    with patch("atm.tasks.dabench.datasets.load_dataset") as mock_ld:
        q_split = _make_mock_hf_split(questions)
        l_split = _make_mock_hf_split(labels)
        mock_ld.side_effect = [q_split, l_split]

        loader = DABenchLoader()
        specs = loader.load(cache_dir=tmp_path)

    assert len(specs) == 1, f"Expected 1 spec, got {len(specs)}"
    spec = specs[0]
    assert spec.expected == "@a_metric[1.0] @b_metric[2.5]", (
        f"Expected '@a_metric[1.0] @b_metric[2.5]', got {spec.expected!r}"
    )


def test_dabench_remote_multi_pair_sorted(tmp_path: Path) -> None:
    """Sort order is deterministic regardless of input order of common_answers."""
    questions = [{"id": 10, "question": "Calculate z, y, x metrics."}]
    labels = [{"id": 10, "common_answers": [["z_val", "3.0"], ["x_val", "1.0"], ["y_val", "2.0"]]}]

    with patch("atm.tasks.dabench.datasets.load_dataset") as mock_ld:
        q_split = _make_mock_hf_split(questions)
        l_split = _make_mock_hf_split(labels)
        mock_ld.side_effect = [q_split, l_split]

        loader = DABenchLoader()
        specs = loader.load(cache_dir=tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.expected == "@x_val[1.0] @y_val[2.0] @z_val[3.0]", (
        f"Expected alphabetically sorted pairs, got {spec.expected!r}"
    )
    pairs = _PAIR_RE.findall(spec.expected)
    names = [p[0] for p in pairs]
    assert names == sorted(names), f"Names not in sorted order: {names}"


def test_dabench_falls_back_on_network_error(tmp_path: Path) -> None:
    """OSError from load_dataset triggers fallback to curated JSONL + structlog warning."""
    import structlog.testing

    with (
        structlog.testing.capture_logs() as cap_logs,
        patch(
            "atm.tasks.dabench.datasets.load_dataset",
            side_effect=OSError("network down"),
        ),
    ):
        loader = DABenchLoader()
        specs = loader.load(cache_dir=tmp_path)

    assert len(specs) >= 8, (
        f"Expected ≥8 specs from curated fallback on network error, got {len(specs)}"
    )
    for spec in specs:
        assert spec.type == "decision"
        assert spec.evaluator_key == "dabench_numeric_exact"

    warning_found = any(
        record.get("log_level") == "warning"
        and (
            "dabench" in str(record.get("event", "")).lower()
            or "unavailable" in str(record.get("event", "")).lower()
            or "fallback" in str(record.get("event", "")).lower()
        )
        for record in cap_logs
    )
    assert warning_found, (
        f"Expected a structlog warning about dabench remote unavailability. Got: {cap_logs}"
    )


def test_dabench_cache_hit(tmp_path: Path) -> None:
    """Pre-written Parquet cache → datasets.load_dataset not called."""
    cache_rows = [
        {
            "id": "dabench/1",
            "question": "What is the mean?",
            "expected": "@mean[1.0]",
            "concepts": ["mean"],
            "constraints": "Round to 2 dp.",
            "format": "@mean[value]",
            "level": "easy",
            "file_name": "data.csv",
        }
    ]
    _cache.write_cache("dabench", cache_rows, cache_dir=tmp_path, dataset_revision=None)

    call_count = 0

    def counting_load_dataset(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return MagicMock()

    with patch("atm.tasks.dabench.datasets.load_dataset", side_effect=counting_load_dataset):
        loader = DABenchLoader()
        specs = loader.load(cache_dir=tmp_path)

    assert call_count == 0, f"Expected 0 network calls (cache hit), got {call_count}"
    assert len(specs) == 1
    assert specs[0].type == "decision"
    assert specs[0].evaluator_key == "dabench_numeric_exact"


@pytest.mark.asyncio
async def test_dabench_evaluator_full_match() -> None:
    """expected '@mean_fare[34.65]', answer contains '@mean_fare[34.65]' → score=1.0, passed=True."""
    spec = _make_dabench_spec("@mean_fare[34.65]")
    evaluator = DABenchEvaluator()
    result = await evaluator.evaluate(spec, "The mean fare is @mean_fare[34.65] per passenger.")

    assert result.passed is True, f"Expected passed=True, got {result.passed}"
    assert math.isclose(result.score, 1.0, abs_tol=1e-9), f"Expected score=1.0, got {result.score}"


@pytest.mark.asyncio
async def test_dabench_evaluator_close_within_tol() -> None:
    """abs_tol=1e-2: diff 0.01 → pass; diff 0.02 → fail."""
    evaluator = DABenchEvaluator()
    spec = _make_dabench_spec("@mean_fare[34.65]")

    result_pass = await evaluator.evaluate(spec, "@mean_fare[34.66]")
    assert result_pass.passed is True, (
        f"Expected passed=True for diff=0.01 (≤ abs_tol=1e-2), got passed={result_pass.passed}"
    )

    result_fail = await evaluator.evaluate(spec, "@mean_fare[34.67]")
    assert result_fail.passed is False, (
        f"Expected passed=False for diff=0.02 (> abs_tol=1e-2), got passed={result_fail.passed}"
    )


@pytest.mark.asyncio
async def test_dabench_evaluator_multi_pair_partial() -> None:
    """expected '@a[1.0] @b[2.0]'; @a correct, @b=3.0 wrong → score=0.5, passed=False."""
    spec = _make_dabench_spec("@a[1.0] @b[2.0]")
    evaluator = DABenchEvaluator()

    result = await evaluator.evaluate(spec, "result: @a[1.0] and @b[3.0]")

    assert math.isclose(result.score, 0.5, abs_tol=1e-9), (
        f"Expected score=0.5 (1/2 pairs correct), got {result.score}"
    )
    assert result.passed is False, f"Expected passed=False (score<1.0), got {result.passed}"
    assert result.details.get("correct") == 1
    assert result.details.get("total") == 2


@pytest.mark.asyncio
async def test_dabench_evaluator_missing_template() -> None:
    """expected '@a[1.0]'; plain text answer with no @a[...] → score=0.0, passed=False."""
    spec = _make_dabench_spec("@a[1.0]")
    evaluator = DABenchEvaluator()

    result = await evaluator.evaluate(spec, "The answer is approximately 1.0 units.")

    assert math.isclose(result.score, 0.0, abs_tol=1e-9), (
        f"Expected score=0.0 (no @a[...] in answer), got {result.score}"
    )
    assert result.passed is False, f"Expected passed=False, got {result.passed}"


@pytest.mark.asyncio
async def test_dabench_evaluator_categorical_fallback() -> None:
    """Categorical fallback: '@cat[YES]' vs '@cat[yes]' → score=1.0, passed=True (N2).

    Sub-assertion: float("YES") raises ValueError, proving categorical path triggered.
    Sub-case: '@n[1.0]' expected, '@n[2.0]' answered → passed=False via numeric comparison
    (not via string equality of '1.0' != '2.0').
    """
    with pytest.raises(ValueError):
        float("YES")

    evaluator = DABenchEvaluator()

    spec_cat = _make_dabench_spec("@cat[YES]")
    result_cat = await evaluator.evaluate(spec_cat, "@cat[yes]")

    assert result_cat.passed is True, (
        f"Expected passed=True for case-insensitive categorical match, got {result_cat.passed}"
    )
    assert math.isclose(result_cat.score, 1.0, abs_tol=1e-9), (
        f"Expected score=1.0 for categorical match, got {result_cat.score}"
    )

    spec_num = _make_dabench_spec("@n[1.0]")
    result_num = await evaluator.evaluate(spec_num, "@n[2.0]")

    assert result_num.passed is False, (
        f"Expected passed=False for numeric mismatch (@n[1.0] vs @n[2.0]), got {result_num.passed}"
    )
    assert float("1.0") == 1.0
    assert float("2.0") == 2.0
    assert not math.isclose(1.0, 2.0, abs_tol=1e-2), (
        "1.0 and 2.0 should not be close with abs_tol=1e-2"
    )


@pytest.mark.asyncio
async def test_dabench_evaluator_vacuous() -> None:
    """expected='' (no pairs) → score=1.0, passed=True (vacuous)."""
    spec = _make_dabench_spec("")
    evaluator = DABenchEvaluator()

    result = await evaluator.evaluate(spec, "Any answer at all.")

    assert result.passed is True, (
        f"Expected passed=True for vacuous (empty) expected, got {result.passed}"
    )
    assert math.isclose(result.score, 1.0, abs_tol=1e-9), (
        f"Expected score=1.0 for vacuous case, got {result.score}"
    )


def test_stage_workspace_for_non_dabench_spec_returns_empty(tmp_path: Path) -> None:
    """A spec whose id does not start with 'dabench/' causes an immediate no-op.

    stage_workspace_for must return [] without touching the filesystem or
    performing any network activity when the spec is not a DABench task.
    """
    spec = TaskSpec(
        id="humaneval/HumanEval/0",
        type="programming",
        input="Write a function that adds two numbers.",
        expected="add(1, 2) == 3",
        evaluator_key="humaneval",
    )

    result = stage_workspace_for(spec, workspace_path=tmp_path)

    assert result == [], f"Expected [] for non-dabench spec, got {result!r}"


def test_stage_workspace_for_downloads_and_caches(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call downloads and caches; second call (fresh workspace) skips urlopen.

    Verifies:
    - Returned list contains exactly the path workspace / file_name.
    - The file in the workspace has the correct byte content.
    - The file is written to cache_dir so subsequent calls avoid re-downloading.
    - urlopen call count does not increase on a cache-hit second call.
    """
    tmp_workspace_1 = tmp_path_factory.mktemp("workspace1")
    tmp_workspace_2 = tmp_path_factory.mktemp("workspace2")
    tmp_cache = tmp_path_factory.mktemp("cache")

    csv_bytes = b"col_a,col_b\n1,2\n"

    mock_response = MagicMock()
    mock_response.read.return_value = csv_bytes
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_urlopen = MagicMock(return_value=mock_response)

    spec = TaskSpec(
        id="dabench/X",
        type="decision",
        input="Analyse the titanic dataset.",
        expected="@survival_rate[0.38]",
        evaluator_key="dabench_numeric_exact",
        metadata={"file_name": "titanic.csv"},
    )

    monkeypatch.delenv("ATM_DABENCH_OFFLINE", raising=False)

    with patch("atm.tasks.dabench.urlopen", mock_urlopen):
        result_1 = stage_workspace_for(spec, workspace_path=tmp_workspace_1, cache_dir=tmp_cache)

        assert result_1 == [tmp_workspace_1 / "titanic.csv"], (
            f"Expected [workspace/titanic.csv], got {result_1!r}"
        )
        assert (tmp_workspace_1 / "titanic.csv").read_bytes() == csv_bytes, (
            "Staged file content does not match downloaded bytes"
        )
        assert (tmp_cache / "titanic.csv").exists(), "Downloaded CSV was not written to cache_dir"
        urlopen_count_after_first = mock_urlopen.call_count
        assert urlopen_count_after_first == 1, (
            f"Expected exactly 1 urlopen call on first (cache miss) call, got {urlopen_count_after_first}"
        )

        result_2 = stage_workspace_for(spec, workspace_path=tmp_workspace_2, cache_dir=tmp_cache)

    assert result_2 == [tmp_workspace_2 / "titanic.csv"], (
        f"Expected [workspace2/titanic.csv] on cache-hit call, got {result_2!r}"
    )
    assert mock_urlopen.call_count == urlopen_count_after_first, (
        "urlopen was called again on cache-hit second call — byte cache not respected"
    )


def test_stage_workspace_for_offline_returns_empty(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATM_DABENCH_OFFLINE=1 causes stage_workspace_for to return [] immediately.

    No urlopen call must be made regardless of whether a cache file exists.
    """
    tmp_workspace = tmp_path_factory.mktemp("workspace")
    tmp_cache = tmp_path_factory.mktemp("cache")

    spec = TaskSpec(
        id="dabench/offline_test",
        type="decision",
        input="Analyse the titanic dataset.",
        expected="@mean[1.0]",
        evaluator_key="dabench_numeric_exact",
        metadata={"file_name": "titanic.csv"},
    )

    monkeypatch.setenv("ATM_DABENCH_OFFLINE", "1")

    mock_urlopen = MagicMock()

    with patch("atm.tasks.dabench.urlopen", mock_urlopen):
        result = stage_workspace_for(spec, workspace_path=tmp_workspace, cache_dir=tmp_cache)

    assert result == [], f"Expected [] when ATM_DABENCH_OFFLINE=1, got {result!r}"
    mock_urlopen.assert_not_called()


def test_stage_workspace_for_network_error_returns_empty_and_logs(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URLError from urlopen is swallowed; function returns [] without raising.

    The error must not propagate to the caller — a degraded-but-running
    experiment is preferable to a crashed run.
    """
    tmp_workspace = tmp_path_factory.mktemp("workspace")
    tmp_cache = tmp_path_factory.mktemp("cache")

    spec = TaskSpec(
        id="dabench/net_error_test",
        type="decision",
        input="Analyse the titanic dataset.",
        expected="@mean[1.0]",
        evaluator_key="dabench_numeric_exact",
        metadata={"file_name": "titanic.csv"},
    )

    monkeypatch.delenv("ATM_DABENCH_OFFLINE", raising=False)

    mock_urlopen = MagicMock(side_effect=urllib.error.URLError("test failure"))

    with patch("atm.tasks.dabench.urlopen", mock_urlopen):
        result = stage_workspace_for(spec, workspace_path=tmp_workspace, cache_dir=tmp_cache)

    assert result == [], f"Expected [] when URLError is raised, got {result!r}"
    assert not (tmp_workspace / "titanic.csv").exists(), (
        "Workspace file should not exist after a network error"
    )


def test_stage_workspace_for_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """file_name containing '../' path traversal sequences is rejected; returns [].

    urlopen must NOT be called — validation must happen before any network or
    filesystem activity.
    """
    monkeypatch.delenv("ATM_DABENCH_OFFLINE", raising=False)

    spec = TaskSpec(
        id="dabench/evil",
        type="decision",
        input="",
        expected="",
        evaluator_key="dabench_numeric_exact",
        metadata={"file_name": "../etc/passwd"},
    )

    mock_urlopen = MagicMock(side_effect=AssertionError("urlopen must not be called"))

    with patch("atm.tasks.dabench.urlopen", mock_urlopen):
        result = stage_workspace_for(spec, workspace_path=tmp_path)

    assert result == [], f"Expected [] for path traversal file_name, got {result!r}"
    mock_urlopen.assert_not_called()


def test_stage_workspace_for_rejects_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """file_name that is an absolute path (e.g. '/etc/passwd') is rejected; returns [].

    urlopen must NOT be called — validation must happen before any network or
    filesystem activity.
    """
    monkeypatch.delenv("ATM_DABENCH_OFFLINE", raising=False)

    spec = TaskSpec(
        id="dabench/evil",
        type="decision",
        input="",
        expected="",
        evaluator_key="dabench_numeric_exact",
        metadata={"file_name": "/etc/passwd"},
    )

    mock_urlopen = MagicMock(side_effect=AssertionError("urlopen must not be called"))

    with patch("atm.tasks.dabench.urlopen", mock_urlopen):
        result = stage_workspace_for(spec, workspace_path=tmp_path)

    assert result == [], f"Expected [] for absolute file_name, got {result!r}"
    mock_urlopen.assert_not_called()
