"""Unit tests for atm.tasks._cache — parquet-based task cache.

Tests cover (TDD):
1. write→read round-trip: rows written can be read back identically
2. is_cached: False before write, True after write
3. no .tmp leftover after successful write (atomic rename)
4. FileNotFoundError raised when reading a missing key
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atm.tasks._cache import (
    cache_dir as default_cache_dir,
)
from atm.tasks._cache import (
    is_cached,
    read_cache,
    write_cache,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sample_rows() -> list[dict[str, object]]:
    return [
        {"task_id": "eval/1", "prompt": "What is 2+2?", "answer": "4"},
        {"task_id": "eval/2", "prompt": "Capital of France?", "answer": "Paris"},
    ]


# ---------------------------------------------------------------------------
# Test 1: write → read round-trip
# ---------------------------------------------------------------------------


def test_write_read_round_trip(tmp_path: Path) -> None:
    """Rows written via write_cache can be read back identically via read_cache."""
    rows = _sample_rows()
    key = "test_dataset_roundtrip"

    write_cache(key, rows, cache_dir=tmp_path)
    result = read_cache(key, cache_dir=tmp_path)

    assert len(result) == len(rows)
    # Check all original fields are preserved
    for original, recovered in zip(rows, result, strict=True):
        for field, value in original.items():
            assert recovered[field] == value


# ---------------------------------------------------------------------------
# Test 2: is_cached — False before write, True after write
# ---------------------------------------------------------------------------


def test_is_cached_false_before_write_true_after(tmp_path: Path) -> None:
    """is_cached returns False before any write and True after write."""
    key = "test_dataset_cached"

    assert is_cached(key, cache_dir=tmp_path) is False

    write_cache(key, _sample_rows(), cache_dir=tmp_path)

    assert is_cached(key, cache_dir=tmp_path) is True


# ---------------------------------------------------------------------------
# Test 3: no .tmp leftover after successful write
# ---------------------------------------------------------------------------


def test_no_tmp_leftover_after_success(tmp_path: Path) -> None:
    """After a successful write_cache, no .tmp files remain in the cache directory."""
    key = "test_dataset_atomic"

    write_cache(key, _sample_rows(), cache_dir=tmp_path)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Test 4: FileNotFoundError on missing key
# ---------------------------------------------------------------------------


def test_read_missing_key_raises_file_not_found(tmp_path: Path) -> None:
    """read_cache raises FileNotFoundError when the key has not been written."""
    with pytest.raises(FileNotFoundError):
        read_cache("nonexistent_key_xyz", cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test 5: write_cache returns Path to the written file
# ---------------------------------------------------------------------------


def test_write_cache_returns_path(tmp_path: Path) -> None:
    """write_cache returns a Path that exists and is a file."""
    key = "test_dataset_returns_path"
    result_path = write_cache(key, _sample_rows(), cache_dir=tmp_path)

    assert isinstance(result_path, Path)
    assert result_path.exists()
    assert result_path.is_file()


# ---------------------------------------------------------------------------
# Test 6: dataset_revision is stored in metadata (optional field)
# ---------------------------------------------------------------------------


def test_write_cache_with_revision(tmp_path: Path) -> None:
    """write_cache accepts optional dataset_revision without error."""
    key = "test_dataset_with_revision"
    rows = _sample_rows()

    result_path = write_cache(key, rows, cache_dir=tmp_path, dataset_revision="abc123")

    assert result_path.exists()
    # Round-trip should still work
    recovered = read_cache(key, cache_dir=tmp_path)
    assert len(recovered) == len(rows)


# ---------------------------------------------------------------------------
# Test 7: cache_dir() default returns a Path object
# ---------------------------------------------------------------------------


def test_default_cache_dir_returns_path() -> None:
    """cache_dir() returns a Path object (data/cache/tasks/ convention)."""
    result = default_cache_dir()
    assert isinstance(result, Path)
    assert "cache" in str(result) or "tasks" in str(result)
