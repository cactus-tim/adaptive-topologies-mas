"""Parquet-based cache for task datasets.

Provides atomic write/read operations for task dataset rows using Apache Parquet.

IMPORTANT — pyarrow metadata footgun:
    All keys AND values in ``pa.schema(...).with_metadata({...})`` must be ``bytes``,
    not ``str``. Passing ``str`` keys or values raises a ``TypeError`` at runtime.
    Always encode with ``b"key"`` or ``"key".encode()`` syntax.

Public API:
- ``cache_dir() -> Path``            — default cache directory (data/cache/tasks/)
- ``write_cache(key, rows, *, ...) -> Path`` — atomic write via os.replace
- ``read_cache(key, cache_dir) -> list[dict]``  — raises FileNotFoundError if missing
- ``is_cached(key, cache_dir) -> bool``         — existence check
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "cache_dir",
    "is_cached",
    "read_cache",
    "write_cache",
]

# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_ROOT = Path("data") / "cache" / "tasks"


def cache_dir() -> Path:
    """Return the default cache directory for task datasets.

    Returns the path ``data/cache/tasks/`` without creating it.
    Directory creation is a caller responsibility — ``write_cache()`` creates it
    before writing.  ``data/`` is globally gitignored so cached files are never
    committed.
    """
    return _DEFAULT_CACHE_ROOT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_path(key: str, cache_directory: Path) -> Path:
    """Return the Parquet file path for ``key`` in ``cache_directory``."""
    # Replace slashes to avoid subdirectory creation; key is dataset name only
    safe_key = key.replace("/", "__").replace("\\", "__")
    return cache_directory / f"{safe_key}.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_cache(
    key: str,
    rows: list[dict[str, Any]],
    *,
    cache_dir: Path | None = None,
    dataset_revision: str | None = None,
) -> Path:
    """Write ``rows`` to a Parquet file for ``key``, atomically via ``os.replace``.

    The write is performed to a ``.tmp`` staging file, then renamed to the final
    path using ``os.replace`` — this guarantees that readers never see a partial
    file even if the process is interrupted.

    Args:
        key:              Dataset identifier (used as filename stem).
        rows:             List of dicts to serialise.  All rows must share the
                          same set of keys (pyarrow infers schema from the first
                          row).
        cache_dir:        Target directory.  Defaults to ``data/cache/tasks/``.
        dataset_revision: Optional HuggingFace dataset revision SHA for
                          reproducibility tracking.  Stored in Parquet metadata.

    Returns:
        Path to the written Parquet file.

    Raises:
        ValueError: if ``rows`` is empty (pyarrow cannot infer schema).
    """
    target_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)

    final_path = _cache_path(key, target_dir)
    tmp_path = final_path.with_suffix(".tmp")

    now_iso = _dt.datetime.now(tz=_dt.UTC).isoformat()

    # Build pyarrow table from rows
    table = pa.Table.from_pylist(rows)

    # IMPORTANT: pyarrow schema metadata keys AND values must be bytes.
    # Passing str raises TypeError at runtime — this is a well-known footgun.
    metadata: dict[bytes, bytes] = {
        b"dataset_id": key.encode(),
        b"row_count": str(len(rows)).encode(),
        b"created_at_utc": now_iso.encode(),
        b"dataset_revision": (dataset_revision or "").encode(),
    }
    schema_with_meta = table.schema.with_metadata(metadata)
    table = table.cast(schema_with_meta)

    # Write to .tmp, then atomically rename
    pq.write_table(table, tmp_path)  # type: ignore[no-untyped-call]
    os.replace(tmp_path, final_path)

    return final_path


def read_cache(
    key: str,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read rows from the cached Parquet file for ``key``.

    Args:
        key:       Dataset identifier.
        cache_dir: Cache directory.  Defaults to ``data/cache/tasks/``.

    Returns:
        List of dicts, one per row.

    Raises:
        FileNotFoundError: if no cache file exists for ``key``.
    """
    target_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_ROOT
    path = _cache_path(key, target_dir)

    if not path.exists():
        raise FileNotFoundError(
            f"No cache file for dataset key {key!r}. "
            f"Expected: {path}"
        )

    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    result: list[dict[str, Any]] = table.to_pylist()
    return result


def is_cached(
    key: str,
    cache_dir: Path | None = None,
) -> bool:
    """Return ``True`` if a cache file exists for ``key``, ``False`` otherwise.

    Args:
        key:       Dataset identifier.
        cache_dir: Cache directory.  Defaults to ``data/cache/tasks/``.
    """
    target_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_ROOT
    return _cache_path(key, target_dir).exists()
