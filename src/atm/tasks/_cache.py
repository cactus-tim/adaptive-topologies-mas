"""Parquet-based cache for task datasets (atomic write/read via os.replace)."""

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

_DEFAULT_CACHE_ROOT = Path("data") / "cache" / "tasks"


def cache_dir() -> Path:
    """Return the default cache directory (``data/cache/tasks/``) without creating it."""
    return _DEFAULT_CACHE_ROOT


def _cache_path(key: str, cache_directory: Path) -> Path:
    """Return the Parquet file path for ``key`` in ``cache_directory``."""
    safe_key = key.replace("/", "__").replace("\\", "__")
    return cache_directory / f"{safe_key}.parquet"


def write_cache(
    key: str,
    rows: list[dict[str, Any]],
    *,
    cache_dir: Path | None = None,
    dataset_revision: str | None = None,
) -> Path:
    """Write ``rows`` to a Parquet file for ``key`` atomically via ``os.replace``."""
    target_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_ROOT
    target_dir.mkdir(parents=True, exist_ok=True)

    final_path = _cache_path(key, target_dir)
    tmp_path = final_path.with_suffix(".tmp")

    now_iso = _dt.datetime.now(tz=_dt.UTC).isoformat()

    table = pa.Table.from_pylist(rows)

    metadata: dict[bytes, bytes] = {
        b"dataset_id": key.encode(),
        b"row_count": str(len(rows)).encode(),
        b"created_at_utc": now_iso.encode(),
        b"dataset_revision": (dataset_revision or "").encode(),
    }
    schema_with_meta = table.schema.with_metadata(metadata)
    table = table.cast(schema_with_meta)

    pq.write_table(table, tmp_path)  # type: ignore[no-untyped-call]
    os.replace(tmp_path, final_path)

    return final_path


def read_cache(
    key: str,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read rows from the cached Parquet file for ``key``; raises FileNotFoundError if missing."""
    target_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_ROOT
    path = _cache_path(key, target_dir)

    if not path.exists():
        raise FileNotFoundError(f"No cache file for dataset key {key!r}. Expected: {path}")

    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    result: list[dict[str, Any]] = table.to_pylist()
    return result


def is_cached(
    key: str,
    cache_dir: Path | None = None,
) -> bool:
    """Return ``True`` if a cache file exists for ``key``, ``False`` otherwise."""
    target_dir = cache_dir if cache_dir is not None else _DEFAULT_CACHE_ROOT
    return _cache_path(key, target_dir).exists()
