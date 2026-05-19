"""LangGraph Postgres checkpointer — two-pool pattern (separate from SQLAlchemy pool)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool


def _to_psycopg_dsn(dsn: str) -> str:
    """Replace ``postgresql+asyncpg`` scheme with ``postgresql``; idempotent."""
    parts = urlsplit(dsn)
    if parts.scheme == "postgresql+asyncpg":
        parts = parts._replace(scheme="postgresql")
    return urlunsplit(parts)


async def build_checkpointer(
    dsn: str,
    *,
    max_size: int = 10,
    min_size: int = 1,
) -> tuple[AsyncPostgresSaver, AsyncConnectionPool[AsyncConnection[DictRow]]]:
    """Open connection pool (autocommit=True), set up saver, return (saver, pool)."""
    raw_pool: AsyncConnectionPool[AsyncConnection[Any]] = AsyncConnectionPool(
        conninfo=_to_psycopg_dsn(dsn),
        min_size=min_size,
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    )
    await raw_pool.open()
    pool = cast(AsyncConnectionPool[AsyncConnection[DictRow]], raw_pool)
    saver = AsyncPostgresSaver(conn=pool)
    saver.supports_pipeline = False
    await saver.setup()
    return saver, pool


@asynccontextmanager
async def checkpointer_scope(
    dsn: str,
    *,
    max_size: int = 10,
    min_size: int = 1,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Context manager: build checkpointer, yield saver, close pool on exit."""
    saver, pool = await build_checkpointer(dsn, max_size=max_size, min_size=min_size)
    try:
        yield saver
    finally:
        await pool.close()
