"""LangGraph Postgres checkpointer — two-pool pattern.

This module provides the checkpointer pool factory for LangGraph's
AsyncPostgresSaver. The checkpointer pool is SEPARATE from the business
SQLAlchemy pool (session.py) because AsyncPostgresSaver requires
autocommit=True (arch.md §11.3, §17/#3, langgraph issue #2755).

Public API
----------
checkpointer_scope  — async context manager (recommended for tests / CLI)
build_checkpointer  — low-level factory that returns (saver, pool)
_to_psycopg_dsn     — DSN normaliser: strips +asyncpg scheme suffix
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast
from urllib.parse import urlunsplit, urlsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool


def _to_psycopg_dsn(dsn: str) -> str:
    """Convert a SQLAlchemy-style DSN to a plain psycopg DSN.

    Replaces the scheme ``postgresql+asyncpg`` with ``postgresql``.
    If the scheme is already ``postgresql`` (or any other value), the
    URL is returned unchanged — the function is idempotent.

    Parameters
    ----------
    dsn:
        A database URL such as
        ``postgresql+asyncpg://user:pwd@host:5432/db?sslmode=require``
        or already-normalised ``postgresql://user:pwd@host:5432/db``.

    Returns
    -------
    str
        A DSN whose scheme is exactly ``postgresql``, e.g.
        ``postgresql://user:pwd@host:5432/db?sslmode=require``.
    """
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
    """Low-level factory: open a connection pool and set up the checkpointer.

    Creates an :class:`psycopg_pool.AsyncConnectionPool` with
    ``autocommit=True`` and ``prepare_threshold=0`` as required by
    LangGraph (arch.md §11.3 / §17/#3, langgraph issue #2755), opens
    it, constructs :class:`~langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`,
    calls ``saver.setup()`` (idempotent DDL), and returns both objects.

    Callers are responsible for closing the pool when done.  For
    short-lived usage prefer :func:`checkpointer_scope`.

    Parameters
    ----------
    dsn:
        Postgres DSN (``postgresql://`` or ``postgresql+asyncpg://``).
    max_size:
        Maximum pool connections (default 10).
    min_size:
        Minimum pool connections kept open (default 1).

    Returns
    -------
    tuple[AsyncPostgresSaver, AsyncConnectionPool]
        ``(saver, pool)`` — both fully initialised.
    """
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
    await saver.setup()
    return saver, pool


@asynccontextmanager
async def checkpointer_scope(
    dsn: str,
    *,
    max_size: int = 10,
    min_size: int = 1,
) -> AsyncIterator[AsyncPostgresSaver]:
    """Async context manager that owns the full checkpointer lifecycle.

    Opens the connection pool, sets up the saver, yields the saver, and
    closes the pool in the ``finally`` block — even if an exception is
    raised inside the ``async with`` block.

    This is the **recommended API** for tests and single-run CLI usage
    where the pool lifetime should be scoped to a single operation.

    Parameters
    ----------
    dsn:
        Postgres DSN (``postgresql://`` or ``postgresql+asyncpg://``).
    max_size:
        Maximum pool connections (default 10).
    min_size:
        Minimum pool connections kept open (default 1).

    Yields
    ------
    AsyncPostgresSaver
        A fully set-up checkpointer backed by the pool.

    Example
    -------
    ::

        async with checkpointer_scope(settings.pg_dsn) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            await graph.ainvoke(state)
    """
    saver, pool = await build_checkpointer(dsn, max_size=max_size, min_size=min_size)
    try:
        yield saver
    finally:
        await pool.close()
