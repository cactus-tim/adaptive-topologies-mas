"""Async SQLAlchemy session factory and lifecycle helpers.

Provides engine construction, session-factory creation, and an async context
manager for transactional scoping.  Intentionally model-agnostic: no ORM
models are imported here.

Architecture: §11.1 of arch.md — one AsyncEngine per process, short-lived
sessions, expire_on_commit=False, pool_pre_ping=True.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine as _sa_create_async_engine,
)


def create_engine(
    dsn: str,
    *,
    echo: bool = False,
    pool_size: int = 10,
    max_overflow: int = 5,
    pool_pre_ping: bool = True,
) -> AsyncEngine:
    """Create an :class:`AsyncEngine` from *dsn*.

    Args:
        dsn: SQLAlchemy async DSN, e.g. ``postgresql+asyncpg://user:pw@host/db``.
        echo: If ``True``, all SQL statements are echoed to the logger.
        pool_size: Number of persistent connections in the pool.
        max_overflow: Extra connections allowed beyond *pool_size*.
        pool_pre_ping: Issue a lightweight ping before handing a connection
            from the pool; avoids stale-connection errors on reconnect.

    Returns:
        A configured :class:`sqlalchemy.ext.asyncio.AsyncEngine`.
    """
    return _sa_create_async_engine(
        dsn,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=pool_pre_ping,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to *engine*.

    The factory is configured with ``expire_on_commit=False`` so that ORM
    objects remain usable after a commit (important for async workflows where
    lazy-loading is not available).

    Args:
        engine: The :class:`AsyncEngine` to bind sessions to.

    Returns:
        An :class:`sqlalchemy.ext.asyncio.async_sessionmaker` producing
        :class:`AsyncSession` instances.
    """
    return async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that provides a transactional :class:`AsyncSession`.

    Commits on clean exit; rolls back and re-raises on any exception.

    Args:
        factory: Session factory created by :func:`create_session_factory`.

    Yields:
        An :class:`AsyncSession` bound to an open transaction.

    Example::

        async with session_scope(session_factory) as session:
            session.add(some_orm_object)
    """
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
