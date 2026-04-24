"""Unit tests for atm.storage.session.

Tests cover:
1. create_engine returns an AsyncEngine instance (no DB connection made).
2. create_session_factory produces a factory with expire_on_commit=False.
3. session_scope is a proper async context manager (has __aenter__ / __aexit__).
4. session_scope rolls back on exception inside the block.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from atm.storage.session import create_engine, create_session_factory, session_scope


# ---------------------------------------------------------------------------
# Test 1: create_engine returns AsyncEngine without connecting
# ---------------------------------------------------------------------------


def test_create_engine_returns_async_engine() -> None:
    """create_engine builds an AsyncEngine without opening any real connection."""
    engine = create_engine("postgresql+asyncpg://fake/db", pool_pre_ping=False)
    assert isinstance(engine, AsyncEngine)


# ---------------------------------------------------------------------------
# Test 2: create_session_factory sets expire_on_commit=False
# ---------------------------------------------------------------------------


def test_create_session_factory_expire_on_commit_false() -> None:
    """create_session_factory produces a factory with expire_on_commit=False."""
    engine = create_engine("postgresql+asyncpg://fake/db", pool_pre_ping=False)
    factory = create_session_factory(engine)
    assert isinstance(factory, async_sessionmaker)
    assert factory.kw["expire_on_commit"] is False


# ---------------------------------------------------------------------------
# Test 3: session_scope is an async context manager
# ---------------------------------------------------------------------------


def test_session_scope_is_async_context_manager() -> None:
    """session_scope result exposes __aenter__ and __aexit__ (asynccontextmanager)."""
    engine = create_engine("postgresql+asyncpg://fake/db", pool_pre_ping=False)
    factory = create_session_factory(engine)
    cm = session_scope(factory)
    assert hasattr(cm, "__aenter__")
    assert hasattr(cm, "__aexit__")


# ---------------------------------------------------------------------------
# Test 4: session_scope calls rollback on exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_scope_rollback_on_exception() -> None:
    """On exception inside session_scope, session.rollback() is awaited."""
    mock_session = AsyncMock(spec=AsyncSession)

    # Build a mock factory whose async-context-manager yields mock_session
    mock_factory = MagicMock(spec=async_sessionmaker)
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_session
    mock_cm.__aexit__.return_value = False
    mock_factory.return_value = mock_cm

    with pytest.raises(ValueError, match="boom"):
        async with session_scope(mock_factory) as session:
            assert session is mock_session
            raise ValueError("boom")

    mock_session.rollback.assert_awaited_once()
    mock_session.commit.assert_not_awaited()
