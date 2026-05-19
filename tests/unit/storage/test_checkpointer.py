"""Unit tests for atm.storage.checkpointer.

Tests cover:
1. _to_psycopg_dsn strips +asyncpg scheme suffix
2. _to_psycopg_dsn is idempotent for plain postgresql:// URLs
3. _to_psycopg_dsn preserves query strings
4. checkpointer_scope lifecycle: open → setup → yield → close
5. build_checkpointer returns (saver, pool) tuple of correct mocked instances
6. Public API imports work
"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from atm.storage.checkpointer import _to_psycopg_dsn, build_checkpointer, checkpointer_scope


def test_to_psycopg_dsn_asyncpg_to_plain() -> None:
    """postgresql+asyncpg scheme is replaced with postgresql."""
    result = _to_psycopg_dsn("postgresql+asyncpg://a:b@h:5432/d")
    assert result == "postgresql://a:b@h:5432/d"


def test_to_psycopg_dsn_idempotent() -> None:
    """Passing a plain postgresql:// URL returns it unchanged."""
    url = "postgresql://a:b@h/d"
    assert _to_psycopg_dsn(url) == url


def test_to_psycopg_dsn_with_query() -> None:
    """Query string is preserved when stripping the +asyncpg suffix."""
    result = _to_psycopg_dsn("postgresql+asyncpg://a@h/d?sslmode=require")
    assert result == "postgresql://a@h/d?sslmode=require"


DSN = "postgresql+asyncpg://user:pwd@localhost:5432/testdb"
PLAIN_DSN = "postgresql://user:pwd@localhost:5432/testdb"


@pytest.mark.asyncio
async def test_checkpointer_scope_lifecycle() -> None:
    """checkpointer_scope: pool opened, saver set up, pool closed in finally."""
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_pool.close = AsyncMock()

    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    mock_pool_cls = MagicMock(return_value=mock_pool)
    mock_saver_cls = MagicMock(return_value=mock_saver)

    with (
        patch("atm.storage.checkpointer.AsyncConnectionPool", mock_pool_cls),
        patch("atm.storage.checkpointer.AsyncPostgresSaver", mock_saver_cls),
    ):
        async with checkpointer_scope(DSN) as saver:
            assert saver is mock_saver

    mock_pool_cls.assert_called_once_with(
        conninfo=PLAIN_DSN,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "row_factory": ANY, "prepare_threshold": 0},
    )
    mock_pool.open.assert_awaited_once()
    mock_saver_cls.assert_called_once_with(conn=mock_pool)
    mock_saver.setup.assert_awaited_once()
    mock_pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_checkpointer_scope_lifecycle_call_order() -> None:
    """Verify strict open→setup→close ordering."""
    call_log: list[str] = []

    mock_pool = MagicMock()

    async def fake_open(**_: object) -> None:
        call_log.append("pool.open")

    async def fake_close() -> None:
        call_log.append("pool.close")

    mock_pool.open = AsyncMock(side_effect=fake_open)
    mock_pool.close = AsyncMock(side_effect=fake_close)

    mock_saver = MagicMock()

    async def fake_setup() -> None:
        call_log.append("saver.setup")

    mock_saver.setup = AsyncMock(side_effect=fake_setup)

    mock_pool_cls = MagicMock(return_value=mock_pool)
    mock_saver_cls = MagicMock(return_value=mock_saver)

    with (
        patch("atm.storage.checkpointer.AsyncConnectionPool", mock_pool_cls),
        patch("atm.storage.checkpointer.AsyncPostgresSaver", mock_saver_cls),
    ):
        async with checkpointer_scope(DSN):
            call_log.append("body")

    assert call_log == ["pool.open", "saver.setup", "body", "pool.close"]


@pytest.mark.asyncio
async def test_build_checkpointer_returns_tuple() -> None:
    """build_checkpointer returns (saver, pool) with correct mock instances."""
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()

    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    mock_pool_cls = MagicMock(return_value=mock_pool)
    mock_saver_cls = MagicMock(return_value=mock_saver)

    with (
        patch("atm.storage.checkpointer.AsyncConnectionPool", mock_pool_cls),
        patch("atm.storage.checkpointer.AsyncPostgresSaver", mock_saver_cls),
    ):
        saver, pool = await build_checkpointer(DSN)

    assert saver is mock_saver
    assert pool is mock_pool
    mock_pool.open.assert_awaited_once()
    mock_saver.setup.assert_awaited_once()


def test_import() -> None:
    """Public API names are importable from atm.storage.checkpointer."""
    from atm.storage.checkpointer import (  # noqa: F401
        _to_psycopg_dsn,
        build_checkpointer,
        checkpointer_scope,
    )
