"""Shared PG fixtures for all integration tests.

Promoted from tests/integration/storage/conftest.py so that evaluation
integration tests can use the same fixtures without duplication.

Fixtures:
  - pg_dsn               (session scope) — returns PG DSN; skips if ATM_ENABLE_PG_TESTS!=1
  - pg_engine_fast       (function scope) — create_all / drop_all DDL
  - pg_engine_alembic    (session scope)  — alembic upgrade head / downgrade base
  - session_factory_fast (function scope) — async_sessionmaker bound to pg_engine_fast

All fixtures skip automatically when ATM_ENABLE_PG_TESTS is not in {"1","true","yes"}.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.storage.models import Base
from atm.storage.session import create_engine, create_session_factory

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """Return the PG DSN from the environment, skip if integration flag not set."""
    if not _PG_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set — skipping PostgreSQL integration tests")
    return os.environ.get("ATM_PG_DSN", _DEFAULT_DSN)


@pytest_asyncio.fixture(scope="function")
async def pg_engine_fast(pg_dsn: str) -> AsyncGenerator[AsyncEngine, None]:
    """Create async engine, run create_all DDL, yield, drop_all on teardown.

    Fast per-test setup for smoke tests — avoids alembic subprocess overhead.

    Resets the ``public`` schema before ``create_all`` so the model DDL
    (with ``server_default``s) actually executes regardless of leftover state
    from a session-scoped ``pg_engine_alembic`` (which uses migration DDL
    without server_defaults). Without this reset, ``create_all`` would be a
    no-op when alembic-style tables already exist, and inserts that rely on
    server defaults (e.g. ``Experiment.started_at``) would fail with
    NotNullViolationError.
    """
    reset_engine = create_engine(pg_dsn, echo=False)
    async with reset_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await reset_engine.dispose()

    engine = create_engine(pg_dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def pg_engine_alembic(pg_dsn: str) -> AsyncGenerator[AsyncEngine, None]:
    """Create async engine, run alembic upgrade head, yield, downgrade base on teardown.

    Session-scoped — used only for the DDL equivalence smoke test.
    DSN is passed via PG_DSN env var (as expected by alembic/env.py).
    alembic/env.py builds an async engine via async_engine_from_config,
    so the asyncpg driver suffix (postgresql+asyncpg://) MUST be kept.

    Resets the ``public`` schema before ``upgrade head`` so the migration
    actually executes regardless of leftover state. Without this reset,
    ``pg_engine_fast`` (function-scoped) drops business tables via
    ``Base.metadata.drop_all`` but leaves the ``alembic_version`` table
    untouched (it is not in ``Base.metadata``); alembic then sees "already at
    head" and skips DDL, and this test fails with missing-tables errors.
    """
    env = os.environ.copy()
    env["PG_DSN"] = pg_dsn

    reset_engine = create_engine(pg_dsn, echo=False)
    async with reset_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await reset_engine.dispose()

    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        check=True,
        env=env,
    )

    engine = create_engine(pg_dsn, echo=False, pool_size=2, max_overflow=1)
    try:
        yield engine
    finally:
        await engine.dispose()
        subprocess.run(
            ["uv", "run", "alembic", "downgrade", "base"],
            check=True,
            env=env,
        )


@pytest.fixture(scope="function")
def session_factory_fast(pg_engine_fast: AsyncEngine):  # type: ignore[no-untyped-def]
    """Synchronous fixture that provides an async_sessionmaker bound to pg_engine_fast."""
    return create_session_factory(pg_engine_fast)
