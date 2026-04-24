"""Fixtures for storage integration tests.

Skips the entire module when ATM_INTEGRATION_PG=1 is not set in the environment.
PG DSN defaults to postgresql+asyncpg://atm:atm@localhost:5432/atm_test.

Two engine fixtures:
- pg_engine_fast (function scope): DDL via Base.metadata.create_all / drop_all — fast setup.
- pg_engine_alembic (session scope): DDL via alembic upgrade head / downgrade base.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from atm.storage.models import Base
from atm.storage.session import create_engine, create_session_factory

# ---------------------------------------------------------------------------
# Module-level skip: only run when ATM_INTEGRATION_PG=1
# ---------------------------------------------------------------------------

_PG_ENABLED = os.environ.get("ATM_INTEGRATION_PG", "") == "1"
_DEFAULT_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """Return the PG DSN from the environment, skip if integration flag not set."""
    if not _PG_ENABLED:
        pytest.skip("ATM_INTEGRATION_PG=1 not set — skipping storage integration tests")
    return os.environ.get("ATM_PG_DSN", _DEFAULT_DSN)


@pytest_asyncio.fixture(scope="function")
async def pg_engine_fast(pg_dsn: str) -> AsyncGenerator[AsyncEngine, None]:
    """Create async engine, run create_all DDL, yield, drop_all on teardown.

    Fast per-test setup for smoke tests — avoids alembic subprocess overhead.
    """
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
    """
    # Derive psycopg-compatible DSN for alembic (it uses synchronous psycopg)
    psycopg_dsn = pg_dsn.replace("postgresql+asyncpg://", "postgresql://")

    env = os.environ.copy()
    env["PG_DSN"] = psycopg_dsn

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
