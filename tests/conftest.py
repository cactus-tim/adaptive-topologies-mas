"""Root conftest.py for the ATM test suite.

Provides shared fixtures used across unit and integration tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# ephemeral_pg_dsn — per-test disposable PostgreSQL DSN
# ---------------------------------------------------------------------------

_PG_TESTS_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_PG_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"


@pytest_asyncio.fixture(scope="function")
async def ephemeral_pg_dsn() -> AsyncGenerator[str, None]:
    """Yield a PostgreSQL DSN backed by an ephemeral schema, then drop all tables.

    Skip behaviour
    --------------
    The fixture skips the test (via ``pytest.skip``) if the ``ATM_ENABLE_PG_TESTS``
    environment variable is not set to "1", "true", or "yes".

    Isolation strategy
    ------------------
    Uses ``Base.metadata.create_all`` to build all ATM tables before each test
    and ``Base.metadata.drop_all`` to tear them down after — ensuring a clean
    state regardless of test ordering.

    Environment variables
    ---------------------
    ATM_ENABLE_PG_TESTS : str
        Must be "1", "true", or "yes" to enable PG tests.
    ATM_PG_DSN : str (optional)
        Override the default DSN (postgresql+asyncpg://atm:atm@localhost:5432/atm_test).
    """
    if not _PG_TESTS_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set — skipping PostgreSQL integration tests")

    dsn = os.environ.get("ATM_PG_DSN", _DEFAULT_PG_DSN)

    from sqlalchemy import text

    from atm.storage.models import Base
    from atm.storage.session import create_engine

    # Reset the public schema BEFORE create_all — defensive against state
    # leaks from prior tests whose subprocess workers (e.g. `atm grid`
    # ProcessPoolExecutor children) may still have pending writes/types in
    # flight during fixture teardown. Without this DROP SCHEMA, a residual
    # row or pg_type entry triggers UniqueViolationError on the next
    # test's `create_all` (pg_type_typname_nsp_index race).
    reset_engine = create_engine(dsn, echo=False, pool_size=1, max_overflow=0)
    async with reset_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await reset_engine.dispose()

    engine = create_engine(dsn, echo=False, pool_size=2, max_overflow=1)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield dsn
    finally:
        # Tear down all tables — ensures clean state for next test
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
