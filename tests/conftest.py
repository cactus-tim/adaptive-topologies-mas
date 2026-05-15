"""Root conftest.py for the ATM test suite.

Provides shared fixtures used across unit and integration tests.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# ephemeral_pg_dsn — per-test disposable PostgreSQL DSN
# ---------------------------------------------------------------------------

_PG_TESTS_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_PG_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"


def _swap_database(dsn: str, new_db: str) -> tuple[str, str]:
    """Return (maintenance_dsn, original_dbname) for the given DSN.

    The maintenance DSN points to ``new_db`` (typically ``postgres``) on the
    same host/credentials. Used for DROP/CREATE DATABASE operations that
    cannot run against the target DB itself.
    """
    parts = urlsplit(dsn)
    original_db = parts.path.lstrip("/") or ""
    new_parts = parts._replace(path=f"/{new_db}")
    return urlunsplit(new_parts), original_db


@pytest_asyncio.fixture(scope="function")
async def ephemeral_pg_dsn() -> AsyncGenerator[str, None]:
    """Yield a PostgreSQL DSN backed by a freshly-recreated database.

    Skip behaviour
    --------------
    The fixture skips the test (via ``pytest.skip``) if the ``ATM_ENABLE_PG_TESTS``
    environment variable is not set to "1", "true", or "yes".

    Isolation strategy
    ------------------
    Between every test the **entire target database is dropped and
    recreated** via a maintenance connection to the ``postgres`` admin DB
    (``DROP DATABASE … WITH (FORCE)``, PG 13+). This kills every backend
    on the target DB and wipes all schema state, eliminating cross-test
    races on pg_catalog (pg_type_typname_nsp_index) that can occur when
    subprocess workers from prior tests (e.g. ``atm grid``
    ProcessPoolExecutor children) leak open connections during fixture
    teardown.

    The Python schema is then re-applied with ``Base.metadata.create_all``.

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
    from sqlalchemy.ext.asyncio import create_async_engine

    from atm.storage.models import Base
    from atm.storage.session import create_engine

    # Build maintenance DSN pointing to the `postgres` admin DB.
    maint_dsn, target_db = _swap_database(dsn, "postgres")
    if not target_db:
        raise RuntimeError(
            f"ephemeral_pg_dsn: cannot determine target DB from DSN {dsn!r} — "
            "expected …/<dbname> at the end"
        )

    # DROP DATABASE / CREATE DATABASE must run with AUTOCOMMIT (no txn block)
    # and cannot target the current DB → use a maintenance connection.
    maint_engine = create_async_engine(
        maint_dsn,
        echo=False,
        pool_size=1,
        max_overflow=0,
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with maint_engine.connect() as conn:
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{target_db}" WITH (FORCE)'))
            await conn.execute(text(f'CREATE DATABASE "{target_db}"'))
    finally:
        await maint_engine.dispose()

    # Now the target DB is brand-new and empty — apply the model schema.
    engine = create_engine(dsn, echo=False, pool_size=2, max_overflow=1)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield dsn
    finally:
        # No teardown DROP needed — the next test (or a final pytest-session
        # hook, if added later) will nuke this DB again. We only dispose
        # the engine here to release this test's connections promptly.
        await engine.dispose()
