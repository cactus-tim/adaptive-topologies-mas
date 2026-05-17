"""Conftest for tests/integration/human — M14 E2E fixtures.

Provides:
  truncate_m14_tables — TRUNCATE the four M14 tables between tests.

All other PG fixtures (pg_dsn, pg_engine_fast, session_factory_fast, etc.)
are inherited automatically from tests/integration/conftest.py via pytest's
conftest hierarchy.
"""

from __future__ import annotations

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@pytest_asyncio.fixture
async def truncate_m14_tables(pg_engine_fast: AsyncEngine) -> None:
    """TRUNCATE the four M14-specific tables before (and after) the test.

    Uses RESTART IDENTITY CASCADE so that any auto-generated sequences are
    reset and FK-referencing rows are deleted atomically.

    Runs as a function-scoped fixture yielded *before* the test body so any
    rows inserted by a previous test (or a dirty DB state) don't leak.
    """
    _tables = [
        "human_request_queue",
        "human_interactions",
        "study_sessions",
        "runs",
        "experiments",
    ]
    async with pg_engine_fast.begin() as conn:
        for tbl in _tables:
            await conn.execute(
                text(f"TRUNCATE TABLE {tbl} RESTART IDENTITY CASCADE")
            )
    yield
    # Post-test cleanup — belt-and-suspenders
    async with pg_engine_fast.begin() as conn:
        for tbl in _tables:
            await conn.execute(
                text(f"TRUNCATE TABLE {tbl} RESTART IDENTITY CASCADE")
            )
