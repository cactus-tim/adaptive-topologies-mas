"""Integration test: Alembic migration 0004 upgrade/downgrade round-trip.

Verifies that migration 0004 (replay_of, host, process_pid columns + FK + index)
applies and reverts cleanly against a real PostgreSQL instance.

Test is gated by ATM_ENABLE_PG_TESTS=1 — skipped in CI unit-only jobs.

Strategy: function-local subprocess-driven upgrade/downgrade so that the test
is self-contained and does not rely on the session-scoped pg_engine_alembic
fixture (which only runs upgrade head once per session).

Migration path exercised:
  upgrade 0003 → inspect → upgrade head (0004) → inspect → downgrade -1 (→ 0003) → inspect

The database is left at upgrade head at the end of the test to avoid breaking
subsequent session-scoped fixtures.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from sqlalchemy import text

from atm.storage.session import create_engine

# ---------------------------------------------------------------------------
# Guard: skip immediately if PG integration flag is not set
# ---------------------------------------------------------------------------

_PG_ENABLED = os.environ.get("ATM_ENABLE_PG_TESTS", "") in ("1", "true", "yes")
_DEFAULT_DSN = "postgresql+asyncpg://atm:atm@localhost:5432/atm_test"


def _pg_dsn() -> str:
    return os.environ.get("ATM_PG_DSN", _DEFAULT_DSN)


# ---------------------------------------------------------------------------
# Helper: run alembic subcommand with PG_DSN forwarded
# ---------------------------------------------------------------------------


def _alembic(cmd: list[str], pg_dsn: str) -> None:
    """Run `uv run alembic <cmd>` with PG_DSN in the environment."""
    env = os.environ.copy()
    env["PG_DSN"] = pg_dsn
    subprocess.run(
        ["uv", "run", "alembic", *cmd],
        check=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Helpers: introspect DB schema via synchronous-friendly async calls
# ---------------------------------------------------------------------------


async def _columns_of_runs(pg_dsn: str) -> set[str]:
    """Return column names of the 'runs' table from information_schema."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'runs'"
                )
            )
            return {row[0] for row in result}
    finally:
        await engine.dispose()


async def _column_info(pg_dsn: str, column_name: str) -> dict[str, object] | None:
    """Return data_type and is_nullable for a specific column in 'runs'."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = 'runs' "
                    "AND column_name = :col"
                ),
                {"col": column_name},
            )
            row = result.fetchone()
            if row is None:
                return None
            return {"data_type": row[0], "is_nullable": row[1]}
    finally:
        await engine.dispose()


async def _fk_exists(pg_dsn: str, constraint_name: str) -> bool:
    """Return True if a FK constraint with given name exists on the 'runs' table."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT pg_constraint.contype, pg_constraint.confdeltype "
                    "FROM pg_constraint "
                    "JOIN pg_class ON pg_class.oid = pg_constraint.conrelid "
                    "WHERE pg_class.relname = 'runs' "
                    "AND pg_constraint.conname = :cname"
                ),
                {"cname": constraint_name},
            )
            row = result.fetchone()
            return row is not None
    finally:
        await engine.dispose()


async def _fk_confdeltype(pg_dsn: str, constraint_name: str) -> str | None:
    """Return the confdeltype code for a FK on 'runs', or None if not found."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT pg_constraint.confdeltype "
                    "FROM pg_constraint "
                    "JOIN pg_class ON pg_class.oid = pg_constraint.conrelid "
                    "WHERE pg_class.relname = 'runs' "
                    "AND pg_constraint.conname = :cname"
                ),
                {"cname": constraint_name},
            )
            row = result.fetchone()
            return row[0].decode() if row is not None else None
    finally:
        await engine.dispose()


async def _index_exists(pg_dsn: str, index_name: str) -> bool:
    """Return True if an index with the given name exists on 'runs'."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public' "
                    "AND tablename = 'runs' "
                    "AND indexname = :iname"
                ),
                {"iname": index_name},
            )
            return result.fetchone() is not None
    finally:
        await engine.dispose()


async def _index_columns(pg_dsn: str, index_name: str) -> str | None:
    """Return the indexdef string for the given index, or None if not found."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname = 'public' "
                    "AND tablename = 'runs' "
                    "AND indexname = :iname"
                ),
                {"iname": index_name},
            )
            row = result.fetchone()
            return str(row[0]) if row is not None else None
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Test: migration 0004 upgrade / downgrade round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_migration_0004_round_trip() -> None:
    """Upgrade to 0004, verify schema, downgrade to 0003, verify removal.

    Requires ATM_ENABLE_PG_TESTS=1 and a live PostgreSQL instance.
    """
    if not _PG_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set — skipping PostgreSQL integration tests")

    pg_dsn = _pg_dsn()

    # -----------------------------------------------------------------------
    # Phase 0: ensure we start from a clean slate at 0003
    # -----------------------------------------------------------------------
    # Reset schema, then upgrade to 0003 so we have the baseline the test
    # requires (this mirrors how conftest.py resets before session-scoped runs).
    reset_engine = create_engine(pg_dsn, echo=False)
    async with reset_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await reset_engine.dispose()

    _alembic(["upgrade", "0003"], pg_dsn)

    # -----------------------------------------------------------------------
    # Phase 1: introspect at 0003 — confirm 0004 columns are ABSENT
    # -----------------------------------------------------------------------
    cols_at_0003 = await _columns_of_runs(pg_dsn)
    assert "replay_of" not in cols_at_0003, "replay_of should not exist at revision 0003"
    assert "host" not in cols_at_0003, "host should not exist at revision 0003"
    assert "process_pid" not in cols_at_0003, "process_pid should not exist at revision 0003"
    assert not await _fk_exists(pg_dsn, "fk_runs_replay_of_runs"), (
        "FK fk_runs_replay_of_runs should not exist at revision 0003"
    )
    assert not await _index_exists(pg_dsn, "runs_exp_status_idx"), (
        "Index runs_exp_status_idx should not exist at revision 0003"
    )

    # -----------------------------------------------------------------------
    # Phase 2: upgrade head → should land at 0004
    # -----------------------------------------------------------------------
    _alembic(["upgrade", "head"], pg_dsn)

    # --- Columns exist and are nullable ---
    replay_of_info = await _column_info(pg_dsn, "replay_of")
    assert replay_of_info is not None, "Column 'replay_of' must exist after upgrade to 0004"
    # PostgreSQL reports UUID as 'uuid'
    assert replay_of_info["data_type"] == "uuid", (
        f"replay_of data_type should be 'uuid', got {replay_of_info['data_type']!r}"
    )
    assert replay_of_info["is_nullable"] == "YES", (
        f"replay_of should be nullable, got is_nullable={replay_of_info['is_nullable']!r}"
    )

    host_info = await _column_info(pg_dsn, "host")
    assert host_info is not None, "Column 'host' must exist after upgrade to 0004"
    # character varying
    assert (
        "character varying" in host_info["data_type"]
        or host_info["data_type"] == "character varying"
    ), f"host data_type should be character varying, got {host_info['data_type']!r}"
    assert host_info["is_nullable"] == "YES", (
        f"host should be nullable, got is_nullable={host_info['is_nullable']!r}"
    )

    pid_info = await _column_info(pg_dsn, "process_pid")
    assert pid_info is not None, "Column 'process_pid' must exist after upgrade to 0004"
    assert pid_info["data_type"] == "integer", (
        f"process_pid data_type should be 'integer', got {pid_info['data_type']!r}"
    )
    assert pid_info["is_nullable"] == "YES", (
        f"process_pid should be nullable, got is_nullable={pid_info['is_nullable']!r}"
    )

    # --- FK exists with confdeltype = 'n' (SET NULL) ---
    assert await _fk_exists(pg_dsn, "fk_runs_replay_of_runs"), (
        "FK fk_runs_replay_of_runs must exist after upgrade to 0004"
    )
    confdeltype = await _fk_confdeltype(pg_dsn, "fk_runs_replay_of_runs")
    assert confdeltype == "n", (
        f"FK fk_runs_replay_of_runs confdeltype should be 'n' (SET NULL), got {confdeltype!r}"
    )

    # --- Index exists and covers (exp_id, status) ---
    assert await _index_exists(pg_dsn, "runs_exp_status_idx"), (
        "Index runs_exp_status_idx must exist after upgrade to 0004"
    )
    indexdef = await _index_columns(pg_dsn, "runs_exp_status_idx")
    assert indexdef is not None
    assert "exp_id" in indexdef and "status" in indexdef, (
        f"Index runs_exp_status_idx should cover (exp_id, status), got: {indexdef!r}"
    )

    # -----------------------------------------------------------------------
    # Phase 3: downgrade -1 → back to 0003
    # -----------------------------------------------------------------------
    _alembic(["downgrade", "-1"], pg_dsn)

    # --- 0004 columns are gone ---
    cols_after_downgrade = await _columns_of_runs(pg_dsn)
    assert "replay_of" not in cols_after_downgrade, (
        "replay_of must be dropped after downgrade to 0003"
    )
    assert "host" not in cols_after_downgrade, "host must be dropped after downgrade to 0003"
    assert "process_pid" not in cols_after_downgrade, (
        "process_pid must be dropped after downgrade to 0003"
    )

    # --- FK and index are gone ---
    assert not await _fk_exists(pg_dsn, "fk_runs_replay_of_runs"), (
        "FK fk_runs_replay_of_runs must be dropped after downgrade to 0003"
    )
    assert not await _index_exists(pg_dsn, "runs_exp_status_idx"), (
        "Index runs_exp_status_idx must be dropped after downgrade to 0003"
    )

    # -----------------------------------------------------------------------
    # Phase 4: restore to head — leave DB in expected state for other tests
    # -----------------------------------------------------------------------
    _alembic(["upgrade", "head"], pg_dsn)
