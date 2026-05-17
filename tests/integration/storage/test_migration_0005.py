"""Integration test: Alembic migration 0005 upgrade/downgrade round-trip.

Verifies that migration 0005 (study_sessions table, human_request_queue table,
study_session_id FK column on human_interactions) applies and reverts cleanly
against a real PostgreSQL instance.

Test is gated by ATM_ENABLE_PG_TESTS=1 — skipped in CI unit-only jobs.

Strategy: function-local subprocess-driven upgrade/downgrade so the test is
self-contained and does not rely on the session-scoped pg_engine_alembic
fixture (which only runs upgrade head once per session).

Migration path exercised:
  upgrade 0004 → inspect → upgrade head (0005) → inspect → downgrade -1 (→ 0004) → inspect
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
# Helpers: introspect DB schema via async engine
# ---------------------------------------------------------------------------


async def _table_exists(pg_dsn: str, table_name: str) -> bool:
    """Return True if a table with the given name exists in the public schema."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = :tname"
                ),
                {"tname": table_name},
            )
            return result.fetchone() is not None
    finally:
        await engine.dispose()


async def _columns_of(pg_dsn: str, table_name: str) -> set[str]:
    """Return column names of the given table from information_schema."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :tname"
                ),
                {"tname": table_name},
            )
            return {row[0] for row in result}
    finally:
        await engine.dispose()


async def _column_info(pg_dsn: str, table_name: str, column_name: str) -> dict[str, object] | None:
    """Return data_type and is_nullable for a specific column in the given table."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :tname "
                    "AND column_name = :col"
                ),
                {"tname": table_name, "col": column_name},
            )
            row = result.fetchone()
            if row is None:
                return None
            return {"data_type": row[0], "is_nullable": row[1]}
    finally:
        await engine.dispose()


async def _fk_exists(pg_dsn: str, table_name: str, constraint_name: str) -> bool:
    """Return True if a FK constraint with given name exists on the given table."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT pg_constraint.contype "
                    "FROM pg_constraint "
                    "JOIN pg_class ON pg_class.oid = pg_constraint.conrelid "
                    "WHERE pg_class.relname = :tname "
                    "AND pg_constraint.conname = :cname"
                ),
                {"tname": table_name, "cname": constraint_name},
            )
            return result.fetchone() is not None
    finally:
        await engine.dispose()


async def _index_exists(pg_dsn: str, table_name: str, index_name: str) -> bool:
    """Return True if an index with the given name exists on the given table."""
    engine = create_engine(pg_dsn, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public' "
                    "AND tablename = :tname "
                    "AND indexname = :iname"
                ),
                {"tname": table_name, "iname": index_name},
            )
            return result.fetchone() is not None
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Test: migration 0005 upgrade / downgrade round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires_postgres
@pytest.mark.integration
async def test_migration_0005_round_trip() -> None:
    """Upgrade to 0005, verify schema, downgrade to 0004, verify removal.

    Requires ATM_ENABLE_PG_TESTS=1 and a live PostgreSQL instance.

    Verifies:
    - study_sessions table is created with expected columns
    - human_request_queue table is created with expected columns
    - human_interactions.study_session_id column is added (nullable UUID FK)
    - FK fk_human_interactions_study_session_id references study_sessions
    - Indexes on both new tables exist
    - Downgrade cleanly removes all additions
    """
    if not _PG_ENABLED:
        pytest.skip("ATM_ENABLE_PG_TESTS not set — skipping PostgreSQL integration tests")

    pg_dsn = _pg_dsn()

    # -----------------------------------------------------------------------
    # Phase 0: ensure we start from a clean slate at 0004
    # -----------------------------------------------------------------------
    reset_engine = create_engine(pg_dsn, echo=False)
    async with reset_engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await reset_engine.dispose()

    _alembic(["upgrade", "0004"], pg_dsn)

    # -----------------------------------------------------------------------
    # Phase 1: introspect at 0004 — confirm 0005 additions are ABSENT
    # -----------------------------------------------------------------------
    assert not await _table_exists(pg_dsn, "study_sessions"), (
        "study_sessions table should not exist at revision 0004"
    )
    assert not await _table_exists(pg_dsn, "human_request_queue"), (
        "human_request_queue table should not exist at revision 0004"
    )

    hi_cols_at_0004 = await _columns_of(pg_dsn, "human_interactions")
    assert "study_session_id" not in hi_cols_at_0004, (
        "study_session_id column should not exist on human_interactions at revision 0004"
    )

    # -----------------------------------------------------------------------
    # Phase 2: upgrade head → should land at 0005
    # -----------------------------------------------------------------------
    _alembic(["upgrade", "head"], pg_dsn)

    # --- study_sessions table exists with expected columns ---
    assert await _table_exists(pg_dsn, "study_sessions"), (
        "study_sessions table must exist after upgrade to 0005"
    )
    ss_cols = await _columns_of(pg_dsn, "study_sessions")
    for expected_col in ("id", "participant_id", "started_at", "status"):
        assert expected_col in ss_cols, (
            f"Column '{expected_col}' must exist in study_sessions after upgrade to 0005"
        )

    # id must be UUID primary key
    id_info = await _column_info(pg_dsn, "study_sessions", "id")
    assert id_info is not None, "study_sessions.id column must exist"
    assert id_info["data_type"] == "uuid", (
        f"study_sessions.id data_type should be 'uuid', got {id_info['data_type']!r}"
    )

    # started_at must be nullable=NO (NOT NULL)
    started_at_info = await _column_info(pg_dsn, "study_sessions", "started_at")
    assert started_at_info is not None, "study_sessions.started_at column must exist"
    assert started_at_info["is_nullable"] == "NO", (
        f"study_sessions.started_at should be NOT NULL, "
        f"got is_nullable={started_at_info['is_nullable']!r}"
    )

    # --- human_request_queue table exists with expected columns ---
    assert await _table_exists(pg_dsn, "human_request_queue"), (
        "human_request_queue table must exist after upgrade to 0005"
    )
    hrq_cols = await _columns_of(pg_dsn, "human_request_queue")
    for expected_col in ("id", "run_id", "request_id", "context_json", "status", "created_at"):
        assert expected_col in hrq_cols, (
            f"Column '{expected_col}' must exist in human_request_queue after upgrade to 0005"
        )

    # response_json should also be present (populated when human responds)
    assert "response_json" in hrq_cols, (
        "Column 'response_json' must exist in human_request_queue after upgrade to 0005"
    )

    # response_json must be nullable (null until human responds)
    rj_info = await _column_info(pg_dsn, "human_request_queue", "response_json")
    assert rj_info is not None
    assert rj_info["is_nullable"] == "YES", (
        f"human_request_queue.response_json should be nullable, "
        f"got is_nullable={rj_info['is_nullable']!r}"
    )

    # --- human_interactions.study_session_id column is added ---
    hi_cols_after = await _columns_of(pg_dsn, "human_interactions")
    assert "study_session_id" in hi_cols_after, (
        "study_session_id column must exist on human_interactions after upgrade to 0005"
    )

    # Must be nullable UUID
    ssi_info = await _column_info(pg_dsn, "human_interactions", "study_session_id")
    assert ssi_info is not None, "human_interactions.study_session_id column must exist"
    assert ssi_info["data_type"] == "uuid", (
        f"study_session_id data_type should be 'uuid', got {ssi_info['data_type']!r}"
    )
    assert ssi_info["is_nullable"] == "YES", (
        f"study_session_id should be nullable, got is_nullable={ssi_info['is_nullable']!r}"
    )

    # --- FK from human_interactions.study_session_id → study_sessions.id exists ---
    assert await _fk_exists(
        pg_dsn, "human_interactions", "fk_human_interactions_study_session_id"
    ), "FK fk_human_interactions_study_session_id must exist after upgrade to 0005"

    # --- Indexes on new tables exist ---
    assert await _index_exists(pg_dsn, "study_sessions", "study_sessions_participant_id_idx"), (
        "Index study_sessions_participant_id_idx must exist after upgrade to 0005"
    )
    assert await _index_exists(pg_dsn, "human_request_queue", "human_request_queue_run_id_idx"), (
        "Index human_request_queue_run_id_idx must exist after upgrade to 0005"
    )
    assert await _index_exists(pg_dsn, "human_request_queue", "human_request_queue_status_idx"), (
        "Index human_request_queue_status_idx must exist after upgrade to 0005"
    )

    # -----------------------------------------------------------------------
    # Phase 3: downgrade -1 → back to 0004
    # -----------------------------------------------------------------------
    _alembic(["downgrade", "-1"], pg_dsn)

    # --- New tables are gone ---
    assert not await _table_exists(pg_dsn, "study_sessions"), (
        "study_sessions table must be dropped after downgrade to 0004"
    )
    assert not await _table_exists(pg_dsn, "human_request_queue"), (
        "human_request_queue table must be dropped after downgrade to 0004"
    )

    # --- study_session_id column is removed from human_interactions ---
    hi_cols_after_downgrade = await _columns_of(pg_dsn, "human_interactions")
    assert "study_session_id" not in hi_cols_after_downgrade, (
        "study_session_id must be dropped from human_interactions after downgrade to 0004"
    )

    # -----------------------------------------------------------------------
    # Phase 4: restore to head — leave DB in expected state for other tests
    # -----------------------------------------------------------------------
    _alembic(["upgrade", "head"], pg_dsn)
