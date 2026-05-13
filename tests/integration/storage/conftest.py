"""Storage integration test fixtures — thin re-export from parent conftest.

All shared PG fixtures (pg_dsn, pg_engine_fast, pg_engine_alembic,
session_factory_fast) are now defined in tests/integration/conftest.py and
automatically available to tests in this subdirectory via pytest's conftest
inheritance.

This file is kept to preserve the original fixture-loading behaviour and
to serve as documentation; no local fixture definitions remain.
"""
