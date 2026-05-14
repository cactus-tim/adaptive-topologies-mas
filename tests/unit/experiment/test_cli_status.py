"""Unit tests for ``atm status`` CLI command.

PG-free: ``_query_status`` is patched to return a fake row.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

from typer.testing import CliRunner

from atm.experiment.cli import app

runner = CliRunner()


def _fake_row() -> dict[str, object]:
    return {
        "id": str(uuid.uuid4()),
        "name": "exp_alpha",
        "status": "running",
        "started_at": datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC),
        "finished_at": None,
        "total": 12,
        "completed": 7,
        "failed": 1,
        "running": 4,
        "avg_quality": 0.85,
        "total_cost": 1.2345,
    }


def test_status_help_exits_zero() -> None:
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_status_requires_pg_dsn(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("ATM_PG_DSN", raising=False)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 3
    assert "pg-dsn" in result.output.lower() or "pg_dsn" in result.output.lower()


def test_status_table_output() -> None:
    row = _fake_row()

    async def fake_query(*_args: object, **_kwargs: object) -> dict[str, object]:
        return row

    async def _dispose() -> None:
        return None

    with (
        patch("atm.experiment.cli._query_status", new=fake_query),
        patch("atm.experiment.cli.create_engine") as mock_engine_factory,
        patch("atm.experiment.cli.create_session_factory"),
    ):
        mock_engine_factory.return_value.dispose = _dispose
        result = runner.invoke(
            app,
            ["status", "--pg-dsn", "postgresql+asyncpg://x", "--exp-name", "exp_alpha"],
        )

    assert result.exit_code == 0, result.output
    assert "exp_alpha" in result.output
    assert "total cells: 12" in result.output
    assert "completed  : 7" in result.output
    assert "failed: 1" in result.output
    assert "running: 4" in result.output
    assert "0.8500" in result.output
    assert "$1.2345" in result.output


def test_status_json_output() -> None:
    row = _fake_row()

    async def fake_query(*_args: object, **_kwargs: object) -> dict[str, object]:
        return row

    async def _dispose() -> None:
        return None

    with (
        patch("atm.experiment.cli._query_status", new=fake_query),
        patch("atm.experiment.cli.create_engine") as mock_engine_factory,
        patch("atm.experiment.cli.create_session_factory"),
    ):
        mock_engine_factory.return_value.dispose = _dispose
        result = runner.invoke(
            app,
            ["status", "--pg-dsn", "postgresql+asyncpg://x", "--json"],
        )

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output.strip())
    assert parsed["name"] == "exp_alpha"
    assert parsed["total"] == 12
    assert parsed["completed"] == 7
    assert parsed["failed"] == 1
    assert parsed["running"] == 4
    assert abs(parsed["total_cost"] - 1.2345) < 1e-9


def test_status_no_match_exits_one() -> None:
    async def fake_query(*_args: object, **_kwargs: object) -> None:
        return None

    async def _dispose() -> None:
        return None

    with (
        patch("atm.experiment.cli._query_status", new=fake_query),
        patch("atm.experiment.cli.create_engine") as mock_engine_factory,
        patch("atm.experiment.cli.create_session_factory"),
    ):
        mock_engine_factory.return_value.dispose = _dispose
        result = runner.invoke(
            app,
            ["status", "--pg-dsn", "postgresql+asyncpg://x", "--exp-name", "missing"],
        )

    assert result.exit_code == 1
    assert "no matching" in result.output.lower()
