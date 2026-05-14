"""Unit tests for the ``atm estimate`` CLI command.

Tests rely on patching ``load_grid_configs``, ``estimate_grid``, and
``_load_pricing`` so no real config / DB / pricing files are required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from atm.experiment.cli import app
from atm.experiment.estimator import CellEstimate, GridEstimate

runner = CliRunner()


def _fake_grid_estimate() -> GridEstimate:
    return GridEstimate(
        cells=[
            CellEstimate(
                topology="star",
                task="humaneval",
                seed=42,
                est_input_tokens=9000,
                est_output_tokens=3000,
                est_cost_usd=0.0234,
                source="historical_avg",
            ),
            CellEstimate(
                topology="chain",
                task="humaneval",
                seed=42,
                est_input_tokens=12000,
                est_output_tokens=4000,
                est_cost_usd=0.0312,
                source="heuristic",
            ),
        ],
        total_cost_usd=0.0546,
        total_input_tokens=21000,
        total_output_tokens=7000,
        per_topology={"star": 0.0234, "chain": 0.0312},
    )


# ---------------------------------------------------------------------------
# Test 1: --help works
# ---------------------------------------------------------------------------


def test_estimate_help_exits_zero() -> None:
    result = runner.invoke(app, ["estimate", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output or "config" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 2: success path → exit 0, table rendered
# ---------------------------------------------------------------------------


def test_estimate_renders_table_and_total(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    # Mock cfg with .observability.pg_dsn + .estimate.use_historical
    mock_cfg = MagicMock()
    mock_cfg.observability.pg_dsn = "postgresql+asyncpg://x"
    mock_cfg.estimate.use_historical = False

    async def fake_estimate(**kwargs: object) -> GridEstimate:
        return _fake_grid_estimate()

    with (
        patch("atm.experiment.cli.load_grid_configs", return_value=[mock_cfg]),
        patch("atm.experiment.cli._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.cli.estimate_grid", new=fake_estimate),
    ):
        result = runner.invoke(app, ["estimate", "--config", str(cfg_file)])

    assert result.exit_code == 0, result.output
    assert "star" in result.output
    assert "chain" in result.output
    assert "TOTAL" in result.output
    assert "0.0546" in result.output
    # source column rendered
    assert "historical_avg" in result.output
    assert "heuristic" in result.output


# ---------------------------------------------------------------------------
# Test 3: missing config file → exit 3
# ---------------------------------------------------------------------------


def test_estimate_missing_config_exits_3(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    result = runner.invoke(app, ["estimate", "--config", str(missing)])
    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# Test 4: empty config list → exit 0 with "no configs" message
# ---------------------------------------------------------------------------


def test_estimate_empty_configs_exits_zero(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    with patch("atm.experiment.cli.load_grid_configs", return_value=[]):
        result = runner.invoke(app, ["estimate", "--config", str(cfg_file)])

    assert result.exit_code == 0
    assert "no configs" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 5: use_historical=True → session_factory is passed through
# ---------------------------------------------------------------------------


def test_estimate_historical_opens_session(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    mock_cfg = MagicMock()
    mock_cfg.observability.pg_dsn = "postgresql+asyncpg://x"
    mock_cfg.estimate.use_historical = True

    seen_factory: list[object] = []

    async def fake_estimate(*, session_factory: object, **_: object) -> GridEstimate:
        seen_factory.append(session_factory)
        return _fake_grid_estimate()

    with (
        patch("atm.experiment.cli.load_grid_configs", return_value=[mock_cfg]),
        patch("atm.experiment.cli._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.cli.estimate_grid", new=fake_estimate),
        patch("atm.experiment.cli.create_engine") as mock_create,
        patch("atm.experiment.cli.create_session_factory") as mock_factory,
    ):
        # The CLI calls engine.dispose() via asyncio.run; provide an async dispose.
        mock_engine = MagicMock()

        async def _dispose() -> None:
            return None

        mock_engine.dispose = _dispose
        mock_create.return_value = mock_engine
        sentinel_factory = MagicMock(name="sentinel_factory")
        mock_factory.return_value = sentinel_factory
        result = runner.invoke(app, ["estimate", "--config", str(cfg_file)])

    assert result.exit_code == 0, result.output
    assert seen_factory == [next((f for f in seen_factory if f is not None), seen_factory[0])]
    # session_factory should be the sentinel
    assert seen_factory[0] is not None
