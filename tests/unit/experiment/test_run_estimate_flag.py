"""Unit tests for ``atm run --estimate`` polish.

Tests that:
  1. Without --estimate: behaviour unchanged (no extra echo, no confirm).
  2. With --estimate and cost <= budget/2: prints estimate line, no confirm.
  3. With --estimate and cost > budget/2 + no --yes: typer.confirm is invoked
     and aborts with exit 3 when user says "n".
  4. With --estimate --yes: confirm path is skipped even when over threshold.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from atm.experiment.cli import app
from atm.experiment.estimator import GridEstimate
from atm.experiment.runner import RunResult

runner = CliRunner()

_RUN_ID = uuid.uuid4()
_EXP_ID = uuid.uuid4()


def _make_cfg(per_experiment_usd: float = 1.0) -> MagicMock:
    cfg = MagicMock()
    cfg.observability.pg_dsn = "postgresql+asyncpg://x"
    cfg.budget.per_experiment_usd = per_experiment_usd
    cfg.estimate.use_historical = False
    return cfg


def _make_grid_estimate(cost: float) -> GridEstimate:
    return GridEstimate(
        cells=[],
        total_cost_usd=cost,
        total_input_tokens=1000,
        total_output_tokens=300,
        per_topology={},
    )


def _make_result(status: str = "completed") -> RunResult:
    return RunResult(
        run_id=_RUN_ID,
        exp_id=_EXP_ID,
        status=status,
        metrics={"quality_score": 1.0, "cost_usd": 0.01, "iters": 3},
        final_answer="ok",
    )


async def _dispose() -> None:
    return None


def test_run_without_estimate_flag(tmp_path: Path) -> None:
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text("name: test\n")
    cfg = _make_cfg()

    async def fake_run_one(*_a: object, **_kw: object) -> RunResult:
        return _make_result()

    with (
        patch("atm.experiment.cli.load_config", return_value=cfg),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
        patch("atm.experiment.cli.estimate_grid") as est_mock,
    ):
        result = runner.invoke(app, ["run", "--config", str(cfg_file)])
    assert result.exit_code == 0
    assert "Estimate:" not in result.output
    est_mock.assert_not_called()


def test_run_estimate_below_threshold_no_confirm(tmp_path: Path) -> None:
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text("name: test\n")
    cfg = _make_cfg(per_experiment_usd=10.0)

    async def fake_estimate(**_kw: object) -> GridEstimate:
        return _make_grid_estimate(cost=1.0)  # well below 5.0

    async def fake_run_one(*_a: object, **_kw: object) -> RunResult:
        return _make_result()

    with (
        patch("atm.experiment.cli.load_config", return_value=cfg),
        patch("atm.experiment.cli._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.cli.estimate_grid", new=fake_estimate),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
        patch("atm.experiment.cli.create_engine") as mock_create,
        patch("atm.experiment.cli.create_session_factory"),
        patch("atm.experiment.cli.typer.confirm") as mock_confirm,
    ):
        mock_create.return_value.dispose = _dispose
        result = runner.invoke(app, ["run", "--config", str(cfg_file), "--estimate"])

    assert result.exit_code == 0, result.output
    assert "Estimate:" in result.output
    mock_confirm.assert_not_called()


def test_run_estimate_above_threshold_aborts_without_yes(tmp_path: Path) -> None:
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text("name: test\n")
    cfg = _make_cfg(per_experiment_usd=1.0)

    async def fake_estimate(**_kw: object) -> GridEstimate:
        return _make_grid_estimate(cost=0.9)  # > 0.5 (1.0 / 2)

    async def fake_run_one(*_a: object, **_kw: object) -> RunResult:
        return _make_result()

    with (
        patch("atm.experiment.cli.load_config", return_value=cfg),
        patch("atm.experiment.cli._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.cli.estimate_grid", new=fake_estimate),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
        patch("atm.experiment.cli.create_engine") as mock_create,
        patch("atm.experiment.cli.create_session_factory"),
        patch("atm.experiment.cli.typer.confirm", return_value=False) as mock_confirm,
    ):
        mock_create.return_value.dispose = _dispose
        result = runner.invoke(app, ["run", "--config", str(cfg_file), "--estimate"])

    assert result.exit_code == 3, result.output
    mock_confirm.assert_called_once()
    assert "Aborted" in result.output or "aborted" in result.output.lower()


def test_run_estimate_above_threshold_yes_skips_confirm(tmp_path: Path) -> None:
    cfg_file = tmp_path / "x.yaml"
    cfg_file.write_text("name: test\n")
    cfg = _make_cfg(per_experiment_usd=1.0)

    async def fake_estimate(**_kw: object) -> GridEstimate:
        return _make_grid_estimate(cost=0.9)

    async def fake_run_one(*_a: object, **_kw: object) -> RunResult:
        return _make_result()

    with (
        patch("atm.experiment.cli.load_config", return_value=cfg),
        patch("atm.experiment.cli._load_pricing", return_value=MagicMock()),
        patch("atm.experiment.cli.estimate_grid", new=fake_estimate),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
        patch("atm.experiment.cli.create_engine") as mock_create,
        patch("atm.experiment.cli.create_session_factory"),
        patch("atm.experiment.cli.typer.confirm") as mock_confirm,
    ):
        mock_create.return_value.dispose = _dispose
        result = runner.invoke(app, ["run", "--config", str(cfg_file), "--estimate", "--yes"])

    assert result.exit_code == 0, result.output
    mock_confirm.assert_not_called()
