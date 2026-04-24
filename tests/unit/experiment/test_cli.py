"""Unit tests for the ATM Typer CLI (atm run).

Tests use typer.testing.CliRunner (synchronous, no real DB/LLM calls).

Test coverage:
  1. --help exits 0 and shows help text
  2. Success path: status="completed" → exit 0
  3. Config error (missing file) → exit 3
  4. Budget exceeded: status="budget_exceeded" → exit 2
  5. Failed run: status="failed" → exit 1
  6. Output format is correct (Run <uuid>: status=..., quality=..., cost=$..., iters=...)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from atm.experiment.cli import app
from atm.experiment.runner import RunResult

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_ID = uuid.uuid4()
_EXP_ID = uuid.uuid4()


def _make_result(
    status: str, quality: float = 1.0, cost: float = 0.01, iters: int = 3
) -> RunResult:
    return RunResult(
        run_id=_RUN_ID,
        exp_id=_EXP_ID,
        status=status,
        metrics={
            "quality_score": quality,
            "cost_usd": cost,
            "iters": iters,
        },
        final_answer="fib(10)=55",
    )


# ---------------------------------------------------------------------------
# Test 1: --help exits 0 and shows help text
# ---------------------------------------------------------------------------


def test_help_exits_zero() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    output_lower = result.output.lower()
    assert "--config" in result.output or "config" in output_lower


# ---------------------------------------------------------------------------
# Test 2: Success path → exit 0
# ---------------------------------------------------------------------------


def test_run_success_exits_zero(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    mock_cfg = MagicMock()
    completed_result = _make_result("completed")

    async def fake_run_one(cfg: object) -> RunResult:
        return completed_result

    with (
        patch("atm.experiment.cli.load_config", return_value=mock_cfg),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
    ):
        result = runner.invoke(app, ["run", "--config", str(cfg_file)])

    assert result.exit_code == 0, (
        f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    )


# ---------------------------------------------------------------------------
# Test 3: Config error (missing file) → exit 3
# ---------------------------------------------------------------------------


def test_config_error_exits_3(tmp_path: Path) -> None:
    nonexistent = tmp_path / "no_such_file.yaml"

    result = runner.invoke(app, ["run", "--config", str(nonexistent)])

    assert result.exit_code == 3, (
        f"Expected exit 3, got {result.exit_code}. Output: {result.output}"
    )


# ---------------------------------------------------------------------------
# Test 4: Budget exceeded → exit 2
# ---------------------------------------------------------------------------


def test_budget_exceeded_exits_2(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    mock_cfg = MagicMock()
    budget_result = _make_result("budget_exceeded")

    async def fake_run_one(cfg: object) -> RunResult:
        return budget_result

    with (
        patch("atm.experiment.cli.load_config", return_value=mock_cfg),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
    ):
        result = runner.invoke(app, ["run", "--config", str(cfg_file)])

    assert result.exit_code == 2, (
        f"Expected exit 2, got {result.exit_code}. Output: {result.output}"
    )


# ---------------------------------------------------------------------------
# Test 5: Failed run → exit 1
# ---------------------------------------------------------------------------


def test_failed_run_exits_1(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    mock_cfg = MagicMock()
    failed_result = _make_result("failed")

    async def fake_run_one(cfg: object) -> RunResult:
        return failed_result

    with (
        patch("atm.experiment.cli.load_config", return_value=mock_cfg),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
    ):
        result = runner.invoke(app, ["run", "--config", str(cfg_file)])

    assert result.exit_code == 1, (
        f"Expected exit 1, got {result.exit_code}. Output: {result.output}"
    )


# ---------------------------------------------------------------------------
# Test 6: Output format is correct
# ---------------------------------------------------------------------------


def test_output_format(tmp_path: Path) -> None:
    cfg_file = tmp_path / "smoke.yaml"
    cfg_file.write_text("name: test\n")

    mock_cfg = MagicMock()
    completed_result = _make_result("completed", quality=0.95, cost=0.0123, iters=5)

    async def fake_run_one(cfg: object) -> RunResult:
        return completed_result

    with (
        patch("atm.experiment.cli.load_config", return_value=mock_cfg),
        patch("atm.experiment.cli.run_one", new=fake_run_one),
    ):
        result = runner.invoke(app, ["run", "--config", str(cfg_file)])

    assert result.exit_code == 0
    output = result.output
    assert f"Run {_RUN_ID}" in output
    assert "status=completed" in output
    assert "quality=0.95" in output
    assert "cost=$0.0123" in output
    assert "iters=5" in output
