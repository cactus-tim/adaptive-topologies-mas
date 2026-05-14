"""Unit tests for `atm resume`, `atm replay`, `atm reconcile` CLI commands.

These tests stub the underlying async functions via `unittest.mock.patch` so
no PG / LLM is needed.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from atm.experiment.cli import app
from atm.experiment.reconcile import ReconcileReport, ZombieRow
from atm.experiment.runner import RunResult

runner = CliRunner()

_RID = uuid.uuid4()
_EID = uuid.uuid4()


def _ok_result(status: str = "completed") -> RunResult:
    return RunResult(
        run_id=_RID,
        exp_id=_EID,
        status=status,
        metrics={"quality_score": 0.9, "cost_usd": 0.05, "iters": 4},
        final_answer="ans",
    )


# ---------------------------------------------------------------------------
# atm resume
# ---------------------------------------------------------------------------


def test_resume_success_exit_0() -> None:
    with patch("atm.experiment.cli.resume_one") as mock_resume:
        mock_resume.return_value = _ok_result("completed")
        result = runner.invoke(app, ["resume", "--run-id", str(_RID)])
    assert result.exit_code == 0
    assert "Resumed run" in result.output
    assert "completed" in result.output


def test_resume_invalid_uuid_exit_3() -> None:
    result = runner.invoke(app, ["resume", "--run-id", "not-a-uuid"])
    assert result.exit_code == 3
    assert "Invalid run_id" in result.output


def test_resume_lookup_error_exit_3() -> None:
    with patch("atm.experiment.cli.resume_one", side_effect=LookupError("nope")):
        result = runner.invoke(app, ["resume", "--run-id", str(_RID)])
    assert result.exit_code == 3
    assert "Resume error" in result.output


def test_resume_runtime_blocked_exit_1() -> None:
    with patch(
        "atm.experiment.cli.resume_one",
        side_effect=RuntimeError("still alive"),
    ):
        result = runner.invoke(app, ["resume", "--run-id", str(_RID)])
    assert result.exit_code == 1
    assert "Resume blocked" in result.output


def test_resume_force_flag_threaded() -> None:
    with patch("atm.experiment.cli.resume_one") as mock_resume:
        mock_resume.return_value = _ok_result()
        runner.invoke(app, ["resume", "--run-id", str(_RID), "--force"])
        # force=True should be passed through.
        _, kwargs = mock_resume.call_args
        assert kwargs.get("force") is True


def test_resume_budget_exceeded_exit_2() -> None:
    with patch("atm.experiment.cli.resume_one") as mock_resume:
        mock_resume.return_value = _ok_result("budget_exceeded")
        result = runner.invoke(app, ["resume", "--run-id", str(_RID)])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# atm replay
# ---------------------------------------------------------------------------


def test_replay_success_exit_0() -> None:
    with patch("atm.experiment.cli.replay_one") as mock_replay:
        mock_replay.return_value = _ok_result("completed")
        result = runner.invoke(app, ["replay", str(_RID)])
    assert result.exit_code == 0
    assert "Replayed run" in result.output


def test_replay_default_mode_is_deterministic() -> None:
    with patch("atm.experiment.cli.replay_one") as mock_replay:
        mock_replay.return_value = _ok_result()
        runner.invoke(app, ["replay", str(_RID)])
        _, kwargs = mock_replay.call_args
        assert kwargs.get("mode") == "deterministic"


def test_replay_semantic_mode_threaded() -> None:
    with patch("atm.experiment.cli.replay_one") as mock_replay:
        mock_replay.return_value = _ok_result()
        runner.invoke(app, ["replay", str(_RID), "--mode", "semantic"])
        _, kwargs = mock_replay.call_args
        assert kwargs.get("mode") == "semantic"


def test_replay_unknown_mode_exit_3() -> None:
    result = runner.invoke(app, ["replay", str(_RID), "--mode", "fancy"])
    assert result.exit_code == 3
    assert "Invalid --mode" in result.output


def test_replay_invalid_uuid_exit_3() -> None:
    result = runner.invoke(app, ["replay", "not-a-uuid"])
    assert result.exit_code == 3
    assert "Invalid run_id" in result.output


def test_replay_missing_parquet_exit_3() -> None:
    with patch(
        "atm.experiment.cli.replay_one",
        side_effect=FileNotFoundError("no parquet"),
    ):
        result = runner.invoke(app, ["replay", str(_RID)])
    assert result.exit_code == 3


def test_replay_value_error_exit_3() -> None:
    with patch(
        "atm.experiment.cli.replay_one",
        side_effect=ValueError("parent mismatch"),
    ):
        result = runner.invoke(app, ["replay", str(_RID)])
    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# atm reconcile
# ---------------------------------------------------------------------------


def test_reconcile_requires_atm_pg_dsn(monkeypatch) -> None:
    monkeypatch.delenv("ATM_PG_DSN", raising=False)
    result = runner.invoke(app, ["reconcile", "--exp-id", str(_EID)])
    assert result.exit_code == 3
    assert "ATM_PG_DSN" in result.output


def test_reconcile_invalid_uuid(monkeypatch) -> None:
    monkeypatch.setenv("ATM_PG_DSN", "postgresql://localhost/x")
    result = runner.invoke(app, ["reconcile", "--exp-id", "bad"])
    assert result.exit_code == 3


def test_reconcile_success_emits_json(monkeypatch) -> None:
    monkeypatch.setenv("ATM_PG_DSN", "postgresql://localhost/x")
    fake_report = ReconcileReport(
        scanned=2,
        zombies=[
            ZombieRow(run_id=_RID, host="h", pid=99, reason="pid_dead"),
        ],
        actions={_RID: "marked_failed"},
    )

    async def _stub(*a, **kw):
        return fake_report

    # MagicMock engine whose .dispose() is async (awaited by CLI).
    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()

    # Patch the inner async fn import path
    with (
        patch("atm.experiment.reconcile.reconcile_zombies", side_effect=_stub),
        patch("atm.storage.session.create_engine", return_value=fake_engine),
        patch("atm.storage.session.create_session_factory"),
    ):
        result = runner.invoke(app, ["reconcile", "--exp-id", str(_EID)])

    assert result.exit_code == 0, f"Output: {result.output}\nExc: {result.exception}"
    assert '"scanned": 2' in result.output
    assert '"reason": "pid_dead"' in result.output


def test_reconcile_dry_run_flag(monkeypatch) -> None:
    monkeypatch.setenv("ATM_PG_DSN", "postgresql://localhost/x")
    fake_report = ReconcileReport(scanned=0)

    async def _stub(session_factory, exp_id, *, allow_force_resume=False, current_host=None):
        # Verify dry_run propagated as allow_force_resume=True.
        assert allow_force_resume is True
        return fake_report

    fake_engine = MagicMock()
    fake_engine.dispose = AsyncMock()

    with (
        patch("atm.experiment.reconcile.reconcile_zombies", side_effect=_stub),
        patch("atm.storage.session.create_engine", return_value=fake_engine),
        patch("atm.storage.session.create_session_factory"),
    ):
        result = runner.invoke(app, ["reconcile", "--exp-id", str(_EID), "--dry-run"])

    assert result.exit_code == 0, f"Output: {result.output}\nExc: {result.exception}"
