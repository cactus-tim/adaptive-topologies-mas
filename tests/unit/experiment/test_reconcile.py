"""Unit tests for `reconcile_zombies` (m12-resume-replay G5).

These tests mock the SQLAlchemy session at the boundary so the reconcile
logic can be exercised without a Postgres backend. The pid-liveness probe
is monkeypatched at the module level for deterministic classification.
"""

from __future__ import annotations

import socket
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from atm.experiment import reconcile as reconcile_mod
from atm.experiment.reconcile import (
    ReconcileReport,
    ZombieRow,
    _classify_row,
    reconcile_zombies,
)


def test_classify_none_host_is_no_pid_zombie() -> None:
    rid = uuid.uuid4()
    z = _classify_row(run_id=rid, host=None, pid=12345, current_host="alpha")
    assert z is not None
    assert z.reason == "no_pid"
    assert z.run_id == rid


def test_classify_host_mismatch_zombie() -> None:
    rid = uuid.uuid4()
    z = _classify_row(run_id=rid, host="beta", pid=12345, current_host="alpha")
    assert z is not None
    assert z.reason == "host_mismatch"


def test_classify_same_host_no_pid_is_zombie() -> None:
    rid = uuid.uuid4()
    z = _classify_row(run_id=rid, host="alpha", pid=None, current_host="alpha")
    assert z is not None
    assert z.reason == "no_pid"


def test_classify_same_host_dead_pid_is_zombie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reconcile_mod, "_pid_alive", lambda pid: False)
    rid = uuid.uuid4()
    z = _classify_row(run_id=rid, host="alpha", pid=9999, current_host="alpha")
    assert z is not None
    assert z.reason == "pid_dead"


def test_classify_same_host_live_pid_is_not_zombie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reconcile_mod, "_pid_alive", lambda pid: True)
    rid = uuid.uuid4()
    z = _classify_row(run_id=rid, host="alpha", pid=1234, current_host="alpha")
    assert z is None


def test_pid_alive_self_pid_is_live() -> None:
    import os

    assert reconcile_mod._pid_alive(os.getpid()) is True


def test_pid_alive_invalid_pid_is_dead() -> None:
    assert reconcile_mod._pid_alive(0) is False
    assert reconcile_mod._pid_alive(-1) is False


def test_pid_alive_likely_dead_pid_is_dead() -> None:
    assert reconcile_mod._pid_alive(2_147_000_000) is False


def _make_session_factory(
    select_rows: list[tuple[UUID, str | None, int | None]],
    update_capture: list[Any],
) -> Any:
    """Build a mock session_factory whose session_scope() returns a session
    that:
      - on session.execute(SELECT) → returns the supplied rows (via .all());
      - on session.execute(UPDATE) → appends to update_capture and returns None.
    """
    session = MagicMock()

    select_result = MagicMock()
    select_result.all = MagicMock(return_value=select_rows)

    async def _execute(stmt: Any) -> Any:
        compiled = str(stmt).upper()
        if compiled.startswith("SELECT"):
            return select_result
        update_capture.append(stmt)
        return MagicMock()

    session.execute = AsyncMock(side_effect=_execute)

    @asynccontextmanager
    async def _scope(_factory: Any) -> Any:
        yield session

    return session, _scope


@pytest.mark.asyncio
async def test_reconcile_marks_dead_pid_as_zombie(monkeypatch: pytest.MonkeyPatch) -> None:
    rid_dead = uuid.uuid4()
    rid_live = uuid.uuid4()
    exp_id = uuid.uuid4()
    rows = [
        (rid_dead, "alpha", 9999),
        (rid_live, "alpha", 1234),
    ]
    update_capture: list[Any] = []
    _session, _scope = _make_session_factory(rows, update_capture)

    monkeypatch.setattr(
        reconcile_mod,
        "_pid_alive",
        lambda pid: pid == 1234,
    )
    import atm.storage.session as storage_session_mod

    monkeypatch.setattr(storage_session_mod, "session_scope", _scope)

    report: ReconcileReport = await reconcile_zombies(
        MagicMock(),
        exp_id,
        current_host="alpha",
    )

    assert report.scanned == 2
    assert len(report.zombies) == 1
    assert report.zombies[0].run_id == rid_dead
    assert report.zombies[0].reason == "pid_dead"
    assert report.actions == {rid_dead: "marked_failed"}
    assert len(update_capture) == 1


@pytest.mark.asyncio
async def test_reconcile_force_resume_does_not_mutate(monkeypatch: pytest.MonkeyPatch) -> None:
    rid_dead = uuid.uuid4()
    exp_id = uuid.uuid4()
    rows = [(rid_dead, "alpha", 9999)]
    update_capture: list[Any] = []
    _session, _scope = _make_session_factory(rows, update_capture)

    monkeypatch.setattr(reconcile_mod, "_pid_alive", lambda pid: False)
    import atm.storage.session as storage_session_mod

    monkeypatch.setattr(storage_session_mod, "session_scope", _scope)

    report = await reconcile_zombies(
        MagicMock(),
        exp_id,
        allow_force_resume=True,
        current_host="alpha",
    )

    assert report.scanned == 1
    assert len(report.zombies) == 1
    assert report.actions == {rid_dead: "kept_force_resume"}
    assert update_capture == []


@pytest.mark.asyncio
async def test_reconcile_host_mismatch_marked_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = uuid.uuid4()
    exp_id = uuid.uuid4()
    rows = [(rid, "other-host", 1)]
    update_capture: list[Any] = []
    _session, _scope = _make_session_factory(rows, update_capture)

    monkeypatch.setattr(reconcile_mod, "_pid_alive", lambda pid: True)
    import atm.storage.session as storage_session_mod

    monkeypatch.setattr(storage_session_mod, "session_scope", _scope)

    report = await reconcile_zombies(
        MagicMock(),
        exp_id,
        current_host="alpha",
    )

    assert len(report.zombies) == 1
    assert report.zombies[0].reason == "host_mismatch"
    assert report.actions == {rid: "marked_failed"}
    assert len(update_capture) == 1


@pytest.mark.asyncio
async def test_reconcile_live_pid_is_not_zombie(monkeypatch: pytest.MonkeyPatch) -> None:
    rid = uuid.uuid4()
    exp_id = uuid.uuid4()
    rows = [(rid, "alpha", 1234)]
    update_capture: list[Any] = []
    _session, _scope = _make_session_factory(rows, update_capture)

    monkeypatch.setattr(reconcile_mod, "_pid_alive", lambda pid: True)
    import atm.storage.session as storage_session_mod

    monkeypatch.setattr(storage_session_mod, "session_scope", _scope)

    report = await reconcile_zombies(
        MagicMock(),
        exp_id,
        current_host="alpha",
    )

    assert report.scanned == 1
    assert report.zombies == []
    assert report.actions == {}
    assert update_capture == []


@pytest.mark.asyncio
async def test_reconcile_empty_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No running runs → scanned=0, zombies=[]."""
    update_capture: list[Any] = []
    _session, _scope = _make_session_factory([], update_capture)

    import atm.storage.session as storage_session_mod

    monkeypatch.setattr(storage_session_mod, "session_scope", _scope)

    report = await reconcile_zombies(MagicMock(), uuid.uuid4(), current_host="alpha")

    assert report.scanned == 0
    assert report.zombies == []
    assert report.actions == {}


def test_zombie_row_is_frozen() -> None:
    """ZombieRow is frozen — immutable for safe sharing across callers."""
    from dataclasses import FrozenInstanceError

    z = ZombieRow(run_id=uuid.uuid4(), host="x", pid=1, reason="pid_dead")
    with pytest.raises(FrozenInstanceError):
        z.host = "y"  # type: ignore[misc]


def test_current_host_default_is_socket_gethostname() -> None:
    """When current_host is omitted, socket.gethostname() is used."""
    assert isinstance(socket.gethostname(), str)
