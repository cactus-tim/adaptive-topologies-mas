"""Step 5.1 — atomic signature extension of `_update_run_success`/`_update_run_failed`.

Verifies that new kwargs `human_role` and `cognitive_load_proxy` are passed through
into the SQL UPDATE statement. Uses AsyncMock-driven session to capture the
compiled values clause.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa

from atm.experiment.runner import _update_run_failed, _update_run_success


@pytest.fixture
def session_factory() -> tuple[MagicMock, AsyncMock]:
    """Build a session_factory that yields a captured AsyncMock session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.close = AsyncMock()
    session.rollback = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock(return_value=session)
    return factory, session


def _captured_update_values(session: AsyncMock) -> dict[str, object]:
    assert session.execute.await_count >= 1
    stmt = session.execute.await_args.args[0]
    assert isinstance(stmt, sa.sql.dml.Update)
    return dict(stmt._values or {})


@pytest.mark.asyncio
async def test_update_run_success_passes_new_kwargs(
    session_factory: tuple[MagicMock, AsyncMock],
) -> None:
    factory, session = session_factory
    run_id = uuid.uuid4()
    exp_id = uuid.uuid4()
    await _update_run_success(
        factory,
        run_id,
        exp_id,
        quality_score=0.75,
        budget_spent_usd=0.12,
        iterations=3,
        human_role="Reviewer",
        cognitive_load_proxy=2.5,
    )
    values = _captured_update_values(session)
    rendered_keys = {str(k) for k in values}
    assert any("cognitive_load_proxy" in k for k in rendered_keys)
    assert any("human_role" in k for k in rendered_keys)


@pytest.mark.asyncio
async def test_update_run_success_defaults_back_compat(
    session_factory: tuple[MagicMock, AsyncMock],
) -> None:
    """Without new kwargs, helper still works and writes cognitive_load_proxy=None."""
    factory, session = session_factory
    run_id = uuid.uuid4()
    exp_id = uuid.uuid4()
    await _update_run_success(
        factory,
        run_id,
        exp_id,
        quality_score=1.0,
        budget_spent_usd=0.05,
        iterations=1,
    )
    values = _captured_update_values(session)
    rendered_keys = {str(k) for k in values}
    assert any("cognitive_load_proxy" in k for k in rendered_keys)
    assert not any("human_role" in k for k in rendered_keys)


@pytest.mark.asyncio
async def test_update_run_failed_passes_new_kwargs(
    session_factory: tuple[MagicMock, AsyncMock],
) -> None:
    factory, session = session_factory
    run_id = uuid.uuid4()
    await _update_run_failed(
        factory,
        run_id,
        status="failed",
        finish_reason="error",
        quality_score=0.0,
        budget_spent_usd=0.01,
        iterations=0,
        error_text="boom",
        human_role="Coordinator",
        cognitive_load_proxy=0.0,
    )
    values = _captured_update_values(session)
    rendered_keys = {str(k) for k in values}
    assert any("cognitive_load_proxy" in k for k in rendered_keys)
    assert any("human_role" in k for k in rendered_keys)
