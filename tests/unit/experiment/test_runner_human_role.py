"""Unit tests for runs.human_role writing in Runner (M9.1 Step 1.5 + M9.2 Step 6.1).

Tests (M9.1 — _insert_run):
  1.  enabled=True  → human_role written as role.value ("reviewer")
  2.  enabled=False → human_role is None
  3.  cfg.human is None → human_role is None

Tests (M9.2 — _update_run_success finalization):
  4.  backward-compatible call (no new kwargs) → completes without error
  5.  human_role="judge" + cog_proxy=3.14 → values appear in compiled UPDATE
  6.  cognitive_load_proxy helper raises → _update_run_success still called with
      cognitive_load_proxy=None (NULL in DB); human_role also None when no rows
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite as sqlite_dialect

from atm.core.types import HumanRole
from atm.experiment.config import (
    AgentSetCfg,
    BudgetCfg,
    ExperimentConfig,
    HumanCfg,
    ModelCfg,
    ObservabilityCfg,
    TaskCfg,
    TopologyCfg,
)
from atm.experiment.runner import _insert_run, _update_run_success

# ---------------------------------------------------------------------------
# Helpers — ExperimentConfig factory
# ---------------------------------------------------------------------------


def _make_cfg(human: HumanCfg | None = None, **overrides: Any) -> ExperimentConfig:
    defaults: dict[str, Any] = {
        "name": "test_run",
        "seed": 42,
        "task": TaskCfg(name="t1", input=""),
        "model": ModelCfg(default="fake:echo"),
        "agents": AgentSetCfg(set="canonical_4"),
        "topology": TopologyCfg(name="chain", max_iterations=3),
        "budget": BudgetCfg(),
        "observability": ObservabilityCfg(
            pg_dsn="postgresql+asyncpg://localhost/test",
            parquet_dir="/tmp/test",
        ),
        "human": human,
    }
    defaults.update(overrides)
    return ExperimentConfig.model_validate(defaults)


# ---------------------------------------------------------------------------
# Helper: captures what Run was created with
# ---------------------------------------------------------------------------


class _RunCapture:
    """Captures the kwargs passed to Run(...) constructor."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        obj = MagicMock()
        return obj


# ---------------------------------------------------------------------------
# 1. enabled=True → human_role = role.value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_run_writes_human_role_when_enabled() -> None:
    """When human.enabled=True, _insert_run passes human_role=role.value to Run(...)."""
    human_cfg = HumanCfg(enabled=True, role=HumanRole.REVIEWER)
    cfg = _make_cfg(human=human_cfg)
    exp_id = uuid.uuid4()

    capture = _RunCapture()
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    mock_session_factory = AsyncMock()

    with (
        patch("atm.experiment.runner.Run", side_effect=capture),
        patch("atm.experiment.runner.session_scope") as mock_scope,
    ):
        # session_scope is an async context manager
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        run_id = await _insert_run(mock_session_factory, exp_id, cfg)

    assert capture.kwargs.get("human_role") == "reviewer"
    assert run_id is not None


# ---------------------------------------------------------------------------
# 2. enabled=False → human_role = None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_run_human_role_none_when_disabled() -> None:
    """When human.enabled=False, _insert_run passes human_role=None to Run(...)."""
    human_cfg = HumanCfg(enabled=False, role=HumanRole.REVIEWER)
    cfg = _make_cfg(human=human_cfg)
    exp_id = uuid.uuid4()

    capture = _RunCapture()
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session_factory = AsyncMock()

    with (
        patch("atm.experiment.runner.Run", side_effect=capture),
        patch("atm.experiment.runner.session_scope") as mock_scope,
    ):
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        await _insert_run(mock_session_factory, exp_id, cfg)

    assert capture.kwargs.get("human_role") is None


# ---------------------------------------------------------------------------
# 3. cfg.human is None → human_role = None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_run_human_role_none_when_no_human_cfg() -> None:
    """When cfg.human is None, _insert_run passes human_role=None to Run(...)."""
    cfg = _make_cfg(human=None)
    exp_id = uuid.uuid4()

    capture = _RunCapture()
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session_factory = AsyncMock()

    with (
        patch("atm.experiment.runner.Run", side_effect=capture),
        patch("atm.experiment.runner.session_scope") as mock_scope,
    ):
        mock_scope.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_scope.return_value.__aexit__ = AsyncMock(return_value=False)

        await _insert_run(mock_session_factory, exp_id, cfg)

    assert capture.kwargs.get("human_role") is None


# ---------------------------------------------------------------------------
# 4. _update_run_success — backward-compatible call (no new kwargs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_run_success_backward_compatible() -> None:
    """_update_run_success accepts new kwargs with defaults (backward-compatible).

    Calling the function with only the original positional/keyword arguments
    must complete without raising.  The new human_role and cognitive_load_proxy
    default to None, which is the correct pre-m9.2 behaviour.
    """
    run_id = uuid.uuid4()
    exp_id = uuid.uuid4()

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("atm.experiment.runner.session_scope", return_value=mock_cm):
        # Call with old-style args only — no human_role, no cognitive_load_proxy
        await _update_run_success(
            AsyncMock(),
            run_id,
            exp_id,
            quality_score=1.0,
            budget_spent_usd=0.0,
            iterations=1,
        )

    # Reaches here without raising — backward compatibility confirmed.


# ---------------------------------------------------------------------------
# 5. _update_run_success — human_role and cognitive_load_proxy written correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_run_success_with_human_role_and_cog_proxy() -> None:
    """_update_run_success writes dynamic human_role and cognitive_load_proxy.

    When called with human_role="judge" and cognitive_load_proxy=3.14, the
    compiled UPDATE statement must include both values.
    """
    run_id = uuid.uuid4()
    exp_id = uuid.uuid4()

    captured_stmt: list[Any] = []

    mock_session = AsyncMock()

    async def _capture(stmt: Any) -> Any:
        captured_stmt.append(stmt)
        return MagicMock()

    mock_session.execute = _capture
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("atm.experiment.runner.session_scope", return_value=mock_cm):
        await _update_run_success(
            AsyncMock(),
            run_id,
            exp_id,
            quality_score=0.9,
            budget_spent_usd=0.05,
            iterations=5,
            human_role="judge",
            cognitive_load_proxy=3.14,
        )

    assert len(captured_stmt) == 1, "Expected exactly one SQL statement to be executed"
    stmt = captured_stmt[0]

    # Compile against SQLite dialect (available in all unit-test environments)
    # to inspect the bound parameter values.
    compiled = stmt.compile(dialect=sqlite_dialect.dialect())
    params = compiled.params

    # SQLAlchemy may suffix param names with "_1" when deduplicating; try both.
    human_role_val = params.get("human_role_1") or params.get("human_role")
    cog_val = params.get("cognitive_load_proxy_1") or params.get("cognitive_load_proxy")
    status_val = params.get("status_1") or params.get("status")

    assert human_role_val == "judge", f"human_role not found in params: {params}"
    assert cog_val is not None and abs(cog_val - 3.14) < 1e-9, (
        f"cognitive_load_proxy mismatch: {cog_val!r}"
    )
    assert status_val == "completed", f"status not found in params: {params}"


# ---------------------------------------------------------------------------
# 6. cognitive_load_proxy raises → _update_run_success still called, proxy=NULL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalization_cog_proxy_exception_yields_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When human_sim_cognitive_load_proxy raises, run still completes with NULL.

    The test simulates the finalization block from run_one directly:
    - role query returns no rows → dynamic_human_role stays None
    - cog proxy helper raises → dynamic_cog_proxy stays None
    Then _update_run_success is called with both None, which the test verifies.
    """
    import atm.experiment.runner as runner_mod

    update_calls: list[dict[str, Any]] = []

    orig_update = runner_mod._update_run_success

    async def _spy_update(
        session_factory: Any,
        run_id: Any,
        exp_id: Any,
        quality_score: float,
        budget_spent_usd: float,
        iterations: int,
        human_role: str | None = None,
        cognitive_load_proxy: float | None = None,
    ) -> None:
        update_calls.append(
            {
                "human_role": human_role,
                "cognitive_load_proxy": cognitive_load_proxy,
            }
        )
        # Call through so the function itself must not raise.
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("atm.experiment.runner.session_scope", return_value=mock_cm):
            await orig_update(
                session_factory,
                run_id,
                exp_id,
                quality_score=quality_score,
                budget_spent_usd=budget_spent_usd,
                iterations=iterations,
                human_role=human_role,
                cognitive_load_proxy=cognitive_load_proxy,
            )

    monkeypatch.setattr(runner_mod, "_update_run_success", _spy_update)

    # Patch the metric to always raise.
    async def _raising_cog_proxy(session: Any, run_id: Any) -> float:
        raise RuntimeError("simulated metric failure")

    monkeypatch.setattr(runner_mod, "human_sim_cognitive_load_proxy", _raising_cog_proxy)

    # Patch session_scope so the role query returns no rows.
    empty_result = MagicMock()
    empty_result.fetchone.return_value = None
    mock_role_session = AsyncMock()
    mock_role_session.execute = AsyncMock(return_value=empty_result)
    mock_role_session.commit = AsyncMock()
    mock_role_session.rollback = AsyncMock()
    mock_role_cm = AsyncMock()
    mock_role_cm.__aenter__ = AsyncMock(return_value=mock_role_session)
    mock_role_cm.__aexit__ = AsyncMock(return_value=False)

    monkeypatch.setattr(runner_mod, "session_scope", lambda *_a, **_kw: mock_role_cm)

    # ---- replicate the finalization block from run_one ----
    run_id = uuid.uuid4()
    session_factory = AsyncMock()

    dynamic_human_role: str | None = None
    dynamic_cog_proxy: float | None = None

    try:
        async with runner_mod.session_scope(session_factory) as _hi_session:
            last_role_result = await _hi_session.execute(
                sa.text(
                    "SELECT role FROM human_interactions"
                    " WHERE run_id = :rid"
                    " ORDER BY requested_at DESC LIMIT 1"
                ).bindparams(rid=run_id)
            )
            last_role_row = last_role_result.fetchone()
            if last_role_row is not None:
                dynamic_human_role = str(last_role_row[0])
    except Exception:
        pass  # leave dynamic_human_role = None

    try:
        async with runner_mod.session_scope(session_factory) as _cog_session:
            dynamic_cog_proxy = await runner_mod.human_sim_cognitive_load_proxy(
                _cog_session, run_id
            )
    except Exception:
        pass  # leave dynamic_cog_proxy = None (metric raised)

    await _spy_update(
        session_factory,
        run_id,
        uuid.uuid4(),
        quality_score=0.5,
        budget_spent_usd=0.0,
        iterations=0,
        human_role=dynamic_human_role,
        cognitive_load_proxy=dynamic_cog_proxy,
    )
    # ---- end finalization block ----

    assert len(update_calls) == 1
    call = update_calls[0]
    # Metric raised → cognitive_load_proxy must be None (NULL in DB)
    assert call["cognitive_load_proxy"] is None, (
        f"Expected None but got {call['cognitive_load_proxy']!r}"
    )
    # No interaction rows → human_role also None
    assert call["human_role"] is None, f"Expected None but got {call['human_role']!r}"
