"""Unit tests for runs.human_role writing in Runner._insert_run (M9.1 Step 1.5).

Tests:
  1.  enabled=True → human_role written as role.value ("reviewer")
  2.  enabled=False → human_role is None
  3.  cfg.human is None → human_role is None
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from atm.experiment.runner import _insert_run

# ---------------------------------------------------------------------------
# Helpers
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
