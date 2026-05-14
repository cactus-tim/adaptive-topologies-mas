"""Unit tests for `_load_cfg_from_snapshot` (m12-resume-replay).

Covers the round-trip: ``cfg.model_dump(mode='json')`` ↔ ``ExperimentConfig``
plus the legacy 4-field snapshot rejection path.
"""

from __future__ import annotations

import pytest

from atm.experiment.config import (
    AgentSetCfg,
    BudgetCfg,
    EvaluationCfg,
    ExperimentConfig,
    ModelCfg,
    ObservabilityCfg,
    TaskCfg,
    TopologyCfg,
)
from atm.experiment.runner import _load_cfg_from_snapshot


def _make_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="round_trip",
        seed=7,
        task=TaskCfg(name="fib_test", input="say hi"),
        model=ModelCfg(default="fake:echo"),
        agents=AgentSetCfg(set="canonical_4"),
        topology=TopologyCfg(name="chain", max_iterations=3),
        budget=BudgetCfg(per_call_usd=0.1, per_run_usd=1.0, per_experiment_usd=5.0),
        observability=ObservabilityCfg(
            pg_dsn="postgresql+asyncpg://localhost/atm",
            parquet_dir="/tmp/atm-parquet",
        ),
        evaluation=EvaluationCfg(judge_model="fake:echo"),
    )


def test_round_trip_via_model_dump() -> None:
    """A widened snapshot round-trips through `_load_cfg_from_snapshot`."""
    cfg = _make_cfg()
    snapshot = cfg.model_dump(mode="json")
    rehydrated = _load_cfg_from_snapshot(snapshot)
    assert rehydrated.name == cfg.name
    assert rehydrated.seed == cfg.seed
    assert rehydrated.topology.name == cfg.topology.name
    assert rehydrated.budget.per_run_usd == cfg.budget.per_run_usd
    assert rehydrated.observability.pg_dsn == cfg.observability.pg_dsn


def test_round_trip_preserves_human_section() -> None:
    """When HumanCfg is present it must round-trip too."""
    from atm.core.types import HumanRole
    from atm.experiment.config import HumanCfg

    cfg = _make_cfg()
    cfg = cfg.model_copy(
        update={
            "human": HumanCfg(
                enabled=True,
                gateway="llm_simulated",
                role=HumanRole.PEER,
            )
        }
    )
    snapshot = cfg.model_dump(mode="json")
    rehydrated = _load_cfg_from_snapshot(snapshot)
    assert rehydrated.human is not None
    assert rehydrated.human.enabled is True
    assert rehydrated.human.role == HumanRole.PEER


def test_legacy_sparse_snapshot_rejected() -> None:
    """Pre-m12 4-field snapshots are explicitly rejected."""
    legacy = {
        "name": "old",
        "topology": "chain",
        "task_name": "fib_test",
        "seed": 1,
    }
    with pytest.raises(ValueError, match="missing keys"):
        _load_cfg_from_snapshot(legacy)


def test_missing_required_key_rejected() -> None:
    """A snapshot missing 'observability' is rejected with a clear message."""
    cfg = _make_cfg()
    snapshot = cfg.model_dump(mode="json")
    snapshot.pop("observability")
    with pytest.raises(ValueError, match="observability"):
        _load_cfg_from_snapshot(snapshot)
