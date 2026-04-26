"""Unit tests for atm.experiment.config — Pydantic schemas + OmegaConf loader.

Tests (≥8):
  1.  load_config with valid_minimal.yaml returns ExperimentConfig
  2.  ExperimentConfig exported from __init__
  3.  load_config exported from __init__
  4.  BudgetCfg defaults
  5.  ModelCfg.get_model_for falls back to default
  6.  dotlist override changes topology.name
  7.  dotlist override changes topology.max_iterations (int coercion)
  8.  missing required field (name) raises ValidationError
  9.  invalid topology name raises ValidationError
  10. ObservabilityCfg pg_dsn env interpolation (${oc.env:...} with default)
  11. ScratchpadCfg defaults are correct
  12. TopologyCfg extra dict carries through
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from atm.experiment import ExperimentConfig, load_config
from atm.experiment.config import (
    AgentSetCfg,
    BudgetCfg,
    ModelCfg,
    ObservabilityCfg,
    ScratchpadCfg,
    TaskCfg,
    TopologyCfg,
)

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "valid_minimal.yaml"


# ---------------------------------------------------------------------------
# 1. Basic load
# ---------------------------------------------------------------------------


def test_load_config_returns_experiment_config() -> None:
    cfg = load_config(str(FIXTURE))
    assert isinstance(cfg, ExperimentConfig)


# ---------------------------------------------------------------------------
# 2-3. Public API re-exports
# ---------------------------------------------------------------------------


def test_experiment_config_exported_from_package() -> None:
    import atm.experiment as pkg

    assert hasattr(pkg, "ExperimentConfig")
    assert pkg.ExperimentConfig is ExperimentConfig


def test_load_config_exported_from_package() -> None:
    import atm.experiment as pkg

    assert hasattr(pkg, "load_config")
    assert pkg.load_config is load_config


# ---------------------------------------------------------------------------
# 4. BudgetCfg defaults
# ---------------------------------------------------------------------------


def test_budget_cfg_defaults() -> None:
    b = BudgetCfg()
    assert b.per_call_usd == pytest.approx(0.10)
    assert b.per_run_usd == pytest.approx(0.50)
    assert b.per_experiment_usd == pytest.approx(50.0)
    assert b.warn_ratio == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 5. ModelCfg.get_model_for fallback
# ---------------------------------------------------------------------------


def test_model_cfg_get_model_for_known_role() -> None:
    m = ModelCfg(default="fake:default", by_role={"planner": "fake:planner"})
    assert m.get_model_for("planner") == "fake:planner"


def test_model_cfg_get_model_for_unknown_role_falls_back() -> None:
    m = ModelCfg(default="fake:default", by_role={})
    assert m.get_model_for("unknown_role") == "fake:default"


# ---------------------------------------------------------------------------
# 6. Dotlist override — topology.name
# ---------------------------------------------------------------------------


def test_load_config_override_topology_name() -> None:
    cfg = load_config(str(FIXTURE), overrides=["+topology.name=star"])
    assert cfg.topology.name == "star"


# ---------------------------------------------------------------------------
# 7. Dotlist override — integer field coercion
# ---------------------------------------------------------------------------


def test_load_config_override_max_iterations() -> None:
    cfg = load_config(str(FIXTURE), overrides=["topology.max_iterations=99"])
    assert cfg.topology.max_iterations == 99


# ---------------------------------------------------------------------------
# 8. Missing required field raises ValidationError
# ---------------------------------------------------------------------------


def test_load_config_missing_name_raises(tmp_path: Path) -> None:
    yaml_content = """
model:
  default: "fake:scripted"
agents:
  set: "canonical_4"
topology:
  name: "chain"
task:
  name: "t1"
observability:
  pg_dsn: "postgresql://localhost/test"
"""
    cfg_file = tmp_path / "no_name.yaml"
    cfg_file.write_text(yaml_content)
    with pytest.raises(ValidationError, match="name"):
        load_config(str(cfg_file))


# ---------------------------------------------------------------------------
# 9. Invalid topology name raises ValidationError
# ---------------------------------------------------------------------------


def test_load_config_invalid_topology_raises(tmp_path: Path) -> None:
    yaml_content = """
name: "bad_topology_test"
model:
  default: "fake:scripted"
agents:
  set: "canonical_4"
topology:
  name: "nonexistent_topology"
task:
  name: "t1"
observability:
  pg_dsn: "postgresql://localhost/test"
"""
    cfg_file = tmp_path / "bad_topology.yaml"
    cfg_file.write_text(yaml_content)
    with pytest.raises(ValidationError):
        load_config(str(cfg_file))


# ---------------------------------------------------------------------------
# 10. Env interpolation via ${oc.env:VAR,default}
# ---------------------------------------------------------------------------


def test_load_config_env_interpolation_with_default(tmp_path: Path) -> None:
    """${oc.env:VAR,fallback} should resolve to fallback when VAR is unset."""
    # Ensure the env var is not set
    os.environ.pop("ATM_TEST_PG_DSN_UNIQUE_XYZ", None)
    yaml_content = """
name: "env_interp_test"
model:
  default: "fake:scripted"
agents:
  set: "canonical_4"
topology:
  name: "chain"
task:
  name: "t1"
observability:
  pg_dsn: "${oc.env:ATM_TEST_PG_DSN_UNIQUE_XYZ,postgresql://fallback:5432/db}"
"""
    cfg_file = tmp_path / "env_interp.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.observability.pg_dsn == "postgresql://fallback:5432/db"


def test_load_config_env_interpolation_reads_env_var(tmp_path: Path) -> None:
    """${oc.env:VAR,default} should read the actual env var when set."""
    os.environ["ATM_TEST_PG_DSN_UNIQUE_XYZ"] = "postgresql://from_env:5432/env_db"
    try:
        yaml_content = """
name: "env_interp_test2"
model:
  default: "fake:scripted"
agents:
  set: "canonical_4"
topology:
  name: "chain"
task:
  name: "t1"
observability:
  pg_dsn: "${oc.env:ATM_TEST_PG_DSN_UNIQUE_XYZ,postgresql://fallback:5432/db}"
"""
        cfg_file = tmp_path / "env_interp2.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(str(cfg_file))
        assert cfg.observability.pg_dsn == "postgresql://from_env:5432/env_db"
    finally:
        os.environ.pop("ATM_TEST_PG_DSN_UNIQUE_XYZ", None)


# ---------------------------------------------------------------------------
# 11. ScratchpadCfg defaults
# ---------------------------------------------------------------------------


def test_scratchpad_cfg_defaults() -> None:
    s = ScratchpadCfg()
    assert s.policy == "window_with_summary"
    assert s.window_size == 3
    assert s.summarizer_model is None
    assert s.context_token_budget == 12000


# ---------------------------------------------------------------------------
# 12. TopologyCfg.extra passthrough
# ---------------------------------------------------------------------------


def test_load_config_topology_extra_passthrough(tmp_path: Path) -> None:
    yaml_content = """
name: "extra_test"
model:
  default: "fake:scripted"
agents:
  set: "canonical_4"
topology:
  name: "star"
  max_iterations: 20
  extra:
    planning_max_iter: 2
    exec_max_iter: 5
    verify_max_iter: 3
task:
  name: "t1"
observability:
  pg_dsn: "postgresql://localhost/test"
"""
    cfg_file = tmp_path / "extra.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.topology.extra["planning_max_iter"] == 2
    assert cfg.topology.extra["exec_max_iter"] == 5
    assert cfg.topology.extra["verify_max_iter"] == 3


# ---------------------------------------------------------------------------
# 13. Full round-trip field checks on valid_minimal.yaml
# ---------------------------------------------------------------------------


def test_load_config_field_values() -> None:
    cfg = load_config(str(FIXTURE))
    assert cfg.name == "test_minimal"
    assert cfg.seed == 0
    assert cfg.topology.name == "chain"
    assert cfg.topology.max_iterations == 5
    assert cfg.agents.set == "canonical_4"
    assert cfg.task.name == "smoke_test"
    assert cfg.observability.callback_sync is True


# ---------------------------------------------------------------------------
# 14. AgentSetCfg and ObservabilityCfg schema validation
# ---------------------------------------------------------------------------


def test_agent_set_cfg_requires_set_field() -> None:
    with pytest.raises(ValidationError):
        AgentSetCfg.model_validate({})  # missing required 'set'


def test_observability_cfg_requires_pg_dsn() -> None:
    with pytest.raises(ValidationError):
        ObservabilityCfg.model_validate({})  # missing required 'pg_dsn'


def test_task_cfg_requires_name() -> None:
    with pytest.raises(ValidationError):
        TaskCfg.model_validate({})  # missing required 'name'


def test_topology_cfg_requires_name() -> None:
    with pytest.raises(ValidationError):
        TopologyCfg.model_validate({})  # missing required 'name'


# ---------------------------------------------------------------------------
# 15. ModelCfg.fake_fixtures — BUG-2 regression
# ---------------------------------------------------------------------------


def test_model_cfg_fake_fixtures_default_is_empty_dict() -> None:
    """ModelCfg.fake_fixtures defaults to an empty dict (BUG-2 fix)."""
    m = ModelCfg(default="fake:echo")
    assert isinstance(m.fake_fixtures, dict)
    assert m.fake_fixtures == {}


def test_model_cfg_fake_fixtures_roundtrip(tmp_path: Path) -> None:
    """ModelCfg.fake_fixtures survives a YAML load → Pydantic roundtrip."""
    yaml_content = """
name: "fixtures_test"
model:
  default: "fake:scripted"
  fake_fixtures:
    planner: "tests/fixtures/llm/m6_chain_planner.yaml"
    executor: "tests/fixtures/llm/m6_chain_executor.yaml"
    critic: "tests/fixtures/llm/m6_chain_critic.yaml"
agents:
  set: "canonical_4"
topology:
  name: "chain"
task:
  name: "t1"
observability:
  pg_dsn: "postgresql://localhost/atm_test"
"""
    cfg_file = tmp_path / "fixtures_test.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.model.fake_fixtures == {
        "planner": "tests/fixtures/llm/m6_chain_planner.yaml",
        "executor": "tests/fixtures/llm/m6_chain_executor.yaml",
        "critic": "tests/fixtures/llm/m6_chain_critic.yaml",
    }
