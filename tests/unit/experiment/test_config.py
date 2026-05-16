"""Unit tests for atm.experiment.config — Pydantic schemas + OmegaConf loader.

Tests (≥8 original + 14 new TopologyExtras tests):
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
  12. TopologyCfg extra passthrough — flat bw-compat via attribute access
  13. Full round-trip field checks on valid_minimal.yaml
  14. AgentSetCfg and ObservabilityCfg schema validation
  15. ModelCfg.fake_fixtures — BUG-2 regression
  --- New TopologyExtras tests ---
  T1. Namespaced form accepted (mesh sub-bucket)
  T2. Namespaced form accepted (debate sub-bucket)
  T3. Flat bw-compat emits DeprecationWarning
  T4. max_rounds scatters to debate AND hierarchical (not mesh)
  T5. max_rounds flat — mesh stays at default 12
  T6. max_rounds flat — adaptive has no max_rounds field (ignored at schema level)
  T7. mesh_max_rounds flat → mesh.max_rounds remap
  T8. Unknown topology name at extra top-level → ValidationError
  T9. Unknown field inside a topology sub-bucket → ValidationError
  T10. AdaptiveExtras defaults are correct (planning=3, exec=10, verify=4)
  T11. StarExtras defaults are correct (planning=2, exec=5, verify=3)
  T12. MeshExtras defaults — max_rounds=12 (starvation-safe)
  T13. DebateExtras and HierarchicalExtras defaults — max_rounds=4
  T14. schema_defaults_match_topology_builder_constants (parity guard)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from atm.experiment import ExperimentConfig, load_config
from atm.experiment.config import (
    AdaptiveExtras,
    AgentSetCfg,
    BudgetCfg,
    ChainExtras,
    DebateExtras,
    HierarchicalExtras,
    MeshExtras,
    ModelCfg,
    ObservabilityCfg,
    ScratchpadCfg,
    StarExtras,
    TaskCfg,
    TopologyCfg,
    TopologyExtras,
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
# 12. TopologyCfg.extra passthrough — flat bw-compat via attribute access
# ---------------------------------------------------------------------------


def test_load_config_topology_extra_passthrough(tmp_path: Path) -> None:
    """Flat extras trigger DeprecationWarning and are accessible via attribute path."""
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
    with pytest.warns(DeprecationWarning):
        cfg = load_config(str(cfg_file))
    # Access via namespaced attribute path (flat keys remapped to star namespace)
    assert cfg.topology.extra.star.planning_max_iter == 2
    assert cfg.topology.extra.star.exec_max_iter == 5
    assert cfg.topology.extra.star.verify_max_iter == 3


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


# ---------------------------------------------------------------------------
# TopologyExtras tests — T1 through T14
# ---------------------------------------------------------------------------


# --- T1. Namespaced form accepted (mesh sub-bucket) ---


def test_topology_extras_namespaced_mesh_accepted() -> None:
    """Namespaced extra {mesh: {max_rounds: 12}} parses cleanly with no warning."""
    cfg = TopologyCfg.model_validate({"name": "mesh", "extra": {"mesh": {"max_rounds": 12}}})
    assert isinstance(cfg.extra, TopologyExtras)
    assert cfg.extra.mesh.max_rounds == 12


# --- T2. Namespaced form accepted (debate sub-bucket) ---


def test_topology_extras_namespaced_debate_accepted() -> None:
    """Namespaced extra {debate: {max_rounds: 6}} parses cleanly with no warning."""
    cfg = TopologyCfg.model_validate({"name": "debate", "extra": {"debate": {"max_rounds": 6}}})
    assert cfg.extra.debate.max_rounds == 6


# --- T3. Flat bw-compat emits DeprecationWarning ---


def test_topology_extras_flat_emits_deprecation_warning() -> None:
    """Flat extra dict triggers DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="flat topology.extra keys are deprecated"):
        TopologyCfg.model_validate({"name": "star", "extra": {"planning_max_iter": 2}})


# --- T4. max_rounds scatters to debate AND hierarchical ---


def test_topology_extras_max_rounds_scatters_to_debate_and_hierarchical() -> None:
    """Flat max_rounds=2 remaps to debate.max_rounds=2 AND hierarchical.max_rounds=2."""
    with pytest.warns(DeprecationWarning):
        cfg = TopologyCfg.model_validate({"name": "debate", "extra": {"max_rounds": 2}})
    assert cfg.extra.debate.max_rounds == 2
    assert cfg.extra.hierarchical.max_rounds == 2


# --- T5. max_rounds flat — mesh stays at default 12 ---


def test_topology_extras_max_rounds_flat_does_not_scatter_to_mesh() -> None:
    """Flat max_rounds=2 does NOT scatter to mesh (mesh stays at default 12)."""
    with pytest.warns(DeprecationWarning):
        cfg = TopologyCfg.model_validate({"name": "debate", "extra": {"max_rounds": 2}})
    assert cfg.extra.mesh.max_rounds == 12  # starvation-safe default preserved


# --- T6. max_rounds flat — adaptive has no max_rounds field ---


def test_topology_extras_max_rounds_flat_does_not_scatter_to_adaptive() -> None:
    """Flat max_rounds=2 does NOT scatter to adaptive (adaptive has no max_rounds field)."""
    with pytest.warns(DeprecationWarning):
        cfg = TopologyCfg.model_validate({"name": "adaptive", "extra": {"max_rounds": 2}})
    # AdaptiveExtras has no max_rounds field; verify no error and defaults intact
    assert not hasattr(cfg.extra.adaptive, "max_rounds")
    assert cfg.extra.adaptive.exec_max_iter == 10  # adaptive default unchanged


# --- T7. mesh_max_rounds flat → mesh.max_rounds remap ---


def test_topology_extras_mesh_max_rounds_remapped_to_mesh_namespace() -> None:
    """Flat mesh_max_rounds=8 is remapped to mesh.max_rounds=8."""
    with pytest.warns(DeprecationWarning):
        cfg = TopologyCfg.model_validate({"name": "mesh", "extra": {"mesh_max_rounds": 8}})
    assert cfg.extra.mesh.max_rounds == 8


# --- T8. Unknown topology name at extra top-level → ValidationError ---


def test_topology_extras_unknown_topology_name_raises() -> None:
    """Extra with an unknown topology name (e.g. 'foo') raises ValidationError.

    'foo' is not a recognised topology name, so the bw-compat validator treats
    it as a flat legacy dict, emits a DeprecationWarning, and then tries to
    build TopologyExtras with 'foo' at the top level — which extra="forbid"
    rejects with a ValidationError.
    """
    with pytest.warns(DeprecationWarning), pytest.raises(ValidationError):
        TopologyCfg.model_validate({"name": "chain", "extra": {"foo": {"some_param": 1}}})


# --- T9. Unknown field inside a topology sub-bucket → ValidationError ---


def test_topology_extras_unknown_field_in_sub_bucket_raises() -> None:
    """Extra with unknown field inside a topology bucket raises ValidationError."""
    with pytest.raises(ValidationError):
        TopologyCfg.model_validate(
            {"name": "mesh", "extra": {"mesh": {"unknown_param": 99}}}
        )


# --- T10. AdaptiveExtras defaults ---


def test_adaptive_extras_defaults() -> None:
    """AdaptiveExtras has correct phase-limit defaults: planning=3, exec=10, verify=4."""
    a = AdaptiveExtras()
    assert a.planning_max_iter == 3
    assert a.exec_max_iter == 10
    assert a.verify_max_iter == 4
    assert a.subgraph_max_iterations == 10
    assert a.switch_guards is True
    assert a.run_id is None


# --- T11. StarExtras defaults ---


def test_star_extras_defaults() -> None:
    """StarExtras has correct defaults: planning=2, exec=5, verify=3."""
    s = StarExtras()
    assert s.planning_max_iter == 2
    assert s.exec_max_iter == 5
    assert s.verify_max_iter == 3


# --- T12. MeshExtras defaults — max_rounds=12 (starvation-safe) ---


def test_mesh_extras_defaults() -> None:
    """MeshExtras has correct starvation-safe defaults."""
    m = MeshExtras()
    assert m.max_rounds == 12
    assert m.consensus_threshold == 3
    assert m.max_messages == 200


# --- T13. DebateExtras and HierarchicalExtras defaults ---


def test_debate_extras_defaults() -> None:
    """DebateExtras defaults: max_rounds=4, role ids are None."""
    d = DebateExtras()
    assert d.max_rounds == 4
    assert d.debater_pro_id is None
    assert d.debater_contra_id is None
    assert d.judge_id is None


def test_hierarchical_extras_defaults() -> None:
    """HierarchicalExtras defaults: max_rounds=4, finalize_signal correct."""
    h = HierarchicalExtras()
    assert h.max_rounds == 4
    assert h.finalize_signal == "top_coord_finalize"
    assert h.sub_teams is None
    assert h.final_answer_strategy is None


# --- T14. Schema-vs-source parity guard ---


def test_schema_defaults_match_topology_builder_constants() -> None:
    """Schema defaults must match the _DEFAULT_* constants in each topology builder.

    This parity guard catches silent drift between config.py schema defaults and
    the topology builder fallback values. If a topology builder constant changes,
    this test fails immediately, preventing silent incorrect defaults in sweeps.
    """
    from atm.topology.debate import _DEFAULT_MAX_ROUNDS as DEBATE_DEFAULT_MAX_ROUNDS
    from atm.topology.hierarchical import (
        _DEFAULT_FINALIZE_SIGNAL,
    )
    from atm.topology.hierarchical import (
        _DEFAULT_MAX_ROUNDS as HIER_DEFAULT_MAX_ROUNDS,
    )
    from atm.topology.mesh import (
        _DEFAULT_BROADCAST_BUS_CAP,
        _DEFAULT_CONSENSUS_THRESHOLD,
    )
    from atm.topology.mesh import (
        _DEFAULT_MAX_ROUNDS as MESH_DEFAULT_MAX_ROUNDS,
    )
    from atm.topology.star import (
        _DEFAULT_EXEC_MAX_ITER,
        _DEFAULT_PLANNING_MAX_ITER,
        _DEFAULT_VERIFY_MAX_ITER,
    )

    # StarExtras parity
    assert StarExtras().planning_max_iter == _DEFAULT_PLANNING_MAX_ITER
    assert StarExtras().exec_max_iter == _DEFAULT_EXEC_MAX_ITER
    assert StarExtras().verify_max_iter == _DEFAULT_VERIFY_MAX_ITER

    # DebateExtras parity
    assert DebateExtras().max_rounds == DEBATE_DEFAULT_MAX_ROUNDS

    # HierarchicalExtras parity
    assert HierarchicalExtras().max_rounds == HIER_DEFAULT_MAX_ROUNDS
    assert HierarchicalExtras().finalize_signal == _DEFAULT_FINALIZE_SIGNAL

    # MeshExtras parity
    assert MeshExtras().max_rounds == MESH_DEFAULT_MAX_ROUNDS
    assert MeshExtras().consensus_threshold == _DEFAULT_CONSENSUS_THRESHOLD
    assert MeshExtras().max_messages == _DEFAULT_BROADCAST_BUS_CAP

    # AdaptiveExtras — no named constants; assert raw integer parity with
    # inline literals in adaptive.py:394-396
    assert AdaptiveExtras().planning_max_iter == 3
    assert AdaptiveExtras().exec_max_iter == 10
    assert AdaptiveExtras().verify_max_iter == 4

    # ChainExtras — no fields (chain reads no extras); verify instance is created
    assert isinstance(ChainExtras(), ChainExtras)
