"""Unit tests for EvaluationCfg in atm.experiment.config.

2 tests:
  1. EvaluationCfg defaults — judge_model and judge_self_consistency_n.
  2. YAML override — evaluation.judge_model and evaluation.judge_self_consistency_n
     can be set from YAML config.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atm.experiment.config import EvaluationCfg


def test_evaluation_cfg_defaults() -> None:
    """EvaluationCfg uses sensible defaults and is backward-compatible."""
    cfg = EvaluationCfg()

    assert cfg.judge_model == "openai:gpt-4o"
    assert cfg.judge_self_consistency_n == 3


def test_experiment_config_evaluation_override(tmp_path: Path) -> None:
    """evaluation.judge_model and judge_self_consistency_n survive YAML → Pydantic."""
    from atm.experiment.config import load_config

    yaml_content = """
name: "eval_cfg_test"
model:
  default: "fake:echo"
agents:
  set: "canonical_4"
topology:
  name: "chain"
task:
  name: "mmlu"
observability:
  pg_dsn: "postgresql+asyncpg://localhost/atm_test"
evaluation:
  judge_model: "fake:echo"
  judge_self_consistency_n: 5
"""
    cfg_file = tmp_path / "eval_cfg.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(str(cfg_file))

    assert cfg.evaluation.judge_model == "fake:echo"
    assert cfg.evaluation.judge_self_consistency_n == 5


def test_experiment_config_evaluation_defaults_when_section_absent(tmp_path: Path) -> None:
    """ExperimentConfig.evaluation defaults to EvaluationCfg() when section absent."""
    from atm.experiment.config import load_config

    yaml_content = """
name: "eval_default_test"
model:
  default: "fake:echo"
agents:
  set: "canonical_4"
topology:
  name: "chain"
task:
  name: "mmlu"
observability:
  pg_dsn: "postgresql+asyncpg://localhost/atm_test"
"""
    cfg_file = tmp_path / "eval_default.yaml"
    cfg_file.write_text(yaml_content)

    cfg = load_config(str(cfg_file))

    assert cfg.evaluation.judge_model == "openai:gpt-4o"
    assert cfg.evaluation.judge_self_consistency_n == 3


def test_evaluation_cfg_consistency_n_validation() -> None:
    """EvaluationCfg rejects judge_self_consistency_n outside [1, 10]."""
    with pytest.raises(ValidationError):
        EvaluationCfg(judge_self_consistency_n=0)

    with pytest.raises(ValidationError):
        EvaluationCfg(judge_self_consistency_n=11)
