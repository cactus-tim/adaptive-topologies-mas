"""Unit tests for HumanCfg Pydantic model + YAML round-trips.

Tests:
  1.  HumanCfg defaults — enabled=False, gateway=llm_simulated, role=reviewer, etc.
  2.  HumanCfg frozen — mutation raises TypeError
  3.  HumanCfg invalid gateway raises ValidationError
  4.  HumanCfg invalid timeout_policy raises ValidationError
  5.  HumanCfg invalid role raises ValidationError
  6.  Round-trip: conf/human/llm_simulated.yaml loads correctly
  7.  Round-trip: conf/human/cli.yaml loads correctly
  8.  ExperimentConfig without human: section → human is None (back-compat)
  9.  ExperimentConfig with human: section inline → HumanCfg parsed
  10. ExperimentConfig human=None does not affect existing fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from atm.core.types import HumanRole
from atm.experiment.config import HumanCfg, ExperimentConfig, load_config

# Absolute path to project root (two levels up from this file's directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CONF_HUMAN = PROJECT_ROOT / "conf" / "human"
FIXTURE_MINIMAL = Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "valid_minimal.yaml"


# ---------------------------------------------------------------------------
# Helper: load a stand-alone conf/human/*.yaml as HumanCfg
# ---------------------------------------------------------------------------


def _load_human_cfg(yaml_path: Path) -> HumanCfg:
    """Load a conf/human YAML file directly into a HumanCfg instance."""
    raw = OmegaConf.load(yaml_path)
    data = OmegaConf.to_container(raw, resolve=True)
    return HumanCfg.model_validate(data)


# ---------------------------------------------------------------------------
# 1. Default values
# ---------------------------------------------------------------------------


def test_human_cfg_defaults() -> None:
    h = HumanCfg()
    assert h.enabled is False
    assert h.gateway == "llm_simulated"
    assert h.role == HumanRole.REVIEWER
    assert h.timeout_s == pytest.approx(900.0)
    assert h.timeout_policy == "llm_fallback"
    assert h.model is None


# ---------------------------------------------------------------------------
# 2. Frozen — mutation raises
# ---------------------------------------------------------------------------


def test_human_cfg_is_frozen() -> None:
    h = HumanCfg()
    with pytest.raises((TypeError, ValidationError)):
        h.enabled = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. Invalid gateway value
# ---------------------------------------------------------------------------


def test_human_cfg_invalid_gateway() -> None:
    with pytest.raises(ValidationError, match="gateway"):
        HumanCfg.model_validate({"gateway": "streamlit"})


# ---------------------------------------------------------------------------
# 4. Invalid timeout_policy value
# ---------------------------------------------------------------------------


def test_human_cfg_invalid_timeout_policy() -> None:
    with pytest.raises(ValidationError, match="timeout_policy"):
        HumanCfg.model_validate({"timeout_policy": "ignore"})


# ---------------------------------------------------------------------------
# 5. Invalid role value
# ---------------------------------------------------------------------------


def test_human_cfg_invalid_role() -> None:
    with pytest.raises(ValidationError):
        HumanCfg.model_validate({"role": "superhero"})


# ---------------------------------------------------------------------------
# 6. Round-trip: conf/human/llm_simulated.yaml
# ---------------------------------------------------------------------------


def test_human_cfg_llm_simulated_yaml() -> None:
    h = _load_human_cfg(CONF_HUMAN / "llm_simulated.yaml")
    assert h.enabled is True
    assert h.gateway == "llm_simulated"
    assert h.role == HumanRole.REVIEWER
    assert h.timeout_s == pytest.approx(900.0)
    assert h.timeout_policy == "llm_fallback"
    assert h.model == "openai:gpt-4o-mini"


# ---------------------------------------------------------------------------
# 7. Round-trip: conf/human/cli.yaml
# ---------------------------------------------------------------------------


def test_human_cfg_cli_yaml() -> None:
    h = _load_human_cfg(CONF_HUMAN / "cli.yaml")
    assert h.enabled is True
    assert h.gateway == "cli"
    assert h.role == HumanRole.REVIEWER
    assert h.timeout_s == pytest.approx(60.0)
    assert h.timeout_policy == "fail"
    assert h.model is None


# ---------------------------------------------------------------------------
# 8. ExperimentConfig without human section — back-compat
# ---------------------------------------------------------------------------


def test_experiment_config_human_none_by_default() -> None:
    cfg = load_config(str(FIXTURE_MINIMAL))
    assert cfg.human is None


# ---------------------------------------------------------------------------
# 9. ExperimentConfig with inline human section
# ---------------------------------------------------------------------------


def test_experiment_config_with_human_inline(tmp_path: Path) -> None:
    yaml_content = """
name: "human_inline_test"
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
human:
  enabled: true
  gateway: llm_simulated
  role: reviewer
  timeout_s: 300
  timeout_policy: skip
  model: openai:gpt-4o
"""
    cfg_file = tmp_path / "with_human.yaml"
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.human is not None
    assert isinstance(cfg.human, HumanCfg)
    assert cfg.human.enabled is True
    assert cfg.human.gateway == "llm_simulated"
    assert cfg.human.role == HumanRole.REVIEWER
    assert cfg.human.timeout_s == pytest.approx(300.0)
    assert cfg.human.timeout_policy == "skip"
    assert cfg.human.model == "openai:gpt-4o"


# ---------------------------------------------------------------------------
# 10. ExperimentConfig.human=None does not affect other fields
# ---------------------------------------------------------------------------


def test_experiment_config_other_fields_unaffected_when_human_none() -> None:
    cfg = load_config(str(FIXTURE_MINIMAL))
    # Existing fields remain intact
    assert cfg.name == "test_minimal"
    assert cfg.topology.name == "chain"
    assert cfg.agents.set == "canonical_4"
    assert cfg.human is None


# ---------------------------------------------------------------------------
# 11. HumanCfg with cli gateway and llm_fallback policy is valid
# ---------------------------------------------------------------------------


def test_human_cfg_cli_with_llm_fallback_is_valid() -> None:
    """Pydantic doesn't cross-validate policy vs gateway — both combinations are valid."""
    h = HumanCfg.model_validate(
        {"enabled": True, "gateway": "cli", "timeout_policy": "llm_fallback"}
    )
    assert h.gateway == "cli"
    assert h.timeout_policy == "llm_fallback"


# ---------------------------------------------------------------------------
# 12. HumanCfg model field accepts None explicitly
# ---------------------------------------------------------------------------


def test_human_cfg_model_none_explicit() -> None:
    h = HumanCfg.model_validate({"model": None})
    assert h.model is None


# ---------------------------------------------------------------------------
# 13. HumanCfg all HumanRole variants are accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role_str, expected",
    [
        ("reviewer", HumanRole.REVIEWER),
        ("coordinator", HumanRole.COORDINATOR),
        ("judge", HumanRole.JUDGE),
        ("peer", HumanRole.PEER),
        ("monitor", HumanRole.MONITOR),
    ],
)
def test_human_cfg_all_roles_accepted(role_str: str, expected: HumanRole) -> None:
    h = HumanCfg.model_validate({"role": role_str})
    assert h.role == expected
