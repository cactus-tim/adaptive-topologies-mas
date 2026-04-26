"""Unit tests for YAML config loading — smoke experiment configs.

Tests (≥3):
  1. load_config("conf/experiments/smoke.yaml") → ExperimentConfig with topology.name == "chain"
  2. Same with override ["+topology.name=star"] → topology.name == "star"
  3. At least 4 agent YAML files exist in conf/agents/
"""

from __future__ import annotations

import glob
from pathlib import Path

from atm.experiment import ExperimentConfig, load_config

SMOKE_YAML = str(Path(__file__).parent.parent.parent.parent / "conf" / "experiments" / "smoke.yaml")
AGENTS_GLOB = str(Path(__file__).parent.parent.parent.parent / "conf" / "agents" / "*.yaml")


# ---------------------------------------------------------------------------
# 1. Default topology is chain
# ---------------------------------------------------------------------------


def test_smoke_yaml_loads_chain() -> None:
    """Smoke YAML loads correctly with default chain topology."""
    cfg = load_config(SMOKE_YAML)

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "m6_smoke"
    assert cfg.topology.name == "chain"
    assert cfg.topology.max_iterations == 12
    assert cfg.seed == 42
    assert cfg.task.name == "fibonacci_smoke"
    assert "fib(10)" in cfg.task.input
    assert cfg.agents.set == "canonical_4"
    assert cfg.model.default == "fake:scripted"
    assert cfg.model.by_role.get("planner") == "fake:scripted"
    assert cfg.model.by_role.get("executor") == "fake:scripted"
    assert cfg.model.by_role.get("critic") == "fake:scripted"


# ---------------------------------------------------------------------------
# 2. Override topology.name=star
# ---------------------------------------------------------------------------


def test_smoke_yaml_loads_star_override() -> None:
    """Smoke YAML with +topology.name=star override yields star topology."""
    cfg = load_config(SMOKE_YAML, overrides=["+topology.name=star"])

    assert isinstance(cfg, ExperimentConfig)
    assert cfg.topology.name == "star"
    # Other fields unchanged
    assert cfg.name == "m6_smoke"
    assert cfg.seed == 42
    assert cfg.agents.set == "canonical_4"


# ---------------------------------------------------------------------------
# 3. Pre-flight: at least 4 agent YAML files exist
# ---------------------------------------------------------------------------


def test_at_least_four_agent_yamls_exist() -> None:
    """At least 4 agent YAML configs must be present in conf/agents/."""
    agent_yamls = glob.glob(AGENTS_GLOB)
    assert len(agent_yamls) >= 4, (
        f"Expected ≥4 agent YAML files in conf/agents/, found {len(agent_yamls)}: {agent_yamls}"
    )
