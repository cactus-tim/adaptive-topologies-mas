"""Unit tests for GridCfg, EstimateCfg, and _resolve_dotpath (M12 Phase 1).

Tests (10 cases):
  1.  GridCfg defaults — parallelism=4, fail_fast=False, seeds=[42]
  2.  GridCfg valid sweep accepts str/int/float/bool scalars
  3.  GridCfg parallelism=0 raises ValidationError (ge=1)
  4.  GridCfg seeds=[] raises ValidationError (min_length=1)
  5.  EstimateCfg defaults — heuristic_tokens_per_call=1500, calls_per_iter=6, use_historical=True
  6.  EstimateCfg heuristic_tokens_per_call=0 raises ValidationError (ge=1)
  7.  ExperimentConfig.grid is None by default (optional)
  8.  ExperimentConfig.estimate has EstimateCfg defaults when not specified in YAML
  9.  GridCfg rejects unknown sweep key (non-existent dotpath) at construction time
  10. GridCfg rejects topology.extra.* keys (untyped leaves) at construction time
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from atm.experiment.config import (
    EstimateCfg,
    ExperimentConfig,
    GridCfg,
    load_config,
)

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "experiment" / "valid_minimal.yaml"


# ---------------------------------------------------------------------------
# 1. GridCfg defaults
# ---------------------------------------------------------------------------


def test_grid_cfg_defaults() -> None:
    """GridCfg with a minimal valid sweep should have correct defaults."""
    g = GridCfg(sweep={"seed": [1, 2, 3]})
    assert g.parallelism == 4
    assert g.fail_fast is False
    assert g.seeds == [42]


# ---------------------------------------------------------------------------
# 2. GridCfg valid sweep — str/int/float/bool scalars accepted
# ---------------------------------------------------------------------------


def test_grid_cfg_valid_sweep_scalar_types() -> None:
    """Sweep dict values may contain str, int, float, and bool scalars."""
    sweep = {
        "seed": [1, 2, 3],
        "topology.name": ["star", "chain"],
        "budget.per_call_usd": [0.05, 0.10],
        "evaluation.judge_self_consistency_n": [1, 3],
    }
    g = GridCfg(sweep=sweep)
    assert g.sweep["seed"] == [1, 2, 3]
    assert g.sweep["topology.name"] == ["star", "chain"]


# ---------------------------------------------------------------------------
# 3. GridCfg parallelism=0 raises ValidationError
# ---------------------------------------------------------------------------


def test_grid_cfg_parallelism_zero_raises() -> None:
    """parallelism must be >= 1."""
    with pytest.raises(ValidationError, match="parallelism"):
        GridCfg(sweep={"seed": [42]}, parallelism=0)


# ---------------------------------------------------------------------------
# 4. GridCfg seeds=[] raises ValidationError
# ---------------------------------------------------------------------------


def test_grid_cfg_seeds_empty_raises() -> None:
    """seeds must have at least one element."""
    with pytest.raises(ValidationError, match="seeds"):
        GridCfg(sweep={"seed": [42]}, seeds=[])


# ---------------------------------------------------------------------------
# 5. EstimateCfg defaults
# ---------------------------------------------------------------------------


def test_estimate_cfg_defaults() -> None:
    """EstimateCfg should have correct default values."""
    e = EstimateCfg()
    assert e.heuristic_tokens_per_call == 1500
    assert e.calls_per_iter == 6
    assert e.use_historical is True


# ---------------------------------------------------------------------------
# 6. EstimateCfg heuristic_tokens_per_call=0 raises ValidationError
# ---------------------------------------------------------------------------


def test_estimate_cfg_tokens_per_call_zero_raises() -> None:
    """heuristic_tokens_per_call must be >= 1."""
    with pytest.raises(ValidationError, match="heuristic_tokens_per_call"):
        EstimateCfg(heuristic_tokens_per_call=0)


# ---------------------------------------------------------------------------
# 7. ExperimentConfig.grid is None by default
# ---------------------------------------------------------------------------


def test_experiment_config_grid_none_by_default() -> None:
    """grid field is optional and defaults to None (backwards-compatible)."""
    cfg = load_config(str(FIXTURE))
    assert cfg.grid is None


# ---------------------------------------------------------------------------
# 8. ExperimentConfig.estimate has defaults when not in YAML
# ---------------------------------------------------------------------------


def test_experiment_config_estimate_defaults() -> None:
    """estimate field has EstimateCfg defaults when YAML doesn't include it."""
    cfg = load_config(str(FIXTURE))
    assert isinstance(cfg.estimate, EstimateCfg)
    assert cfg.estimate.heuristic_tokens_per_call == 1500
    assert cfg.estimate.calls_per_iter == 6
    assert cfg.estimate.use_historical is True


# ---------------------------------------------------------------------------
# 9. GridCfg rejects unknown dotpath keys at construction time
# ---------------------------------------------------------------------------


def test_grid_cfg_rejects_unknown_sweep_key() -> None:
    """GridCfg must reject sweep keys that don't resolve to a valid config field."""
    with pytest.raises((ValueError, ValidationError)):
        GridCfg(sweep={"nonexistent.field.path": [1, 2, 3]})


# ---------------------------------------------------------------------------
# 10. GridCfg rejects topology.extra.* keys (untyped leaves)
# ---------------------------------------------------------------------------


def test_grid_cfg_rejects_topology_extra_keys() -> None:
    """topology.extra.* keys should be rejected as untyped leaves."""
    with pytest.raises((ValueError, ValidationError)):
        GridCfg(sweep={"topology.extra.custom_param": ["a", "b"]})
