"""Unit tests for load_grid_configs — grid sweep expansion (M12 Phase 3).

Tests (8 cases):
  1. Import from atm.experiment works (load_grid_configs accessible)
  2. Import from atm.experiment.loader works (load_grid_configs accessible)
  3. grid_minimal.yaml → load_grid_configs returns exactly 8 configs
     (2 topologies × 2 tasks × 2 seeds)
  4. All 8 (topology, task, seed) combinations are present and correct
  5. Without a grid: block, load_grid_configs returns [cfg] (single-element list)
  6. Invalid sweep dotpath in GridCfg raises ValidationError at construction time
  7. Sweep value type validation rejects non-scalar (list-of-list) — raises
     ValidationError when constructing ExperimentConfig / GridCfg
  8. Each returned cell preserves the grid block (parallelism, fail_fast readable)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Path to the fixture files
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "experiment"
GRID_MINIMAL = FIXTURES_DIR / "grid_minimal.yaml"
VALID_MINIMAL = FIXTURES_DIR / "valid_minimal.yaml"


# ---------------------------------------------------------------------------
# Test 1: Import path — atm.experiment
# ---------------------------------------------------------------------------


def test_import_from_atm_experiment() -> None:
    """load_grid_configs is importable from atm.experiment."""
    from atm.experiment import load_grid_configs  # noqa: F401

    assert callable(load_grid_configs)


# ---------------------------------------------------------------------------
# Test 2: Import path — atm.experiment.loader
# ---------------------------------------------------------------------------


def test_import_from_atm_experiment_loader() -> None:
    """load_grid_configs is importable from atm.experiment.loader."""
    from atm.experiment.loader import load_grid_configs  # noqa: F401

    assert callable(load_grid_configs)


# ---------------------------------------------------------------------------
# Test 3: Grid expansion returns exactly 8 configs
# ---------------------------------------------------------------------------


def test_grid_minimal_returns_eight_configs() -> None:
    """grid_minimal.yaml with 2 topologies × 2 tasks × 2 seeds = 8 ExperimentConfigs."""
    from atm.experiment import load_grid_configs

    configs = load_grid_configs(str(GRID_MINIMAL))
    assert len(configs) == 8


# ---------------------------------------------------------------------------
# Test 4: All 8 (topology, task, seed) combinations are present
# ---------------------------------------------------------------------------


def test_grid_minimal_all_combinations_present() -> None:
    """All 8 expected (topology.name, task.name, seed) tuples must be present."""
    from atm.experiment import load_grid_configs

    configs = load_grid_configs(str(GRID_MINIMAL))

    expected = {
        ("star", "humaneval", 42),
        ("star", "humaneval", 43),
        ("star", "gsm8k", 42),
        ("star", "gsm8k", 43),
        ("chain", "humaneval", 42),
        ("chain", "humaneval", 43),
        ("chain", "gsm8k", 42),
        ("chain", "gsm8k", 43),
    }

    actual = {(c.topology.name, c.task.name, c.seed) for c in configs}
    assert actual == expected


# ---------------------------------------------------------------------------
# Test 5: No grid block → single-element list returned (back-compat)
# ---------------------------------------------------------------------------


def test_no_grid_block_returns_single_config() -> None:
    """Without a grid: block, load_grid_configs returns a list with exactly one config."""
    from atm.experiment import load_grid_configs, load_config

    configs = load_grid_configs(str(VALID_MINIMAL))
    assert len(configs) == 1

    base = load_config(str(VALID_MINIMAL))
    assert configs[0].name == base.name
    assert configs[0].seed == base.seed


# ---------------------------------------------------------------------------
# Test 6: Invalid sweep dotpath raises ValidationError at GridCfg construction
# ---------------------------------------------------------------------------


def test_invalid_dotpath_raises_validation_error() -> None:
    """GridCfg must raise ValidationError for unknown sweep dotpaths."""
    from atm.experiment.config import GridCfg

    with pytest.raises((ValueError, ValidationError)):
        GridCfg(sweep={"nonexistent.field": ["a", "b"]})


# ---------------------------------------------------------------------------
# Test 7: Non-scalar sweep values raise ValidationError
# ---------------------------------------------------------------------------


def test_non_scalar_sweep_values_raise_error() -> None:
    """Sweep values must be scalars; list-of-list must be rejected."""
    from atm.experiment.config import GridCfg

    # list-of-list: each element of the sweep list is itself a list
    with pytest.raises((ValueError, ValidationError)):
        GridCfg(sweep={"seed": [[1, 2], [3, 4]]})


# ---------------------------------------------------------------------------
# Test 8: Each returned cell preserves the grid block (parallelism, fail_fast)
# ---------------------------------------------------------------------------


def test_grid_cells_preserve_grid_block() -> None:
    """Every returned ExperimentConfig preserves the original grid block."""
    from atm.experiment import load_grid_configs

    configs = load_grid_configs(str(GRID_MINIMAL))
    assert len(configs) == 8

    for cfg in configs:
        assert cfg.grid is not None
        assert cfg.grid.parallelism == 2
        assert cfg.grid.fail_fast is False
        assert cfg.grid.seeds == [42, 43]
