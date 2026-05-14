"""OmegaConf loader and grid sweep expander for experiment configs.

This module was split from ``atm.experiment.config`` (M12 Phase 2) to keep
the schema definitions separate from I/O concerns.

Public API:
  - load_config(path, overrides) -> ExperimentConfig
  - load_grid_configs(path, overrides) -> list[ExperimentConfig]
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from atm.experiment.config import ExperimentConfig, GridCfg


def _merge_includes(cfg: DictConfig, base_dir: Path) -> DictConfig:
    """Merge any ``include`` key found in the config (simplified include support).

    Supports:
      include: conf/topology/star.yaml     # single file
      include:                             # list of files
        - conf/topology/star.yaml
        - conf/agents/canonical_4.yaml

    The include files are merged INTO the base config (merge-left precedence).
    After merging, the ``include`` key is removed from the result.
    """
    if "include" not in cfg:
        return cfg

    include_val: Any = OmegaConf.select(cfg, "include")

    include_paths: list[str] = (
        [str(include_val)] if isinstance(include_val, str) else [str(p) for p in include_val]
    )

    keys: list[str] = [str(k) for k in cfg if k != "include"]
    merged: DictConfig = OmegaConf.masked_copy(cfg, keys)

    for inc_path_str in include_paths:
        inc_path = Path(inc_path_str)
        if not inc_path.is_absolute():
            inc_path = base_dir / inc_path_str
        inc_cfg: DictConfig = OmegaConf.load(inc_path)  # type: ignore[assignment]
        # include contents are the base; main cfg values take precedence
        merged = OmegaConf.merge(inc_cfg, merged)  # type: ignore[assignment]

    return merged


def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Pipeline (arch.md §12.2):
      1. OmegaConf.load(path)
      2. Merge ``include`` files (simplified include mechanism)
      3. OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
      4. OmegaConf.to_container(resolve=True)
      5. ExperimentConfig.model_validate(data)

    Args:
        path:      Path to the base YAML config file.
        overrides: List of dotlist override strings, e.g. ["+topology.name=star"].
                   Leading ``+`` is stripped before passing to OmegaConf.from_dotlist.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        pydantic.ValidationError: If the resolved config does not match the schema.
        FileNotFoundError: If the config file does not exist.
        omegaconf.OmegaConfBaseException: If interpolation resolution fails.
    """
    cfg_path = Path(path)
    base_dir = cfg_path.parent

    # Step 1: load base YAML
    cfg: DictConfig = OmegaConf.load(cfg_path)  # type: ignore[assignment]

    # Step 2: merge includes
    cfg = _merge_includes(cfg, base_dir)

    # Step 3: apply CLI overrides
    if overrides:
        # Strip leading '+' — OmegaConf.from_dotlist does not support it
        clean_overrides = [o.lstrip("+") for o in overrides]
        override_cfg = OmegaConf.from_dotlist(clean_overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)  # type: ignore[assignment]

    # Step 4 + 5: resolve and convert to plain dict, then validate with Pydantic
    data: Any = OmegaConf.to_container(cfg, resolve=True)
    return ExperimentConfig.model_validate(data)


def load_grid_configs(
    path: str | Path,
    overrides: list[str] | None = None,
) -> list[ExperimentConfig]:
    """Load a config and expand any ``grid:`` block into a list of configs.

    If the loaded config has no ``grid`` block, returns ``[cfg]`` (single-run
    mode — fully backward-compatible with existing call sites).

    When a ``grid`` block is present, computes the Cartesian product of:
      - all sweep parameter combinations (one value per key per cell)
      - all seeds listed in ``grid.seeds``

    Ordering (deterministic):
      ``(sweep_cell_0_value, sweep_cell_1_value, ..., seed)``
      where sweep keys are iterated in alphabetical order.

    Each returned ``ExperimentConfig`` preserves the original ``grid`` block so
    callers can read ``cfg.grid.parallelism`` and ``cfg.grid.fail_fast``.

    Args:
        path:      Path to the base YAML config file (str or Path).
        overrides: OmegaConf dotlist overrides applied before grid expansion.

    Returns:
        List of validated ``ExperimentConfig`` instances (length >= 1).

    Raises:
        ValueError: If any sweep value is a non-scalar (list or dict).
        pydantic.ValidationError: If a generated cell config fails schema validation.
        FileNotFoundError: If the config file does not exist.

    Example::

        configs = load_grid_configs("conf/sweep.yaml")
        # → one ExperimentConfig per (sweep_combination × seed) cell
    """
    base_cfg = load_config(str(path), overrides)

    grid: GridCfg | None = base_cfg.grid
    if grid is None:
        return [base_cfg]

    # Validate sweep values are all scalars (str/int/float/bool).
    _SCALAR_TYPES = (str, int, float, bool)
    for key, values in grid.sweep.items():
        for v in values:
            if not isinstance(v, _SCALAR_TYPES):
                raise ValueError(
                    f"load_grid_configs: sweep value for key '{key}' must be a scalar "
                    f"(str/int/float/bool), got {type(v).__name__}: {v!r}"
                )

    # Sorted keys for deterministic ordering.
    sweep_keys = sorted(grid.sweep.keys())
    sweep_value_lists = [grid.sweep[k] for k in sweep_keys]

    configs: list[ExperimentConfig] = []

    for sweep_combo in itertools.product(*sweep_value_lists):
        for seed in grid.seeds:
            # Start from a fresh copy of the base dict each cell.
            cell_dict: dict[str, Any] = base_cfg.model_dump(mode="python")
            cell_oc: DictConfig = OmegaConf.create(cell_dict)

            # Apply sweep overrides.
            for key, value in zip(sweep_keys, sweep_combo):
                OmegaConf.update(cell_oc, key, value, merge=True)

            # Apply seed override.
            OmegaConf.update(cell_oc, "seed", seed, merge=True)

            # Resolve and validate.
            cell_data: Any = OmegaConf.to_container(cell_oc, resolve=True)
            cell_cfg = ExperimentConfig.model_validate(cell_data)
            configs.append(cell_cfg)

    return configs
