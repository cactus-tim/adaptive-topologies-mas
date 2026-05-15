"""Public API for the atm.experiment package.

Exports:
  - ExperimentConfig: top-level Pydantic config schema
  - GridCfg:          grid sweep configuration (M12)
  - EstimateCfg:      cost estimation configuration (M12)
  - load_config:      OmegaConf -> Pydantic loader
  - load_grid_configs: load + expand grid sweep configs (M12)
  - RunResult:        result of a single experiment run
  - run_one:          main experiment orchestrator
"""

from atm.experiment.config import EstimateCfg, ExperimentConfig, GridCfg
from atm.experiment.loader import load_config, load_grid_configs
from atm.experiment.runner import RunResult, run_one

__all__ = [
    "EstimateCfg",
    "ExperimentConfig",
    "GridCfg",
    "RunResult",
    "load_config",
    "load_grid_configs",
    "run_one",
]
