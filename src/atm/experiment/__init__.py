"""Public API for the atm.experiment package.

Exports:
  - ExperimentConfig: top-level Pydantic config schema
  - load_config:      OmegaConf → Pydantic loader
  - RunResult:        result of a single experiment run
  - run_one:          main experiment orchestrator
"""

from atm.experiment.config import ExperimentConfig, load_config
from atm.experiment.runner import RunResult, run_one

__all__ = ["ExperimentConfig", "RunResult", "load_config", "run_one"]
