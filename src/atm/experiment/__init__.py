"""Public API for the atm.experiment package.

Exports:
  - ExperimentConfig: top-level Pydantic config schema
  - load_config:      OmegaConf → Pydantic loader
"""

from atm.experiment.config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
