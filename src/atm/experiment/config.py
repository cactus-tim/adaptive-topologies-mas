"""Pydantic v2 schemas for experiment configuration + OmegaConf loader.

Public API:
  - BudgetCfg, ModelCfg, ScratchpadCfg, AgentSetCfg, TopologyCfg, TaskCfg,
    ObservabilityCfg, ExperimentConfig
  - load_config(path, overrides) -> ExperimentConfig

Pipeline (arch.md §12.2):
  OmegaConf.load(path)
  → merge _includes
  → merge OmegaConf.from_dotlist(overrides)
  → OmegaConf.to_container(resolve=True)
  → ExperimentConfig.model_validate(dict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------


class BudgetCfg(BaseModel):
    """Per-call, per-run, and per-experiment budget limits (arch.md §12.1)."""

    per_call_usd: float = 0.10
    per_run_usd: float = 0.50
    per_experiment_usd: float = 50.0
    warn_ratio: float = 0.8


class ModelCfg(BaseModel):
    """Per-role model assignment (arch.md §12.1).

    Resolution: get_model_for(role) -> by_role.get(role) or default.
    """

    default: str = "openai:gpt-4o-mini"
    by_role: dict[str, str] = Field(default_factory=dict)
    judge: str = "openai:gpt-4o"
    summarizer: str = "openai:gpt-4o-mini"
    router: str = "openai:gpt-4o-mini"
    provider_opts: dict[str, Any] = Field(default_factory=dict)
    prompt_cache_scope: Literal["per_run", "per_task", "off"] = "per_run"

    def get_model_for(self, role: str) -> str:
        """Return model id for the given role, falling back to default."""
        return self.by_role.get(role, self.default)


class ScratchpadCfg(BaseModel):
    """Scratchpad memory policy configuration (arch.md §12.1)."""

    policy: Literal["window_with_summary", "window_only", "full"] = "window_with_summary"
    window_size: int = 3
    summarizer_model: str | None = None
    context_token_budget: int = 12000


class AgentSetCfg(BaseModel):
    """Agent set selection + scratchpad policy (arch.md §12.1)."""

    set: str  # e.g. "canonical_4", "debate", "hier"
    scratchpad: ScratchpadCfg = Field(default_factory=ScratchpadCfg)


class TopologyCfg(BaseModel):
    """Topology selection and stopping parameters (arch.md §12.1)."""

    name: Literal["star", "chain", "mesh", "debate", "hierarchical", "adaptive"]
    max_iterations: int = 10
    extra: dict[str, Any] = Field(default_factory=dict)


class TaskCfg(BaseModel):
    """Task specification (arch.md §12.1, M6 simplified)."""

    name: str
    input: str = ""  # direct task input string (M6 smoke: inline prompt)
    split: str = "test"
    limit: int | None = None
    shuffle_seed: int = 0


class ObservabilityCfg(BaseModel):
    """Storage/observability configuration (arch.md §12.1)."""

    parquet_dir: str = "data/experiments"
    pg_dsn: str  # required; supports ${oc.env:PG_DSN,...} interpolation
    callback_sync: bool = False


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration schema (arch.md §12.1).

    No sweep/grid/dry-run in M6 scope — those are M12.
    """

    name: str
    seed: int = 42
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    model: ModelCfg
    agents: AgentSetCfg
    topology: TopologyCfg
    task: TaskCfg
    observability: ObservabilityCfg

    @model_validator(mode="after")
    def _check_topology(self) -> ExperimentConfig:
        if self.topology.name == "adaptive" and self.topology.max_iterations <= 0:
            raise ValueError("adaptive topology requires max_iterations > 0")
        return self


# ---------------------------------------------------------------------------
# OmegaConf loader
# ---------------------------------------------------------------------------


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
