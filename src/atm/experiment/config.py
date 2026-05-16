"""Pydantic v2 schemas for experiment configuration.

The OmegaConf loader (``load_config``) and grid sweep expander
(``load_grid_configs``) have been split into ``atm.experiment.loader``.
This module re-exports ``load_config`` for full back-compat.

Public API:
  - BudgetCfg, ModelCfg, ScratchpadCfg, AgentSetCfg, TopologyCfg, TaskCfg,
    ObservabilityCfg, ExperimentConfig, GridCfg, EstimateCfg
  - TopologyExtras, StarExtras, ChainExtras, DebateExtras, HierarchicalExtras,
    MeshExtras, AdaptiveExtras
  - load_config(path, overrides) -> ExperimentConfig  (re-exported from loader)
"""

from __future__ import annotations

import types
import warnings
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atm.core.types import HumanRole

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

    fake_fixtures: Optional mapping from role name to fixture file path.
    Used by _build_llm_wrappers when model_id starts with "fake:scripted".
    If empty for scripted mode, falls back to FakeLLM(mode="echo") with a warning.

    Example::

        model:
          default: "fake:scripted"
          fake_fixtures:
            planner: "tests/fixtures/llm/m6_chain_planner.yaml"
            executor: "tests/fixtures/llm/m6_chain_executor.yaml"
            critic:   "tests/fixtures/llm/m6_chain_critic.yaml"
    """

    default: str = "openai:gpt-4o-mini"
    by_role: dict[str, str] = Field(default_factory=dict)
    fake_fixtures: dict[str, str] = Field(default_factory=dict)
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


# ---------------------------------------------------------------------------
# Topology extras sub-schemas
# ---------------------------------------------------------------------------

# Canonical set of topology names; used by both TopologyExtras and the
# bw-compat remapper to determine whether an extras dict is namespaced.
_TOPOLOGY_EXTRA_NAMES: frozenset[str] = frozenset(
    {"star", "chain", "debate", "hierarchical", "mesh", "adaptive"}
)

# Flat extra keys that are remapped to a specific namespace on detection.
# Any flat key that is NOT in this mapping raises ValidationError after
# remapping (because TopologyExtras has extra="forbid").
_FLAT_TO_NAMESPACE: dict[str, tuple[str, str]] = {
    # shared round-counters → debate + hierarchical only (NOT mesh, NOT adaptive)
    # (handled separately because they scatter to two namespaces)
    # "max_rounds": handled inline in _remap_flat_extras
    # star-specific phase limits
    "planning_max_iter": ("star", "planning_max_iter"),
    "exec_max_iter": ("star", "exec_max_iter"),
    "verify_max_iter": ("star", "verify_max_iter"),
    # mesh-specific (aliased key in legacy form)
    "mesh_max_rounds": ("mesh", "max_rounds"),
    "consensus_threshold": ("mesh", "consensus_threshold"),
    "broadcast_bus_cap": ("mesh", "broadcast_bus_cap"),
    "activation_policy": ("mesh", "activation_policy"),
    "agent_order": ("mesh", "agent_order"),
    # debate-specific
    "debater_pro_id": ("debate", "debater_pro_id"),
    "debater_contra_id": ("debate", "debater_contra_id"),
    "judge_id": ("debate", "judge_id"),
    # hierarchical-specific
    "sub_teams": ("hierarchical", "sub_teams"),
    "final_answer_strategy": ("hierarchical", "final_answer_strategy"),
    # adaptive-specific
    "phase_router": ("adaptive", "phase_router"),
    "topology_router": ("adaptive", "topology_router"),
    "subgraph_max_iterations": ("adaptive", "subgraph_max_iterations"),
    "switch_guards": ("adaptive", "switch_guards"),
    "switch_guards_config": ("adaptive", "switch_guards_config"),
    "run_id": ("adaptive", "run_id"),
}


class StarExtras(BaseModel):
    """Extra parameters for the Star topology.

    Defaults mirror ``star.py`` constants ``_DEFAULT_PLANNING_MAX_ITER``,
    ``_DEFAULT_EXEC_MAX_ITER``, and ``_DEFAULT_VERIFY_MAX_ITER``.
    """

    model_config = ConfigDict(extra="forbid")

    planning_max_iter: int = 2
    exec_max_iter: int = 5
    verify_max_iter: int = 3


class ChainExtras(BaseModel):
    """Extra parameters for the Chain topology (reserved; chain reads no extras)."""

    model_config = ConfigDict(extra="forbid")


class DebateExtras(BaseModel):
    """Extra parameters for the Debate topology.

    Defaults mirror ``debate.py`` constant ``_DEFAULT_MAX_ROUNDS``.
    """

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = 4
    debater_pro_id: str | None = None
    debater_contra_id: str | None = None
    judge_id: str | None = None


class HierarchicalExtras(BaseModel):
    """Extra parameters for the Hierarchical topology.

    Defaults mirror ``hierarchical.py`` constants ``_DEFAULT_MAX_ROUNDS``
    and ``_DEFAULT_FINALIZE_SIGNAL``.
    """

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = 4
    finalize_signal: str = "top_coord_finalize"
    sub_teams: list[Any] | None = None
    final_answer_strategy: str | None = None


class MeshExtras(BaseModel):
    """Extra parameters for the Mesh topology.

    Defaults mirror ``mesh.py`` constants:
      - ``_DEFAULT_MAX_ROUNDS = 12``
      - ``_DEFAULT_CONSENSUS_THRESHOLD = 3``
      - ``_DEFAULT_BROADCAST_BUS_CAP = 200``
      - ``_DEFAULT_ACTIVATION_POLICY = "round_robin"``
      - ``_DEFAULT_AGENT_ORDER = ["planner","researcher","executor","critic"]``

    ``max_rounds`` corresponds to the legacy ``mesh_max_rounds`` flat key.
    The bw-compat validator in ``TopologyCfg`` remaps ``mesh_max_rounds``
    to ``mesh.max_rounds`` transparently.

    Field names match the keys read by ``mesh.py`` builder exactly so that
    YAML values set via ``topology.extra.mesh.<key>`` propagate correctly.
    """

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = 12
    consensus_threshold: int = 3
    broadcast_bus_cap: int = 200
    activation_policy: str = "round_robin"
    agent_order: list[str] = Field(
        default_factory=lambda: ["planner", "researcher", "executor", "critic"]
    )


class AdaptiveExtras(BaseModel):
    """Extra parameters for the Adaptive topology.

    Phase-limit defaults (3/10/4) match the inline fallback literals in
    ``adaptive.py:394-396`` — they are intentionally LARGER than star's 2/5/3
    because adaptive runs longer reasoning phases.

    ``run_id`` defaults to ``None`` so that the runner's ``model_dump(
    exclude_none=True)`` omits it entirely, allowing the builder's own
    ``uuid.uuid4()`` fallback to fire (plain ``None`` would produce the
    literal string ``"None"`` via ``str(None)``).

    ``phase_router`` and ``topology_router`` are parsed but NOT consumed by
    the current builder implementation — kept for forward-compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    planning_max_iter: int = 3
    exec_max_iter: int = 10
    verify_max_iter: int = 4
    subgraph_max_iterations: int = 10
    switch_guards: bool = True
    switch_guards_config: dict[str, Any] | None = None
    run_id: str | None = None
    phase_router: str = "rule"  # NOTE: parsed but NOT consumed by builder
    topology_router: str = "rule"  # NOTE: parsed but NOT consumed by builder


# ---------------------------------------------------------------------------
# TopologyExtras — schema rationale
#
# Single source of truth: default values in each sub-model below mirror the
# ``_DEFAULT_*`` constants defined in the corresponding topology module.  Any
# drift between schema defaults and module constants is caught immediately by
# the schema-vs-source parity guard test in tests/unit/experiment/test_config.py.
#
# Dotpath access in sweep configs follows the namespaced shape:
#   topology.extra.mesh.max_rounds: 12
#   topology.extra.debate.max_rounds: 4
#   topology.extra.star.planning_max_iter: 3
# Flat (legacy) extras are auto-remapped by the bw-compat pre-validator on
# ``TopologyCfg``; see ``_remap_flat_extras`` and the ``@model_validator``
# below for the full scattering rules.
# ---------------------------------------------------------------------------


class TopologyExtras(BaseModel):
    """Namespaced container for per-topology extra parameters.

    Each sub-field holds the extras for exactly one topology.  Sweeps and
    YAML configs access them via dotpath, e.g.::

        topology.extra.mesh.max_rounds: 12
        topology.extra.debate.max_rounds: 4

    Unknown top-level keys (i.e. topology names not in this schema) are
    rejected by ``extra="forbid"`` to surface typos early.
    """

    model_config = ConfigDict(extra="forbid")

    star: StarExtras = Field(default_factory=StarExtras)
    chain: ChainExtras = Field(default_factory=ChainExtras)
    debate: DebateExtras = Field(default_factory=DebateExtras)
    hierarchical: HierarchicalExtras = Field(default_factory=HierarchicalExtras)
    mesh: MeshExtras = Field(default_factory=MeshExtras)
    adaptive: AdaptiveExtras = Field(default_factory=AdaptiveExtras)


def _remap_flat_extras(flat: dict[str, Any]) -> dict[str, Any]:
    """Remap a flat legacy extras dict to the namespaced ``TopologyExtras`` shape.

    Emits a ``DeprecationWarning`` describing the flat keys found.

    Scattering rules:
    - ``max_rounds`` → ``debate.max_rounds`` AND ``hierarchical.max_rounds``
      (NOT mesh — mesh default of 12 is starvation-safe; NOT adaptive — no
      such field).
    - All other flat keys → their declared namespace per ``_FLAT_TO_NAMESPACE``.

    Unknown keys (not in ``_FLAT_TO_NAMESPACE`` and not ``max_rounds``) are
    placed in a synthetic top-level key so that Pydantic's ``extra="forbid"``
    on ``TopologyExtras`` raises a ``ValidationError`` with a clear message.
    """
    namespaced: dict[str, dict[str, Any]] = {}
    unknown_keys: list[str] = []

    for key, value in flat.items():
        if key == "max_rounds":
            namespaced.setdefault("debate", {})["max_rounds"] = value
            namespaced.setdefault("hierarchical", {})["max_rounds"] = value
        elif key in _FLAT_TO_NAMESPACE:
            ns, field = _FLAT_TO_NAMESPACE[key]
            namespaced.setdefault(ns, {})[field] = value
        else:
            unknown_keys.append(key)

    if unknown_keys:
        # Include unknown keys verbatim at top-level so TopologyExtras'
        # extra="forbid" produces an informative ValidationError.
        for key in unknown_keys:
            namespaced[key] = flat[key]

    return namespaced


class TopologyCfg(BaseModel):
    """Topology selection and stopping parameters (arch.md §12.1).

    ``extra`` is a typed, namespaced container (``TopologyExtras``).  Each
    topology reads only its own sub-namespace, eliminating shared-key
    collisions (e.g. ``max_rounds`` in debate vs hierarchical vs mesh).

    **Legacy flat form** (bw-compat):
    YAML configs written before this refactor may use a flat ``extra:`` block
    such as ``extra: {max_rounds: 4}``.  The ``@model_validator(mode="before")``
    detects flat-shaped dicts and remaps them to the namespaced form while
    emitting a ``DeprecationWarning``.  See ``_remap_flat_extras`` for the
    scattering rules.
    """

    name: Literal["star", "chain", "mesh", "debate", "hierarchical", "adaptive"]
    max_iterations: int = 10
    extra: TopologyExtras = Field(default_factory=TopologyExtras)

    @model_validator(mode="before")
    @classmethod
    def _bw_compat_flat_extras(cls, values: Any) -> Any:
        """Detect flat legacy ``extra`` dicts and remap to namespaced form.

        A dict is considered FLAT (legacy) if it has at least one key that is
        NOT a recognised topology name.  A dict whose keys are ALL topology
        names is already in namespaced form and is passed through unchanged.
        An empty dict is also passed through unchanged.
        """
        if not isinstance(values, dict):
            return values

        raw_extra = values.get("extra")
        if not isinstance(raw_extra, dict) or not raw_extra:
            return values

        # Check if all keys are known topology names (namespaced form).
        if raw_extra.keys() <= _TOPOLOGY_EXTRA_NAMES:
            # Already namespaced — pass through without warning.
            return values

        # Flat (legacy) form detected.
        warnings.warn(
            "flat topology.extra keys are deprecated; use namespaced form "
            "(e.g. topology.extra.mesh.max_rounds)",
            DeprecationWarning,
            stacklevel=2,
        )
        values = dict(values)
        values["extra"] = _remap_flat_extras(raw_extra)
        return values


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


class EvaluationCfg(BaseModel):
    """Post-hoc evaluation configuration (M11).

    Controls the LLM judge used by the aggregator and self-consistency N.
    Defaults are backward-compatible — existing YAML configs that do not
    include an ``evaluation:`` section will use these values automatically.

    Fields:
        judge_model:             Model ID for LLM judge calls
                                 (format: ``provider:model``).
        judge_self_consistency_n: Number of independent judge calls when
                                 using SelfConsistentJudge (1 = disabled).
                                 Range: 1..10.
    """

    judge_model: str = "openai:gpt-4o"
    judge_self_consistency_n: int = Field(default=3, ge=1, le=10)


class HumanCfg(BaseModel):
    """HITL gateway configuration (arch.md M9/M9.1/M9.2).

    Controls whether human-in-the-loop is active, which gateway to use,
    which role the human plays, and how timeouts are handled.

    When ``enabled=False`` (the default), the topology behaves exactly
    as before — no ``human_reviewer`` node is inserted.

    ``model`` is an optional override for the LLM model used by
    ``LLMSimulatedGateway``. ``None`` -> derive from ModelCfg.default.

    ``extra`` is per-topology HITL configuration (M9.1). Keys are
    topology-specific (e.g. ``judge``, ``scope``, ``activation_round``,
    ``override_coordinator``, ``human_can_override_router``).

    M9.2 — Adaptive Role Router fields:
      ``role_router`` — strategy for selecting active HumanRole per-phase:
        ``"fixed"`` (default) → use ``role`` directly; back-compat byte-identical.
        ``"rule"`` → RuleBasedRoleRouter (table phase → HumanRole, optional override).
        ``"llm"`` → LLMRoleRouter (LLM decides; falls back to rule on error).
      ``role_table`` — optional override for the rule router's phase → HumanRole table.
      ``role_router_model`` — optional LLM model override for LLMRoleRouter.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    gateway: Literal["llm_simulated", "cli"] = "llm_simulated"
    role: HumanRole = HumanRole.REVIEWER
    timeout_s: float | None = 900.0
    timeout_policy: Literal["fail", "llm_fallback", "skip"] = "llm_fallback"
    model: str | None = None
    extra: dict[str, Any] | None = None
    role_router: Literal["fixed", "rule", "llm"] = "fixed"
    role_table: dict[str, str] | None = None
    role_router_model: str | None = None


# ---------------------------------------------------------------------------
# M12: Grid sweep helpers
# ---------------------------------------------------------------------------


def _resolve_dotpath(path: str) -> None:
    """Validate that *path* resolves to a typed scalar leaf in ExperimentConfig.

    Walks ``ExperimentConfig.model_fields`` segment-by-segment.  Uses
    ``typing.get_origin`` / ``get_args`` to strip ``T | None`` wrappers so
    that optional sub-schemas are still reachable.

    Raises:
        ValueError: If any path segment does not resolve or the leaf field is
                    an untyped dict (e.g. ``topology.extra.*``).
    """
    segments = path.split(".")

    # We need ExperimentConfig to be defined — defer to a late-binding approach
    # via _EXPERIMENT_CONFIG_REF.  GridCfg validators run at instance
    # construction time (after the module finishes loading), so the ref is
    # always populated by then.
    cfg_cls = _EXPERIMENT_CONFIG_REF.get()
    if cfg_cls is None:
        # ExperimentConfig not yet defined — skip validation (happens only
        # during module load before the class body finishes).
        return

    fields = cfg_cls.model_fields

    for i, segment in enumerate(segments):
        if segment not in fields:
            raise ValueError(
                f"_resolve_dotpath: unknown field '{segment}' at segment {i} of '{path}'"
            )

        field_info = fields[segment]
        annotation = field_info.annotation

        # Strip Optional / X | None wrappers to get the inner type
        inner = _unwrap_optional(annotation)

        # If this is the last segment — it must be a scalar (not a BaseModel
        # or an untyped dict).
        if i == len(segments) - 1:
            # Reject untyped dict leaves (e.g. topology.extra, human.extra)
            origin = get_origin(inner)
            if origin is dict or inner is dict:
                raise ValueError(
                    f"_resolve_dotpath: '{path}' resolves to an untyped dict leaf — "
                    "sweep keys must target typed scalar fields"
                )
            # Allow scalar types (str, int, float, bool, Literal, etc.)
            return

        # Descend into sub-model
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            fields = inner.model_fields
        else:
            raise ValueError(
                f"_resolve_dotpath: cannot descend into non-model type at segment "
                f"'{segment}' (index {i}) of '{path}'"
            )


def _unwrap_optional(annotation: Any) -> Any:
    """Strip ``T | None`` (Union[T, None]) wrappers, return inner type T.

    Handles both ``X | None`` (Python 3.10+ native union via ``types.UnionType``)
    and ``Optional[X]`` / ``Union[X, None]`` (``typing.Union``).  Returns the
    annotation unchanged if it is not a nullable union, or if it is a union of
    multiple non-None types (not unwrappable).
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        args = tuple(a for a in get_args(annotation) if a is not type(None))
        if len(args) == 1:
            return args[0]
        # union of multiple non-None types — not unwrappable, return as-is
        return annotation
    return annotation


class _ExperimentConfigRef:
    """Simple mutable cell holding a forward reference to ExperimentConfig."""

    def __init__(self) -> None:
        self._cls: type[BaseModel] | None = None

    def set(self, cls: type[BaseModel]) -> None:
        self._cls = cls

    def get(self) -> type[BaseModel] | None:
        return self._cls


_EXPERIMENT_CONFIG_REF = _ExperimentConfigRef()


# ---------------------------------------------------------------------------
# M12: New sub-schemas
# ---------------------------------------------------------------------------


class GridCfg(BaseModel):
    """Grid sweep configuration (M12).

    ``sweep`` maps dotpath keys (e.g. ``"topology.name"``) to a list of
    scalar values to try.  Keys are validated against ``ExperimentConfig``
    model fields at construction time via ``_resolve_dotpath``.

    Fields:
        sweep:       Mapping of config dotpath → list of scalar values.
        parallelism: Max concurrent sweep runs (>= 1, default 4).
        fail_fast:   Stop the sweep on first failure (default False).
        seeds:       List of random seeds to cross-product with sweep
                     (>= 1 element, default [42]).
    """

    sweep: dict[str, list[str | int | float | bool]]
    parallelism: int = Field(default=4, ge=1)
    fail_fast: bool = False
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1)

    @field_validator("sweep")
    @classmethod
    def _validate_sweep_keys(
        cls, v: dict[str, list[str | int | float | bool]]
    ) -> dict[str, list[str | int | float | bool]]:
        """Validate that every sweep key is a resolvable typed scalar dotpath."""
        for key in v:
            _resolve_dotpath(key)
        return v


class EstimateCfg(BaseModel):
    """Cost/token estimation configuration (M12).

    Fields:
        heuristic_tokens_per_call: Average token count per LLM call for
                                   cost estimation (>= 1, default 1500).
        calls_per_iter:            Estimated LLM calls per iteration
                                   (>= 1, default 6).
        use_historical:            If True, supplement heuristic with
                                   historical run data when available
                                   (default True).
    """

    heuristic_tokens_per_call: int = Field(default=1500, ge=1)
    calls_per_iter: int = Field(default=6, ge=1)
    use_historical: bool = True


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration schema (arch.md §12.1).

    M12 additions: optional ``grid`` for sweep runs; ``estimate`` for
    pre-run cost estimation.
    """

    name: str
    seed: int = 42
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    model: ModelCfg
    agents: AgentSetCfg
    topology: TopologyCfg
    task: TaskCfg
    observability: ObservabilityCfg
    evaluation: EvaluationCfg = Field(default_factory=lambda: EvaluationCfg())
    human: HumanCfg | None = None
    grid: GridCfg | None = None
    estimate: EstimateCfg = Field(default_factory=EstimateCfg)

    @model_validator(mode="after")
    def _check_topology(self) -> ExperimentConfig:
        if self.topology.name == "adaptive" and self.topology.max_iterations <= 0:
            raise ValueError("adaptive topology requires max_iterations > 0")
        return self


# Register ExperimentConfig so _resolve_dotpath can walk its fields.
_EXPERIMENT_CONFIG_REF.set(ExperimentConfig)


# ---------------------------------------------------------------------------
# Back-compat re-export: loader functions live in atm.experiment.loader
# ---------------------------------------------------------------------------

from atm.experiment.loader import load_config as load_config  # noqa: E402
