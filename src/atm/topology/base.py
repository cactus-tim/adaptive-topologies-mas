"""Topology Protocol, TopologyConfig, TopologyRegistry, _should_stop helper,
and get_topology_extras indirection helper.

Key contracts (arch.md §6, §7.1):
  - Topology is a @runtime_checkable Protocol with:
      name: str
      build(agents, cfg) -> CompiledStateGraph
  - TopologyConfig holds name + max_iterations + optional extra dict.
  - TopologyRegistry validates presence of 'build' and 'name' on registration.
  - _should_stop implements stopping-precedence:
      budget → (NOT here — raised by LLMWrapper as BudgetExceededError)
      max_iter → topology_success → topology_max → continue
    Returns (bool, reason_str) where reason_str matches FinishReason.value.
  - get_topology_extras extracts the per-topology bucket from cfg.extra,
    handling both the namespaced shape (production) and the legacy flat
    shape (test fixtures / legacy callers) for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from atm.storage.models import FinishReason

# ---------------------------------------------------------------------------
# Topology name constants — used by get_topology_extras detection logic
# ---------------------------------------------------------------------------

#: Frozenset of the canonical topology names that form valid namespace keys
#: in the structured ``cfg.extra`` dict produced by the experiment runner.
#: A ``cfg.extra`` whose *every* top-level key belongs to this set is treated
#: as a namespaced dict; otherwise it is treated as a legacy-flat dict.
_TOPOLOGY_NAMES: frozenset[str] = frozenset(
    {"star", "chain", "debate", "hierarchical", "mesh", "adaptive"}
)

# ---------------------------------------------------------------------------
# TopologyConfig
# ---------------------------------------------------------------------------


class TopologyConfig(BaseModel):
    """Configuration for a topology instance.

    Attributes:
        name:           Topology identifier (e.g. "star", "chain").
        max_iterations: Global hard cap on iter_total. Enforced by _should_stop.
        extra:          Topology-specific parameters (phase caps, etc.).
    """

    name: str
    max_iterations: int = 20
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Topology Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Topology(Protocol):
    """Duck-typing contract for all topology implementations (arch.md §6).

    Concrete classes do NOT inherit from this Protocol — LangGraph duck-typing
    style. TopologyRegistry validates presence of 'build' and 'name' on
    registration as an explicit guard.
    """

    name: str

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a LangGraph CompiledStateGraph.

        Args:
            agents: Dict mapping agent_id → BaseAgent instance.
            cfg:    Topology-specific configuration.
            **kwargs: Optional extras (e.g. checkpointer=...).

        Returns:
            A compiled LangGraph graph ready to ainvoke.
        """
        ...


# ---------------------------------------------------------------------------
# TopologyRegistry
# ---------------------------------------------------------------------------


class TopologyRegistry:
    """Class-level registry that maps topology name → topology class.

    Usage:
        @TopologyRegistry.register("star")
        class StarTopology:
            name = "star"
            def build(self, agents, cfg, **kwargs): ...

    Design:
        - register() is a decorator factory; it validates that the class has
          callable 'build' and non-empty 'name' attribute before storing.
        - get() raises KeyError for unknown names.
        - list_names() returns a copy of registered names.
    """

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """Decorator factory — register a topology class under *name*.

        Args:
            name: Key to store the topology class under.

        Returns:
            A decorator that validates and stores the class.

        Raises:
            ValueError: If the class lacks a callable 'build' or 'name' attribute.
        """

        def decorator(topology_cls: type) -> type:
            # Validate 'build' is a callable attribute
            build = getattr(topology_cls, "build", None)
            if build is None or not callable(build):
                raise ValueError(
                    f"Topology class {topology_cls.__name__!r} must define a callable 'build' method."
                )
            # Validate 'name' is a non-empty string attribute
            cls_name = getattr(topology_cls, "name", None)
            if cls_name is None:
                raise ValueError(
                    f"Topology class {topology_cls.__name__!r} must define a 'name' class attribute."
                )
            cls._registry[name] = topology_cls
            return topology_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieve a registered topology class by name.

        Raises:
            KeyError: If no topology is registered under *name*.
        """
        if name not in cls._registry:
            raise KeyError(
                f"No topology registered under name {name!r}. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list[str]:
        """Return a list of all registered topology names."""
        return list(cls._registry.keys())


# ---------------------------------------------------------------------------
# _should_stop helper
# ---------------------------------------------------------------------------


def _should_stop(
    state: dict[str, Any],
    cfg: TopologyConfig,
    *,
    topology_success: bool,
    topology_max_reached: bool,
) -> tuple[bool, str]:
    """Evaluate stopping conditions according to arch.md §7.1 precedence.

    Precedence (highest to lowest):
      1. max_iter   — iter_total >= max_iterations → FinishReason.MAX_ITER
      2. topology_success  → FinishReason.SUCCESS
      3. topology_max_reached → FinishReason.TOPOLOGY_MAX
      4. continue   → (False, "")

    Budget is NOT detected here. BudgetExceededError is raised by LLMWrapper
    and caught by the runner at a higher level.

    Args:
        state:                 GraphState dict; reads state["shared"]["iter_total"].
        cfg:                   TopologyConfig with max_iterations.
        topology_success:      True if topology-specific success criterion met.
        topology_max_reached:  True if topology-specific max (e.g. verify_max_iter) hit.

    Returns:
        (should_stop: bool, reason: str) — reason matches FinishReason.value or "".
    """
    shared = state.get("shared", {})
    iter_total: int = shared.get("iter_total", 0)

    # Priority 1: global max_iterations guard
    if iter_total >= cfg.max_iterations:
        return True, FinishReason.MAX_ITER.value

    # Priority 2: topology-level success
    if topology_success:
        return True, FinishReason.SUCCESS.value

    # Priority 3: topology-internal max reached (e.g. verify_max_iter)
    if topology_max_reached:
        return True, FinishReason.TOPOLOGY_MAX.value

    # Priority 4: continue
    return False, ""


# ---------------------------------------------------------------------------
# get_topology_extras — per-topology bucket indirection helper
# ---------------------------------------------------------------------------


def get_topology_extras(cfg: TopologyConfig, topology_name: str) -> dict[str, Any]:
    """Return the per-topology bucket from ``cfg.extra``.

    Handles two forms transparently so that topology builders work correctly
    with both production configs (namespaced) and legacy test fixtures (flat):

    **Form 1 — Typed TopologyExtras (production path):**
    The experiment runner serialises ``TopologyCfg.extra`` (a Pydantic
    ``TopologyExtras`` model) to a dict via ``model_dump(exclude_none=True)``
    before constructing the internal ``TopologyConfig``. The resulting dict has
    every top-level key in ``_TOPOLOGY_NAMES`` (e.g.
    ``{"star": {...}, "mesh": {...}}``). In this case the function returns
    ``cfg.extra[topology_name]``, or ``{}`` if the key is absent.

    **Form 2 — Legacy flat dict (test fixtures / legacy callers):**
    Many unit tests construct ``TopologyConfig(extra={"max_rounds": 5})``
    directly. If *any* top-level key falls outside ``_TOPOLOGY_NAMES`` the
    entire dict is treated as a flat bucket belonging to the requested topology
    and returned verbatim. This preserves backwards compatibility without test
    churn.

    Detection invariant:
        A ``cfg.extra`` dict is treated as **namespaced** iff *every* top-level
        key is a member of ``_TOPOLOGY_NAMES``. Any other shape (including a
        mix of namespace and non-namespace keys) is treated as **legacy-flat**.

    Args:
        cfg:           ``TopologyConfig`` passed to the topology builder.
        topology_name: The canonical topology identifier (e.g. ``"mesh"``).

    Returns:
        A plain ``dict[str, Any]`` ready for builder consumption.
        Always returns ``{}`` for an empty or absent ``cfg.extra``.
    """
    extra: dict[str, Any] = cfg.extra or {}

    if not extra:
        return {}

    # Check whether every top-level key is a known topology namespace key.
    is_namespaced = all(k in _TOPOLOGY_NAMES for k in extra)

    if is_namespaced:
        # Namespaced form: return the bucket for this topology (default {}).
        bucket = extra.get(topology_name, {})
        # Guard: bucket must be a dict (not a stray scalar from a buggy dump).
        return dict(bucket) if isinstance(bucket, dict) else {}
    else:
        # Legacy-flat form: treat the whole dict as this topology's bucket.
        return dict(extra)
