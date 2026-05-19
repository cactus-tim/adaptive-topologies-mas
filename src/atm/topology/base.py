"""Topology Protocol, TopologyConfig, TopologyRegistry, _should_stop, get_topology_extras."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from atm.storage.models import FinishReason

_TOPOLOGY_NAMES: frozenset[str] = frozenset(
    {"star", "chain", "debate", "hierarchical", "mesh", "adaptive"}
)


class TopologyConfig(BaseModel):
    """Configuration for a topology instance (name, max_iterations, extra dict)."""

    name: str
    max_iterations: int = 20
    extra: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Topology(Protocol):
    """Duck-typing contract for all topology implementations."""

    name: str

    def build(
        self,
        agents: dict[str, Any],
        cfg: TopologyConfig,
        **kwargs: Any,
    ) -> Any:
        """Compile and return a LangGraph CompiledStateGraph."""
        ...


class TopologyRegistry:
    """Class-level registry mapping topology name → class."""

    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type], type]:
        """Decorator factory: validate and register topology class under *name*."""

        def decorator(topology_cls: type) -> type:
            build = getattr(topology_cls, "build", None)
            if build is None or not callable(build):
                raise ValueError(
                    f"Topology class {topology_cls.__name__!r} must define a callable 'build' method."
                )
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
        """Retrieve a registered topology class by name; raises KeyError if absent."""
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


def _should_stop(
    state: dict[str, Any],
    cfg: TopologyConfig,
    *,
    topology_success: bool,
    topology_max_reached: bool,
) -> tuple[bool, str]:
    """Return (should_stop, reason) following stopping precedence."""
    shared = state.get("shared", {})
    iter_total: int = shared.get("iter_total", 0)

    if iter_total >= cfg.max_iterations:
        return True, FinishReason.MAX_ITER.value

    if topology_success:
        return True, FinishReason.SUCCESS.value

    if topology_max_reached:
        return True, FinishReason.TOPOLOGY_MAX.value

    return False, ""


def get_topology_extras(cfg: TopologyConfig, topology_name: str) -> dict[str, Any]:
    """Return the per-topology bucket from ``cfg.extra``.

    Handles both namespaced form (``{"star": {...}, "mesh": {...}}``) and
    legacy-flat form (``{"max_rounds": 5}``). Detection: if every top-level
    key is in ``_TOPOLOGY_NAMES``, the dict is namespaced; otherwise flat.
    """
    extra: dict[str, Any] = cfg.extra or {}

    if not extra:
        return {}

    is_namespaced = all(k in _TOPOLOGY_NAMES for k in extra)

    if is_namespaced:
        bucket = extra.get(topology_name, {})
        return dict(bucket) if isinstance(bucket, dict) else {}
    else:
        return dict(extra)
