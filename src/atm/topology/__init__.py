"""Public API for the atm.topology package.

Exports:
  - Topology        — @runtime_checkable Protocol (arch.md §6)
  - TopologyConfig  — Pydantic config model
  - TopologyRegistry — class-level registry for topology lookup
  - _should_stop    — stopping-precedence helper (arch.md §7.1)
  - AdaptiveTopology — L2 adaptive meta-graph (M8)
  - build_adaptive_graph — convenience factory for AdaptiveTopology

Side-effect imports (guarded):
  - star.py, chain.py, mesh.py, debate.py, and hierarchical.py register
    themselves via @TopologyRegistry.register() when imported. Guards ensure
    this package loads cleanly even if a module does not yet exist.
  - adaptive.py registers under "adaptive" via @TopologyRegistry.register().

Registration list finalized at M6+M7+M8.
"""

from __future__ import annotations

import contextlib
import importlib

from atm.topology.base import (
    Topology,
    TopologyConfig,
    TopologyRegistry,
    _should_stop,
)

__all__ = [
    "AdaptiveTopology",
    "Topology",
    "TopologyConfig",
    "TopologyRegistry",
    "_should_stop",
    "build_adaptive_graph",
]

# ---------------------------------------------------------------------------
# Side-effect registration — guarded imports so the package loads cleanly
# even when star.py / chain.py do not yet exist.
# ---------------------------------------------------------------------------

with contextlib.suppress(ImportError):
    importlib.import_module("atm.topology.star")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.topology.chain")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.topology.mesh")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.topology.debate")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.topology.hierarchical")

with contextlib.suppress(ImportError):
    importlib.import_module("atm.topology.adaptive")

# Expose AdaptiveTopology and build_adaptive_graph only if the module exists.
with contextlib.suppress(ImportError, AttributeError):
    from atm.topology.adaptive import AdaptiveTopology, build_adaptive_graph
