"""Public API for the atm.topology package.

Exports:
  - Topology        — @runtime_checkable Protocol (arch.md §6)
  - TopologyConfig  — Pydantic config model
  - TopologyRegistry — class-level registry for topology lookup
  - _should_stop    — stopping-precedence helper (arch.md §7.1)

Side-effect imports (guarded):
  - star.py, chain.py, mesh.py, debate.py, and hierarchical.py register
    themselves via @TopologyRegistry.register() when imported. Guards ensure
    this package loads cleanly even if a module does not yet exist.

Registration list finalized at M6+M7. Future topologies must be added explicitly here.
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
    "Topology",
    "TopologyConfig",
    "TopologyRegistry",
    "_should_stop",
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
