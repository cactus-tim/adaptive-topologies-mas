"""Public API for the atm.topology package.

Exports:
  - Topology        — @runtime_checkable Protocol (arch.md §6)
  - TopologyConfig  — Pydantic config model
  - TopologyRegistry — class-level registry for topology lookup
  - _should_stop    — stopping-precedence helper (arch.md §7.1)

Side-effect imports (guarded):
  - star.py and chain.py register themselves via @TopologyRegistry.register()
    when imported. Guards ensure this package loads cleanly even before
    those modules are created (Steps 2 and 3 are parallel to this step).

IMPORTANT: Steps 2 and 3 must NOT edit this file.
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
