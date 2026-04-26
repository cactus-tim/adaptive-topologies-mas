"""Conftest for topology unit tests.

Provides a fixture to ensure all topology classes are always registered in
TopologyRegistry before each test. This is needed because test_base.py's
TestTopologyRegistry.teardown_method() clears _registry, and Python's module
cache (sys.modules) prevents the side-effect re-import from firing again
automatically.
"""

from __future__ import annotations

import contextlib
import importlib
import sys

import pytest

from atm.topology.base import TopologyRegistry

# Mapping of topology name → (module_name, class_attribute_name)
_TOPOLOGY_MODULES: list[tuple[str, str, str]] = [
    ("star", "atm.topology.star", "StarTopology"),
    ("chain", "atm.topology.chain", "ChainTopology"),
    ("hierarchical", "atm.topology.hierarchical", "HierarchicalTopology"),
]


@pytest.fixture(autouse=True)
def ensure_star_registered() -> None:
    """Re-register all topology classes if they were cleared from the registry.

    If the module is already in sys.modules, directly registers the class
    (avoids reload which creates new class objects, breaking identity checks).
    If the module is not imported yet, imports it (triggering the decorator).
    """
    for topology_name, mod_name, cls_attr in _TOPOLOGY_MODULES:
        if topology_name not in TopologyRegistry.list_names():
            if mod_name in sys.modules:
                # Module already loaded — register the class directly from module
                mod = sys.modules[mod_name]
                cls = getattr(mod, cls_attr, None)
                if cls is not None:
                    TopologyRegistry._registry[topology_name] = cls
            else:
                with contextlib.suppress(ImportError):
                    importlib.import_module(mod_name)
