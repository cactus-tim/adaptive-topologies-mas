"""Conftest for topology unit tests.

Provides a fixture to ensure StarTopology (and future topology classes) are
always registered in TopologyRegistry before each test. This is needed because
test_base.py's TestTopologyRegistry.teardown_method() clears _registry, and
Python's module cache (sys.modules) prevents the side-effect re-import from
firing again automatically.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from atm.topology.base import TopologyRegistry


@pytest.fixture(autouse=True)
def ensure_star_registered() -> None:
    """Re-register star topology if it was cleared from the registry.

    Forces reload of atm.topology.star so the @TopologyRegistry.register("star")
    decorator fires again. Only reloads if star is not currently registered.
    """
    if "star" not in TopologyRegistry.list_names():
        # Force re-execution of star.py module-level code
        mod_name = "atm.topology.star"
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)
