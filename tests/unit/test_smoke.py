"""Smoke tests for the atm package scaffold (M0).

Intentionally minimal: verifies that the package is importable under the
build configuration from pyproject.toml and that the test runner works.
"""

import atm


def test_atm_package_importable() -> None:
    assert atm.__version__ == "0.1.0"


def test_python_arithmetic_sanity() -> None:
    assert 1 + 1 == 2
