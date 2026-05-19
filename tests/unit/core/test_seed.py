"""Unit tests for atm.core.seed — seed_all() reproducibility helper."""

from __future__ import annotations

import os
import random

import pytest

from atm.core.seed import seed_all


class TestSeedAllPythonRandom:
    """Two calls with the same seed produce the same random.random() value."""

    def test_same_seed_same_value(self) -> None:
        seed_all(42)
        first = random.random()
        seed_all(42)
        second = random.random()
        assert first == second

    def test_different_seeds_different_values(self) -> None:
        seed_all(42)
        val_42 = random.random()
        seed_all(7)
        val_7 = random.random()
        assert val_42 != val_7


class TestSeedAllNumpy:
    """Two calls with the same seed produce the same np.random.rand() value."""

    def test_same_seed_same_value(self) -> None:
        np = pytest.importorskip("numpy")
        seed_all(42)
        first = np.random.rand()
        seed_all(42)
        second = np.random.rand()
        assert first == second

    def test_different_seeds_different_values(self) -> None:
        np = pytest.importorskip("numpy")
        seed_all(42)
        val_42 = np.random.rand()
        seed_all(7)
        val_7 = np.random.rand()
        assert val_42 != val_7


class TestSeedAllEnv:
    """os.environ["PYTHONHASHSEED"] is set to str(seed) after call."""

    def test_pythonhashseed_set(self) -> None:
        seed_all(42)
        assert os.environ.get("PYTHONHASHSEED") == "42"

    def test_pythonhashseed_updates_on_second_call(self) -> None:
        seed_all(42)
        assert os.environ.get("PYTHONHASHSEED") == "42"
        seed_all(99)
        assert os.environ.get("PYTHONHASHSEED") == "99"


class TestSeedAllTorchAbsent:
    """seed_all() must not raise ImportError even when torch is not installed."""

    def test_no_raise_when_torch_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        real_torch = sys.modules.pop("torch", None)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            seed_all(42)
        finally:
            if real_torch is not None:
                sys.modules["torch"] = real_torch
            else:
                sys.modules.pop("torch", None)

    def test_idempotent_double_call(self) -> None:
        """Calling seed_all twice with the same seed is safe (idempotent)."""
        seed_all(42)
        seed_all(42)


class TestSeedAllPublicApi:
    """seed_all is exported from atm.core package."""

    def test_importable_from_atm_core(self) -> None:
        from atm.core import seed_all as _seed_all

        assert callable(_seed_all)
