"""Smoke tests for the TaskRegistry with pre-seeded fixture cache.

Verifies:
1. All four loaders are registered after importing atm.tasks.
2. TASKS.sample("humaneval", n=10, seed=42) returns exactly 10 TaskSpec
   without any network calls (HumanEvalLoader.load is patched).
3. Determinism: two calls with the same seed return identical results.
"""

from __future__ import annotations

import pytest

from atm.core.types import TaskSpec
from atm.tasks import TASKS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_specs(n: int) -> list[TaskSpec]:
    """Create ``n`` fake TaskSpec instances with deterministic IDs."""
    return [
        TaskSpec(
            id=f"humaneval/{i:03d}",
            type="programming",
            input=f"def solve_{i}(x: int) -> int:\n    # TODO\n    pass",
            expected=f"    return {i}",
            evaluator_key="humaneval_pytest",
            metadata={
                "test": f"def check(candidate):\n    assert candidate({i}) == {i}",
                "entry_point": f"solve_{i}",
            },
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 1: All four loaders are registered
# ---------------------------------------------------------------------------


def test_all_loaders_registered() -> None:
    """All four loader modules are registered in TASKS after import."""
    for loader_name in ("humaneval", "mmlu", "creative", "analysis"):
        loader_cls = TASKS.get(loader_name)
        assert loader_cls is not None, f"Loader '{loader_name}' not found in TASKS"
        assert loader_cls.name == loader_name


# ---------------------------------------------------------------------------
# Test 2: TASKS.sample("humaneval") returns 10 TaskSpec — no network
# ---------------------------------------------------------------------------


def test_sample_humaneval_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample("humaneval", n=10, seed=42) returns exactly 10 TaskSpec.

    HumanEvalLoader.load is patched to return 20 fake TaskSpec, so no
    network or filesystem access is required.
    """
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    # Clear the internal cache so the monkeypatched load is actually called
    TASKS._cache.pop("humaneval", None)

    result = TASKS.sample("humaneval", n=10, seed=42)

    assert len(result) == 10
    for spec in result:
        assert isinstance(spec, TaskSpec)
        assert spec.evaluator_key == "humaneval_pytest"


# ---------------------------------------------------------------------------
# Test 3: Determinism — same seed → same sample
# ---------------------------------------------------------------------------


def test_sample_humaneval_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls with the same seed produce identical results."""
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    # Clear cache to force a fresh load (uses monkeypatched method)
    TASKS._cache.pop("humaneval", None)

    first = TASKS.sample("humaneval", n=10, seed=42)

    # Second call uses the populated cache — same seed → same result
    second = TASKS.sample("humaneval", n=10, seed=42)

    assert first == second, "Same seed must produce identical sample"


# ---------------------------------------------------------------------------
# Test 4: Different seeds → different samples (with enough items)
# ---------------------------------------------------------------------------


def test_sample_humaneval_different_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different seeds produce different samples (probabilistic with 20 items)."""
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    TASKS._cache.pop("humaneval", None)

    sample_a = TASKS.sample("humaneval", n=10, seed=1)
    sample_b = TASKS.sample("humaneval", n=10, seed=99)

    # With 20 items and n=10, different seeds almost certainly pick different subsets
    ids_a = {s.id for s in sample_a}
    ids_b = {s.id for s in sample_b}
    assert ids_a != ids_b, "Different seeds should produce different samples"


# ---------------------------------------------------------------------------
# Test 5: True sampler determinism — cache cleared between calls
# ---------------------------------------------------------------------------


def test_sample_humaneval_determinism_fresh_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same seed produces identical results even when cache is cleared between calls.

    This test verifies sampler determinism — not merely a cache hit.
    The loader is called twice (cache is evicted between the two sample calls),
    and the same seed must still produce the same ordered sample.
    """
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    TASKS._cache.pop("humaneval", None)
    first = TASKS.sample("humaneval", n=10, seed=42)

    # Clear the registry cache to force a second fresh load
    TASKS._cache.clear()
    second = TASKS.sample("humaneval", n=10, seed=42)

    assert first == second, "Same seed must produce identical sample even after cache clear"
