"""Smoke tests for the TaskRegistry with pre-seeded fixture cache.

Verifies:
1. All four loaders/evaluators are registered after importing atm.tasks.
2. TASKS.sample("humaneval", n=10, seed=42) returns exactly 10 TaskSpec
   without any network calls (HumanEvalLoader.load is patched).
3. Determinism: two calls with the same seed return identical results.
4. Sampling tests for gsm8k, commongen, and dabench — each monkeypatches
   the respective loader.load and verifies correct type/evaluator_key.
"""

from __future__ import annotations

import pytest

from atm.core.types import TaskSpec
from atm.tasks import EVALUATORS, TASKS


def _make_fake_specs(n: int) -> list[TaskSpec]:
    """Create ``n`` fake humaneval TaskSpec instances with deterministic IDs."""
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


def _make_fake_gsm8k_specs(n: int = 20) -> list[TaskSpec]:
    """Create ``n`` fake GSM8K TaskSpec instances with deterministic IDs."""
    return [
        TaskSpec(
            id=f"gsm8k/{i:03d}",
            type="reasoning",
            input=f"Question {i}: If you have {i} apples and give away {i // 2}, how many remain?",
            expected=f"{i - i // 2}",
            evaluator_key="gsm8k_numeric",
            metadata={},
        )
        for i in range(n)
    ]


def _make_fake_commongen_specs(n: int = 20) -> list[TaskSpec]:
    """Create ``n`` fake CommonGen TaskSpec instances with deterministic IDs."""
    return [
        TaskSpec(
            id=f"commongen/{i:03d}",
            type="creative",
            input=f"concepts: word_{i}, action_{i}",
            expected=None,
            evaluator_key="commongen_rouge_coverage",
            metadata={
                "concepts": [f"word_{i}", f"action_{i}"],
                "references": [f"A sentence using word_{i} and action_{i}."],
            },
        )
        for i in range(n)
    ]


def _make_fake_dabench_specs(n: int = 20) -> list[TaskSpec]:
    """Create ``n`` fake DABench TaskSpec instances with deterministic IDs."""
    return [
        TaskSpec(
            id=f"dabench/{i:03d}",
            type="decision",
            input=f"Calculate the mean of column_{i} in the dataset.",
            expected=f"@mean_{i}[{float(i):.2f}]",
            evaluator_key="dabench_numeric_exact",
            metadata={
                "concepts": [f"mean_{i}"],
                "constraints": "Round to 2 decimals",
                "format": f"@mean_{i}[value]",
                "level": "easy",
                "file_name": f"dataset_{i}.csv",
            },
        )
        for i in range(n)
    ]


def test_all_loaders_registered() -> None:
    """All four loader/evaluator modules are registered in TASKS/EVALUATORS after import."""
    assert set(TASKS._registry.keys()) >= {
        "humaneval",
        "gsm8k",
        "commongen",
        "dabench",
    }, f"Expected loaders not found. Got: {sorted(TASKS._registry.keys())}"

    assert set(EVALUATORS._registry.keys()) >= {
        "humaneval_pytest",
        "gsm8k_numeric",
        "commongen_rouge_coverage",
        "dabench_numeric_exact",
    }, f"Expected evaluators not found. Got: {sorted(EVALUATORS._registry.keys())}"


def test_sample_humaneval_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample("humaneval", n=10, seed=42) returns exactly 10 TaskSpec.

    HumanEvalLoader.load is patched to return 20 fake TaskSpec, so no
    network or filesystem access is required.
    """
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    TASKS._cache.pop("humaneval", None)

    result = TASKS.sample("humaneval", n=10, seed=42)

    assert len(result) == 10
    for spec in result:
        assert isinstance(spec, TaskSpec)
        assert spec.evaluator_key == "humaneval_pytest"


def test_sample_humaneval_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls with the same seed produce identical results."""
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    TASKS._cache.pop("humaneval", None)

    first = TASKS.sample("humaneval", n=10, seed=42)

    second = TASKS.sample("humaneval", n=10, seed=42)

    assert first == second, "Same seed must produce identical sample"


def test_sample_humaneval_different_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different seeds produce different samples (probabilistic with 20 items)."""
    fake_specs = _make_fake_specs(20)

    from atm.tasks.humaneval import HumanEvalLoader

    monkeypatch.setattr(HumanEvalLoader, "load", lambda self, cache_dir=None: fake_specs)

    TASKS._cache.pop("humaneval", None)

    sample_a = TASKS.sample("humaneval", n=10, seed=1)
    sample_b = TASKS.sample("humaneval", n=10, seed=99)

    ids_a = {s.id for s in sample_a}
    ids_b = {s.id for s in sample_b}
    assert ids_a != ids_b, "Different seeds should produce different samples"


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

    TASKS._cache.clear()
    second = TASKS.sample("humaneval", n=10, seed=42)

    assert first == second, "Same seed must produce identical sample even after cache clear"


def test_gsm8k_sample_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample("gsm8k", n=10, seed=42) returns 10 reasoning specs; same seed is stable."""
    TASKS._cache.pop("gsm8k", None)
    fake_specs = _make_fake_gsm8k_specs(20)
    monkeypatch.setattr(
        "atm.tasks.gsm8k.GSM8KLoader.load",
        lambda self, cache_dir=None: fake_specs,
    )
    out = TASKS.sample("gsm8k", n=10, seed=42)
    assert len(out) == 10
    assert all(s.type == "reasoning" for s in out)
    assert all(s.evaluator_key == "gsm8k_numeric" for s in out)
    TASKS._cache.pop("gsm8k", None)
    out2 = TASKS.sample("gsm8k", n=10, seed=42)
    assert [s.id for s in out] == [s.id for s in out2]


def test_commongen_sample_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample("commongen", n=10, seed=42) returns 10 creative specs; same seed is stable."""
    TASKS._cache.pop("commongen", None)
    fake_specs = _make_fake_commongen_specs(20)
    monkeypatch.setattr(
        "atm.tasks.commongen.CommonGenLoader.load",
        lambda self, cache_dir=None: fake_specs,
    )
    out = TASKS.sample("commongen", n=10, seed=42)
    assert len(out) == 10
    assert all(s.type == "creative" for s in out)
    assert all(s.evaluator_key == "commongen_rouge_coverage" for s in out)
    TASKS._cache.pop("commongen", None)
    out2 = TASKS.sample("commongen", n=10, seed=42)
    assert [s.id for s in out] == [s.id for s in out2]


def test_dabench_sample_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample("dabench", n=10, seed=42) returns 10 decision specs; same seed is stable."""
    TASKS._cache.pop("dabench", None)
    fake_specs = _make_fake_dabench_specs(20)
    monkeypatch.setattr(
        "atm.tasks.dabench.DABenchLoader.load",
        lambda self, cache_dir=None: fake_specs,
    )
    out = TASKS.sample("dabench", n=10, seed=42)
    assert len(out) == 10
    assert all(s.type == "decision" for s in out)
    assert all(s.evaluator_key == "dabench_numeric_exact" for s in out)
    TASKS._cache.pop("dabench", None)
    out2 = TASKS.sample("dabench", n=10, seed=42)
    assert [s.id for s in out] == [s.id for s in out2]
