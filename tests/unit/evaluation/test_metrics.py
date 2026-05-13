"""Unit tests for atm.evaluation.metrics — pure metric functions.

12 tests covering:
1.  humaneval_pass_at_k: reference value n=20,c=2,k=10
2.  humaneval_pass_at_k: edge case n-c < k → 1.0
3.  humaneval_pass_at_k: c=0 → 0.0
4.  humaneval_pass_at_k: c==n → 1.0
5.  cost_per_quality: happy path
6.  cost_per_quality: quality=0 uses eps (no ZeroDivisionError)
7.  time_per_quality: happy path
8.  time_per_quality: quality=0 uses eps
9.  aggregate_quality: non-empty list
10. aggregate_quality: empty list → 0.0
11. aggregate_human_load: single NasaTLX delegates to aggregate_tlx
12. aggregate_human_load: empty list → 0.0
"""

from __future__ import annotations

import pytest

from atm.evaluation.metrics import (
    aggregate_human_load,
    aggregate_quality,
    cost_per_quality,
    humaneval_pass_at_k,
    time_per_quality,
)
from atm.evaluation.tlx import NasaTLX


class TestHumanevalPassAtK:
    def test_reference_value_n20_c2_k10(self) -> None:
        """Chen et al. 2021 unbiased estimator: n=20,c=2,k=10 → ~0.7632."""
        result = humaneval_pass_at_k(n=20, c=2, k=10)
        # Correct reference from Chen et al. 2021: 1 - C(18,10)/C(20,10) ≈ 0.7632
        assert result == pytest.approx(0.7632, abs=1e-4)

    def test_n_minus_c_less_than_k_returns_one(self) -> None:
        """When n-c < k, guaranteed to find at least one correct → 1.0."""
        # n=5, c=4, k=3: n-c=1 < k=3 → 1.0
        assert humaneval_pass_at_k(n=5, c=4, k=3) == pytest.approx(1.0)

    def test_no_correct_samples_returns_zero(self) -> None:
        """c=0 means no correct samples; pass@k = 0.0."""
        assert humaneval_pass_at_k(n=10, c=0, k=5) == pytest.approx(0.0)

    def test_all_correct_returns_one(self) -> None:
        """c==n means every sample is correct; pass@k = 1.0."""
        assert humaneval_pass_at_k(n=10, c=10, k=5) == pytest.approx(1.0)


class TestCostPerQuality:
    def test_happy_path(self) -> None:
        result = cost_per_quality(cost=3.0, quality=0.75)
        assert result == pytest.approx(4.0)

    def test_quality_zero_uses_eps(self) -> None:
        """quality=0 should not raise ZeroDivisionError; uses eps=1e-6."""
        result = cost_per_quality(cost=1.0, quality=0.0)
        assert result == pytest.approx(1.0 / 1e-6)

    def test_cost_zero(self) -> None:
        result = cost_per_quality(cost=0.0, quality=0.5)
        assert result == pytest.approx(0.0)


class TestTimePerQuality:
    def test_happy_path(self) -> None:
        result = time_per_quality(time_s=30.0, quality=0.5)
        assert result == pytest.approx(60.0)

    def test_quality_zero_uses_eps(self) -> None:
        """quality=0 should not raise ZeroDivisionError; uses eps=1e-6."""
        result = time_per_quality(time_s=5.0, quality=0.0)
        assert result == pytest.approx(5.0 / 1e-6)


class TestAggregateQuality:
    def test_non_empty_list_mean(self) -> None:
        scores = [0.2, 0.8, 0.6]
        result = aggregate_quality(scores)
        assert result == pytest.approx(sum(scores) / len(scores))

    def test_empty_list_returns_zero(self) -> None:
        assert aggregate_quality([]) == pytest.approx(0.0)

    def test_single_element(self) -> None:
        assert aggregate_quality([0.42]) == pytest.approx(0.42)


class TestAggregateHumanLoad:
    def _make_tlx(
        self,
        mental: int = 50,
        physical: int = 50,
        temporal: int = 50,
        performance: int = 50,
        effort: int = 50,
        frustration: int = 50,
    ) -> NasaTLX:
        return NasaTLX(
            mental_demand=mental,
            physical_demand=physical,
            temporal_demand=temporal,
            performance=performance,
            effort=effort,
            frustration=frustration,
        )

    def test_delegates_to_aggregate_tlx(self) -> None:
        """aggregate_human_load([tlx]) should equal tlx.raw_score."""
        tlx = self._make_tlx(mental=80, performance=20)
        result = aggregate_human_load([tlx])
        assert result == pytest.approx(tlx.raw_score)

    def test_empty_list_returns_zero(self) -> None:
        """Empty list → 0.0 (no division)."""
        assert aggregate_human_load([]) == pytest.approx(0.0)
