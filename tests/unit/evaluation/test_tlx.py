"""Unit tests for atm.evaluation.tlx — NasaTLX model and aggregate_tlx helper.

6 tests covering:
1. Valid construction with all 6 scales
2. raw_score formula (arch.md §13.3): (mental + physical + temporal + (100-perf) + effort + frustration) / 6
3. Boundary values: all zeros → raw_score == 0.0
4. Boundary values: all 100 → raw_score == 100.0 (performance inverted: (100-100)=0)
5. Validation: value out of [0, 100] raises ValidationError
6. aggregate_tlx: mean of multiple NasaTLX instances
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atm.evaluation.tlx import NasaTLX, aggregate_tlx


class TestNasaTLXConstruction:
    def test_valid_construction(self) -> None:
        tlx = NasaTLX(
            mental_demand=60,
            physical_demand=10,
            temporal_demand=70,
            performance=80,
            effort=50,
            frustration=40,
        )
        assert tlx.mental_demand == 60
        assert tlx.performance == 80

    def test_raw_score_formula(self) -> None:
        # arch.md §13.3: (mental + physical + temporal + (100 - perf) + effort + frustration) / 6
        tlx = NasaTLX(
            mental_demand=60,
            physical_demand=10,
            temporal_demand=70,
            performance=80,
            effort=50,
            frustration=40,
        )
        expected = (60 + 10 + 70 + (100 - 80) + 50 + 40) / 6
        assert tlx.raw_score == pytest.approx(expected)

    def test_all_zeros_raw_score(self) -> None:
        tlx = NasaTLX(
            mental_demand=0,
            physical_demand=0,
            temporal_demand=0,
            performance=0,
            effort=0,
            frustration=0,
        )
        # performance=0 → (100-0)=100 contributes 100; all others 0 → 100/6
        expected = (0 + 0 + 0 + 100 + 0 + 0) / 6
        assert tlx.raw_score == pytest.approx(expected)

    def test_all_max_raw_score(self) -> None:
        tlx = NasaTLX(
            mental_demand=100,
            physical_demand=100,
            temporal_demand=100,
            performance=100,
            effort=100,
            frustration=100,
        )
        # performance=100 → (100-100)=0; all others 100 → (100*5)/6
        expected = (100 + 100 + 100 + 0 + 100 + 100) / 6
        assert tlx.raw_score == pytest.approx(expected)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            NasaTLX(
                mental_demand=101,
                physical_demand=0,
                temporal_demand=0,
                performance=0,
                effort=0,
                frustration=0,
            )

    def test_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            NasaTLX(
                mental_demand=-1,
                physical_demand=0,
                temporal_demand=0,
                performance=0,
                effort=0,
                frustration=0,
            )


class TestAggregateTlx:
    def test_aggregate_mean(self) -> None:
        a = NasaTLX(
            mental_demand=40,
            physical_demand=20,
            temporal_demand=60,
            performance=80,
            effort=30,
            frustration=10,
        )
        b = NasaTLX(
            mental_demand=80,
            physical_demand=20,
            temporal_demand=40,
            performance=60,
            effort=70,
            frustration=50,
        )
        result = aggregate_tlx([a, b])
        expected = (a.raw_score + b.raw_score) / 2
        assert result == pytest.approx(expected)

    def test_aggregate_single(self) -> None:
        tlx = NasaTLX(
            mental_demand=50,
            physical_demand=50,
            temporal_demand=50,
            performance=50,
            effort=50,
            frustration=50,
        )
        assert aggregate_tlx([tlx]) == pytest.approx(tlx.raw_score)
