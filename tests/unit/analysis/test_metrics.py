"""Unit tests for atm.analysis.metrics — Step 6.1.

Covers all 7 metric functions with ≥2 tests each (≥14 total):
  - compute_guard_override_rate
  - compute_router_cost_share
  - compute_time_per_topology
  - compute_oracle_gap_loo
  - compute_oracle_gap_manual
  - compute_hurt_rate
  - compute_topology_switch_counts
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures — shared DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def transitions_basic() -> pd.DataFrame:
    """5 transitions for 2 runs: 2 guard overrides, 3 non-overrides."""
    return pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r1", "r2", "r2"],
            "from_topology": [None, "linear", "linear", None, "mesh"],
            "to_topology": ["linear", "mesh", "linear", "mesh", "linear"],
            "decided_by": [
                "initial",
                "guard_override",
                "rule",
                "initial",
                "guard_override",
            ],
            "guards_applied": [[], ["budget"], [], [], ["time"]],
            "router_cost_usd": [0.00, 0.01, 0.005, 0.00, 0.02],
            "at": pd.to_datetime(
                [
                    "2024-01-01 10:00:00+00:00",
                    "2024-01-01 10:01:00+00:00",
                    "2024-01-01 10:03:00+00:00",
                    "2024-01-02 10:00:00+00:00",
                    "2024-01-02 10:05:00+00:00",
                ]
            ),
        }
    )


@pytest.fixture
def runs_basic() -> pd.DataFrame:
    """4 runs: 2 tasks x 2 topologies."""
    return pd.DataFrame(
        {
            "id": ["run-1", "run-2", "run-3", "run-4"],
            "task_id": [
                "HumanEval/0",
                "HumanEval/0",
                "HumanEval/1",
                "HumanEval/1",
            ],
            "topology": ["linear", "mesh", "linear", "mesh"],
            "quality_score": [0.60, 0.80, 0.70, 0.90],
            "budget_spent_usd": [0.10, 0.12, 0.08, 0.15],
        }
    )


# ---------------------------------------------------------------------------
# compute_guard_override_rate
# ---------------------------------------------------------------------------


class TestComputeGuardOverrideRate:
    """Tests for compute_guard_override_rate."""

    def test_basic_rate(self, transitions_basic: pd.DataFrame) -> None:
        """2 guard_overrides out of 5 total → 0.40."""
        from atm.analysis.metrics import compute_guard_override_rate

        rate = compute_guard_override_rate(transitions_basic)
        assert math.isclose(rate, 2 / 5, rel_tol=1e-9)

    def test_empty_dataframe_returns_zero(self) -> None:
        """Empty transitions_df → 0.0."""
        from atm.analysis.metrics import compute_guard_override_rate

        empty = pd.DataFrame(columns=["decided_by"])
        assert compute_guard_override_rate(empty) == 0.0

    def test_no_overrides(self) -> None:
        """No guard_override rows → 0.0."""
        from atm.analysis.metrics import compute_guard_override_rate

        df = pd.DataFrame({"decided_by": ["initial", "rule", "llm_router", "oracle"]})
        assert compute_guard_override_rate(df) == 0.0

    def test_all_overrides(self) -> None:
        """All rows are guard_override → 1.0."""
        from atm.analysis.metrics import compute_guard_override_rate

        df = pd.DataFrame({"decided_by": ["guard_override"] * 3})
        assert compute_guard_override_rate(df) == 1.0


# ---------------------------------------------------------------------------
# compute_router_cost_share
# ---------------------------------------------------------------------------


class TestComputeRouterCostShare:
    """Tests for compute_router_cost_share."""

    def test_basic_share(self, transitions_basic: pd.DataFrame, runs_basic: pd.DataFrame) -> None:
        """router total = 0.035; run total = 0.45 → share ≈ 0.0778."""
        from atm.analysis.metrics import compute_router_cost_share

        share = compute_router_cost_share(transitions_basic, runs_basic)
        expected = 0.035 / 0.45
        assert math.isclose(share, expected, rel_tol=1e-6)

    def test_empty_transitions_returns_zero(self, runs_basic: pd.DataFrame) -> None:
        """Empty transitions → 0.0."""
        from atm.analysis.metrics import compute_router_cost_share

        empty = pd.DataFrame(columns=["router_cost_usd"])
        assert compute_router_cost_share(empty, runs_basic) == 0.0

    def test_empty_runs_returns_zero(self, transitions_basic: pd.DataFrame) -> None:
        """Empty runs_df → 0.0 (denominator is zero)."""
        from atm.analysis.metrics import compute_router_cost_share

        empty = pd.DataFrame(columns=["budget_spent_usd"])
        assert compute_router_cost_share(transitions_basic, empty) == 0.0

    def test_zero_run_cost_returns_zero(self, transitions_basic: pd.DataFrame) -> None:
        """All run costs zero → 0.0 (avoid division by zero)."""
        from atm.analysis.metrics import compute_router_cost_share

        runs_zero = pd.DataFrame({"budget_spent_usd": [0.0, 0.0]})
        assert compute_router_cost_share(transitions_basic, runs_zero) == 0.0


# ---------------------------------------------------------------------------
# compute_time_per_topology
# ---------------------------------------------------------------------------


class TestComputeTimePerTopology:
    """Tests for compute_time_per_topology."""

    def test_single_run_two_transitions(self) -> None:
        """3 events in one run: topology A holds 60s, topology B holds 120s.

        t0=0s  to=A  (initial)
        t1=60s to=B  (switch: A was active for 60s)
        t2=180s to=C  (switch: B was active for 120s; C is open-ended, not counted)
        """
        from atm.analysis.metrics import compute_time_per_topology

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r1", "r1"],
                "to_topology": ["linear", "mesh", "debate"],
                "at": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00+00:00",
                        "2024-01-01 10:01:00+00:00",
                        "2024-01-01 10:03:00+00:00",
                    ]
                ),
            }
        )
        result = compute_time_per_topology(df)
        total = sum(result.values())
        assert math.isclose(total, 1.0, rel_tol=1e-9), f"fractions must sum to 1.0, got {total}"
        # linear: 60s / 180s = 1/3; mesh: 120s / 180s = 2/3
        assert math.isclose(result["linear"], 60 / 180, rel_tol=1e-6)
        assert math.isclose(result["mesh"], 120 / 180, rel_tol=1e-6)
        # debate is the last row — no successor → not counted
        assert "debate" not in result

    def test_empty_returns_empty_dict(self) -> None:
        """Empty transitions_df → {}."""
        from atm.analysis.metrics import compute_time_per_topology

        empty = pd.DataFrame(columns=["run_id", "to_topology", "at"])
        assert compute_time_per_topology(empty) == {}

    def test_single_event_per_run_returns_empty(self) -> None:
        """Only one event per run → no delta computable → empty dict."""
        from atm.analysis.metrics import compute_time_per_topology

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r2"],
                "to_topology": ["linear", "mesh"],
                "at": pd.to_datetime(["2024-01-01 10:00:00+00:00", "2024-01-01 11:00:00+00:00"]),
            }
        )
        assert compute_time_per_topology(df) == {}

    def test_two_runs_combined(self) -> None:
        """Two runs each contributing different topology times are merged."""
        from atm.analysis.metrics import compute_time_per_topology

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r1", "r2", "r2"],
                "to_topology": ["linear", "mesh", "mesh", "linear"],
                "at": pd.to_datetime(
                    [
                        "2024-01-01 10:00:00+00:00",
                        "2024-01-01 10:01:00+00:00",  # r1: linear 60s
                        "2024-01-02 10:00:00+00:00",
                        "2024-01-02 10:02:00+00:00",  # r2: mesh 120s
                    ]
                ),
            }
        )
        result = compute_time_per_topology(df)
        total = sum(result.values())
        assert math.isclose(total, 1.0, rel_tol=1e-9)
        # linear: 60s; mesh: 120s → linear share = 60/180 = 1/3
        assert math.isclose(result["linear"], 60 / 180, rel_tol=1e-6)
        assert math.isclose(result["mesh"], 120 / 180, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# compute_oracle_gap_loo
# ---------------------------------------------------------------------------


class TestComputeOracleGapLoo:
    """Tests for compute_oracle_gap_loo — per-task_id strict filter (reviewer N2)."""

    def _make_oracle(self, by_task_id: dict) -> object:
        from atm.analysis.oracle import OracleTable

        return OracleTable(by_task_id=by_task_id, by_task_type={})

    def test_basic_gap(self, runs_basic: pd.DataFrame) -> None:
        """task HumanEval/0: oracle=mesh (quality=0.80), router_mean=(0.60+0.80)/2=0.70 → gap=0.10."""
        from atm.analysis.metrics import compute_oracle_gap_loo

        oracle = self._make_oracle({"HumanEval/0": "mesh", "HumanEval/1": "mesh"})
        gaps = compute_oracle_gap_loo(runs_basic, oracle)

        assert "HumanEval/0" in gaps.index
        assert math.isclose(gaps["HumanEval/0"], 0.80 - 0.70, rel_tol=1e-9)

    def test_oracle_topology_not_in_runs_gives_nan(self, runs_basic: pd.DataFrame) -> None:
        """Oracle says 'debate' but no debate runs → NaN."""
        from atm.analysis.metrics import compute_oracle_gap_loo

        oracle = self._make_oracle({"HumanEval/0": "debate"})
        gaps = compute_oracle_gap_loo(runs_basic, oracle)

        assert math.isnan(gaps["HumanEval/0"])

    def test_empty_runs_returns_empty_series(self) -> None:
        """Empty runs_df → empty Series."""
        from atm.analysis.metrics import compute_oracle_gap_loo

        oracle = self._make_oracle({"HumanEval/0": "mesh"})
        result = compute_oracle_gap_loo(pd.DataFrame(), oracle)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_divergent_fixture_per_task_id_filter(self) -> None:
        """Divergent fixture: LOO oracle differs per task_id.

        task A (HumanEval/0) oracle → 'mesh'    quality_oracle=0.90
        task B (HumanEval/1) oracle → 'linear'  quality_oracle=0.70
        Router quality per task = mean of all runs for that task.
        """
        from atm.analysis.metrics import compute_oracle_gap_loo

        runs = pd.DataFrame(
            {
                "task_id": [
                    "HumanEval/0",
                    "HumanEval/0",
                    "HumanEval/1",
                    "HumanEval/1",
                ],
                "topology": ["mesh", "linear", "linear", "mesh"],
                "quality_score": [0.90, 0.50, 0.70, 0.40],
            }
        )
        oracle = self._make_oracle(
            {
                "HumanEval/0": "mesh",  # oracle_quality = 0.90
                "HumanEval/1": "linear",  # oracle_quality = 0.70
            }
        )
        gaps = compute_oracle_gap_loo(runs, oracle)

        # HumanEval/0: oracle=0.90, router_mean=(0.90+0.50)/2=0.70 → gap=0.20
        assert math.isclose(gaps["HumanEval/0"], 0.20, rel_tol=1e-9)
        # HumanEval/1: oracle=0.70, router_mean=(0.70+0.40)/2=0.55 → gap=0.15
        assert math.isclose(gaps["HumanEval/1"], 0.15, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_oracle_gap_manual
# ---------------------------------------------------------------------------


class TestComputeOracleGapManual:
    """Tests for compute_oracle_gap_manual — symmetric to LOO variant."""

    def _make_oracle(self, by_task_id: dict, by_task_type: dict | None = None) -> object:
        from atm.analysis.oracle import OracleTable

        return OracleTable(
            by_task_id=by_task_id,
            by_task_type=by_task_type or {},
        )

    def test_basic_gap_from_task_id(self, runs_basic: pd.DataFrame) -> None:
        """Manual oracle per task_id works identically to LOO variant."""
        from atm.analysis.metrics import compute_oracle_gap_manual

        oracle = self._make_oracle({"HumanEval/0": "mesh", "HumanEval/1": "mesh"})
        gaps = compute_oracle_gap_manual(runs_basic, oracle)

        assert "HumanEval/0" in gaps.index
        # oracle quality = 0.80, router_mean = 0.70 → gap = 0.10
        assert math.isclose(gaps["HumanEval/0"], 0.10, rel_tol=1e-9)

    def test_fallback_to_task_type(self) -> None:
        """When task_id not in by_task_id, falls back to by_task_type."""
        from atm.analysis.metrics import compute_oracle_gap_manual

        runs = pd.DataFrame(
            {
                "task_id": ["HumanEval/0", "HumanEval/0"],
                "topology": ["mesh", "linear"],
                "quality_score": [0.80, 0.60],
            }
        )
        # by_task_id is empty, but by_task_type has programming → mesh
        oracle = self._make_oracle(by_task_id={}, by_task_type={"programming": "mesh"})
        gaps = compute_oracle_gap_manual(runs, oracle)

        # oracle_quality = 0.80 (mesh), router_mean = 0.70 → gap = 0.10
        assert math.isclose(gaps["HumanEval/0"], 0.10, rel_tol=1e-9)

    def test_missing_oracle_gives_nan(self, runs_basic: pd.DataFrame) -> None:
        """Task absent from both by_task_id and by_task_type → NaN."""
        from atm.analysis.metrics import compute_oracle_gap_manual

        oracle = self._make_oracle(by_task_id={}, by_task_type={})
        gaps = compute_oracle_gap_manual(runs_basic, oracle)

        # No oracle for any task → all NaN
        assert all(math.isnan(v) for v in gaps.values)

    def test_empty_runs_returns_empty_series(self) -> None:
        """Empty runs_df → empty Series."""
        from atm.analysis.metrics import compute_oracle_gap_manual

        oracle = self._make_oracle(by_task_id={}, by_task_type={})
        result = compute_oracle_gap_manual(pd.DataFrame(), oracle)
        assert isinstance(result, pd.Series)
        assert result.empty


# ---------------------------------------------------------------------------
# compute_hurt_rate
# ---------------------------------------------------------------------------


class TestComputeHurtRate:
    """Tests for compute_hurt_rate — safety indicator."""

    def test_all_hurt(self) -> None:
        """Adaptive worse on all tasks → 1.0."""
        from atm.analysis.metrics import compute_hurt_rate

        adaptive = pd.DataFrame(
            {
                "task_id": ["t1", "t1", "t2", "t2"],
                "quality_score": [0.5, 0.5, 0.4, 0.4],
            }
        )
        best_static = pd.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "quality_score": [0.8, 0.9],
            }
        )
        assert compute_hurt_rate(adaptive, best_static) == 1.0

    def test_none_hurt(self) -> None:
        """Adaptive better on all tasks → 0.0."""
        from atm.analysis.metrics import compute_hurt_rate

        adaptive = pd.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "quality_score": [0.9, 0.95],
            }
        )
        best_static = pd.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "quality_score": [0.5, 0.5],
            }
        )
        assert compute_hurt_rate(adaptive, best_static) == 0.0

    def test_half_hurt(self) -> None:
        """Adaptive worse on 1 of 2 tasks → 0.5."""
        from atm.analysis.metrics import compute_hurt_rate

        adaptive = pd.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "quality_score": [0.9, 0.3],
            }
        )
        best_static = pd.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "quality_score": [0.5, 0.8],
            }
        )
        assert math.isclose(compute_hurt_rate(adaptive, best_static), 0.5, rel_tol=1e-9)

    def test_empty_adaptive_returns_zero(self) -> None:
        """Empty runs_df → 0.0."""
        from atm.analysis.metrics import compute_hurt_rate

        best_static = pd.DataFrame({"task_id": ["t1"], "quality_score": [0.8]})
        assert compute_hurt_rate(pd.DataFrame(), best_static) == 0.0

    def test_empty_best_static_returns_zero(self) -> None:
        """Empty best_static_df → 0.0."""
        from atm.analysis.metrics import compute_hurt_rate

        adaptive = pd.DataFrame({"task_id": ["t1"], "quality_score": [0.8]})
        assert compute_hurt_rate(adaptive, pd.DataFrame()) == 0.0

    def test_task_not_in_best_static_is_skipped(self) -> None:
        """Tasks absent from best_static_df are not counted as hurt."""
        from atm.analysis.metrics import compute_hurt_rate

        adaptive = pd.DataFrame(
            {
                "task_id": ["t1", "t2", "t3"],
                "quality_score": [0.3, 0.9, 0.5],
            }
        )
        # Only t1 and t2 have a best_static entry; t3 should be skipped
        best_static = pd.DataFrame(
            {
                "task_id": ["t1", "t2"],
                "quality_score": [0.8, 0.5],
            }
        )
        # t1 is hurt (0.3 < 0.8), t2 is not (0.9 > 0.5), t3 skipped
        assert math.isclose(compute_hurt_rate(adaptive, best_static), 0.5, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compute_topology_switch_counts
# ---------------------------------------------------------------------------


class TestComputeTopologySwitchCounts:
    """Tests for compute_topology_switch_counts."""

    def test_basic_counts(self, transitions_basic: pd.DataFrame) -> None:
        """r1: 1 switch (linear→mesh); r2: 1 switch (mesh→linear)."""
        from atm.analysis.metrics import compute_topology_switch_counts

        counts = compute_topology_switch_counts(transitions_basic)
        assert counts["r1"] == 1
        assert counts["r2"] == 1

    def test_no_switches(self) -> None:
        """All transitions are initial or no-change → 0 switches."""
        from atm.analysis.metrics import compute_topology_switch_counts

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r1"],
                "from_topology": [None, "linear"],
                "to_topology": ["linear", "linear"],
                "decided_by": ["initial", "rule"],
            }
        )
        counts = compute_topology_switch_counts(df)
        assert counts["r1"] == 0

    def test_empty_returns_empty_series(self) -> None:
        """Empty transitions_df → empty Series."""
        from atm.analysis.metrics import compute_topology_switch_counts

        empty = pd.DataFrame(columns=["run_id", "from_topology", "to_topology"])
        result = compute_topology_switch_counts(empty)
        assert isinstance(result, pd.Series)
        assert result.empty

    def test_multiple_switches_same_run(self) -> None:
        """A run with 3 actual topology switches."""
        from atm.analysis.metrics import compute_topology_switch_counts

        df = pd.DataFrame(
            {
                "run_id": ["r1"] * 5,
                "from_topology": [None, "linear", "mesh", "debate", "linear"],
                "to_topology": ["linear", "mesh", "debate", "linear", "linear"],
                "decided_by": ["initial", "rule", "rule", "rule", "rule"],
            }
        )
        counts = compute_topology_switch_counts(df)
        # linear→mesh, mesh→debate, debate→linear = 3 switches
        # last row: linear→linear = no switch
        assert counts["r1"] == 3

    def test_initial_row_not_counted(self) -> None:
        """Initial row (from_topology=None) is never a switch."""
        from atm.analysis.metrics import compute_topology_switch_counts

        df = pd.DataFrame(
            {
                "run_id": ["r1"],
                "from_topology": [None],
                "to_topology": ["mesh"],
                "decided_by": ["initial"],
            }
        )
        counts = compute_topology_switch_counts(df)
        assert counts["r1"] == 0

    def test_all_run_ids_present_even_zero_switch(self) -> None:
        """Runs with 0 switches still appear in the result Series."""
        from atm.analysis.metrics import compute_topology_switch_counts

        df = pd.DataFrame(
            {
                "run_id": ["r1", "r1", "r2", "r2"],
                "from_topology": [None, "linear", None, "mesh"],
                "to_topology": ["linear", "linear", "mesh", "mesh"],
                "decided_by": ["initial", "rule", "initial", "rule"],
            }
        )
        counts = compute_topology_switch_counts(df)
        # r1: no switch (linear→linear); r2: no switch (mesh→mesh)
        assert "r1" in counts.index
        assert "r2" in counts.index
        assert counts["r1"] == 0
        assert counts["r2"] == 0
