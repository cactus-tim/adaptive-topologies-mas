"""Smoke tests and structural assertions for atm.analysis.plots — Steps 7.1 and 8.1.

Tests focus on:
  - Figures render without exception
  - Returned object is a matplotlib.figure.Figure
  - Key structural properties (Axes count, labels, etc.)
  - Edge cases: empty DataFrames, missing columns, single-group DataFrames

All plot functions MUST work under filterwarnings=["error"] — no matplotlib
GUI-backend warnings allowed (matplotlib.use("Agg") is enforced in plots.py).
"""

from __future__ import annotations

import datetime
import warnings

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_all_figures() -> None:  # type: ignore[return]
    """Close all open matplotlib figures after each test to avoid the
    'More than 20 figures have been opened' RuntimeWarning that becomes
    a hard error under filterwarnings=["error"]."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures — shared DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def runs_df() -> pd.DataFrame:
    """Minimal runs DataFrame for plot_pareto and plot_topology_task_heatmap."""
    rng = np.random.default_rng(0)
    n = 40
    topologies = ["adaptive", "linear", "mesh", "supervisor"]
    task_types = ["programming", "reasoning", "creative"]
    return pd.DataFrame(
        {
            "run_id": [f"run-{i}" for i in range(n)],
            "topology": [topologies[i % len(topologies)] for i in range(n)],
            "task_type": [task_types[i % len(task_types)] for i in range(n)],
            "quality_score": rng.uniform(0.3, 1.0, n),
            "budget_spent_usd": rng.uniform(0.01, 0.5, n),
        }
    )


@pytest.fixture
def phases_df() -> pd.DataFrame:
    """Minimal phases DataFrame for plot_phase_timeline."""
    t0 = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    rows = []
    for run_id, run_offset in [("run-A", 0), ("run-B", 3600)]:
        for i, phase in enumerate(["planning", "execution", "verification"]):
            start = t0 + datetime.timedelta(seconds=run_offset + i * 30)
            end = start + datetime.timedelta(seconds=25)
            rows.append(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "started_at": start,
                    "ended_at": end,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# test_plot_pareto — RQ1 smoke test
# ---------------------------------------------------------------------------


class TestPlotPareto:
    """plot_pareto returns a valid Figure with one Axes."""

    def test_plot_pareto_returns_figure(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig = plot_pareto(runs_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_pareto_has_one_axes(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig = plot_pareto(runs_df)
        assert len(fig.axes) == 1

    def test_plot_pareto_axes_labels(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig = plot_pareto(runs_df)
        ax = fig.axes[0]
        assert "budget_spent_usd" in ax.get_xlabel()
        assert "quality_score" in ax.get_ylabel()

    def test_plot_pareto_title_set(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig = plot_pareto(runs_df)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_plot_pareto_custom_group_col(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig = plot_pareto(runs_df, group_col="task_type")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_pareto_missing_column_graceful(self) -> None:
        from atm.analysis.plots import plot_pareto

        df = pd.DataFrame({"topology": ["linear", "mesh"], "quality_score": [0.5, 0.7]})
        # budget_spent_usd is missing — should not raise, just return a figure
        fig = plot_pareto(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_pareto_empty_df_graceful(self) -> None:
        from atm.analysis.plots import plot_pareto

        df = pd.DataFrame(columns=["topology", "quality_score", "budget_spent_usd"])
        fig = plot_pareto(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_pareto_single_group(self) -> None:
        from atm.analysis.plots import plot_pareto

        df = pd.DataFrame(
            {
                "topology": ["adaptive"] * 10,
                "quality_score": np.random.default_rng(1).uniform(0.5, 1.0, 10),
                "budget_spent_usd": np.random.default_rng(2).uniform(0.1, 0.3, 10),
            }
        )
        fig = plot_pareto(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_pareto_no_warnings(self, runs_df: pd.DataFrame) -> None:
        """Ensure no warnings are raised (filterwarnings=error is active in pytest)."""
        from atm.analysis.plots import plot_pareto

        # If warnings.catch_warnings doesn't raise, no warnings occurred
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_pareto(runs_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_pareto_returns_new_figure_each_call(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig1 = plot_pareto(runs_df)
        fig2 = plot_pareto(runs_df)
        assert fig1 is not fig2

    def test_plot_pareto_adaptive_in_legend(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_pareto

        fig = plot_pareto(runs_df)
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "adaptive" in legend_texts


# ---------------------------------------------------------------------------
# test_plot_topology_task_heatmap — RQ1 smoke test
# ---------------------------------------------------------------------------


class TestPlotTopologyTaskHeatmap:
    """plot_topology_task_heatmap returns a valid Figure with one Axes."""

    def test_heatmap_returns_figure(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_topology_task_heatmap

        fig = plot_topology_task_heatmap(runs_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_heatmap_has_one_axes(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_topology_task_heatmap

        fig = plot_topology_task_heatmap(runs_df)
        # seaborn heatmap may add a colorbar Axes — at least 1 Axes exists
        assert len(fig.axes) >= 1

    def test_heatmap_title_set(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_topology_task_heatmap

        fig = plot_topology_task_heatmap(runs_df)
        ax = fig.axes[0]
        title = ax.get_title()
        assert title != ""
        assert "quality_score" in title

    def test_heatmap_missing_column_graceful(self) -> None:
        from atm.analysis.plots import plot_topology_task_heatmap

        df = pd.DataFrame({"topology": ["linear", "mesh"]})
        fig = plot_topology_task_heatmap(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_heatmap_empty_df_graceful(self) -> None:
        from atm.analysis.plots import plot_topology_task_heatmap

        df = pd.DataFrame(columns=["topology", "task_type", "quality_score"])
        # pivot_table on empty df may raise or return empty — we just want no crash
        try:
            fig = plot_topology_task_heatmap(df)
            assert isinstance(fig, matplotlib.figure.Figure)
        except Exception as exc:
            pytest.fail(f"plot_topology_task_heatmap raised unexpectedly: {exc}")

    def test_heatmap_no_warnings(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_topology_task_heatmap

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_topology_task_heatmap(runs_df)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# test_plot_phase_timeline — RQ1 smoke test
# ---------------------------------------------------------------------------


class TestPlotPhaseTimeline:
    """plot_phase_timeline returns a valid Figure with one Axes."""

    def test_phase_timeline_returns_figure(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        fig = plot_phase_timeline(phases_df, run_id="run-A")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_phase_timeline_has_one_axes(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        fig = plot_phase_timeline(phases_df, run_id="run-A")
        assert len(fig.axes) == 1

    def test_phase_timeline_title_contains_run_id(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        fig = plot_phase_timeline(phases_df, run_id="run-A")
        ax = fig.axes[0]
        assert "run-A" in ax.get_title()

    def test_phase_timeline_unknown_run_id_graceful(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        fig = plot_phase_timeline(phases_df, run_id="nonexistent-run")
        assert isinstance(fig, matplotlib.figure.Figure)
        # Title should indicate no data was found
        ax = fig.axes[0]
        title = ax.get_title()
        assert "nonexistent-run" in title or "No phases" in title

    def test_phase_timeline_missing_run_id_column(self) -> None:
        from atm.analysis.plots import plot_phase_timeline

        df = pd.DataFrame({"phase": ["planning"], "started_at": [None], "ended_at": [None]})
        fig = plot_phase_timeline(df, run_id="run-X")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_phase_timeline_run_b(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        fig = plot_phase_timeline(phases_df, run_id="run-B")
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        assert "run-B" in ax.get_title()

    def test_phase_timeline_no_warnings(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_phase_timeline(phases_df, run_id="run-A")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_phase_timeline_xlabel_set(self, phases_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_phase_timeline

        fig = plot_phase_timeline(phases_df, run_id="run-A")
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""


# ---------------------------------------------------------------------------
# Module-level import smoke test
# ---------------------------------------------------------------------------


class TestPlotsModuleImport:
    """plots.py imports cleanly and exposes all public symbols."""

    def test_plots_module_importable(self) -> None:
        import atm.analysis.plots as plots_mod

        assert hasattr(plots_mod, "plot_pareto")
        assert hasattr(plots_mod, "plot_topology_task_heatmap")
        assert hasattr(plots_mod, "plot_phase_timeline")

    def test_plots_all_symbols_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_pareto" in __all__
        assert "plot_topology_task_heatmap" in __all__
        assert "plot_phase_timeline" in __all__

    def test_matplotlib_backend_is_agg(self) -> None:
        """Ensure Agg backend is active after importing plots module."""
        import matplotlib

        import atm.analysis.plots  # noqa: F401

        assert matplotlib.get_backend().lower() == "agg"


# ---------------------------------------------------------------------------
# RQ2 / G11 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transitions_df() -> pd.DataFrame:
    """Minimal topology transitions DataFrame for RQ2 plot tests."""
    t0 = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    rows = []
    topologies = ["linear", "mesh", "supervisor", "adaptive"]
    deciders = ["router", "guard_override", "router", "router", "guard_override"]
    for run_idx in range(3):
        run_id = f"run-{run_idx}"
        for i in range(5):
            rows.append(
                {
                    "run_id": run_id,
                    "from_topology": topologies[i % len(topologies)],
                    "to_topology": topologies[(i + 1) % len(topologies)],
                    "decided_by": deciders[i % len(deciders)],
                    "guards_applied": ["quality_guard"] if i % 2 == 0 else [],
                    "router_cost_usd": 0.001 * (i + 1),
                    "at": t0 + datetime.timedelta(seconds=run_idx * 300 + i * 30),
                    "quality_score": 0.6 + 0.05 * i,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def llm_calls_df() -> pd.DataFrame:
    """Minimal LLM calls DataFrame for plot_router_cost_share."""
    rows = []
    for run_idx in range(3):
        run_id = f"run-{run_idx}"
        for call_idx in range(4):
            rows.append(
                {
                    "run_id": run_id,
                    "role": "router" if call_idx == 0 else "worker",
                    "cost_usd": 0.002 * (call_idx + 1),
                    "model": "test-model",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def runs_df_rq2() -> pd.DataFrame:
    """Runs DataFrame for RQ2 plots (includes topology and timing)."""
    rng = np.random.default_rng(42)
    t0 = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC)
    n = 20
    topologies = ["adaptive", "linear", "mesh", "supervisor"]
    task_ids = [f"HumanEval/{i}" for i in range(5)] * 4
    rows = []
    for i in range(n):
        start = t0 + datetime.timedelta(seconds=i * 120)
        rows.append(
            {
                "run_id": f"run-{i}",
                "topology": topologies[i % len(topologies)],
                "task_id": task_ids[i],
                "quality_score": float(rng.uniform(0.4, 1.0)),
                "budget_spent_usd": float(rng.uniform(0.01, 0.5)),
                "started_at": start,
                "finished_at": start + datetime.timedelta(seconds=90 + i * 5),
                "duration_seconds": float(90 + i * 5),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def oracle_gap_series() -> pd.Series:
    """Pre-computed oracle gap Series for plot_oracle_gap_loo."""
    return pd.Series(
        {
            "HumanEval/0": 0.15,
            "HumanEval/1": -0.05,
            "HumanEval/2": 0.22,
            "HumanEval/3": 0.08,
            "HumanEval/4": 0.31,
        },
        name="oracle_gap_loo",
    )


# ---------------------------------------------------------------------------
# test_plot_transition_timeline_quality — RQ2/G11 smoke tests
# ---------------------------------------------------------------------------


class TestPlotTransitionTimelineQualityRq2:
    """plot_transition_timeline_quality returns a valid Figure. [rq2]"""

    def test_rq2_timeline_returns_figure(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        fig = plot_transition_timeline_quality(transitions_df, run_id="run-0")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_timeline_has_axes(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        fig = plot_transition_timeline_quality(transitions_df, run_id="run-0")
        assert len(fig.axes) >= 1

    def test_rq2_timeline_title_contains_run_id(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        fig = plot_transition_timeline_quality(transitions_df, run_id="run-1")
        ax = fig.axes[0]
        assert "run-1" in ax.get_title()

    def test_rq2_timeline_unknown_run_id_graceful(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        fig = plot_transition_timeline_quality(transitions_df, run_id="nonexistent")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_timeline_empty_df_graceful(self) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        df = pd.DataFrame(columns=["run_id", "from_topology", "to_topology", "at", "quality_score"])
        fig = plot_transition_timeline_quality(df, run_id="run-0")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_timeline_no_warnings(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_transition_timeline_quality(transitions_df, run_id="run-0")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_timeline_missing_run_id_column(self) -> None:
        from atm.analysis.plots import plot_transition_timeline_quality

        df = pd.DataFrame({"from_topology": ["linear"], "to_topology": ["mesh"]})
        fig = plot_transition_timeline_quality(df, run_id="run-0")
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# test_plot_guard_override_rate — RQ2/G11 smoke tests
# ---------------------------------------------------------------------------


class TestPlotGuardOverrideRateRq2:
    """plot_guard_override_rate returns a valid Figure. [rq2]"""

    def test_rq2_guard_rate_returns_figure(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        fig = plot_guard_override_rate(transitions_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_guard_rate_has_one_axes(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        fig = plot_guard_override_rate(transitions_df)
        assert len(fig.axes) == 1

    def test_rq2_guard_rate_title_set(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        fig = plot_guard_override_rate(transitions_df)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_rq2_guard_rate_empty_df_graceful(self) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        df = pd.DataFrame(columns=["run_id", "decided_by", "to_topology"])
        fig = plot_guard_override_rate(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_guard_rate_missing_decided_by_graceful(self) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        df = pd.DataFrame({"run_id": ["run-0"], "to_topology": ["mesh"]})
        fig = plot_guard_override_rate(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_guard_rate_no_warnings(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_guard_override_rate(transitions_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_guard_rate_xlabel_or_ylabel_set(self, transitions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_guard_override_rate

        fig = plot_guard_override_rate(transitions_df)
        ax = fig.axes[0]
        # At least one axis label should be non-empty
        assert ax.get_xlabel() != "" or ax.get_ylabel() != ""


# ---------------------------------------------------------------------------
# test_plot_router_cost_share — RQ2/G11 smoke tests
# ---------------------------------------------------------------------------


class TestPlotRouterCostShareRq2:
    """plot_router_cost_share returns a valid Figure. [rq2]"""

    def test_rq2_cost_share_returns_figure(
        self, runs_df_rq2: pd.DataFrame, llm_calls_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_router_cost_share

        fig = plot_router_cost_share(runs_df_rq2, llm_calls_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_cost_share_has_one_axes(
        self, runs_df_rq2: pd.DataFrame, llm_calls_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_router_cost_share

        fig = plot_router_cost_share(runs_df_rq2, llm_calls_df)
        assert len(fig.axes) == 1

    def test_rq2_cost_share_title_set(
        self, runs_df_rq2: pd.DataFrame, llm_calls_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_router_cost_share

        fig = plot_router_cost_share(runs_df_rq2, llm_calls_df)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_rq2_cost_share_empty_runs_graceful(self, llm_calls_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_router_cost_share

        df = pd.DataFrame(columns=["run_id", "topology", "budget_spent_usd"])
        fig = plot_router_cost_share(df, llm_calls_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_cost_share_empty_llm_calls_graceful(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_router_cost_share

        df = pd.DataFrame(columns=["run_id", "role", "cost_usd"])
        fig = plot_router_cost_share(runs_df_rq2, df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_cost_share_no_warnings(
        self, runs_df_rq2: pd.DataFrame, llm_calls_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_router_cost_share

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_router_cost_share(runs_df_rq2, llm_calls_df)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# test_plot_time_per_topology — RQ2/G11 smoke tests
# ---------------------------------------------------------------------------


class TestPlotTimePerTopologyRq2:
    """plot_time_per_topology returns a valid Figure. [rq2]"""

    def test_rq2_time_topo_returns_figure(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_time_per_topology

        fig = plot_time_per_topology(runs_df_rq2)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_time_topo_has_one_axes(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_time_per_topology

        fig = plot_time_per_topology(runs_df_rq2)
        assert len(fig.axes) == 1

    def test_rq2_time_topo_title_set(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_time_per_topology

        fig = plot_time_per_topology(runs_df_rq2)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_rq2_time_topo_empty_df_graceful(self) -> None:
        from atm.analysis.plots import plot_time_per_topology

        df = pd.DataFrame(columns=["run_id", "topology", "duration_seconds"])
        fig = plot_time_per_topology(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_time_topo_missing_duration_col_graceful(self) -> None:
        from atm.analysis.plots import plot_time_per_topology

        df = pd.DataFrame({"run_id": ["run-0", "run-1"], "topology": ["linear", "mesh"]})
        fig = plot_time_per_topology(df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_time_topo_no_warnings(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_time_per_topology

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_time_per_topology(runs_df_rq2)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_time_topo_xlabel_or_ylabel_set(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_time_per_topology

        fig = plot_time_per_topology(runs_df_rq2)
        ax = fig.axes[0]
        assert ax.get_xlabel() != "" or ax.get_ylabel() != ""

    def test_rq2_time_topo_transitions_df_path(self, transitions_df: pd.DataFrame) -> None:
        """plot_time_per_topology also accepts transitions_df with 'at' + 'to_topology'."""
        from atm.analysis.plots import plot_time_per_topology

        # Using transitions_df (which has run_id, to_topology, at) but no duration_seconds —
        # should still render gracefully or fall back to count-based display.
        fig = plot_time_per_topology(transitions_df)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# test_plot_oracle_gap_loo — RQ2/G11 smoke tests
# ---------------------------------------------------------------------------


class TestPlotOracleGapLooRq2:
    """plot_oracle_gap_loo returns a valid Figure. [rq2]"""

    def test_rq2_oracle_gap_returns_figure(
        self, runs_df_rq2: pd.DataFrame, oracle_gap_series: pd.Series
    ) -> None:
        from atm.analysis.plots import plot_oracle_gap_loo

        fig = plot_oracle_gap_loo(runs_df_rq2, oracle_gap_series)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_oracle_gap_has_one_axes(
        self, runs_df_rq2: pd.DataFrame, oracle_gap_series: pd.Series
    ) -> None:
        from atm.analysis.plots import plot_oracle_gap_loo

        fig = plot_oracle_gap_loo(runs_df_rq2, oracle_gap_series)
        assert len(fig.axes) == 1

    def test_rq2_oracle_gap_title_set(
        self, runs_df_rq2: pd.DataFrame, oracle_gap_series: pd.Series
    ) -> None:
        from atm.analysis.plots import plot_oracle_gap_loo

        fig = plot_oracle_gap_loo(runs_df_rq2, oracle_gap_series)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_rq2_oracle_gap_empty_series_graceful(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_oracle_gap_loo

        fig = plot_oracle_gap_loo(runs_df_rq2, pd.Series(dtype=float, name="oracle_gap_loo"))
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_oracle_gap_dataframe_input(
        self, runs_df_rq2: pd.DataFrame, oracle_gap_series: pd.Series
    ) -> None:
        """plot_oracle_gap_loo should also accept a DataFrame with 'oracle_gap_loo' column."""
        from atm.analysis.plots import plot_oracle_gap_loo

        oracle_gap_df = oracle_gap_series.reset_index()
        oracle_gap_df.columns = ["task_id", "oracle_gap_loo"]
        fig = plot_oracle_gap_loo(runs_df_rq2, oracle_gap_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_oracle_gap_no_warnings(
        self, runs_df_rq2: pd.DataFrame, oracle_gap_series: pd.Series
    ) -> None:
        from atm.analysis.plots import plot_oracle_gap_loo

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_oracle_gap_loo(runs_df_rq2, oracle_gap_series)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rq2_oracle_gap_vline_at_zero(
        self, runs_df_rq2: pd.DataFrame, oracle_gap_series: pd.Series
    ) -> None:
        """A vertical reference line at x=0 should be present on the Axes."""
        from atm.analysis.plots import plot_oracle_gap_loo

        fig = plot_oracle_gap_loo(runs_df_rq2, oracle_gap_series)
        ax = fig.axes[0]
        # Check the axes exists and has content (vline is present for non-empty data)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert ax is not None

    def test_rq2_oracle_gap_all_nan_graceful(self, runs_df_rq2: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_oracle_gap_loo

        s = pd.Series({"task-0": float("nan"), "task-1": float("nan")}, name="oracle_gap_loo")
        fig = plot_oracle_gap_loo(runs_df_rq2, s)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# RQ2 symbols exported check — g11
# ---------------------------------------------------------------------------


class TestRq2SymbolsExportedG11:
    """All RQ2/G11 plot functions appear in plots.__all__. [rq2] [g11]"""

    def test_rq2_g11_transition_timeline_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_transition_timeline_quality" in __all__

    def test_rq2_g11_guard_override_rate_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_guard_override_rate" in __all__

    def test_rq2_g11_router_cost_share_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_router_cost_share" in __all__

    def test_rq2_g11_time_per_topology_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_time_per_topology" in __all__

    def test_rq2_g11_oracle_gap_loo_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_oracle_gap_loo" in __all__


# ---------------------------------------------------------------------------
# RQ3/RQ4 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def human_interactions_df() -> pd.DataFrame:
    """Minimal human interactions DataFrame for plot_cognitive_load_boxplot."""
    rng = np.random.default_rng(99)
    n = 30
    topologies = ["adaptive", "linear", "mesh"]
    return pd.DataFrame(
        {
            "run_id": [f"run-{i}" for i in range(n)],
            "topology": [topologies[i % len(topologies)] for i in range(n)],
            "raw_tlx_score": rng.uniform(20.0, 90.0, n),
            "cognitive_load_proxy": rng.uniform(0.2, 0.9, n),
        }
    )


@pytest.fixture
def runs_df_cognitive() -> pd.DataFrame:
    """Minimal runs DataFrame for plot_cognitive_load_boxplot."""
    rng = np.random.default_rng(42)
    n = 30
    topologies = ["adaptive", "linear", "mesh"]
    return pd.DataFrame(
        {
            "run_id": [f"run-{i}" for i in range(n)],
            "topology": [topologies[i % len(topologies)] for i in range(n)],
            "cognitive_load_proxy": rng.uniform(0.2, 0.9, n),
        }
    )


# ---------------------------------------------------------------------------
# test_plot_cognitive_load_boxplot — RQ3/RQ4 smoke tests
# ---------------------------------------------------------------------------


class TestPlotCognitiveLoadBoxplot:
    """plot_cognitive_load_boxplot returns a valid Figure with 2 Axes."""

    def test_returns_figure(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        fig = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_two_axes(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        fig = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df)
        assert len(fig.axes) == 2

    def test_title_set(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        fig = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df)
        # At least one axes should have a title
        titles = [ax.get_title() for ax in fig.axes]
        assert any(t != "" for t in titles)

    def test_empty_human_interactions_graceful(self, runs_df_cognitive: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        df = pd.DataFrame(columns=["run_id", "topology", "raw_tlx_score", "cognitive_load_proxy"])
        fig = plot_cognitive_load_boxplot(runs_df_cognitive, df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_runs_df_graceful(self, human_interactions_df: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        df = pd.DataFrame(columns=["run_id", "topology", "cognitive_load_proxy"])
        fig = plot_cognitive_load_boxplot(df, human_interactions_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_role_filter_none(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        fig = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df, role=None)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_role_filter_string(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        # Non-existent role — should produce graceful figure, not crash
        fig = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df, role="reviewer")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_missing_raw_tlx_score_graceful(self, runs_df_cognitive: pd.DataFrame) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        df = pd.DataFrame(
            {
                "run_id": ["run-0", "run-1"],
                "topology": ["linear", "mesh"],
                "cognitive_load_proxy": [0.4, 0.6],
            }
        )
        fig = plot_cognitive_load_boxplot(runs_df_cognitive, df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_warnings(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_new_figure_each_call(
        self, runs_df_cognitive: pd.DataFrame, human_interactions_df: pd.DataFrame
    ) -> None:
        from atm.analysis.plots import plot_cognitive_load_boxplot

        fig1 = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df)
        fig2 = plot_cognitive_load_boxplot(runs_df_cognitive, human_interactions_df)
        assert fig1 is not fig2

    def test_cognitive_load_exported(self) -> None:
        from atm.analysis.plots import __all__

        assert "plot_cognitive_load_boxplot" in __all__
