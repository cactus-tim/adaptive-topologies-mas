"""Smoke tests and structural assertions for atm.analysis.plots — Step 7.1.

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
