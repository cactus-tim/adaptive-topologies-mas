"""Tests for plot_oracle_vs_router in atm.analysis.oracle.

This function lives in oracle.py (not plots.py) so that oracle-related
visualisation stays co-located with the data type. matplotlib is imported
lazily inside the function body; oracle.py itself stays import-lightweight.
"""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from atm.analysis.oracle import OracleTable


@pytest.fixture(autouse=True)
def close_all_figures() -> None:  # type: ignore[return]
    """Close all open matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runs_df() -> pd.DataFrame:
    """Minimal runs DataFrame for plot_oracle_vs_router."""
    rng = np.random.default_rng(7)
    n = 20
    topologies = ["adaptive", "linear", "mesh", "supervisor"]
    task_ids = [f"HumanEval/{i % 5}" for i in range(n)]
    return pd.DataFrame(
        {
            "run_id": [f"run-{i}" for i in range(n)],
            "topology": [topologies[i % len(topologies)] for i in range(n)],
            "task_id": task_ids,
            "quality_score": rng.uniform(0.4, 1.0, n),
        }
    )


@pytest.fixture
def oracle_table() -> OracleTable:
    """OracleTable with by_task_id and by_task_type entries."""
    return OracleTable(
        by_task_type={"programming": "adaptive", "reasoning": "linear"},
        by_task_id={f"HumanEval/{i}": "adaptive" for i in range(5)},
        **{"_default": "linear"},
    )


@pytest.fixture
def oracle_table_empty() -> OracleTable:
    """OracleTable with no entries (all default)."""
    return OracleTable(
        by_task_type={},
        by_task_id={},
        **{"_default": "linear"},
    )


# ---------------------------------------------------------------------------
# TestPlotOracleVsRouter
# ---------------------------------------------------------------------------


class TestPlotOracleVsRouter:
    """plot_oracle_vs_router returns a valid Figure with ≥1 Axes."""

    def test_returns_figure(self, runs_df: pd.DataFrame, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        fig = plot_oracle_vs_router(runs_df, oracle_table)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_at_least_one_axes(self, runs_df: pd.DataFrame, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        fig = plot_oracle_vs_router(runs_df, oracle_table)
        assert len(fig.axes) >= 1

    def test_title_set(self, runs_df: pd.DataFrame, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        fig = plot_oracle_vs_router(runs_df, oracle_table)
        ax = fig.axes[0]
        assert ax.get_title() != ""

    def test_empty_runs_df_graceful(self, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        df = pd.DataFrame(columns=["run_id", "topology", "task_id", "quality_score"])
        fig = plot_oracle_vs_router(df, oracle_table)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_empty_oracle_table_graceful(self, runs_df: pd.DataFrame) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        empty_oracle = OracleTable(by_task_type={}, by_task_id={}, **{"_default": "linear"})
        fig = plot_oracle_vs_router(runs_df, empty_oracle)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_missing_topology_column_graceful(self, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        df = pd.DataFrame({"run_id": ["run-0"], "task_id": ["HumanEval/0"], "quality_score": [0.7]})
        fig = plot_oracle_vs_router(df, oracle_table)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_router_col(self, runs_df: pd.DataFrame, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        df = runs_df.rename(columns={"topology": "router_topology"})
        fig = plot_oracle_vs_router(df, oracle_table, router_col="router_topology")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_no_warnings(self, runs_df: pd.DataFrame, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = plot_oracle_vs_router(runs_df, oracle_table)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_new_figure_each_call(
        self, runs_df: pd.DataFrame, oracle_table: OracleTable
    ) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        fig1 = plot_oracle_vs_router(runs_df, oracle_table)
        fig2 = plot_oracle_vs_router(runs_df, oracle_table)
        assert fig1 is not fig2

    def test_missing_task_id_col_graceful(self, oracle_table: OracleTable) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        df = pd.DataFrame(
            {
                "run_id": ["run-0", "run-1"],
                "topology": ["linear", "adaptive"],
                "quality_score": [0.6, 0.8],
            }
        )
        fig = plot_oracle_vs_router(df, oracle_table)
        assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


class TestOracleExports:
    """plot_oracle_vs_router is exported from oracle.py."""

    def test_plot_oracle_vs_router_in_dir(self) -> None:
        import atm.analysis.oracle as mod

        assert "plot_oracle_vs_router" in dir(mod)

    def test_callable(self) -> None:
        from atm.analysis.oracle import plot_oracle_vs_router

        assert callable(plot_oracle_vs_router)
