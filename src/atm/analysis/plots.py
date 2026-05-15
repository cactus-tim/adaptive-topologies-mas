"""Plot functions for RQ1 / RQ2 / RQ3 / RQ4 -- all accept DataFrames, return Figures.

The Agg backend is forced before any pyplot import to prevent GUI warnings
in headless CI environments (pytest filterwarnings=["error"] would turn them
into failures).

Public surface (to be implemented in Steps 7-9):
    plot_pareto                      -- RQ1 quality vs. cost Pareto frontier
    plot_topology_task_heatmap       -- RQ1 topology x task-type quality heatmap
    plot_phase_timeline              -- RQ1 Gantt-style phase timeline
    plot_transition_timeline_quality -- RQ2 transition events overlaid on quality curve
    plot_guard_override_rate         -- RQ2 guard override rate bar chart
    plot_router_cost_share           -- RQ2 routing cost share pie/bar chart
    plot_time_per_topology           -- RQ2 time spent per topology
    plot_oracle_gap_loo              -- G11 per-task LOO oracle gap
    plot_cognitive_load_boxplot      -- RQ3/RQ4 cognitive load distribution (2 axes)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_pareto(
    runs_df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
) -> plt.Figure:
    """Plot the quality vs. cost Pareto frontier across topologies.

    Confidence bands are computed via numpy bootstrap (no scipy dependency).

    Args:
        runs_df:     DataFrame from ``load_runs`` with ``quality_score`` and
                     ``budget_spent_usd`` columns.
        n_bootstrap: Number of bootstrap samples for CI bands (default 1000).

    Returns:
        matplotlib Figure with at least one Axes.
    """
    raise NotImplementedError("plot_pareto -- implemented in Step 7")


def plot_topology_task_heatmap(
    runs_df: pd.DataFrame,
) -> plt.Figure:
    """Plot a heatmap of mean quality_score for each topology x task_type cell.

    Args:
        runs_df: DataFrame from ``load_runs`` with ``topology``, ``task_type``,
                 and ``quality_score`` columns.

    Returns:
        matplotlib Figure with one Axes (heatmap).
    """
    raise NotImplementedError("plot_topology_task_heatmap -- implemented in Step 7")


def plot_phase_timeline(
    phases_df: pd.DataFrame,
    *,
    run_id: str | None = None,
) -> plt.Figure:
    """Plot a Gantt-style timeline of phases for one or all runs.

    Args:
        phases_df: DataFrame from ``load_phases``.
        run_id:    If provided, filter to only this run; otherwise show all.

    Returns:
        matplotlib Figure with one Axes.
    """
    raise NotImplementedError("plot_phase_timeline -- implemented in Step 7")


def plot_transition_timeline_quality(
    runs_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
) -> plt.Figure:
    """Plot quality curve over time with topology-transition events overlaid.

    Uses a twinx axis so the transition events and quality curve share the
    same x-axis (time). The legend contains at least 2 entries.

    Args:
        runs_df:        DataFrame from ``load_runs``.
        transitions_df: DataFrame from ``load_topology_transitions``.

    Returns:
        matplotlib Figure with two Axes (twinx).
    """
    raise NotImplementedError("plot_transition_timeline_quality -- implemented in Step 8")


def plot_guard_override_rate(
    transitions_df: pd.DataFrame,
) -> plt.Figure:
    """Plot the guard override rate as a bar chart.

    Args:
        transitions_df: DataFrame from ``load_topology_transitions``.

    Returns:
        matplotlib Figure with one Axes.
    """
    raise NotImplementedError("plot_guard_override_rate -- implemented in Step 8")


def plot_router_cost_share(
    runs_df: pd.DataFrame,
    llm_calls_df: pd.DataFrame,
) -> plt.Figure:
    """Plot the routing cost share relative to total LLM budget.

    Args:
        runs_df:      DataFrame from ``load_runs``.
        llm_calls_df: DataFrame from ``load_llm_calls_for_experiment``.

    Returns:
        matplotlib Figure with one Axes.
    """
    raise NotImplementedError("plot_router_cost_share -- implemented in Step 8")


def plot_time_per_topology(
    phases_df: pd.DataFrame,
) -> plt.Figure:
    """Plot mean wall-clock time spent in each topology.

    Args:
        phases_df: DataFrame from ``load_phases``.

    Returns:
        matplotlib Figure with one Axes.
    """
    raise NotImplementedError("plot_time_per_topology -- implemented in Step 8")


def plot_oracle_gap_loo(
    gap_df: pd.DataFrame,
) -> plt.Figure:
    """Plot per-task LOO oracle quality gap (G11 metric).

    Args:
        gap_df: DataFrame from ``compute_oracle_gap_loo`` with columns
                ``task_id``, ``oracle_quality``, ``actual_quality``, ``gap``.

    Returns:
        matplotlib Figure with one Axes.
    """
    raise NotImplementedError("plot_oracle_gap_loo -- implemented in Step 8")


def plot_cognitive_load_boxplot(
    human_interactions_df: pd.DataFrame,
) -> plt.Figure:
    """Plot cognitive load (raw_tlx_score) distribution across topology conditions.

    NaN rows in ``raw_tlx_score`` are silently dropped before plotting.

    Args:
        human_interactions_df: DataFrame from ``load_human_interactions``.

    Returns:
        matplotlib Figure with 2 Axes (one per comparison group).
    """
    raise NotImplementedError("plot_cognitive_load_boxplot -- implemented in Step 9")
