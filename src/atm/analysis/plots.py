"""RQ1/RQ2/RQ3/RQ4 plot functions for ATM analysis.

All functions accept pre-loaded DataFrames and return matplotlib Figure objects.

IMPORTANT: matplotlib.use("Agg") MUST be the first matplotlib call so that
headless CI environments do not trigger GUI backend warnings (which become errors
under filterwarnings=["error"]).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must be called before pyplot is imported

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_cognitive_load_boxplot",
    "plot_guard_override_rate",
    "plot_oracle_gap_loo",
    "plot_pareto",
    "plot_phase_timeline",
    "plot_router_cost_share",
    "plot_time_per_topology",
    "plot_topology_task_heatmap",
    "plot_transition_timeline_quality",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RNG_SEED = 42
_N_BOOTSTRAP = 1000


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = _N_BOOTSTRAP,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for the mean.

    Uses numpy bootstrap resampling (no scipy dependency).
    """
    if rng is None:
        rng = np.random.default_rng(_RNG_SEED)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v
    boot_means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    return lower, upper


# ---------------------------------------------------------------------------
# RQ1 plots
# ---------------------------------------------------------------------------


def plot_pareto(
    runs_df: pd.DataFrame,
    *,
    group_col: str = "topology",
    quality_col: str = "quality_score",
    cost_col: str = "budget_spent_usd",
    adaptive_label: str = "adaptive",
    n_bootstrap: int = _N_BOOTSTRAP,
    figsize: tuple[float, float] = (8, 6),
) -> matplotlib.figure.Figure:
    """Plot quality vs cost Pareto scatter with bootstrap confidence bands.

    Each group (topology) is plotted as a point at (mean cost, mean quality).
    Bootstrap 95% CI bands are drawn as error bars for each group.
    The adaptive topology group (if present) is highlighted with a distinct marker.

    Args:
        runs_df:        DataFrame with at minimum ``group_col``, ``quality_col``,
                        and ``cost_col`` columns.
        group_col:      Column used to split runs into groups. Default: "topology".
        quality_col:    Column for quality metric. Default: "quality_score".
        cost_col:       Column for cost metric. Default: "budget_spent_usd".
        adaptive_label: Topology label treated as "adaptive" (highlighted marker).
        n_bootstrap:    Number of bootstrap resamples for CI computation.
        figsize:        Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (quality vs cost scatter + error bars).
    """
    fig, ax = plt.subplots(figsize=figsize)

    required_cols = {group_col, quality_col, cost_col}
    missing = required_cols - set(runs_df.columns)
    if missing:
        ax.set_title("plot_pareto: missing columns")
        ax.text(0.5, 0.5, f"Missing: {missing}", transform=ax.transAxes, ha="center")
        return fig

    rng = np.random.default_rng(_RNG_SEED)
    groups = sorted(runs_df[group_col].dropna().unique())

    for group in groups:
        mask = runs_df[group_col] == group
        q_vals = runs_df.loc[mask, quality_col].dropna().to_numpy(dtype=float)
        c_vals = runs_df.loc[mask, cost_col].dropna().to_numpy(dtype=float)

        if len(q_vals) == 0 or len(c_vals) == 0:
            continue

        q_mean = float(q_vals.mean())
        c_mean = float(c_vals.mean())
        q_lo, q_hi = _bootstrap_ci(q_vals, n_bootstrap=n_bootstrap, rng=rng)
        c_lo, c_hi = _bootstrap_ci(c_vals, n_bootstrap=n_bootstrap, rng=rng)

        is_adaptive = str(group).lower() == adaptive_label.lower()
        marker = "*" if is_adaptive else "o"
        markersize = 14 if is_adaptive else 8
        zorder = 5 if is_adaptive else 3

        ax.errorbar(
            c_mean,
            q_mean,
            xerr=[[c_mean - c_lo], [c_hi - c_mean]],
            yerr=[[q_mean - q_lo], [q_hi - q_mean]],
            fmt=marker,
            markersize=markersize,
            capsize=4,
            label=str(group),
            zorder=zorder,
        )

    ax.set_xlabel(f"Mean {cost_col}")
    ax.set_ylabel(f"Mean {quality_col}")
    ax.set_title("Quality vs Cost (Pareto)")
    # Only add legend if there are labeled artists (avoid UserWarning on empty data)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_topology_task_heatmap(
    runs_df: pd.DataFrame,
    *,
    topology_col: str = "topology",
    task_type_col: str = "task_type",
    value_col: str = "quality_score",
    aggfunc: str = "mean",
    figsize: tuple[float, float] = (8, 5),
    cmap: str = "YlOrRd",
) -> matplotlib.figure.Figure:
    """Plot a topology x task_type heatmap of mean quality score.

    Args:
        runs_df:       DataFrame with topology, task_type, and quality_score columns.
        topology_col:  Column for topology labels. Default: "topology".
        task_type_col: Column for task type labels. Default: "task_type".
        value_col:     Column to aggregate. Default: "quality_score".
        aggfunc:       Aggregation function name (passed to pivot_table). Default: "mean".
        figsize:       Figure (width, height) in inches.
        cmap:          Colormap name.

    Returns:
        matplotlib Figure with one Axes (heatmap).
    """
    import seaborn as sns  # lazy import — seaborn is optional for headless

    fig, ax = plt.subplots(figsize=figsize)

    required_cols = {topology_col, task_type_col, value_col}
    missing = required_cols - set(runs_df.columns)
    if missing:
        ax.set_title("plot_topology_task_heatmap: missing columns")
        ax.text(0.5, 0.5, f"Missing: {missing}", transform=ax.transAxes, ha="center")
        return fig

    pivot = runs_df.pivot_table(
        values=value_col,
        index=topology_col,
        columns=task_type_col,
        aggfunc=aggfunc,
    )

    if pivot.empty:
        ax.set_title(f"Topology x Task Type - {aggfunc} {value_col} (no data)")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        fig.tight_layout()
        return fig

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": f"{aggfunc} {value_col}"},
    )
    ax.set_title(f"Topology x Task Type - {aggfunc} {value_col}")
    ax.set_xlabel(task_type_col)
    ax.set_ylabel(topology_col)
    fig.tight_layout()
    return fig


def plot_phase_timeline(
    phases_df: pd.DataFrame,
    *,
    run_id: str,
    phase_col: str = "phase",
    started_col: str = "started_at",
    ended_col: str = "ended_at",
    figsize: tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    """Plot a horizontal Gantt-style timeline of phases for a single run.

    Args:
        phases_df:   DataFrame with phase records including timing columns.
        run_id:      Filter to this run_id value (must be present in ``run_id`` column).
        phase_col:   Column for phase name. Default: "phase".
        started_col: Column for phase start timestamp. Default: "started_at".
        ended_col:   Column for phase end timestamp. Default: "ended_at".
        figsize:     Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (horizontal bar / broken_barh Gantt chart).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if "run_id" not in phases_df.columns:
        ax.set_title("plot_phase_timeline: 'run_id' column missing")
        return fig

    run_phases = phases_df[phases_df["run_id"] == run_id].copy()

    if run_phases.empty:
        ax.set_title(f"No phases found for run_id={run_id!r}")
        return fig

    required_cols = {phase_col, started_col, ended_col}
    missing = required_cols - set(run_phases.columns)
    if missing:
        ax.set_title(f"plot_phase_timeline: missing columns {missing}")
        return fig

    # Convert timestamps to numeric (seconds from first event) for plotting
    run_phases = run_phases.sort_values(started_col).reset_index(drop=True)

    # Get reference time as earliest started_at
    t0 = pd.to_datetime(run_phases[started_col]).min()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    phases = run_phases[phase_col].dropna().unique().tolist()
    phase_color = {p: colors[i % len(colors)] for i, p in enumerate(phases)}

    for i, row in run_phases.iterrows():
        phase_name = str(row[phase_col])
        t_start = pd.to_datetime(row[started_col])
        t_end = pd.to_datetime(row[ended_col])

        if pd.isna(t_start) or pd.isna(t_end):
            continue

        start_s = (t_start - t0).total_seconds()
        duration_s = max((t_end - t_start).total_seconds(), 0.0)

        ax.broken_barh(
            [(start_s, duration_s)],
            (int(i) - 0.4, 0.8),
            facecolors=phase_color.get(phase_name, "steelblue"),
            label=phase_name,
            alpha=0.8,
        )
        ax.text(
            start_s + duration_s / 2,
            int(i),
            phase_name,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    ax.set_xlabel("Time (seconds from start)")
    ax.set_ylabel("Phase event index")
    ax.set_title(f"Phase Timeline — run {run_id}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RQ2 plots (stubs for Steps 8+; implemented here as NotImplementedError)
# ---------------------------------------------------------------------------


def plot_transition_timeline_quality(
    transitions_df: pd.DataFrame,
    *,
    run_id: str,
    figsize: tuple[float, float] = (10, 5),
) -> matplotlib.figure.Figure:
    """Plot quality-weighted topology transition timeline for a single run.

    Args:
        transitions_df: DataFrame with topology transition records.
        run_id:         Filter to this run_id value.
        figsize:        Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes.

    Raises:
        NotImplementedError: This function is implemented in Step 8.
    """
    raise NotImplementedError("plot_transition_timeline_quality: implemented in Step 8")


def plot_guard_override_rate(
    transitions_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7, 5),
) -> matplotlib.figure.Figure:
    """Plot guard override rate per topology.

    Args:
        transitions_df: DataFrame with topology transition records.
        figsize:        Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes.

    Raises:
        NotImplementedError: This function is implemented in Step 8.
    """
    raise NotImplementedError("plot_guard_override_rate: implemented in Step 8")


def plot_router_cost_share(
    runs_df: pd.DataFrame,
    llm_calls_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7, 5),
) -> matplotlib.figure.Figure:
    """Plot router LLM cost as share of total cost per experiment.

    Args:
        runs_df:      DataFrame with run-level records.
        llm_calls_df: DataFrame with LLM call records.
        figsize:      Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes.

    Raises:
        NotImplementedError: This function is implemented in Step 8.
    """
    raise NotImplementedError("plot_router_cost_share: implemented in Step 8")


def plot_time_per_topology(
    runs_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot distribution of wall-clock time per topology.

    Args:
        runs_df: DataFrame with run-level records including timing columns.
        figsize: Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes.

    Raises:
        NotImplementedError: This function is implemented in Step 8.
    """
    raise NotImplementedError("plot_time_per_topology: implemented in Step 8")


def plot_oracle_gap_loo(
    runs_df: pd.DataFrame,
    oracle_gap_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot oracle gap (LOO) distribution: adaptive vs static topologies.

    Args:
        runs_df:       DataFrame with run-level records.
        oracle_gap_df: DataFrame with pre-computed oracle gap values.
        figsize:       Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes.

    Raises:
        NotImplementedError: This function is implemented in Step 8.
    """
    raise NotImplementedError("plot_oracle_gap_loo: implemented in Step 8")


# ---------------------------------------------------------------------------
# RQ3/RQ4 plots (stubs for Steps 9+)
# ---------------------------------------------------------------------------


def plot_cognitive_load_boxplot(
    human_interactions_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot cognitive load (TLX proxy) boxplot per topology.

    Args:
        human_interactions_df: DataFrame with human interaction records including
                               ``raw_tlx_score`` and ``topology`` columns.
        figsize:               Figure (width, height) in inches.

    Returns:
        matplotlib Figure with two Axes (boxplot + swarm overlay).

    Raises:
        NotImplementedError: This function is implemented in Step 9.
    """
    raise NotImplementedError("plot_cognitive_load_boxplot: implemented in Step 9")
