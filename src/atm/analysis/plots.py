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
    import seaborn as sns  # type: ignore[import-untyped]  # lazy import — seaborn is optional for headless

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
        aggfunc=aggfunc,  # type: ignore[arg-type]
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
    phase_col: str = "phase_name",
    started_col: str = "started_at",
    ended_col: str = "ended_at",
    figsize: tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    """Plot a horizontal Gantt-style timeline of phases for a single run.

    Args:
        phases_df:   DataFrame with phase records including timing columns.
        run_id:      Filter to this run_id value (must be present in ``run_id`` column).
        phase_col:   Column for phase name. Default: "phase_name" (matches
                     ``load_phases`` output column name).
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

        row_idx = int(i)  # type: ignore[call-overload]
        ax.broken_barh(
            [(start_s, duration_s)],
            (row_idx - 0.4, 0.8),
            facecolors=phase_color.get(phase_name, "steelblue"),
            label=phase_name,
            alpha=0.8,
        )
        ax.text(
            start_s + duration_s / 2,
            row_idx,
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
# RQ2 / G11 plots
# ---------------------------------------------------------------------------


def plot_transition_timeline_quality(
    transitions_df: pd.DataFrame,
    *,
    run_id: str,
    figsize: tuple[float, float] = (10, 5),
) -> matplotlib.figure.Figure:
    """Plot quality-weighted topology transition timeline for a single run.

    Renders a step plot of topology transitions over time, with quality score
    overlaid as a secondary line (if ``quality_score`` column is present).

    Args:
        transitions_df: DataFrame with topology transition records including
                        ``run_id``, ``at`` (datetime), ``to_topology``, and
                        optionally ``quality_score`` columns.
        run_id:         Filter to this run_id value.
        figsize:        Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes showing the timeline.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if "run_id" not in transitions_df.columns:
        ax.set_title(f"plot_transition_timeline_quality: 'run_id' column missing — {run_id}")
        return fig

    run_df = transitions_df[transitions_df["run_id"] == run_id].copy()

    if run_df.empty:
        ax.set_title(f"No transitions found for run_id={run_id!r}")
        return fig

    if "at" not in run_df.columns or "to_topology" not in run_df.columns:
        ax.set_title(f"Transition Timeline — run {run_id} (missing columns)")
        return fig

    run_df["at"] = pd.to_datetime(run_df["at"], utc=True, errors="coerce")
    run_df = run_df.dropna(subset=["at"]).sort_values("at").reset_index(drop=True)

    if run_df.empty:
        ax.set_title(f"Transition Timeline — run {run_id} (no valid timestamps)")
        return fig

    t0 = run_df["at"].min()
    times_s = [(t - t0).total_seconds() for t in run_df["at"]]

    # Encode topology as integer y-axis for step plot
    topos = run_df["to_topology"].tolist()
    unique_topos = sorted(set(topos))
    topo_idx = {t: i for i, t in enumerate(unique_topos)}
    y_vals = [topo_idx[t] for t in topos]

    ax.step(times_s, y_vals, where="post", linewidth=2, label="topology")
    ax.set_yticks(list(topo_idx.values()))
    ax.set_yticklabels(list(topo_idx.keys()))
    ax.set_xlabel("Time (seconds from start)")
    ax.set_ylabel("Topology")

    # Overlay quality score if available
    if "quality_score" in run_df.columns:
        q_vals = pd.to_numeric(run_df["quality_score"], errors="coerce")
        if q_vals.notna().any():
            ax2 = ax.twinx()
            ax2.plot(
                times_s,
                q_vals.tolist(),
                color="orange",
                linestyle="--",
                marker="o",
                markersize=4,
                label="quality",
                alpha=0.7,
            )
            ax2.set_ylabel("Quality Score", color="orange")
            ax2.tick_params(axis="y", labelcolor="orange")

    ax.set_title(f"Transition Timeline — run {run_id}")
    fig.tight_layout()
    return fig


def plot_guard_override_rate(
    transitions_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7, 5),
) -> matplotlib.figure.Figure:
    """Plot guard override rate per destination topology as a bar chart.

    Computes fraction of transitions decided by ``"guard_override"`` for each
    ``to_topology``.  An empty ``transitions_df`` or missing ``decided_by``
    column returns a graceful empty figure.

    Args:
        transitions_df: DataFrame with at least ``decided_by`` and
                        ``to_topology`` columns.
        figsize:        Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (horizontal bar chart).
    """
    fig, ax = plt.subplots(figsize=figsize)

    required = {"decided_by", "to_topology"}
    if transitions_df.empty or not required.issubset(transitions_df.columns):
        ax.set_title("Guard Override Rate per Topology (no data)")
        ax.set_xlabel("Override Rate")
        ax.set_ylabel("Topology")
        fig.tight_layout()
        return fig

    df = transitions_df.copy()
    df["is_override"] = df["decided_by"] == "guard_override"

    rates = (
        df.groupby("to_topology")["is_override"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_overrides", "count": "n_total"})
    )
    rates["rate"] = rates["n_overrides"] / rates["n_total"].clip(lower=1)
    rates = rates.sort_values("rate", ascending=True)

    if rates.empty:
        ax.set_title("Guard Override Rate per Topology (no data)")
        ax.set_xlabel("Override Rate")
        ax.set_ylabel("Topology")
        fig.tight_layout()
        return fig

    ax.barh(rates.index.tolist(), rates["rate"].tolist(), color="steelblue", alpha=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Override Rate")
    ax.set_ylabel("Topology")
    ax.set_title("Guard Override Rate per Topology")
    fig.tight_layout()
    return fig


def plot_router_cost_share(
    runs_df: pd.DataFrame,
    llm_calls_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (7, 5),
) -> matplotlib.figure.Figure:
    """Plot router LLM cost share vs worker cost share as a stacked bar chart.

    Router calls are identified by ``role == "router"`` in ``llm_calls_df``.
    Total cost per run comes from ``llm_calls_df["cost_usd"]``; fallback is
    ``runs_df["budget_spent_usd"]``.

    An empty input returns a graceful empty figure.

    Args:
        runs_df:      DataFrame with run-level records (used for fallback cost).
        llm_calls_df: DataFrame with LLM call records including ``role`` and
                      ``cost_usd`` columns.
        figsize:      Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (stacked bar chart).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if llm_calls_df.empty or "cost_usd" not in llm_calls_df.columns:
        ax.set_title("Router Cost Share (no data)")
        ax.set_xlabel("Run")
        ax.set_ylabel("Cost (USD)")
        fig.tight_layout()
        return fig

    df = llm_calls_df.copy()
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    df["is_router"] = df.get("role", pd.Series(dtype=str)) == "router"

    if "run_id" not in df.columns:
        # No run_id column — aggregate globally
        router_cost = float(df.loc[df["is_router"], "cost_usd"].sum())
        worker_cost = float(df.loc[~df["is_router"], "cost_usd"].sum())
        labels = ["all_runs"]
        router_costs = [router_cost]
        worker_costs = [worker_cost]
    else:
        run_agg = df.groupby(["run_id", "is_router"])["cost_usd"].sum().unstack(fill_value=0)
        router_costs_series = run_agg.get(True, pd.Series(dtype=float))
        worker_costs_series = run_agg.get(False, pd.Series(dtype=float))
        labels = run_agg.index.tolist()
        router_costs = router_costs_series.reindex(labels, fill_value=0.0).tolist()
        worker_costs = worker_costs_series.reindex(labels, fill_value=0.0).tolist()

    x = list(range(len(labels)))
    ax.bar(x, router_costs, label="Router", color="steelblue", alpha=0.8)
    ax.bar(x, worker_costs, bottom=router_costs, label="Worker", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Run")
    ax.set_ylabel("Cost (USD)")
    ax.set_title("Router vs Worker LLM Cost per Run")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_time_per_topology(
    runs_df: pd.DataFrame,
    *,
    topology_col: str = "topology",
    duration_col: str = "duration_seconds",
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot mean wall-clock time per topology as a bar chart.

    Uses ``duration_seconds`` (or ``duration_col``) from ``runs_df``.
    If neither column is present, falls back to counting the number of
    rows per topology (i.e., number of runs).

    Works with both a ``runs_df`` (one row per run with topology + duration)
    and a ``transitions_df`` (one row per transition event with ``to_topology``),
    as long as ``topology_col`` resolves.

    Args:
        runs_df:       DataFrame with topology and optional duration column.
        topology_col:  Column for topology name. Default: "topology".
        duration_col:  Column for duration in seconds. Default: "duration_seconds".
        figsize:       Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (bar chart).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Support transitions_df path: use to_topology if topology col is missing
    if topology_col not in runs_df.columns and "to_topology" in runs_df.columns:
        topology_col = "to_topology"

    if runs_df.empty or topology_col not in runs_df.columns:
        ax.set_title("Time per Topology (no data)")
        ax.set_xlabel("Topology")
        ax.set_ylabel("Mean Duration (s)")
        fig.tight_layout()
        return fig

    df = runs_df.copy()

    if duration_col in df.columns:
        df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
        agg = df.groupby(topology_col)[duration_col].mean().dropna().sort_values(ascending=False)
        ylabel = "Mean Duration (s)"
    else:
        # Fallback: count rows per topology
        agg = df.groupby(topology_col).size().sort_values(ascending=False).astype(float)
        ylabel = "Run Count"

    if agg.empty:
        ax.set_title("Time per Topology (no data)")
        ax.set_xlabel("Topology")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return fig

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = [colors[i % len(colors)] for i in range(len(agg))]

    ax.bar(agg.index.tolist(), agg.tolist(), color=bar_colors, alpha=0.8)
    ax.set_xlabel("Topology")
    ax.set_ylabel(ylabel)
    ax.set_title("Mean Time per Topology")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_oracle_gap_loo(
    runs_df: pd.DataFrame,
    oracle_gap: pd.Series | pd.DataFrame,
    *,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot oracle gap (LOO) distribution as a histogram with a reference line at 0.

    Positive values mean the oracle topology beats the router; negative means
    the router outperforms the leave-one-out oracle for that task.

    Args:
        runs_df:     DataFrame with run-level records (used for context; not
                     directly plotted, but kept for API symmetry with the
                     metrics function).
        oracle_gap:  Pre-computed oracle gap values.  Either:
                     - ``pd.Series`` (indexed by task_id, values are float), or
                     - ``pd.DataFrame`` with an ``oracle_gap_loo`` column.
        figsize:     Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (histogram + vline at 0).
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Normalise input to a Series of float gap values
    if isinstance(oracle_gap, pd.DataFrame):
        if "oracle_gap_loo" in oracle_gap.columns:
            gap_series = pd.to_numeric(oracle_gap["oracle_gap_loo"], errors="coerce")
        else:
            # Try the first numeric column
            numeric_cols = oracle_gap.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                gap_series = pd.Series(dtype=float)
            else:
                gap_series = pd.to_numeric(oracle_gap[numeric_cols[0]], errors="coerce")
    else:
        gap_series = pd.to_numeric(oracle_gap, errors="coerce")

    gap_values = gap_series.dropna()

    if gap_values.empty:
        ax.set_title("Oracle Gap LOO Distribution (no data)")
        ax.set_xlabel("Oracle Gap (oracle - router quality)")
        ax.set_ylabel("Count")
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="zero gap")
        fig.tight_layout()
        return fig

    ax.hist(gap_values.tolist(), bins="auto", color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="zero gap")
    ax.set_xlabel("Oracle Gap (oracle - router quality)")
    ax.set_ylabel("Count")
    ax.set_title("Oracle Gap LOO Distribution")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# RQ3/RQ4 plots (stubs for Steps 9+)
# ---------------------------------------------------------------------------


def plot_cognitive_load_boxplot(
    runs_df: pd.DataFrame,
    human_interactions_df: pd.DataFrame,
    *,
    role: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot cognitive load (TLX proxy) boxplot per topology.

    Produces a two-panel figure:
    - Left Axes: boxplot of ``raw_tlx_score`` from ``human_interactions_df``
      grouped by topology.
    - Right Axes: boxplot of ``cognitive_load_proxy`` from ``runs_df``
      grouped by topology.

    Args:
        runs_df:               DataFrame with run-level records including
                               ``topology`` and ``cognitive_load_proxy`` columns.
        human_interactions_df: DataFrame with human interaction records including
                               ``raw_tlx_score`` and ``topology`` columns.
        role:                  If given, filter ``human_interactions_df`` to rows
                               where ``role == role`` before plotting.  ``None``
                               means all rows are included.
        figsize:               Figure (width, height) in inches.

    Returns:
        matplotlib Figure with two Axes (TLX boxplot | cognitive_load_proxy boxplot).
    """
    fig, (ax_tlx, ax_proxy) = plt.subplots(1, 2, figsize=figsize)

    # ------------------------------------------------------------------
    # Helper: draw a boxplot on a given axes using pure matplotlib
    # ------------------------------------------------------------------
    def _draw_boxplot(
        ax: matplotlib.axes.Axes,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        title: str,
        ylabel: str,
    ) -> None:
        groups = sorted(df[group_col].dropna().unique())
        data = [
            df.loc[df[group_col] == g, value_col].dropna().to_numpy(dtype=float) for g in groups
        ]
        # Filter out empty groups
        valid = [(g, d) for g, d in zip(groups, data, strict=False) if len(d) > 0]
        if not valid:
            ax.set_title(f"{title} (no data)")
            ax.set_xlabel(group_col)
            ax.set_ylabel(ylabel)
            return

        valid_groups, valid_data = zip(*valid, strict=False)
        ax.boxplot(valid_data, tick_labels=list(valid_groups), patch_artist=True)
        ax.set_title(title)
        ax.set_xlabel(group_col)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # ------------------------------------------------------------------
    # Left Axes: raw_tlx_score from human_interactions_df
    # ------------------------------------------------------------------
    hi_df = human_interactions_df.copy()

    # Apply role filter if requested
    if role is not None and "role" in hi_df.columns:
        hi_df = hi_df[hi_df["role"] == role]

    if hi_df.empty or "raw_tlx_score" not in hi_df.columns or "topology" not in hi_df.columns:
        ax_tlx.set_title("Raw TLX Score per Topology (no data)")
        ax_tlx.set_xlabel("Topology")
        ax_tlx.set_ylabel("Raw TLX Score")
    else:
        tlx_plot_df = hi_df[["topology", "raw_tlx_score"]].dropna()
        _draw_boxplot(
            ax_tlx,
            tlx_plot_df,
            "topology",
            "raw_tlx_score",
            "Raw TLX Score per Topology",
            "Raw TLX Score",
        )

    # ------------------------------------------------------------------
    # Right Axes: cognitive_load_proxy from runs_df
    # ------------------------------------------------------------------
    if (
        runs_df.empty
        or "cognitive_load_proxy" not in runs_df.columns
        or "topology" not in runs_df.columns
    ):
        ax_proxy.set_title("Cognitive Load Proxy per Topology (no data)")
        ax_proxy.set_xlabel("Topology")
        ax_proxy.set_ylabel("Cognitive Load Proxy")
    else:
        proxy_plot_df = runs_df[["topology", "cognitive_load_proxy"]].dropna()
        _draw_boxplot(
            ax_proxy,
            proxy_plot_df,
            "topology",
            "cognitive_load_proxy",
            "Cognitive Load Proxy per Topology",
            "Cognitive Load Proxy",
        )

    fig.tight_layout()
    return fig
