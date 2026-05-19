"""RQ1/RQ2/RQ3/RQ4 plot functions — accept DataFrames, return matplotlib Figures."""

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
    """Plot quality vs cost Pareto scatter per topology with 95% bootstrap CI error bars."""
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
    """Plot a topology x task_type heatmap aggregated by ``aggfunc`` on ``value_col``."""
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
    """Plot a horizontal Gantt-style broken_barh phase timeline for a single run."""
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

    run_phases = run_phases.sort_values(started_col).reset_index(drop=True)

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


def plot_transition_timeline_quality(
    transitions_df: pd.DataFrame,
    *,
    run_id: str,
    figsize: tuple[float, float] = (10, 5),
) -> matplotlib.figure.Figure:
    """Plot topology step-transitions over time for a run, with quality overlaid on twin axis."""
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

    topos = run_df["to_topology"].tolist()
    unique_topos = sorted(set(topos))
    topo_idx = {t: i for i, t in enumerate(unique_topos)}
    y_vals = [topo_idx[t] for t in topos]

    ax.step(times_s, y_vals, where="post", linewidth=2, label="topology")
    ax.set_yticks(list(topo_idx.values()))
    ax.set_yticklabels(list(topo_idx.keys()))
    ax.set_xlabel("Time (seconds from start)")
    ax.set_ylabel("Topology")

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
    """Plot horizontal bar chart of guard_override fraction per destination topology."""
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
    """Plot stacked bar chart of router vs worker LLM cost per run."""
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
    """Plot mean duration per topology; falls back to run count if duration_col absent."""
    fig, ax = plt.subplots(figsize=figsize)

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
    """Plot oracle gap LOO histogram with a zero-reference line."""
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(oracle_gap, pd.DataFrame):
        if "oracle_gap_loo" in oracle_gap.columns:
            gap_series = pd.to_numeric(oracle_gap["oracle_gap_loo"], errors="coerce")
        else:
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


def plot_cognitive_load_boxplot(
    runs_df: pd.DataFrame,
    human_interactions_df: pd.DataFrame,
    *,
    role: str | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot two-panel boxplot: raw TLX score and cognitive_load_proxy, both per topology."""
    fig, (ax_tlx, ax_proxy) = plt.subplots(1, 2, figsize=figsize)

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

    hi_df = human_interactions_df.copy()

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
