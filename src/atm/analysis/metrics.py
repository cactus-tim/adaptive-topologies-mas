"""Derived metrics for RQ2 / safety analysis — pure, side-effect-free functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from atm.analysis.oracle import OracleTable


def compute_guard_override_rate(transitions_df: pd.DataFrame) -> float:
    """Return fraction of transitions where guards_applied is non-empty; 0.0 on empty input."""
    if transitions_df.empty:
        return 0.0

    total = len(transitions_df)
    if total == 0:
        return 0.0

    overrides = (
        transitions_df["guards_applied"].apply(lambda x: bool(x) if x is not None else False).sum()
    )
    return float(overrides) / float(total)


def compute_router_cost_share(
    transitions_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> float:
    """Return sum(router_cost_usd) / sum(budget_spent_usd); 0.0 on empty or zero denominator."""
    if transitions_df.empty or runs_df.empty:
        return 0.0

    router_total = float(pd.to_numeric(transitions_df["router_cost_usd"], errors="coerce").sum())
    run_total = float(pd.to_numeric(runs_df["budget_spent_usd"], errors="coerce").sum())

    if run_total == 0.0 or np.isnan(run_total):
        return 0.0

    return router_total / run_total


def compute_time_per_topology(
    transitions_df: pd.DataFrame,
) -> dict[str, float]:
    """Return topology → fractional share of total active time across all runs."""
    if transitions_df.empty:
        return {}

    df = transitions_df.copy()
    df["at"] = pd.to_datetime(df["at"], utc=True, errors="coerce")
    df = df.dropna(subset=["at"])

    if df.empty:
        return {}

    time_per_topo: dict[str, float] = {}
    total_seconds = 0.0

    for _run_id, group in df.groupby("run_id", sort=False):
        group_sorted = group.sort_values("at")
        timestamps = group_sorted["at"].to_numpy()
        topologies = group_sorted["to_topology"].to_list()

        for i in range(len(timestamps) - 1):
            delta = (timestamps[i + 1] - timestamps[i]) / np.timedelta64(1, "s")
            if delta < 0.0:
                delta = 0.0
            topo = str(topologies[i])
            time_per_topo[topo] = time_per_topo.get(topo, 0.0) + delta
            total_seconds += delta

    if total_seconds == 0.0:
        return {}

    return {topo: secs / total_seconds for topo, secs in time_per_topo.items()}


def compute_oracle_gap_loo(
    runs_df: pd.DataFrame,
    oracle_table: OracleTable,
) -> pd.Series:
    """Return per-task oracle gap (oracle quality - router quality) using LOO oracle."""
    if runs_df.empty:
        return pd.Series(dtype=float)

    df = runs_df.copy()
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")

    results: dict[str, float] = {}

    for task_id, group in df.groupby("task_id", sort=False):
        task_id_str = str(task_id)
        oracle_topo = oracle_table.by_task_id.get(task_id_str)

        if oracle_topo is None:
            results[task_id_str] = float("nan")
            continue

        oracle_runs = group[group["topology"] == oracle_topo]["quality_score"].dropna()
        if oracle_runs.empty:
            results[task_id_str] = float("nan")
            continue

        oracle_quality = float(oracle_runs.mean())

        router_runs = group["quality_score"].dropna()
        if router_runs.empty:
            results[task_id_str] = float("nan")
            continue

        router_quality = float(router_runs.mean())
        results[task_id_str] = oracle_quality - router_quality

    return pd.Series(results, name="oracle_gap_loo")


def compute_oracle_gap_manual(
    runs_df: pd.DataFrame,
    manual_oracle_table: OracleTable,
) -> pd.Series:
    """Return per-task oracle gap using the manual oracle; by_task_type fallback applied."""
    if runs_df.empty:
        return pd.Series(dtype=float)

    df = runs_df.copy()
    df["quality_score"] = pd.to_numeric(df["quality_score"], errors="coerce")

    results: dict[str, float] = {}

    for task_id, group in df.groupby("task_id", sort=False):
        task_id_str = str(task_id)
        # Manual oracle may be stored in by_task_id or by_task_type
        oracle_topo = manual_oracle_table.by_task_id.get(task_id_str)
        if oracle_topo is None:
            task_type = _infer_task_type_from_group(group, task_id_str)
            oracle_topo = manual_oracle_table.by_task_type.get(task_type)

        if oracle_topo is None:
            results[task_id_str] = float("nan")
            continue

        oracle_runs = group[group["topology"] == oracle_topo]["quality_score"].dropna()
        if oracle_runs.empty:
            results[task_id_str] = float("nan")
            continue

        oracle_quality = float(oracle_runs.mean())

        router_runs = group["quality_score"].dropna()
        if router_runs.empty:
            results[task_id_str] = float("nan")
            continue

        router_quality = float(router_runs.mean())
        results[task_id_str] = oracle_quality - router_quality

    return pd.Series(results, name="oracle_gap_manual")


def compute_hurt_rate(
    runs_df: pd.DataFrame,
    best_static_df: pd.DataFrame,
) -> float:
    """Return fraction of tasks where mean adaptive quality < best static quality."""
    if runs_df.empty or best_static_df.empty:
        return 0.0

    adaptive = runs_df.copy()
    adaptive["quality_score"] = pd.to_numeric(adaptive["quality_score"], errors="coerce")
    static = best_static_df.copy()
    static["quality_score"] = pd.to_numeric(static["quality_score"], errors="coerce")

    static_lookup = static.set_index("task_id")["quality_score"].to_dict()

    adaptive_mean = adaptive.groupby("task_id")["quality_score"].mean().dropna()

    if adaptive_mean.empty:
        return 0.0

    hurt_count = 0
    evaluated = 0
    for task_id, adap_q in adaptive_mean.items():
        if task_id not in static_lookup:
            continue
        static_q = static_lookup[task_id]
        if pd.isna(static_q):
            continue
        evaluated += 1
        if adap_q < static_q:
            hurt_count += 1

    if evaluated == 0:
        return 0.0

    return float(hurt_count) / float(evaluated)


def compute_topology_switch_counts(
    transitions_df: pd.DataFrame,
) -> pd.Series:
    """Return per-run topology switch counts (rows where to_topology != from_topology)."""
    if transitions_df.empty:
        return pd.Series(dtype=int)

    df = transitions_df.copy()
    has_from = df["from_topology"].notna() & (df["from_topology"] != "")
    is_switch = has_from & (df["to_topology"] != df["from_topology"])

    switch_counts = df[is_switch].groupby("run_id").size()

    all_run_ids = df["run_id"].unique()
    switch_counts = switch_counts.reindex(all_run_ids, fill_value=0)
    switch_counts.name = "topology_switch_count"

    return switch_counts.astype(int)


def _infer_task_type_from_group(group: pd.DataFrame, task_id: str) -> str:
    """Infer task_type from group DataFrame ``task_type`` column, or task_id prefix fallback."""
    if "task_type" in group.columns:
        non_null = group["task_type"].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])

    prefixes = [
        ("HumanEval/", "programming"),
        ("humaneval/", "programming"),
        ("MBPP/", "programming"),
        ("mbpp/", "programming"),
        ("GSM8K/", "reasoning"),
        ("gsm8k/", "reasoning"),
        ("MATH/", "reasoning"),
        ("math/", "reasoning"),
        ("ARC/", "reasoning"),
        ("arc/", "reasoning"),
        ("commongen/", "creative"),
        ("CommonGen/", "creative"),
        ("dabench/", "decision"),
        ("DABench/", "decision"),
    ]
    for prefix, task_type in prefixes:
        if task_id.startswith(prefix):
            return task_type
    return "unknown"
