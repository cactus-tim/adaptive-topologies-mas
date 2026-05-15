"""Derived metrics for RQ2 / safety analysis — pure functions.

All functions are side-effect-free and accept pandas DataFrames produced by
the loaders in ``atm.analysis.loaders``.  No I/O, no database, no plotting.

Public surface (7 functions):
    compute_guard_override_rate        — share of router decisions that were guard overrides
    compute_router_cost_share          — router cost as fraction of total run cost
    compute_time_per_topology          — active wall-time fraction spent in each topology
    compute_oracle_gap_loo             — quality gap vs leave-one-out oracle (per task_id)
    compute_oracle_gap_manual          — quality gap vs manual oracle (per task_id)
    compute_hurt_rate                  — fraction of tasks where Adaptive < best static
    compute_topology_switch_counts     — actual switch count per run

Column name contracts (match loaders.py output):
    transitions_df columns used:
        run_id, from_topology, to_topology, decided_by, guards_applied,
        router_cost_usd, at
    runs_df columns used:
        id (or run_id), task_id, quality_score, budget_spent_usd
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from atm.analysis.oracle import OracleTable


# ---------------------------------------------------------------------------
# G11 / RQ2 metric helpers
# ---------------------------------------------------------------------------


def compute_guard_override_rate(transitions_df: pd.DataFrame) -> float:
    """Compute the fraction of router decisions that were guard overrides.

    Formula (experiment_plan.md §4):
        guard_override_rate = |decided_by == "guard_override"| / |all decisions|

    An empty ``transitions_df`` returns 0.0 (no decisions → no overrides).

    Args:
        transitions_df: DataFrame with at least a ``decided_by`` column.

    Returns:
        Float in [0.0, 1.0].
    """
    if transitions_df.empty:
        return 0.0

    total = len(transitions_df)
    if total == 0:
        return 0.0

    overrides = (transitions_df["decided_by"] == "guard_override").sum()
    return float(overrides) / float(total)


def compute_router_cost_share(
    transitions_df: pd.DataFrame,
    runs_df: pd.DataFrame,
) -> float:
    """Compute router cost as a fraction of total run cost.

    Formula (experiment_plan.md §4):
        router_cost_share = sum(router_cost_usd) / sum(total_usd)

    where ``total_usd`` is taken from ``runs_df["budget_spent_usd"]`` and
    ``router_cost_usd`` from ``transitions_df["router_cost_usd"]``.

    Returns 0.0 when either denominator or numerator is zero/absent.

    Args:
        transitions_df: DataFrame with ``router_cost_usd`` column.
        runs_df:        DataFrame with ``budget_spent_usd`` column.

    Returns:
        Float in [0.0, ∞) (can exceed 1.0 if data is inconsistent, but
        should be well below 1.0 in practice).
    """
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
    """Compute the fraction of total active time spent in each topology.

    Algorithm (from plan Risks section):
        For row i (sorted ascending by ``at`` within each run), the active
        topology during the delta between row i and row i+1 is the
        ``from_topology`` of row i+1 (= ``to_topology`` of row i).
        The last row in each run has no next row, so its final topology
        (``to_topology``) holds until run completion — but without a
        successor timestamp we cannot measure this final delta.  We
        therefore only account for deltas between consecutive decisions.

    Timestamps are taken from the ``at`` column (datetime-compatible).
    Rows without a valid ``at`` value (NaT / NaN) are skipped.

    Returns a dict topology → fractional share summing to 1.0.
    An empty or single-row DataFrame returns an empty dict.

    Args:
        transitions_df: DataFrame with columns ``run_id``, ``to_topology``,
                        and ``at`` (datetime or datetime-compatible).

    Returns:
        Dict[str, float] mapping topology name → fraction of total time.
    """
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

        # Delta between row i and row i+1: active topology is to_topology of row i
        # (equivalently from_topology of row i+1 once switching occurs).
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
    """Compute per-task oracle gap vs leave-one-out oracle.

    Formula (experiment_plan.md §4):
        oracle_gap_loo[task_id] = quality_oracle_loo[task_id] - quality_<router>[task_id]

    For each unique ``task_id`` in ``runs_df``:
      1. Look up the oracle topology via ``oracle_table.by_task_id[task_id]``
         (strict per-task_id filter — reviewer N2 requirement).
      2. Compute mean ``quality_score`` of runs with that topology for this task_id.
      3. Compute mean ``quality_score`` of all runs for this task_id (router quality).
      4. Gap = oracle_quality - router_quality.

    Rows with null ``quality_score`` are excluded from means.
    Tasks where the oracle topology is not present in runs_df get NaN.

    Args:
        runs_df:      DataFrame with columns ``task_id``, ``topology``,
                      ``quality_score``.
        oracle_table: OracleTable with ``by_task_id`` mapping.

    Returns:
        pd.Series indexed by ``task_id``, values are float (NaN where undetermined).
    """
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

        # Oracle quality: mean quality of runs that used the oracle topology for this task
        oracle_runs = group[group["topology"] == oracle_topo]["quality_score"].dropna()
        if oracle_runs.empty:
            results[task_id_str] = float("nan")
            continue

        oracle_quality = float(oracle_runs.mean())

        # Router quality: mean quality across all runs for this task
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
    """Compute per-task oracle gap vs manual oracle.

    Symmetric to ``compute_oracle_gap_loo`` — same algorithm, using
    ``manual_oracle_table`` instead of the LOO table.

    Formula:
        oracle_gap_manual[task_id] = quality_oracle_manual[task_id] - quality_<router>[task_id]

    Tasks where the manual oracle has a TODO/None entry get NaN (documented
    behaviour — conf/oracle/type_level_manual.yaml may contain stubs until M14+).

    Args:
        runs_df:             DataFrame with columns ``task_id``, ``topology``,
                             ``quality_score``.
        manual_oracle_table: OracleTable with ``by_task_id`` (or ``by_task_type``)
                             mapping for the manual oracle.

    Returns:
        pd.Series indexed by ``task_id``, values are float (NaN where undetermined).
    """
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
            # Try by_task_type fallback
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
    """Compute the fraction of tasks where Adaptive is worse than best static.

    Formula (experiment_plan.md §4):
        hurt_rate = |tasks where mean(adaptive_quality) < best_static_quality| / |tasks|

    The caller pre-computes ``best_static_df`` (e.g., groupby topology, pick
    top-1 per task_id).

    ``best_static_df`` must have columns:
        task_id, quality_score   (best static quality for that task)

    ``runs_df`` must have columns:
        task_id, quality_score   (adaptive router quality — all runs for each task)

    For each task_id present in ``runs_df``:
      - Adaptive quality = mean of quality_score for that task_id.
      - Best static quality = quality_score from ``best_static_df`` for that task_id.
      - Hurt = adaptive_quality < best_static_quality.

    Tasks absent from ``best_static_df`` are skipped (not counted as hurt).
    Returns 0.0 if ``runs_df`` is empty or no tasks can be evaluated.

    Args:
        runs_df:         DataFrame with adaptive run quality per task.
        best_static_df:  DataFrame with best static quality per task (one row per task_id).

    Returns:
        Float in [0.0, 1.0].
    """
    if runs_df.empty or best_static_df.empty:
        return 0.0

    adaptive = runs_df.copy()
    adaptive["quality_score"] = pd.to_numeric(adaptive["quality_score"], errors="coerce")
    static = best_static_df.copy()
    static["quality_score"] = pd.to_numeric(static["quality_score"], errors="coerce")

    # Build lookup: task_id → best static quality
    static_lookup = static.set_index("task_id")["quality_score"].to_dict()

    adaptive_mean = (
        adaptive.groupby("task_id")["quality_score"]
        .mean()
        .dropna()
    )

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
    """Compute the number of actual topology switches per run.

    A switch is a row where ``to_topology != from_topology`` AND
    ``from_topology`` is not null.

    Initial rows (``from_topology`` is null / NaN) are never counted as switches.

    Returns a pd.Series indexed by ``run_id`` with integer switch counts.
    An empty DataFrame returns an empty Series.

    Args:
        transitions_df: DataFrame with columns ``run_id``, ``from_topology``,
                        ``to_topology``.

    Returns:
        pd.Series[int] indexed by ``run_id``.
    """
    if transitions_df.empty:
        return pd.Series(dtype=int)

    df = transitions_df.copy()
    # Mark initial rows (from_topology is null)
    has_from = df["from_topology"].notna() & (df["from_topology"] != "")
    is_switch = has_from & (df["to_topology"] != df["from_topology"])

    switch_counts = (
        df[is_switch]
        .groupby("run_id")
        .size()
    )

    # Ensure all run_ids in the df appear (even if they have 0 switches)
    all_run_ids = df["run_id"].unique()
    switch_counts = switch_counts.reindex(all_run_ids, fill_value=0)
    switch_counts.name = "topology_switch_count"

    return switch_counts.astype(int)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _infer_task_type_from_group(group: pd.DataFrame, task_id: str) -> str:
    """Infer task_type from group DataFrame or task_id prefix.

    Looks for a ``task_type`` column first; falls back to prefix-based inference.
    """
    if "task_type" in group.columns:
        non_null = group["task_type"].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])

    # Prefix-based fallback (mirrors oracle._infer_task_type)
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
