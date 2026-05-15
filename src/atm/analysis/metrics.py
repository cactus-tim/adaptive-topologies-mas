"""Derived metrics for RQ2 / G11 analysis — pure functions, no side effects.

All functions accept pre-loaded pandas DataFrames (produced by loaders.py)
and return scalar values or new DataFrames.

Public surface (to be implemented in Step 6):
    compute_guard_override_rate      — fraction of transitions that overrode the guard
    compute_router_cost_share        — fraction of total budget spent on routing calls
    compute_time_per_topology        — mean wall-time per topology (non-contiguous sums)
    compute_oracle_gap_loo           — per-task_id oracle vs. router quality delta
    compute_oracle_gap_manual        — per-task_type oracle vs. router quality delta
    compute_hurt_rate                — fraction of tasks where adaptive < best_static
    compute_topology_switch_counts   — switch frequency per (run_id, topology) pair
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from atm.analysis.oracle import OracleTable


def compute_guard_override_rate(transitions_df: pd.DataFrame) -> float:
    """Compute the fraction of topology transitions that overrode the guard.

    A transition is considered a guard override when the guard signal would have
    rejected the proposed topology but the transition happened anyway.

    Args:
        transitions_df: DataFrame from ``load_topology_transitions``.

    Returns:
        Float in [0.0, 1.0]. Returns NaN if ``transitions_df`` is empty.
    """
    raise NotImplementedError("compute_guard_override_rate — implemented in Step 6")


def compute_router_cost_share(
    runs_df: pd.DataFrame,
    llm_calls_df: pd.DataFrame,
) -> float:
    """Compute the fraction of total LLM budget spent on routing calls.

    Args:
        runs_df:      DataFrame from ``load_runs`` (contains ``budget_spent_usd``).
        llm_calls_df: DataFrame from ``load_llm_calls_for_experiment``.

    Returns:
        Float in [0.0, 1.0]. Returns NaN if total budget is zero.
    """
    raise NotImplementedError("compute_router_cost_share — implemented in Step 6")


def compute_time_per_topology(phases_df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """Compute mean wall-clock time spent in each topology across all runs.

    Non-contiguous time slices for the same topology within a run are summed
    before computing the cross-run mean. This avoids double-counting topologies
    that appear multiple times within a single run.

    Args:
        phases_df: DataFrame from ``load_phases`` with ``topology`` and
                   timing columns (``started_at`` / ``ended_at``).

    Returns:
        pd.Series indexed by topology name, values are mean seconds (float).
    """
    raise NotImplementedError("compute_time_per_topology — implemented in Step 6")


def compute_oracle_gap_loo(
    runs_df: pd.DataFrame,
    oracle: OracleTable,
) -> pd.DataFrame:
    """Compute per-task_id quality gap between the LOO oracle and the actual router.

    For each run, the oracle recommends the topology for that ``task_id`` (LOO).
    The gap is ``oracle_quality - actual_quality`` (positive = oracle is better).

    Filtering is per-``task_id``, not per-``task_type``.

    Args:
        runs_df: DataFrame from ``load_runs``.
        oracle:  OracleTable built via ``build_leave_one_out_oracle``.

    Returns:
        DataFrame with columns: ``task_id``, ``oracle_topology``,
        ``oracle_quality``, ``actual_quality``, ``gap``.
    """
    raise NotImplementedError("compute_oracle_gap_loo — implemented in Step 6")


def compute_oracle_gap_manual(
    runs_df: pd.DataFrame,
    oracle: OracleTable,
) -> pd.DataFrame:
    """Compute per-task_type quality gap between the manual oracle and the actual router.

    The manual oracle is read from ``conf/oracle/type_level_manual.yaml``.
    Tasks whose task_type is not in the manual oracle receive NaN gap values
    (documented behaviour until M14+ fills the YAML).

    Args:
        runs_df: DataFrame from ``load_runs``.
        oracle:  OracleTable loaded from the manual YAML file.

    Returns:
        DataFrame with columns: ``task_type``, ``oracle_topology``,
        ``oracle_quality``, ``actual_quality``, ``gap``.
    """
    raise NotImplementedError("compute_oracle_gap_manual — implemented in Step 6")


def compute_hurt_rate(
    runs_df: pd.DataFrame,
    best_static_df: pd.DataFrame,
) -> float:
    """Compute the fraction of tasks where the adaptive router under-performed the best static.

    The caller is responsible for pre-computing ``best_static_df`` (e.g., the single
    topology with highest mean quality_score per task_type across all static runs).

    Args:
        runs_df:        DataFrame from ``load_runs`` (adaptive runs only).
        best_static_df: DataFrame with columns ``task_id`` and ``best_static_quality``.

    Returns:
        Float in [0.0, 1.0]. Returns NaN if ``runs_df`` is empty.
    """
    raise NotImplementedError("compute_hurt_rate — implemented in Step 6")


def compute_topology_switch_counts(
    transitions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count topology switches per (run_id, from_topology, to_topology) triple.

    Args:
        transitions_df: DataFrame from ``load_topology_transitions``.

    Returns:
        DataFrame with columns: ``run_id``, ``from_topology``, ``to_topology``,
        ``count``. Sorted descending by ``count``.
    """
    raise NotImplementedError("compute_topology_switch_counts — implemented in Step 6")
