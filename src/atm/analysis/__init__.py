"""Analysis layer — oracle labels pipeline, data loaders, metrics, and plots."""

from atm.analysis.loaders import (
    load_experiment,
    load_human_interactions,
    load_llm_calls,
    load_llm_calls_for_experiment,
    load_phases,
    load_runs,
    load_topology_transitions,
)
from atm.analysis.metrics import (
    compute_guard_override_rate,
    compute_hurt_rate,
    compute_oracle_gap_loo,
    compute_oracle_gap_manual,
    compute_router_cost_share,
    compute_time_per_topology,
    compute_topology_switch_counts,
)
from atm.analysis.oracle import (
    OracleTable,
    build_leave_one_out_oracle,
    build_loo_from_rows,
    load_oracle_table,
)
from atm.analysis.plots import (
    plot_cognitive_load_boxplot,
    plot_guard_override_rate,
    plot_oracle_gap_loo,
    plot_pareto,
    plot_phase_timeline,
    plot_router_cost_share,
    plot_time_per_topology,
    plot_topology_task_heatmap,
    plot_transition_timeline_quality,
)

__all__ = [
    "OracleTable",
    "build_leave_one_out_oracle",
    "build_loo_from_rows",
    "compute_guard_override_rate",
    "compute_hurt_rate",
    "compute_oracle_gap_loo",
    "compute_oracle_gap_manual",
    "compute_router_cost_share",
    "compute_time_per_topology",
    "compute_topology_switch_counts",
    "load_experiment",
    "load_human_interactions",
    "load_llm_calls",
    "load_llm_calls_for_experiment",
    "load_oracle_table",
    "load_phases",
    "load_runs",
    "load_topology_transitions",
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
