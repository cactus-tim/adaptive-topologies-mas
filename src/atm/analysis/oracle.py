"""Oracle labels pipeline for M8.7 — OracleTable producer/consumer.

Public surface:
    OracleTable         — Pydantic v2 frozen model
    build_loo_from_rows — pure LOO aggregation helper
    load_oracle_table   — sync JSON/YAML loader
    build_leave_one_out_oracle — async Postgres-backed builder
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

if TYPE_CHECKING:
    import matplotlib.figure

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_TOPOLOGIES: frozenset[str] = frozenset(
    {"linear", "supervisor", "mesh", "debate", "hierarchical"}
)

_DEFAULT_TOPOLOGY: str = "linear"

# Phases emitted by the router (must match arch.md §3.4)
_PHASES: tuple[str, ...] = ("planning", "execution", "verification")

# Task-type inference rules — longest prefix first
_TASK_TYPE_PREFIX_MAP: list[tuple[str, str]] = [
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
    # CommonGen loader emits `commongen/{idx}` (real task IDs) — type=creative
    ("commongen/", "creative"),
    ("CommonGen/", "creative"),
    # DABench loader emits `dabench/{qid}` — type=decision
    ("dabench/", "decision"),
    ("DABench/", "decision"),
]


# ---------------------------------------------------------------------------
# OracleTable — Pydantic v2 frozen model
# ---------------------------------------------------------------------------


@pydantic_dataclass(config=ConfigDict(frozen=True, populate_by_name=True))
class OracleTable:
    """Flat oracle table: task_type/task_id → best topology.

    Fields
    ------
    by_task_type : dict[str, str]
        Maps task_type → best topology (global LOO aggregate).
    by_task_id : dict[str, str]
        Maps task_id → best topology (per-task LOO winner).
    default_topology : str
        Fallback topology when neither task_id nor task_type is found.
        Stored under the ``_default`` key in the JSON representation.
    """

    by_task_type: dict[str, str] = Field(default_factory=dict)
    by_task_id: dict[str, str] = Field(default_factory=dict)
    default_topology: str = Field(
        default=_DEFAULT_TOPOLOGY,
        alias="_default",
        serialization_alias="_default",
    )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, *, task_id: str = "", task_type: str = "") -> str:
        """Return best topology with priority: by_task_id > by_task_type > default.

        Args:
            task_id:   Specific task identifier.
            task_type: Task category (programming, reasoning, …).

        Returns:
            Topology name string.
        """
        if task_id and task_id in self.by_task_id:
            return self.by_task_id[task_id]
        if task_type and task_type in self.by_task_type:
            return self.by_task_type[task_type]
        return self.default_topology

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (round-trip via from_dict)."""
        return {
            "by_task_type": dict(self.by_task_type),
            "by_task_id": dict(self.by_task_id),
            "_default": self.default_topology,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OracleTable:
        """Deserialize from a plain dict (produced by to_dict)."""
        return cls(
            by_task_type=dict(data.get("by_task_type", {})),
            by_task_id=dict(data.get("by_task_id", {})),
            **{"_default": data.get("_default", _DEFAULT_TOPOLOGY)},
        )

    def to_router_dict(self) -> dict[str, Any]:
        """Convert to the nested phase-keyed format consumed by OracleTopologyRouter.

        The router expects:
        {
            "by_task_type": {"<type>": {"planning": "<topo>", "execution": "<topo>", ...}},
            "by_task_id":   {"<id>":   {"planning": "<topo>", "execution": "<topo>", ...}},
            "_default": "<topo>"
        }

        Because OracleTable stores a single topology per task (same for all phases),
        this expands each flat mapping into the phase-keyed dict.
        """
        by_task_type_nested: dict[str, Any] = {
            t: dict.fromkeys(_PHASES, topo) for t, topo in self.by_task_type.items()
        }
        by_task_id_nested: dict[str, Any] = {
            tid: dict.fromkeys(_PHASES, topo) for tid, topo in self.by_task_id.items()
        }
        return {
            "by_task_type": by_task_type_nested,
            "by_task_id": by_task_id_nested,
            "_default": self.default_topology,
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Alias for to_router_dict — returns the shape consumed by OracleTopologyRouter."""
        return self.to_router_dict()

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> OracleTable:
        """Deserialize from the nested router format (inverse of to_json_dict).

        Unknown phase keys are silently dropped (DEBUG log).
        """
        by_task_type: dict[str, str] = {}
        for t, phase_map in data.get("by_task_type", {}).items():
            if not isinstance(phase_map, dict):
                _log.debug("from_json_dict: skipping non-dict phase_map for task_type=%r", t)
                continue
            # Pick the first known phase's topology as the canonical value
            topo = None
            for phase in _PHASES:
                if phase in phase_map:
                    topo = phase_map[phase]
                    break
            if topo is not None:
                by_task_type[t] = str(topo)
            else:
                _log.debug("from_json_dict: no known phase keys for task_type=%r, dropping", t)

        by_task_id: dict[str, str] = {}
        for tid, phase_map in data.get("by_task_id", {}).items():
            if not isinstance(phase_map, dict):
                _log.debug("from_json_dict: skipping non-dict phase_map for task_id=%r", tid)
                continue
            topo = None
            for phase in _PHASES:
                if phase in phase_map:
                    topo = phase_map[phase]
                    break
            if topo is not None:
                by_task_id[tid] = str(topo)
            else:
                _log.debug("from_json_dict: no known phase keys for task_id=%r, dropping", tid)

        return cls(
            by_task_type=by_task_type,
            by_task_id=by_task_id,
            **{"_default": str(data.get("_default", _DEFAULT_TOPOLOGY))},
        )


# ---------------------------------------------------------------------------
# Task-type inference
# ---------------------------------------------------------------------------


def _infer_task_type(task_id: str) -> str:
    """Infer task_type from task_id prefix.

    Examples:
        HumanEval/0  → programming
        GSM8K/42     → reasoning

    Returns:
        Inferred task_type string, or "unknown" if no prefix matches.
    """
    for prefix, task_type in _TASK_TYPE_PREFIX_MAP:
        if task_id.startswith(prefix):
            return task_type
    return "unknown"


# ---------------------------------------------------------------------------
# LOO algorithm — pure helper
# ---------------------------------------------------------------------------


def build_loo_from_rows(rows: list[dict[str, Any]]) -> OracleTable:
    """Build an OracleTable as a per-task TOP-1 upper-bound table.

    Despite the historical function name ("loo"), this builder now picks the
    empirically best topology per task_id (and per task_type) — i.e. the
    ceiling assuming perfect topology knowledge. This matches the RQ2
    "oracle = upper bound" semantics used in arch.md and the e3 acceptance
    criteria. The earlier leave-one-out semantics measured generalisation
    from sibling tasks and conflated with predictor performance.

    Each row must have at minimum:
        task_id       : str
        task_type     : str  (used for grouping; inferred from task_id if missing)
        topology      : str
        quality_score : float

    Two-pass algorithm
    ------------------
    Pass 1 (by_task_type):
        Group all rows by task_type.
        For each (task_type, topology) compute mean quality_score (primary)
        and mean budget_spent_usd (secondary tiebreak, if present).
        Pick topology with highest mean quality_score.

    Pass 2 (by_task_id) — TOP-1 PER TASK:
        For each target task_id:
            Take only rows whose task_id == target.
            Aggregate quality_score per topology.
            Pick topology with highest mean quality_score.
            If no rows for that task_id → fall back to _default topology.

    Returns:
        OracleTable with by_task_type, by_task_id, and default_topology set.
    """
    if not rows:
        return OracleTable(
            by_task_type={},
            by_task_id={},
            **{"_default": _DEFAULT_TOPOLOGY},
        )

    # Normalize rows: ensure task_type is present
    normalized: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        if not r.get("task_type"):
            r["task_type"] = _infer_task_type(str(r.get("task_id", "")))
        normalized.append(r)

    # ------------------------------------------------------------------
    # Pass 1: global by_task_type (all rows, no exclusion)
    # ------------------------------------------------------------------
    # type_topo_scores[task_type][topology] = list[float]
    type_topo_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in normalized:
        tt = str(row.get("task_type", "unknown"))
        topo = str(row.get("topology", ""))
        score = float(row.get("quality_score") or 0.0)
        if topo in _VALID_TOPOLOGIES:
            type_topo_scores[tt][topo].append(score)

    by_task_type: dict[str, str] = {}
    for tt, topo_scores in type_topo_scores.items():
        best = _pick_best_topology(topo_scores)
        if best is not None:
            by_task_type[tt] = best

    # ------------------------------------------------------------------
    # Pass 2: per-task_id TOP-1 (upper-bound oracle)
    # ------------------------------------------------------------------
    all_task_ids = sorted({str(row.get("task_id", "")) for row in normalized if row.get("task_id")})

    by_task_id: dict[str, str] = {}
    for target_id in all_task_ids:
        # Take ONLY rows for this exact task_id (no LOO exclusion)
        target_rows: list[dict[str, Any]] = [
            r
            for r in normalized
            if str(r.get("task_id")) == target_id
            and str(r.get("topology", "")) in _VALID_TOPOLOGIES
        ]

        if not target_rows:
            # No data for this task → fall back to default
            by_task_id[target_id] = _DEFAULT_TOPOLOGY
            continue

        # Aggregate scores by topology over rows of THIS task
        task_topo_scores: dict[str, list[float]] = defaultdict(list)
        for r in target_rows:
            topo = str(r.get("topology", ""))
            score = float(r.get("quality_score") or 0.0)
            task_topo_scores[topo].append(score)

        best = _pick_best_topology(task_topo_scores)
        by_task_id[target_id] = best if best is not None else _DEFAULT_TOPOLOGY

    return OracleTable(
        by_task_type=by_task_type,
        by_task_id=by_task_id,
        **{"_default": _DEFAULT_TOPOLOGY},
    )


def _pick_best_topology(
    topo_scores: dict[str, list[float]],
) -> str | None:
    """Return topology with highest mean quality_score; None if dict is empty.

    Ties on quality are broken by topology name (lexicographic ascending) so that
    oracle generation is deterministic across runs and Python dict orderings.
    """
    if not topo_scores:
        return None

    ranked: list[tuple[float, str]] = []
    for topo, scores in topo_scores.items():
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        ranked.append((mean, topo))

    if not ranked:
        return None

    # Sort by (-mean, topo) — highest mean first, lex name as tiebreak.
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][1]


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------


def load_oracle_table(path: Path | str) -> OracleTable:
    """Load an OracleTable from a JSON or YAML file.

    The file format must match the router dict shape (nested phase maps)
    as produced by OracleTable.to_json_dict().

    Args:
        path: Path to a .json or .yaml / .yml file.

    Returns:
        OracleTable deserialized via from_json_dict.

    Raises:
        ValueError: If file suffix is not .json, .yaml, or .yml.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        import json

        with path.open(encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
    elif suffix in {".yaml", ".yml"}:
        import yaml

        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    else:
        raise ValueError(
            f"load_oracle_table: unsupported file suffix {suffix!r} for {path}. "
            "Expected .json, .yaml, or .yml."
        )

    return OracleTable.from_json_dict(data)


# ---------------------------------------------------------------------------
# Async Postgres-backed builder
# ---------------------------------------------------------------------------


async def build_leave_one_out_oracle(
    exp_id: str,
    *,
    session_factory: Any,
) -> OracleTable:
    """Build OracleTable (per-task TOP-1 upper bound) over all runs for an exp.

    Despite the historical "leave_one_out" function name (kept for API
    stability), the underlying builder now picks the empirically best
    topology per task_id — the upper-bound semantics required by RQ2.

    Queries the ``runs`` table (SQLAlchemy 2.x async session) for all rows
    matching ``exp_id``, extracts relevant columns as dicts, and delegates
    to build_loo_from_rows.

    Args:
        exp_id:          Experiment UUID string (or UUID object).
        session_factory: Async SQLAlchemy sessionmaker / async_sessionmaker.

    Returns:
        OracleTable populated by LOO aggregation.
    """
    from sqlalchemy import select

    from atm.storage.models import Run

    rows: list[dict[str, Any]] = []

    async with session_factory() as session:
        result = await session.execute(select(Run).where(Run.exp_id == exp_id))
        run_objects: list[Run] = list(result.scalars().all())

    for run in run_objects:
        rows.append(
            {
                "task_id": run.task_id,
                "task_type": _infer_task_type(run.task_id),
                "topology": run.topology,
                "quality_score": float(run.quality_score) if run.quality_score is not None else 0.0,
                "budget_spent_usd": float(run.budget_spent_usd)
                if run.budget_spent_usd is not None
                else 0.0,
            }
        )

    return build_loo_from_rows(rows)


# ---------------------------------------------------------------------------
# Plot — oracle vs router (TYPE_CHECKING guard: matplotlib imported lazily)
# ---------------------------------------------------------------------------


def plot_oracle_vs_router(
    runs_df: pd.DataFrame,
    oracle_table: OracleTable,
    *,
    router_col: str = "topology",
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot oracle topology recommendation vs actual router choice per task.

    For each run (identified by ``task_id``), computes the oracle-recommended
    topology via ``oracle_table.lookup(task_id=...)`` and compares it to the
    actual topology chosen by the router (``router_col`` column).

    Produces a grouped bar chart: for each unique topology, shows how often
    the oracle recommended it vs how often the router actually chose it.

    Args:
        runs_df:      DataFrame with at minimum ``task_id``, ``router_col``, and
                      optionally ``quality_score`` columns.
        oracle_table: Pre-built OracleTable mapping task_id → oracle topology.
        router_col:   Column in ``runs_df`` holding the router's topology choice.
                      Default: ``"topology"``.
        figsize:      Figure (width, height) in inches.

    Returns:
        matplotlib Figure with one Axes (grouped bar chart).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # no-op if already set; safe to call again

    fig, ax = plt.subplots(figsize=figsize)

    # Validate required columns
    required_cols = {router_col}
    if runs_df.empty or not required_cols.issubset(runs_df.columns):
        ax.set_title("Oracle vs Router Topology Choice (no data)")
        ax.set_xlabel("Topology")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    df = runs_df.copy()

    # Compute oracle recommendation per row (using task_id if available)
    if "task_id" in df.columns:
        df["oracle_topology"] = df["task_id"].apply(
            lambda tid: oracle_table.lookup(task_id=str(tid))
        )
    else:
        # No task_id: use the table's default for all rows
        default_topo = oracle_table.default_topology
        df["oracle_topology"] = default_topo

    # Aggregate counts
    router_counts = df[router_col].value_counts().rename("router")
    oracle_counts = df["oracle_topology"].value_counts().rename("oracle")

    counts_df = pd.concat([router_counts, oracle_counts], axis=1).fillna(0).astype(int)
    counts_df = counts_df.sort_index()

    if counts_df.empty:
        ax.set_title("Oracle vs Router Topology Choice (no data)")
        ax.set_xlabel("Topology")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    topologies = counts_df.index.tolist()
    n = len(topologies)
    x = list(range(n))
    bar_width = 0.35

    router_vals = counts_df.get("router", pd.Series(0, index=counts_df.index)).tolist()
    oracle_vals = counts_df.get("oracle", pd.Series(0, index=counts_df.index)).tolist()

    ax.bar(
        [xi - bar_width / 2 for xi in x],
        router_vals,
        width=bar_width,
        label="Router",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        [xi + bar_width / 2 for xi in x],
        oracle_vals,
        width=bar_width,
        label="Oracle",
        color="coral",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(topologies, rotation=30, ha="right")
    ax.set_xlabel("Topology")
    ax.set_ylabel("Count")
    ax.set_title("Oracle vs Router Topology Choice")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig
