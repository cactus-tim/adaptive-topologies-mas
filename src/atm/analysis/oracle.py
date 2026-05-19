"""Oracle labels pipeline for M8.7 — OracleTable producer/consumer."""

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

_VALID_TOPOLOGIES: frozenset[str] = frozenset(
    {"linear", "supervisor", "mesh", "debate", "hierarchical"}
)

_DEFAULT_TOPOLOGY: str = "linear"

_PHASES: tuple[str, ...] = ("planning", "execution", "verification")

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
    ("commongen/", "creative"),
    ("CommonGen/", "creative"),
    ("dabench/", "decision"),
    ("DABench/", "decision"),
]


@pydantic_dataclass(config=ConfigDict(frozen=True, populate_by_name=True))
class OracleTable:
    """Flat oracle table mapping task_type/task_id → best topology."""

    by_task_type: dict[str, str] = Field(default_factory=dict)
    by_task_id: dict[str, str] = Field(default_factory=dict)
    default_topology: str = Field(
        default=_DEFAULT_TOPOLOGY,
        alias="_default",
        serialization_alias="_default",
    )

    def lookup(self, *, task_id: str = "", task_type: str = "") -> str:
        """Return best topology: by_task_id > by_task_type > default_topology."""
        if task_id and task_id in self.by_task_id:
            return self.by_task_id[task_id]
        if task_type and task_type in self.by_task_type:
            return self.by_task_type[task_type]
        return self.default_topology

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
        """Convert to the nested phase-keyed format consumed by OracleTopologyRouter."""
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
        """Deserialize from the nested router format (inverse of to_json_dict); drops unknown phases."""
        by_task_type: dict[str, str] = {}
        for t, phase_map in data.get("by_task_type", {}).items():
            if not isinstance(phase_map, dict):
                _log.debug("from_json_dict: skipping non-dict phase_map for task_type=%r", t)
                continue
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


def _infer_task_type(task_id: str) -> str:
    """Infer task_type from task_id prefix; returns "unknown" if no prefix matches."""
    for prefix, task_type in _TASK_TYPE_PREFIX_MAP:
        if task_id.startswith(prefix):
            return task_type
    return "unknown"


def build_loo_from_rows(rows: list[dict[str, Any]]) -> OracleTable:
    """Build OracleTable: best topology per task_id (and per task_type) by mean quality_score."""
    if not rows:
        return OracleTable(
            by_task_type={},
            by_task_id={},
            **{"_default": _DEFAULT_TOPOLOGY},
        )

    normalized: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        if not r.get("task_type"):
            r["task_type"] = _infer_task_type(str(r.get("task_id", "")))
        normalized.append(r)

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
            by_task_id[target_id] = _DEFAULT_TOPOLOGY
            continue

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

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked[0][1]


def load_oracle_table(path: Path | str) -> OracleTable:
    """Load OracleTable from a .json/.yaml/.yml file; raises ValueError on unsupported suffix."""
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


async def build_leave_one_out_oracle(
    exp_id: str,
    *,
    session_factory: Any,
) -> OracleTable:
    """Build OracleTable (per-task TOP-1 upper bound) over all runs for an experiment."""
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


def plot_oracle_vs_router(
    runs_df: pd.DataFrame,
    oracle_table: OracleTable,
    *,
    router_col: str = "topology",
    figsize: tuple[float, float] = (8, 5),
) -> matplotlib.figure.Figure:
    """Plot grouped bar chart: oracle topology recommendation vs actual router choice."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # no-op if already set; safe to call again

    fig, ax = plt.subplots(figsize=figsize)

    required_cols = {router_col}
    if runs_df.empty or not required_cols.issubset(runs_df.columns):
        ax.set_title("Oracle vs Router Topology Choice (no data)")
        ax.set_xlabel("Topology")
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    df = runs_df.copy()

    if "task_id" in df.columns:
        df["oracle_topology"] = df["task_id"].apply(
            lambda tid: oracle_table.lookup(task_id=str(tid))
        )
    else:
        default_topo = oracle_table.default_topology
        df["oracle_topology"] = default_topo

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
