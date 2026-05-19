"""Compute top-3 topologies per task_type from E1 results.

Reads completed Run rows for the given exp_id, computes mean ``quality_score``
per (topology, task_type), and emits:

  1. A console-friendly table.
  2. A JSON artifact at ``analysis/e1_top3.json`` consumed by
     ``scripts/run_e2.py`` to filter E2 cells down to ~2160 from the
     full cross-product.

Usage::

    uv run python -m analysis.e1_top3 --exp-id <UUID>
    uv run python -m analysis.e1_top3 --exp-id <UUID> --top 3 --out analysis/e1_top3.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from atm.analysis.loaders import load_runs
from atm.storage.session import create_engine, create_session_factory

_DEFAULT_OUT = Path("analysis/e1_top3.json")


async def _run(exp_id: str, top_n: int, out_path: Path) -> int:
    pg_dsn = os.environ.get("PG_DSN") or os.environ.get("ATM_PG_DSN")
    if not pg_dsn:
        sys.stderr.write(
            "PG_DSN / ATM_PG_DSN not set. Source .env first: `set -a && source .env && set +a`\n"
        )
        return 2

    engine = create_engine(pg_dsn)
    session_factory = create_session_factory(engine)
    df = await load_runs(exp_id, session_factory=session_factory)
    await engine.dispose()

    if df.empty:
        sys.stderr.write(f"No runs for exp_id={exp_id}\n")
        return 3

    completed = df[df["status"] == "completed"].copy()
    if completed.empty:
        sys.stderr.write("No completed runs.\n")
        return 4

    matrix = (
        completed.groupby(["task_id", "topology"], as_index=False)
        .agg(avg_quality=("quality_score", "mean"), n=("quality_score", "size"))
        .sort_values(["task_id", "avg_quality"], ascending=[True, False])
    )

    print(f"Matrix (exp_id={exp_id}, completed={len(completed)}):")
    print(matrix.to_string(index=False))
    print()

    top_per_task: dict[str, list[dict[str, float | int | str]]] = {}
    for task_name, group in matrix.groupby("task_id"):
        top = group.head(top_n)
        top_per_task[str(task_name)] = [
            {
                "topology": str(row.topology),
                "avg_quality": round(float(row.avg_quality), 4),
                "n": int(row.n),
            }
            for row in top.itertuples()
        ]

    print(f"Top-{top_n} per task_type:")
    for task, entries in top_per_task.items():
        winners = ", ".join(f"{e['topology']} ({e['avg_quality']:.3f})" for e in entries)
        print(f"  {task}: {winners}")

    artifact = {
        "exp_id": exp_id,
        "top_n": top_n,
        "tasks": top_per_task,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True, help="E1 experiment UUID")
    parser.add_argument("--top", type=int, default=3, help="Top-N topologies per task")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    return asyncio.run(_run(args.exp_id, args.top, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
