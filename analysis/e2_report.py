"""E2 results report — print + JSON artifact.

Aggregates ``runs`` rows for an E2 experiment into a multi-axis table:

  - status totals (completed / failed / running) + cost
  - by task_id, by human_role, by topology
  - role x task matrix (avg quality)
  - failures broken down by topology+task and role

Emits a JSON artifact at ``analysis/e2_results.json`` that captures the
per-role winners per task — consumed by E3 (RQ2 baseline) and downstream
analysis.

Usage::

    set -a && source .env && set +a && export ATM_PG_DSN="$PG_DSN"
    uv run python -m analysis.e2_report --exp-id <UUID>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_DEFAULT_OUT = Path("analysis/e2_results.json")

_ROLES = ("coordinator", "reviewer", "judge", "peer", "monitor")
_TASKS = ("humaneval", "gsm8k", "commongen", "dabench")


async def _run(exp_id: str, out_path: Path) -> int:
    pg_dsn = os.environ.get("PG_DSN") or os.environ.get("ATM_PG_DSN")
    if not pg_dsn:
        sys.stderr.write("PG_DSN / ATM_PG_DSN not set.\n")
        return 2

    engine = create_async_engine(pg_dsn)
    try:
        async with engine.connect() as conn:
            print("=" * 70)
            print(f"E2 REPORT — exp_id={exp_id}")
            print("=" * 70)

            r = (await conn.execute(text("""
                SELECT MIN(started_at) AS first_at, MAX(finished_at) AS last_at,
                       EXTRACT(EPOCH FROM (MAX(finished_at) - MIN(started_at))) AS wall_s
                FROM runs WHERE exp_id = :e
            """), {"e": exp_id})).mappings().one()
            wall_h = float(r["wall_s"]) / 3600.0 if r["wall_s"] is not None else None
            print(f"\nWALL TIME: {wall_h:.2f}h  ({r['first_at']} → {r['last_at']})")

            r = (await conn.execute(text("""
                SELECT status, COUNT(*) AS n,
                       AVG(quality_score) AS q,
                       SUM(budget_spent_usd) AS spent
                FROM runs WHERE exp_id = :e GROUP BY status
            """), {"e": exp_id})).mappings().all()
            print("\nSTATUS TOTALS:")
            status_totals: dict[str, dict] = {}
            for row in r:
                q = None if row["q"] is None else float(row["q"])
                cost = float(row["spent"] or 0)
                qstr = "NA" if q is None else f"{q:.3f}"
                print(f"  {row['status']:14} n={row['n']:4d}  q={qstr}  ${cost:.2f}")
                status_totals[row["status"]] = {"n": int(row["n"]), "q": q, "cost": cost}

            print("\nBY TASK (completed):")
            by_task: dict[str, dict] = {}
            r = (await conn.execute(text("""
                SELECT task_id, COUNT(*) AS n, AVG(quality_score) AS q
                FROM runs WHERE exp_id = :e AND status='completed'
                GROUP BY task_id ORDER BY task_id
            """), {"e": exp_id})).mappings().all()
            for row in r:
                q = float(row["q"])
                print(f"  {row['task_id']:10}  n={row['n']:4d}  q={q:.3f}")
                by_task[row["task_id"]] = {"n": int(row["n"]), "q": round(q, 4)}

            print("\nBY ROLE (completed, sorted by q):")
            by_role: dict[str, dict] = {}
            r = (await conn.execute(text("""
                SELECT human_role, COUNT(*) AS n, AVG(quality_score) AS q
                FROM runs WHERE exp_id = :e AND status='completed'
                GROUP BY human_role ORDER BY q DESC
            """), {"e": exp_id})).mappings().all()
            for row in r:
                q = float(row["q"])
                print(f"  {row['human_role']:12}  n={row['n']:4d}  q={q:.3f}")
                by_role[row["human_role"]] = {"n": int(row["n"]), "q": round(q, 4)}

            print("\nBY TOPOLOGY (completed):")
            by_topology: dict[str, dict] = {}
            r = (await conn.execute(text("""
                SELECT topology, COUNT(*) AS n, AVG(quality_score) AS q
                FROM runs WHERE exp_id = :e AND status='completed'
                GROUP BY topology ORDER BY q DESC
            """), {"e": exp_id})).mappings().all()
            for row in r:
                q = float(row["q"])
                print(f"  {row['topology']:14}  n={row['n']:4d}  q={q:.3f}")
                by_topology[row["topology"]] = {"n": int(row["n"]), "q": round(q, 4)}

            print("\nMATRIX: ROLE × TASK avg_q:")
            r = (await conn.execute(text("""
                SELECT human_role, task_id, AVG(quality_score) AS q, COUNT(*) AS n
                FROM runs WHERE exp_id = :e AND status='completed'
                GROUP BY human_role, task_id
            """), {"e": exp_id})).mappings().all()
            cell: dict[tuple[str, str], dict] = {}
            for row in r:
                cell[(row["human_role"], row["task_id"])] = {
                    "q": round(float(row["q"]), 4), "n": int(row["n"]),
                }
            print(f"  {'role':12} | {' '.join(f'{t:>10}' for t in _TASKS)}")
            print(f"  {'-'*12}-+-{'-'*(11*len(_TASKS))}")
            role_task_matrix: dict[str, dict[str, dict]] = {}
            for role in _ROLES:
                line = f"  {role:12} |"
                role_task_matrix[role] = {}
                for task in _TASKS:
                    entry = cell.get((role, task))
                    if entry is not None:
                        line += f"  {entry['q']:6.3f}({entry['n']:>2})"
                        role_task_matrix[role][task] = entry
                    else:
                        line += "     -    "
                print(line)

            print("\nPER-TASK WINNERS (role with max q):")
            best_role_per_task: dict[str, dict] = {}
            for task in _TASKS:
                best = max(
                    (cell.get((role, task)) for role in _ROLES if (role, task) in cell),
                    key=lambda e: e["q"] if e else -1,
                    default=None,
                )
                winner_role = next(
                    (role for role in _ROLES if cell.get((role, task)) is best),
                    None,
                )
                if winner_role is not None and best is not None:
                    print(f"  {task:10} → {winner_role:12}  q={best['q']:.3f}")
                    best_role_per_task[task] = {"role": winner_role, **best}

            print("\nFAILS by topology+task:")
            fails_by_pair: list[dict] = []
            r = (await conn.execute(text("""
                SELECT topology, task_id, COUNT(*) AS n
                FROM runs WHERE exp_id = :e AND status='failed'
                GROUP BY topology, task_id ORDER BY n DESC
            """), {"e": exp_id})).mappings().all()
            for row in r:
                print(f"  {row['topology']:14} {row['task_id']:10}  n={row['n']}")
                fails_by_pair.append(
                    {"topology": row["topology"], "task_id": row["task_id"], "n": int(row["n"])}
                )

            print("\nFAILS by role:")
            fails_by_role: dict[str, int] = {}
            r = (await conn.execute(text("""
                SELECT human_role, COUNT(*) AS n
                FROM runs WHERE exp_id = :e AND status='failed'
                GROUP BY human_role ORDER BY n DESC
            """), {"e": exp_id})).mappings().all()
            for row in r:
                print(f"  {row['human_role']:12}  n={row['n']}")
                fails_by_role[row["human_role"]] = int(row["n"])

            artifact = {
                "exp_id": exp_id,
                "wall_time_h": round(wall_h, 4) if wall_h is not None else None,
                "status_totals": status_totals,
                "by_task": by_task,
                "by_role": by_role,
                "by_topology": by_topology,
                "role_task_matrix": role_task_matrix,
                "best_role_per_task": best_role_per_task,
                "fails_by_topology_task": fails_by_pair,
                "fails_by_role": fails_by_role,
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nWrote {out_path}")
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True, help="E2 experiment UUID")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    return asyncio.run(_run(args.exp_id, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
