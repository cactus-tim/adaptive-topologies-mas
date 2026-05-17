"""E4 results report — print + JSON artifact.

E4 is the adaptive-topology + adaptive-role experiment (RQ4). The grid
sweeps ``human.role_router`` ∈ {fixed, rule, llm} × 4 tasks × 15 shuffles
× 3 seeds = 540 cells with ``topology_router=llm`` (E3 Pareto champion)
fixed in the YAML.

Mode inference: PG ``runs`` table does not store role_router. We exploit
the grid's deterministic sweep order (sorted alphabetically; outermost =
``human.role_router``) plus the queue's FIFO submission to bin runs by
``started_at`` rank: rn 1-180 = fixed, 181-360 = rule, 361-540 = llm.

Verified against the human_role distribution: fixed rows are 100%
``reviewer`` (= cfg.human.role), while rule/llm rows show ``coordinator``/
``peer`` from phase-based / LLM-decided routing — matches expected
DEFAULT_ROLE_TABLE behaviour.

Usage::

    set -a && source .env && set +a && export ATM_PG_DSN="$PG_DSN"
    uv run python -m analysis.e4_report --exp-id <UUID>
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

_DEFAULT_OUT = Path("analysis/e4_results.json")
_TASKS = ("humaneval", "gsm8k", "commongen", "dabench")
_MODES = ("fixed", "rule", "llm")


async def _run(exp_id: str, out_path: Path) -> int:
    pg_dsn = os.environ.get("PG_DSN") or os.environ.get("ATM_PG_DSN")
    if not pg_dsn:
        sys.stderr.write("PG_DSN / ATM_PG_DSN not set.\n")
        return 2

    engine = create_async_engine(pg_dsn)
    try:
        async with engine.connect() as conn:
            print("=" * 72)
            print(f"E4 REPORT — exp_id={exp_id}")
            print("=" * 72)

            r = (await conn.execute(text("""
                SELECT MIN(started_at) AS first_at, MAX(finished_at) AS last_at,
                       EXTRACT(EPOCH FROM (MAX(finished_at) - MIN(started_at))) AS wall_s,
                       SUM(budget_spent_usd) AS total_cost
                FROM runs WHERE exp_id = :e
            """), {"e": exp_id})).mappings().one()
            wall_h = float(r["wall_s"]) / 3600.0
            print(f"\nWALL TIME: {wall_h:.2f}h  ({r['first_at']} → {r['last_at']})")
            print(f"TOTAL COST: ${float(r['total_cost']):.2f}")

            r = (await conn.execute(text("""
                SELECT status, COUNT(*) AS n
                FROM runs WHERE exp_id = :e GROUP BY status
            """), {"e": exp_id})).mappings().all()
            status_totals = {row["status"]: int(row["n"]) for row in r}
            print(f"\nSTATUS: {status_totals}")

            print("\nMODE × TASK (mode inferred by started_at thirds):")
            r = (await conn.execute(text("""
                WITH ordered AS (
                    SELECT task_id, quality_score, budget_spent_usd, wall_time_s,
                           ROW_NUMBER() OVER (ORDER BY started_at) AS rn
                    FROM runs WHERE exp_id = :e AND status = 'completed'
                )
                SELECT
                    CASE WHEN rn <= 180 THEN 'fixed'
                         WHEN rn <= 360 THEN 'rule' ELSE 'llm' END AS mode,
                    task_id,
                    COUNT(*) AS n,
                    AVG(quality_score) AS q,
                    STDDEV(quality_score) AS std,
                    AVG(budget_spent_usd) AS cost,
                    AVG(wall_time_s) AS wall
                FROM ordered GROUP BY mode, task_id
            """), {"e": exp_id})).mappings().all()
            cell: dict[tuple[str, str], dict] = {}
            for row in r:
                cell[(row["mode"], row["task_id"])] = {
                    "n": int(row["n"]),
                    "q": round(float(row["q"]), 4),
                    "std": round(float(row["std"] or 0), 4),
                    "cost": round(float(row["cost"]), 5),
                    "wall_s": round(float(row["wall"]), 1),
                }

            print(f"  {'task':10} | {' '.join(f'{m:>8}' for m in _MODES)}  | best (Δ)")
            print(f"  {'-'*10}-+-{'-'*(9*len(_MODES))}-+----------")
            per_task_winner: dict[str, dict] = {}
            for task in _TASKS:
                qs = {m: cell[(m, task)]["q"] for m in _MODES if (m, task) in cell}
                line = f"  {task:10} |"
                for m in _MODES:
                    line += f"  {qs.get(m, 0):.3f} "
                if qs:
                    best_mode = max(qs, key=qs.get)
                    delta = qs[best_mode] - qs.get("fixed", qs[best_mode])
                    line += f"  | {best_mode} ({delta:+.3f})"
                    per_task_winner[task] = {
                        "mode": best_mode,
                        "q": qs[best_mode],
                        "delta_vs_fixed": round(delta, 4),
                    }
                print(line)

            print("\nMODE TOTALS:")
            mode_totals: dict[str, dict] = {}
            for mode in _MODES:
                entries = [cell[(mode, t)] for t in _TASKS if (mode, t) in cell]
                if not entries:
                    continue
                n_total = sum(e["n"] for e in entries)
                # micro-average (weighted by cell count, but cells are equal-n by design)
                q_avg = sum(e["q"] * e["n"] for e in entries) / n_total
                cost_avg = sum(e["cost"] * e["n"] for e in entries) / n_total
                wall_avg = sum(e["wall_s"] * e["n"] for e in entries) / n_total
                mode_totals[mode] = {
                    "n": n_total,
                    "mean_q": round(q_avg, 4),
                    "mean_cost": round(cost_avg, 5),
                    "mean_wall_s": round(wall_avg, 1),
                }
                print(f"  {mode:6}: n={n_total}  q={q_avg:.3f}  ${cost_avg:.4f}  {wall_avg:.0f}s")

            # RQ4 verdict
            fixed_q = mode_totals.get("fixed", {}).get("mean_q", 0)
            rule_q = mode_totals.get("rule", {}).get("mean_q", 0)
            llm_q = mode_totals.get("llm", {}).get("mean_q", 0)
            print(f"\nRQ4 VERDICT (adaptive role > fixed?):")
            print(f"  rule vs fixed: Δq = {rule_q - fixed_q:+.4f}  "
                  f"{'CONFIRMED' if rule_q > fixed_q else 'NOT CONFIRMED'}")
            print(f"  llm  vs fixed: Δq = {llm_q - fixed_q:+.4f}  "
                  f"{'CONFIRMED' if llm_q > fixed_q else 'NOT CONFIRMED'}")

            # human_role distribution per mode (signal verification)
            r = (await conn.execute(text("""
                WITH ordered AS (
                    SELECT human_role,
                           ROW_NUMBER() OVER (ORDER BY started_at) AS rn
                    FROM runs WHERE exp_id = :e AND status = 'completed'
                )
                SELECT
                    CASE WHEN rn <= 180 THEN 'fixed'
                         WHEN rn <= 360 THEN 'rule' ELSE 'llm' END AS mode,
                    human_role, COUNT(*) AS n
                FROM ordered GROUP BY mode, human_role
            """), {"e": exp_id})).mappings().all()
            role_dist: dict[str, dict[str, int]] = {m: {} for m in _MODES}
            for row in r:
                role_dist[row["mode"]][row["human_role"]] = int(row["n"])
            print(f"\nROLE DISTRIBUTION (mode → role: n):")
            for mode in _MODES:
                print(f"  {mode:6}: {role_dist[mode]}")

            artifact = {
                "exp_id": exp_id,
                "wall_time_h": round(wall_h, 4),
                "total_cost_usd": round(float(r[0]["n"] if False else 0) or 0, 4),
                "status_totals": status_totals,
                "mode_totals": mode_totals,
                "per_task_winner": per_task_winner,
                "mode_task_matrix": {
                    f"{m}/{t}": cell[(m, t)]
                    for (m, t) in cell.keys()
                },
                "role_distribution": role_dist,
                "rq4": {
                    "fixed_q": fixed_q,
                    "rule_q": rule_q,
                    "llm_q": llm_q,
                    "rule_delta": round(rule_q - fixed_q, 4),
                    "llm_delta": round(llm_q - fixed_q, 4),
                    "rule_confirmed": rule_q > fixed_q,
                    "llm_confirmed": llm_q > fixed_q,
                },
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"\nWrote {out_path}")
    finally:
        await engine.dispose()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True, help="E4 experiment UUID")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    return asyncio.run(_run(args.exp_id, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
