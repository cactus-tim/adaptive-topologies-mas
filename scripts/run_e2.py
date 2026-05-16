"""Launch the E2 grid with top-3 (per task) + bad-combo filtering.

``conf/experiments/e2_full.yaml`` emits the full 5x4x5x15x3 = 4500-cell
cross-product. This script:

  1. Loads ``analysis/e1_top3.json`` (produced by ``analysis/e1_top3.py``).
  2. Keeps only cells whose ``(task.name, topology.name)`` pair appears in
     the per-task top-3 winners.
  3. Drops three structurally non-sensical ``(topology, human.role)`` pairs:
       - debate x coordinator       — coordinator ≡ judge in debate
       - chain x peer               — no slot in a linear pipeline
       - hierarchical x peer        — peer breaks the hierarchy
  4. Hands the filtered ~2295 cells to ``run_grid`` with a live-progress
     callback matching the ``atm grid`` UX.

Usage::

    set -a && source .env && set +a && export ATM_PG_DSN="$PG_DSN"
    uv run python -m scripts.run_e2 [--config conf/experiments/e2_full.yaml]
                                    [--top3 analysis/e1_top3.json]
                                    [--parallelism 30]
                                    [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from atm.experiment.grid import GridProgress, run_grid
from atm.experiment.loader import load_grid_configs

_BAD_TOPO_ROLE: set[tuple[str, str]] = {
    ("debate", "coordinator"),
    ("chain", "peer"),
    ("hierarchical", "peer"),
}


def _load_top3(path: Path) -> dict[str, set[str]]:
    """Return {task_name: {topology, ...}} from the e1_top3.json artifact."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for task, entries in data["tasks"].items():
        out[str(task)] = {str(e["topology"]) for e in entries}
    return out


def _filter_cells(configs: list[Any], top3: dict[str, set[str]]) -> list[Any]:
    """Return the subset of *configs* that should actually run in E2."""
    kept: list[Any] = []
    for cfg in configs:
        task = cfg.task.name
        topo = cfg.topology.name
        role = cfg.human.role if cfg.human is not None else None

        allowed_topos = top3.get(task)
        if allowed_topos is None or topo not in allowed_topos:
            continue
        if role is not None and (topo, role) in _BAD_TOPO_ROLE:
            continue
        kept.append(cfg)
    return kept


def _summarize(configs: list[Any]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    for cfg in configs:
        role = cfg.human.role if cfg.human is not None else "none"
        key = (cfg.task.name, cfg.topology.name, role)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _on_progress(p: GridProgress) -> None:
    eta = f"{p.eta_s:.0f}s" if p.eta_s is not None else "?"
    msg = (
        f"\r[{p.done}/{p.total}] done={p.done - p.failed} failed={p.failed} "
        f"in_progress={p.in_progress} eta={eta}    "
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("conf/experiments/e2_full.yaml")
    )
    parser.add_argument("--top3", type=Path, default=Path("analysis/e1_top3.json"))
    parser.add_argument("--parallelism", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the filtered cell count + breakdown and exit without launching.",
    )
    args = parser.parse_args()

    if not args.top3.exists():
        sys.stderr.write(
            f"{args.top3} not found. Run analysis/e1_top3.py first.\n"
        )
        return 2

    top3 = _load_top3(args.top3)
    print("Top-3 per task (from e1_top3.json):")
    for task, topos in sorted(top3.items()):
        print(f"  {task}: {sorted(topos)}")

    configs = load_grid_configs(str(args.config))
    pre = len(configs)
    configs = _filter_cells(configs, top3)
    post = len(configs)
    print(f"\nCells: {pre} → {post} after top-3 + bad-combo filter")

    if not configs:
        sys.stderr.write("Empty grid after filtering.\n")
        return 3

    # Per-(task, topology) cell counts (sanity check; expect 5 roles x 15 sh
    # x 3 seed = 225 each, minus dropped bad combos).
    by_pair: dict[tuple[str, str], int] = {}
    for cfg in configs:
        by_pair[(cfg.task.name, cfg.topology.name)] = (
            by_pair.get((cfg.task.name, cfg.topology.name), 0) + 1
        )
    print("\nCells per (task, topology):")
    for (task, topo), n in sorted(by_pair.items()):
        print(f"  {task:10s} x {topo:14s}: {n}")

    if args.dry_run:
        print("\n--dry-run: not launching.")
        return 0

    parallelism = args.parallelism if args.parallelism is not None else (
        configs[0].grid.parallelism if configs[0].grid is not None else 30
    )
    print(f"\nLaunching run_grid (parallelism={parallelism}, fail_fast=False) …")

    result = asyncio.run(
        run_grid(
            configs,
            parallelism=parallelism,
            fail_fast=False,
            progress_callback=_on_progress,
        )
    )
    sys.stdout.write("\n")
    print(
        f"GridResult: exp_id={result.exp_id} total={result.total} "
        f"completed={result.completed} failed={result.failed} "
        f"budget_exceeded={result.budget_exceeded}"
    )

    failed_total = result.failed + result.budget_exceeded
    if result.completed == result.total:
        return 0
    if failed_total == result.total:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
