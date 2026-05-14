# m12-estimate-status-cli — atm estimate, atm status, atm run polish

## What is this block

Read-only / pre-flight CLI cluster. Не трогает execution pipeline — только конфиг-walk и PG-read. Можно гнать параллельно с m12-grid-runner и m12-resume-replay.

Реализует две команды и небольшой polish:
1. **`atm estimate`** — dry-run по grid'у: проходит по всем cells, для каждой задачи оценивает ожидаемые токены (по preceding runs того же `(topology, task)` пары в PG; если их нет — heuristic из `EstimateCfg`), считает cost через `Pricing`. Печатает таблицу `topology × task × seed → est_cost_usd` + total.
2. **`atm status`** — query: `--exp-id <uuid>` или `--exp-name <name>` или (без аргументов) последний эксперимент. Печатает агрегированную сводку: total cells, done/failed/running, avg quality, total cost, ETA.
3. **`atm run` polish** — добавить `--estimate` флаг, который перед запуском печатает estimated cost и просит подтверждение если > `BudgetCfg.per_experiment_usd * 0.5`.

## In scope

1. **`src/atm/experiment/estimator.py` (NEW FILE):**

   ```python
   @dataclass
   class CellEstimate:
       topology: str
       task: str
       seed: int
       est_input_tokens: int
       est_output_tokens: int
       est_cost_usd: float
       source: Literal["historical_avg", "heuristic"]

   @dataclass
   class GridEstimate:
       cells: list[CellEstimate]
       total_cost_usd: float
       total_input_tokens: int
       total_output_tokens: int
       per_topology: dict[str, float]  # topology → total cost

   async def estimate_grid(
       configs: list[ExperimentConfig],
       session_factory,
       pricing: Pricing,
       cfg: EstimateCfg,
   ) -> GridEstimate: ...
   ```

   - **Historical-avg path** (`EstimateCfg.use_historical=True`):
     - For each `(topology, task)` pair in the grid, query:
       ```sql
       SELECT AVG(input_tokens), AVG(output_tokens)
       FROM llm_calls
       JOIN runs ON llm_calls.run_id = runs.id
       WHERE runs.topology = :topology AND runs.task_id = :task
         AND runs.status = 'completed'
       ```
       (Adjust column names per actual `llm_calls` schema — see codebase-map; `llm_calls` is Parquet, NOT PG. So this must read from parquet files or from `runs.budget_spent_usd` + `runs.iterations` as a proxy. **Implementation choice:** use `runs` aggregate columns — much simpler and PG-only.)
     - Revised query:
       ```sql
       SELECT AVG(budget_spent_usd), AVG(iterations)
       FROM runs WHERE topology=:t AND task_id=:task AND status='completed'
       ```
       Use `avg_cost_usd` directly as per-cell estimate.
   - **Heuristic fallback** when no historical data:
     - `est_input_tokens = cfg.calls_per_iter * cfg.heuristic_tokens_per_call * topology.max_iterations`.
     - Output tokens = input / 3 (typical ratio).
     - Cost = `pricing.estimate_cost(model_id, est_input_tokens, est_output_tokens)`.

2. **`atm estimate` CLI:**
   - Signature: `atm estimate --config <path> [+key=val ...]`.
   - Loads via `load_grid_configs` (single-cell configs also OK — returns 1-element list).
   - Calls `estimate_grid`, prints formatted table:
     ```
     topology    task        seed   tokens_in  tokens_out  cost_usd   source
     star        humaneval   42     9000       3000        $0.0234    historical_avg
     chain       humaneval   42     12000      4000        $0.0312    heuristic
     ...
     TOTAL                                                  $4.21
     ```
   - Exit 0 unless config error (exit 3).

3. **`atm status` CLI:**
   - Signature: `atm status [--exp-id <uuid> | --exp-name <name>] [--json]`.
   - SELECT from `experiments` + COUNT/AVG from `runs`:
     ```sql
     SELECT
       e.id, e.name, e.status, e.started_at, e.finished_at,
       COUNT(r.id) AS total,
       SUM(CASE WHEN r.status='completed' THEN 1 ELSE 0 END) AS completed,
       SUM(CASE WHEN r.status='failed' THEN 1 ELSE 0 END) AS failed,
       SUM(CASE WHEN r.status='running' THEN 1 ELSE 0 END) AS running,
       AVG(r.quality_score) AS avg_quality,
       SUM(r.budget_spent_usd) AS total_cost
     FROM experiments e LEFT JOIN runs r ON r.exp_id=e.id
     WHERE e.id=:eid OR e.name=:ename
     GROUP BY e.id
     ```
   - `--json` flag: emit JSON instead of formatted table (machine-readable for scripts/CI).
   - If no args: SELECT most recent experiment.

4. **`atm run --estimate` polish:**
   - Before invoking `run_one`, call `estimate_grid([cfg], ...)`.
   - If `est_cost_usd > cfg.budget.per_experiment_usd * 0.5` → `typer.confirm(...)` to abort unless `--yes`.
   - Print one-line estimate even without the threshold trigger.

5. **Tests:**
   - `tests/unit/experiment/test_estimator_heuristic.py`: synthetic configs → heuristic path → assert calc matches `pricing.estimate_cost`.
   - `tests/integration/experiment/test_estimator_historical.py` (PG): seed a couple completed `runs` rows; assert estimator picks historical avg.
   - `tests/integration/experiment/test_status_cli.py` (PG): seed experiment + runs; invoke `atm status --exp-id <uuid> --json`; parse output; assert counts.
   - `tests/unit/experiment/test_run_estimate_flag.py`: invoke `atm run --estimate --yes` with budget-blowing cfg; ensure typer.confirm path triggers.

## Out of scope

- `atm grid --estimate` pre-flight (using this block's estimator inside grid) — m12-grid-integration.
- LLM-call-level token forecasting (would need to actually run a planner pass) — heuristic + historical avg is enough for M12.
- Cost tracking dashboards / plots — M13.

## Inputs / preconditions

- m12-config-schema merged: needs `EstimateCfg`, `load_grid_configs`.
- Existing `Pricing` class — `src/atm/llm/pricing.py` (already has `estimate_cost`).
- PG schema with `runs.budget_spent_usd`, `runs.topology`, `runs.task_id`, `runs.iterations`, `runs.status`, `runs.quality_score`.

## Outputs / contract

```python
# Public API
from atm.experiment.estimator import estimate_grid, GridEstimate, CellEstimate

# CLI
$ atm estimate --config exp1.yaml
$ atm status --exp-id <uuid>
$ atm status --exp-name exp1 --json
$ atm run --config exp1.yaml --estimate [--yes]
```

`estimate_grid` is the function m12-grid-integration imports for the `atm grid --estimate` flow.

## References

- `/home/cactustim/agents/feat/m12/arch/PLAN.md` §M12 line 656-657.
- `/home/cactustim/agents/feat/m12/arch/token_budget_openai.md` — token/cost reference numbers (use these in heuristics).
- `/home/cactustim/agents/feat/m12/src/atm/llm/pricing.py` — `Pricing.estimate_cost`.
- `/home/cactustim/agents/feat/m12/src/atm/storage/models.py` — Run/Experiment models for SELECTs.

## Depends On

- `m12-config-schema` — needs `EstimateCfg`, `GridCfg`, `load_grid_configs`.

## Suggested run-task class

**standard** — read-only queries + heuristic math + two new CLI commands. ~400-600 LOC including tests.

## Notes / open questions

- **Decision:** use `runs.budget_spent_usd` aggregate instead of querying `llm_calls.parquet` for historical estimates. Avoids parquet IO from CLI; sufficient accuracy.
- **Decision:** if no historical data for a `(topology, task)` pair and no heuristic numbers in `EstimateCfg`, fall back to global default `(heuristic_tokens_per_call=1500, calls_per_iter=6)`. Document in `--help`.
- Assumption: `Pricing.estimate_cost` signature already supports prompt+completion split. If not, extend or use a wrapper.
- `atm status` reuses `engine.dispose()` pattern from runner; keep DB session short-lived.
- For `--json` output, use `model_dump_json` if a Pydantic wrapper exists; otherwise plain `json.dumps`.
