# Plan: m12-estimate-status-cli

Class: standard
Needs Integration Tests: yes (PG-gated: estimator-historical + status-cli)

## Steps

### Step 1 (tdd) — Estimator core: `src/atm/experiment/estimator.py`
- Public dataclasses (frozen, slots): `CellEstimate(topology, task, seed, est_input_tokens, est_output_tokens, est_cost_usd, source)` and `GridEstimate(cells, total_cost_usd, total_input_tokens, total_output_tokens, per_topology)`.
- `source: Literal["historical_avg", "heuristic"]`.
- `async def estimate_grid(configs, session_factory, pricing, cfg) -> GridEstimate`.
- Historical path: SELECT AVG(budget_spent_usd), AVG(iterations) FROM runs WHERE topology=:t AND task_id=:task AND status='completed' GROUP BY topology, task. Use parameterized bound params (no SQL injection). Build per-(topology, task) dict.
- Heuristic fallback: `est_input = cfg.calls_per_iter * cfg.heuristic_tokens_per_call * topology.max_iterations`; `est_output = est_input // 3`; cost via `pricing.estimate(model_id=cfg.model.default, prompt_tokens=..., completion_tokens=...)`.
- If `cfg.estimate.use_historical=False` → skip the SQL entirely (use heuristic for all).
- If historical AVG present and >0 → `est_cost_usd = avg_budget`, derive est_input/est_output from avg_iterations*calls_per_iter*heuristic_tokens_per_call ratio (just for display); source=`historical_avg`.
- If `session_factory is None` → heuristic-only (used by `atm estimate` when DB unreachable; documented).
- Tests: `tests/unit/experiment/test_estimator_heuristic.py` covering:
  - heuristic-only path with no DB (session_factory=None) → cost matches `pricing.estimate`.
  - empty configs → empty cells, total=0.
  - per_topology aggregation.
  - source label correctness.

### Step 2 (tdd) — `atm estimate` CLI
- New `@app.command("estimate")` in `src/atm/experiment/cli.py`.
- Signature: `atm estimate --config <path> [overrides ...]`.
- Use `load_grid_configs`. Build engine+session_factory from cfg[0].observability.pg_dsn if `cfg.estimate.use_historical`; else None.
- Call `estimate_grid(...)`. Print table (typer.echo) with columns: topology, task, seed, tokens_in, tokens_out, cost_usd, source. Footer: `TOTAL $X.YY`.
- Exit 0 on success, 3 on config error.
- Tests: extend `tests/unit/experiment/test_cli.py` OR new `tests/unit/experiment/test_cli_estimate.py`:
  - --help exits 0.
  - patched `load_grid_configs` + `estimate_grid` returns a small GridEstimate → output contains expected rows + TOTAL.
  - missing config → exit 3.

### Step 3 (tdd) — `atm status` CLI
- New `@app.command("status")` with `--exp-id` / `--exp-name` / no-args (latest) + `--json`.
- Uses `select(Experiment, ...).outerjoin(Run)` with SQLAlchemy 2.x — strictly parameterized (no SQL string injection).
- Aggregates: COUNT/SUM(CASE WHEN status=...). Use `func.coalesce`.
- Output dict shape: `{id, name, status, started_at, finished_at, total, completed, failed, running, avg_quality, total_cost}`.
- `--json` → `json.dumps(dict, default=str)` (datetimes/UUIDs → str).
- Tests: unit test with mocked session_factory that returns a row object; integration PG test seeds data and parses JSON.

### Step 4 (tdd) — `atm run --estimate` polish
- Add `--estimate / --no-estimate` flag (default False) and `--yes` to `atm run`.
- If `--estimate`: run `estimate_grid([cfg], session_factory_or_none, pricing, cfg.estimate)` BEFORE `run_one(cfg)`; print one-line estimate.
- If `total_cost_usd > cfg.budget.per_experiment_usd * 0.5` AND not `--yes` → `typer.confirm(...)`; abort with exit 3 on no.
- Tests: `tests/unit/experiment/test_run_estimate_flag.py` — patches `estimate_grid` to return high cost, asserts confirm path triggered; with `--yes` proceeds to run.

### Step 5 (integration) — PG estimator historical + status CLI
- `tests/integration/experiment/test_estimator_historical.py`: seed Experiment + 3 completed runs (topology=star, task=humaneval, budget_spent_usd=0.05/0.06/0.07, iterations=4/5/6). Call estimate_grid with a config whose topology=star, task=humaneval. Assert source=historical_avg, est_cost_usd ≈ 0.06.
- `tests/integration/experiment/test_status_cli.py`: seed experiment with mixed run statuses; invoke `atm status --exp-id <uuid> --json` via CliRunner; parse JSON; assert counts/avg_quality/total_cost.
- Both PG-gated (use `pg_engine_fast` fixture; skip when `ATM_ENABLE_PG_TESTS` unset).

## Verification gates
- Lint: ruff format, ruff check --fix, mypy on new/modified files.
- Security: SQL parameterization, no shell construction.
- Code-review (sonnet).
