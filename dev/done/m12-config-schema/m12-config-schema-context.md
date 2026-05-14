# m12-config-schema — Context

## SESSION PROGRESS (2026-05-14)

### COMPLETED
- Lint/format pass: ruff format + ruff check --fix + mypy strict; all clean. Committed as chore(m12-config-schema): ruff auto-fix (cd85c42).

### IN PROGRESS
- All steps complete

### BLOCKERS
- None

## Quick Resume

1. Read this file
2. Check `m12-config-schema-tasks.md` for what's next
3. Read `m12-config-schema-plan.md` Phase 1 for strategy
4. Start with: write `tests/unit/experiment/test_config_grid.py` (TDD red phase), then implement `GridCfg`/`EstimateCfg`/`_resolve_dotpath` in `src/atm/experiment/config.py`

## Key Files

**`src/atm/experiment/config.py`**
- Role: Pydantic schemas for experiment configuration; currently also contains OmegaConf loader code (lines ~197-282)
- Planned change: Add `GridCfg`, `EstimateCfg` sub-schemas and `_resolve_dotpath` helper between `HumanCfg` and `ExperimentConfig`; extend `ExperimentConfig` with `grid: GridCfg | None = None` and `estimate: EstimateCfg`; remove loader body and replace with back-compat re-export line
- Status: NOT STARTED

**`src/atm/experiment/loader.py`**
- Role: New file — OmegaConf loader logic moved from `config.py`
- Planned change: Create with `_merge_includes` (private), `load_config` (public), and new `load_grid_configs` (public) using `itertools.product` over sweep dimensions × seeds
- Status: NOT STARTED

**`src/atm/experiment/__init__.py`**
- Role: Public package surface re-exporting `ExperimentConfig`, `load_config`, etc.
- Planned change: Add `load_grid_configs` to imports and `__all__`
- Status: NOT STARTED

**`src/atm/storage/models.py`**
- Role: SQLAlchemy ORM models including `Run` at line ~106
- Planned change: Add `replay_of` (UUID nullable self-FK), `host` (String(64) nullable), `process_pid` (Integer nullable) columns to `Run`; add `Index("runs_exp_status_idx", "exp_id", "status")` to `__table_args__`
- Status: NOT STARTED

**`alembic/versions/0004_m12_runs_replay_and_pid.py`**
- Role: New Alembic migration to apply the `runs` table changes to Postgres
- Planned change: Create with `revision = "0004"`, `down_revision = "0003"`; upgrade adds 3 columns + self-FK + index; downgrade reverses in correct order (index, FK, columns)
- Status: NOT STARTED

**`tests/unit/experiment/test_config_grid.py`**
- Role: New unit tests for `GridCfg`/`EstimateCfg` schema validation
- Planned change: 10 test cases covering: valid dotpaths, unknown root rejected, deep valid path, `extra.*` rejected, empty sweep, `parallelism=0`, `seeds=[]`, back-compat (no `grid:` block), `EstimateCfg` defaults, `EstimateCfg` overrides
- Status: NOT STARTED

**`tests/unit/experiment/test_loader_grid_expansion.py`**
- Role: New unit tests for `load_grid_configs` cartesian expansion
- Planned change: 6+ test cases covering 2×2 sweep, 2×2×2, no-grid, parallelism preserved, invalid dotpath raises, import paths
- Status: NOT STARTED

**`tests/fixtures/experiment/grid_minimal.yaml`**
- Role: New fixture YAML with a `grid:` block for expansion tests
- Planned change: Clone of `valid_minimal.yaml` plus `grid: {sweep: {topology.name: [star, chain], task.name: [t1, t2]}, seeds: [42]}`
- Status: NOT STARTED

**`tests/integration/storage/test_migration_0004.py`**
- Role: New PG integration test for migration `0004` round-trip
- Planned change: Subprocess-driven `upgrade 0003 → head`, column/index/FK introspection via `information_schema` + `pg_constraint`, then `downgrade -1` verification; gated by `ATM_ENABLE_PG_TESTS`
- Status: NOT STARTED

## Decisions

- **`ResumeCfg` deferred**: Resume parameters will be CLI flags only, not a schema sub-block. Keeps schema surface minimal and back-compat risk low. Can be added in m12-resume-replay if needed.
- **`load_grid_configs` preserves `grid` block in each cell**: Each returned `ExperimentConfig` retains the original `grid` config verbatim. Per-cell selection is reflected in `topology.name`, `task.name`, `seed`, etc. Downstream callers can still introspect `cfg.grid.parallelism`.
- **Sweep values restricted to scalars**: str/int/float/bool only for M12. Nested-dict/list sweep values require richer OmegaConf merging — out of scope.
- **Self-FK `replay_of` as raw column**: No `relationship()` added to avoid cyclic SQLAlchemy relationship headaches. Downstream block can add it.
- **Named FK constraint**: `fk_runs_replay_of_runs` — enables clean `op.drop_constraint` by name on downgrade.
- **Index convention**: `runs_exp_status_idx` follows existing `runs_*_idx` naming in `models.py`.
- **Dotpath validator restriction**: Sweep keys must resolve to declared `model_fields` on `ExperimentConfig` (recursive into nested `BaseModel`). `extra.*` paths (where `extra` is `dict[str, Any]`, not a sub-model) are rejected.

## Constraints

- Must not break: `from atm.experiment.config import load_config` (back-compat re-export required).
- Must not break: existing tests in `tests/unit/experiment/test_config.py` and `tests/unit/experiment/test_smoke_yaml_loads.py`.
- Must not break: `tests/integration/storage/test_smoke_run.py` DDL equivalence (ORM must match migration column names/types byte-for-byte).
- Alembic head must advance from `0003` to `0004` — no skipping, no branching.
- PG integration tests are auto-skipped when `ATM_ENABLE_PG_TESTS` is unset; no additional pytest marker needed.
- All `uv run alembic` invocations assume `alembic.ini` is discoverable from the project root cwd.
