# m12-config-schema — Tasks

## Phase 1: Schema and ORM foundation - NOT STARTED
- [ ] 1.1 Write unit tests for `GridCfg`/`EstimateCfg` (TDD red phase) — `tests/unit/experiment/test_config_grid.py`
  - Acceptance: 10 test cases defined; all fail before 1.2 implementation lands
- [ ] 1.2 Add `GridCfg`, `EstimateCfg`, `_resolve_dotpath` to `config.py` — `src/atm/experiment/config.py`
  - Acceptance: `uv run pytest tests/unit/experiment/test_config_grid.py -v` all green; `uv run pytest tests/unit/experiment/test_config.py -v` still green
- [ ] 1.3 Extend `Run` ORM with `replay_of`, `host`, `process_pid` + composite index — `src/atm/storage/models.py`
  - Acceptance: `python -c "from atm.storage.models import Run; print([c.name for c in Run.__table__.c])"` shows 3 new columns; `uv run mypy src/atm/storage/models.py` passes

## Phase 2: Loader split and grid expansion - NOT STARTED
- [ ] 2.1 Create `src/atm/experiment/loader.py` with moved + extended loader code — `src/atm/experiment/loader.py`
  - Acceptance: `from atm.experiment.loader import load_config, load_grid_configs` works without error
- [ ] 2.2 Remove loader body from `config.py`, add back-compat re-export — `src/atm/experiment/config.py`
  - Acceptance: `from atm.experiment.config import load_config` still works; module docstring updated
- [ ] 2.3 Add `load_grid_configs` to package exports — `src/atm/experiment/__init__.py`
  - Acceptance: `from atm.experiment import load_config, load_grid_configs, ExperimentConfig` works; `uv run pytest tests/unit/experiment/ -v` all green

## Phase 3: Grid expansion tests - NOT STARTED
- [ ] 3.1 Create fixture YAML with `grid:` block — `tests/fixtures/experiment/grid_minimal.yaml`
  - Acceptance: YAML is valid `ExperimentConfig`; `load_config("tests/fixtures/experiment/grid_minimal.yaml")` does not raise
- [ ] 3.2 Write unit tests for `load_grid_configs` — `tests/unit/experiment/test_loader_grid_expansion.py`
  - Acceptance: 6+ tests passing; 2×2 sweep → 4 cells; 2×2×2 → 8 cells; no-grid → 1 cell; invalid dotpath raises `ValidationError`; import from both `atm.experiment` and `atm.experiment.loader` succeeds

## Phase 4: Alembic migration + integration test - NOT STARTED
- [ ] 4.1 Create Alembic migration `0004` — `alembic/versions/0004_m12_runs_replay_and_pid.py`
  - Acceptance: `uv run alembic upgrade head` → current rev `0004`; `uv run alembic downgrade -1` → current rev `0003` with no errors
- [ ] 4.2 Write PG integration test for migration `0004` — `tests/integration/storage/test_migration_0004.py`
  - Acceptance: `ATM_ENABLE_PG_TESTS=1 uv run pytest tests/integration/storage/test_migration_0004.py -v` green; columns, index, and FK `confdeltype='n'` all asserted; downgrade verified

---
## Stats
- Total: 9 tasks · ~5.5h
- Done: 0 / 9

## How to Update
Check off tasks with [x] and update `m12-config-schema-context.md` SESSION PROGRESS after each milestone.
