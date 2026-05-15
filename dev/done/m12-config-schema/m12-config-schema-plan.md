# m12-config-schema — Plan

## Executive Summary

Foundation block of M12. Extend the `ExperimentConfig` Pydantic schema with optional `grid` (sweep/parallelism/seeds/fail_fast) and `estimate` (heuristics) sub-schemas; split the OmegaConf loader out of `config.py` into a dedicated `loader.py`; add a `load_grid_configs` helper that expands a sweep into a flat list of concrete `ExperimentConfig` instances; ship Alembic migration `0004_m12_runs_replay_and_pid` that adds `runs.replay_of` (self-FK), `runs.host`, `runs.process_pid`, and a composite index `(exp_id, status)`; sync ORM `Run` model. Cover with unit tests (schema, expansion, back-compat) and one PG integration test for the migration.

## Current State

- `src/atm/experiment/config.py` holds both Pydantic schemas and the OmegaConf loader (`load_config`, `_merge_includes`). Lines ~197-282 are the loader code to be split out.
- `ExperimentConfig` uses Pydantic v2 idioms (`Field(default_factory=...)`, `model_validator`, `ConfigDict(frozen=True)` on `HumanCfg`). All field names are `snake_case`; literal-constrained enums use `Literal[...]`.
- `Run` ORM model lives at `src/atm/storage/models.py:106` with `__table_args__` listing 5 existing indexes.
- Alembic head is currently `0003`; existing migrations follow the `revision = "000N"` / `down_revision = "000(N-1)"` pattern.
- `tests/unit/experiment/test_config.py` already covers `load_config` paths; existing `tests/integration/conftest.py` exposes `pg_engine_alembic` (session-scoped).

## Proposed Approach

1. Add `GridCfg` and `EstimateCfg` Pydantic v2 sub-schemas to `config.py`, with a module-level `_resolve_dotpath` validator that walks `ExperimentConfig.model_fields` segment-by-segment and rejects unknown roots and `extra.*` paths.
2. Create `src/atm/experiment/loader.py` by moving `_merge_includes` and `load_config` from `config.py`, then add `load_grid_configs` which uses `itertools.product` over sweep dimensions × seeds and calls `load_config` once per cell.
3. Keep full backwards compatibility: `from atm.experiment.config import load_config` continues to work via a re-export line.
4. Extend the `Run` ORM class with three nullable columns and a composite index, then ship `0004_m12_runs_replay_and_pid.py` to apply the same changes in Postgres.
5. Validate with layered tests: unit (no PG) for schema/expansion, integration (PG-gated) for migration round-trip.

Key design decision: `ResumeCfg` is deferred — resume parameters will be CLI flags, not schema fields. `load_grid_configs` preserves the `grid` block verbatim in each returned cell so callers can introspect `parallelism`/`fail_fast`. Sweep values are restricted to scalars (str/int/float/bool) for M12.

## Implementation Phases

### Phase 1: Schema and ORM foundation (~2h)
**Goal:** Add new sub-schemas to `config.py` and extend the `Run` ORM model — no DB yet.

- [ ] 1.1 Write unit tests for `GridCfg`/`EstimateCfg` (TDD — red first)
  - File: `tests/unit/experiment/test_config_grid.py`
  - Acceptance: 10 test cases defined; all fail before Step 1.2 implementation

- [ ] 1.2 Add `GridCfg`, `EstimateCfg`, `_resolve_dotpath` to `config.py`
  - File: `src/atm/experiment/config.py`
  - Acceptance: `uv run pytest tests/unit/experiment/test_config_grid.py -v` all green; existing `tests/unit/experiment/test_config.py` still green

- [ ] 1.3 Extend `Run` ORM with `replay_of`, `host`, `process_pid` + composite index
  - File: `src/atm/storage/models.py`
  - Acceptance: `uv run python -c "from atm.storage.models import Run; print([c.name for c in Run.__table__.c])"` shows the three new columns; `mypy src/atm/storage/models.py` passes

### Phase 2: Loader split and grid expansion (~1.5h)
**Goal:** Create `loader.py`, add `load_grid_configs`, keep back-compat re-exports.

- [ ] 2.1 Create `src/atm/experiment/loader.py` (move loader code + add `load_grid_configs`)
  - File: `src/atm/experiment/loader.py`
  - Acceptance: `from atm.experiment.loader import load_config, load_grid_configs` works; `from atm.experiment.config import load_config` still works

- [ ] 2.2 Update `src/atm/experiment/config.py` (remove loader body, add re-export line)
  - File: `src/atm/experiment/config.py`
  - Acceptance: no `ImportError`; module docstring updated to note split

- [ ] 2.3 Update `src/atm/experiment/__init__.py` (add `load_grid_configs` to exports)
  - File: `src/atm/experiment/__init__.py`
  - Acceptance: `from atm.experiment import load_config, load_grid_configs, ExperimentConfig` works

### Phase 3: Grid expansion tests (~1h)
**Goal:** Cover `load_grid_configs` cartesian expansion with unit tests + fixture YAML.

- [ ] 3.1 Create fixture YAML `tests/fixtures/experiment/grid_minimal.yaml`
  - File: `tests/fixtures/experiment/grid_minimal.yaml`
  - Acceptance: YAML is a valid `ExperimentConfig` with a `grid:` block containing `sweep: {topology.name: [star, chain], task.name: [t1, t2]}`

- [ ] 3.2 Write unit tests for grid expansion
  - File: `tests/unit/experiment/test_loader_grid_expansion.py`
  - Acceptance: 6+ tests passing; 2×2 sweep yields 4 configs, 2×2×2 yields 8, no-grid returns 1, invalid dotpath raises `ValidationError`, import from both `atm.experiment` and `atm.experiment.loader` succeeds

### Phase 4: Alembic migration + integration test (~1h)
**Goal:** Ship `0004` migration and verify it with a PG round-trip test.

- [ ] 4.1 Create Alembic migration `0004_m12_runs_replay_and_pid.py`
  - File: `alembic/versions/0004_m12_runs_replay_and_pid.py`
  - Acceptance: `uv run alembic upgrade head` brings head to `0004`; `uv run alembic downgrade -1` returns to `0003` cleanly

- [ ] 4.2 Write PG integration test for migration 0004
  - File: `tests/integration/storage/test_migration_0004.py`
  - Acceptance: `ATM_ENABLE_PG_TESTS=1 uv run pytest tests/integration/storage/test_migration_0004.py -v` passes; upgrade/downgrade round-trip verified; column types, nullability, index, and FK delete-action all asserted

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/experiment/config.py` | Add `GridCfg`, `EstimateCfg`, `_resolve_dotpath`; remove loader body; add re-export | Core schema extension + back-compat |
| `src/atm/experiment/loader.py` | New file — moved `load_config`, `_merge_includes` + new `load_grid_configs` | Separation of concerns; new public API |
| `src/atm/experiment/__init__.py` | Add `load_grid_configs` to `__all__` and import | Public package surface |
| `src/atm/storage/models.py` | `Run` class: 3 new columns + 1 composite index | ORM must match DB schema |
| `alembic/versions/0004_m12_runs_replay_and_pid.py` | New migration: 3 columns, 1 self-FK, 1 index + downgrade | DB contract for downstream M12 blocks |
| `tests/unit/experiment/test_config_grid.py` | New — 10 unit tests for `GridCfg`/`EstimateCfg` | TDD coverage of new sub-schemas |
| `tests/unit/experiment/test_loader_grid_expansion.py` | New — 6+ unit tests for `load_grid_configs` | Coverage of cartesian expansion logic |
| `tests/fixtures/experiment/grid_minimal.yaml` | New fixture YAML with `grid:` block | Shared test fixture for expansion tests |
| `tests/integration/storage/test_migration_0004.py` | New — PG round-trip test for migration | Validates published DB contract |

## Dependencies & Order Constraints

- Phase 1 steps 1.1/1.2/1.3 are fully parallel (different files).
- Phase 2 depends on Phase 1.2 (needs `GridCfg` to exist for type-checking in loader).
- Phase 3 depends on Phases 1.2 + 2 (needs schema and loader both present).
- Phase 4.1 depends on Phase 1.3 (ORM and migration must be byte-identical in column names/types).
- Phase 4.2 depends on Phases 1.3 + 4.1.
- Phase 3 and Phase 4 can run in parallel with each other (different files).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dotpath validator over-restricts due to sloppy `Optional[T]` unwrapping | Medium | Medium | Use `typing.get_origin`/`get_args`; strip `NoneType`; test `model.default` (deep valid) and `topology.extra.foo` (rejected) explicitly |
| `from __future__ import annotations` makes annotation strings not types | Low | High | Pydantic v2 `model_fields[name].annotation` returns resolved type objects — verify in REPL before writing the walker |
| ORM columns drift from migration (different name/type) | Low | High | Review both files side-by-side before commit; `pg_engine_alembic` DDL equivalence smoke test catches this in CI |
| `config.py` re-export line omitted — breaks all existing `from atm.experiment.config import load_config` call sites | Low | High | Explicit `from atm.experiment.loader import load_config as load_config  # noqa: F401` |
| `replay_of` column missing `nullable=True` blocks `SET NULL` FK action | Low | High | Checklist in Step 1.3 acceptance criterion |
| Migration downgrade drops column before FK constraint — Postgres error | Low | High | Downgrade reverses order: drop index, drop FK, drop columns (plan already specifies this) |
| Integration test pollutes shared DB state for `pg_engine_alembic` fixture | Medium | Low | Teardown to `base`; tests run serially per file |

## Out of Scope

- `ResumeCfg` sub-schema — resume parameters will be CLI flags, not schema fields. Deferred to m12-resume-replay block.
- `experiments.status` column changes — `partial` status is left to a future block.
- Nested-dict / list sweep values — scalars only (str/int/float/bool) for M12.
- `replay_of` ORM relationship (`relationship()`) — raw FK column only; downstream block adds relationship if needed.
- Alembic `branch_labels` / `depends_on` metadata — not used in this project.

## Timeline

- Total: ~5.5h
- Created: 2026-05-14
