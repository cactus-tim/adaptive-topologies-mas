# m12-config-schema — Pydantic sweep schema + DB migration

## What is this block

Foundation layer of M12. Расширяет `experiment/config.py` Pydantic-схему чтобы поддержать grid/sweep, resume, replay и estimate секции; обновляет OmegaConf loader (`load_config`) чтобы корректно резолвить новые ключи; добавляет Alembic-миграцию `0004` для новых колонок (`runs.replay_of`, `runs.host`, `runs.process_pid`) и индексов, которые нужны последующим блокам (reconcile, replay verification).

Этот блок целиком об контрактах. После него B/C/D могут стартовать параллельно — каждый использует только публичные импорты `from atm.experiment.config import ...` и DB-схему из `Run`/`Experiment`.

## In scope

1. **`src/atm/experiment/config.py` extensions:**
   - `GridCfg` Pydantic-модель:
     ```python
     class GridCfg(BaseModel):
         sweep: dict[str, list[Any]]  # e.g. {"topology.name": ["star","chain"], "task.name": ["humaneval","gsm8k"]}
         parallelism: int = 4          # max ProcessPoolExecutor workers
         fail_fast: bool = False       # if True, abort grid on first failure
         seeds: list[int] = Field(default_factory=lambda: [42])
     ```
   - `EstimateCfg` Pydantic-модель:
     ```python
     class EstimateCfg(BaseModel):
         heuristic_tokens_per_call: int = 1500   # fallback when no historical data
         calls_per_iter: int = 6                  # heuristic agent calls per iter
         use_historical: bool = True              # query past runs for averages
     ```
   - `ResumeCfg` (optional sub-block; can also live as flag-only — implementer's call):
     - `force: bool = False`
   - `ExperimentConfig` gains optional fields:
     ```python
     grid: GridCfg | None = None
     estimate: EstimateCfg = Field(default_factory=EstimateCfg)
     ```
   - Backward-compat: configs without `grid:` section continue to validate (`atm run` path unaffected).

2. **`src/atm/experiment/loader.py`** (NEW FILE — split from `config.py`):
   - Move `load_config` and `_merge_includes` from `config.py` to dedicated `loader.py`.
   - Re-export from `config.py` for back-compat: `from atm.experiment.loader import load_config as load_config`.
   - Add `load_grid_configs(path, overrides) -> list[ExperimentConfig]` helper:
     - Parses `grid.sweep` cartesian product (using `itertools.product`).
     - Each cell + each seed yields one `ExperimentConfig` with corresponding override applied (e.g. `topology.name=chain`, `task.name=gsm8k`, `seed=42`).
     - Returns flat list, ordered (topology, task, seed) for stable enumeration.
   - Validate sweep keys via dotpath: must resolve to a real `ExperimentConfig` field (e.g. `topology.name`, `task.name`, `model.default`). Unknown dotpath → raise `ValueError` with the path that failed.

3. **Alembic migration `0004_m12_runs_replay_and_pid.py`:**
   - `runs.replay_of: UUID NULL` (FK runs.id, on delete SET NULL) — populated by `atm replay`.
   - `runs.host: VARCHAR(64) NULL` — hostname where the worker process runs (populated at run start).
   - `runs.process_pid: INTEGER NULL` — OS PID of worker process.
   - Index on `(exp_id, status)` for reconcile-scan: `CREATE INDEX runs_exp_status_idx ON runs(exp_id, status)`.
   - `experiments.status` extension: NO enum constraint change required (column is `VARCHAR(16)`). Document allowed values informally in docstring: `running | completed | failed | partial`.
   - Update `src/atm/storage/models.py::Run` mappings to add the three new columns (Mapped types).
   - Head bumps from `0003_runs_cognitive_load_proxy` → `0004_m12_runs_replay_and_pid`.

4. **Tests:**
   - `tests/unit/experiment/test_config_grid.py`: sweep section parses; invalid dotpath raises; cartesian product cardinality matches; back-compat (no `grid:` → still validates).
   - `tests/unit/experiment/test_loader_grid_expansion.py`: 2×2×2 sweep → 8 ExperimentConfig instances with correct overrides; seeds list multiplies.
   - `tests/integration/storage/test_migration_0004.py` (PG): `alembic upgrade head` from `0003` succeeds; new columns exist; downgrade works.

## Out of scope (handled in later blocks)

- ProcessPoolExecutor / grid driver → m12-grid-runner.
- `atm grid` / `atm estimate` / `atm resume` / `atm replay` / `atm status` CLI commands → respective blocks.
- Reconcile logic (zombie detection) → m12-resume-replay.
- Population of `replay_of`/`host`/`process_pid` at runtime → m12-grid-runner sets host/pid, m12-resume-replay sets replay_of.

## Inputs / preconditions

- M11 complete (current state on `feat/m12` branch). Codebase map confirms: alembic head = `0003_runs_cognitive_load_proxy`, M11 reproducibility bundle in place.
- `ExperimentConfig` exists with current shape — see `/home/cactustim/agents/feat/m12/src/atm/experiment/config.py`.
- `Run` model exists — see `/home/cactustim/agents/feat/m12/src/atm/storage/models.py:106`.

## Outputs / contract

Downstream blocks depend on this contract:

```python
# 1. Pydantic
from atm.experiment.config import ExperimentConfig, GridCfg, EstimateCfg
cfg: ExperimentConfig = ...        # cfg.grid: GridCfg | None ; cfg.estimate: EstimateCfg

# 2. Loader API
from atm.experiment.loader import load_config, load_grid_configs
cells: list[ExperimentConfig] = load_grid_configs("conf/experiments/exp1.yaml", overrides=[])

# 3. ORM model — Run gains:
#     replay_of: UUID | None
#     host: str | None
#     process_pid: int | None
```

Sweep YAML shape (canonical):

```yaml
name: exp1
seed: 42
grid:
  parallelism: 4
  fail_fast: false
  seeds: [42, 43, 44]
  sweep:
    topology.name: [star, chain, mesh, debate]
    task.name: [humaneval, gsm8k]
# ... rest of ExperimentConfig (model, agents, observability, ...)
```

`load_grid_configs` returns `len(seeds) * prod(len(v) for v in sweep.values())` configs.

## References

- `/home/cactustim/agents/feat/m12/arch/PLAN.md` §M12 (lines 644-668) — feature list.
- `/home/cactustim/agents/feat/m12/arch/arch.md` §12 (Experiment layer) — config schema decisions.
- `/home/cactustim/agents/feat/m12/src/atm/experiment/config.py` — current schema (extend, do not break).
- `/home/cactustim/agents/feat/m12/alembic/versions/0003_runs_cognitive_load_proxy.py` — template for new migration.
- `/home/cactustim/agents/feat/m12/src/atm/storage/models.py` — Run/Experiment ORM.

## Depends On

(none — starting block)

## Suggested run-task class

**standard** — schema + migration + loader. Mostly mechanical; the only design decision is the sweep dotpath validator. ~400-600 LOC including tests.

## Notes / open questions

- **Decision:** sweep accepts dotpath keys (`topology.name`) rather than nested dict. Simpler parser, plays well with OmegaConf overrides.
- **Decision:** `seeds` list is grid-level, not topology-level. If users want per-topology seeds they can use a wrapper script — out of M12 scope.
- **Decision:** `experiments.status` stays VARCHAR; adding `partial` value semantically (run when some grid cells fail but not all). No DB constraint change.
- Assumption: `0004` migration name slug — adjust if alembic auto-gen prefers different naming.
