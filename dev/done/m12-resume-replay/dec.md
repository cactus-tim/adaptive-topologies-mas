# m12-resume-replay — atm resume, atm replay, reconcile (G5)

## What is this block

Recovery + reproducibility cluster of M12. Реализует три тесно связанные фичи поверх LangGraph PG checkpointer:

1. **`atm resume --run-id <uuid> [--force]`** — продолжает run из последнего checkpoint'а через `AsyncPostgresSaver`. Требуется для HITL E2/E5 (живой человек отвечает, процесс упал из-за OOM/SIGKILL).
2. **`atm replay <run_id> [--mode deterministic|semantic]`** — детерминированный реплей через `FakeLLM(mode="replay")`, читая `data/experiments/{exp_id}/runs/{run_id}/llm_calls.parquet`. Новый run пишется с `replay_of=<original_run_id>`. Verifies pin через `runs.model_version_snapshot`.
3. **Reconcile** — функция `reconcile_zombies(exp_id, session_factory) -> ReconcileReport`, которую m12-grid-integration зовёт перед стартом grid'а. Сканирует `runs` где `status='running' AND exp_id=:exp_id`; для каждой строки проверяет `process_pid` на `host` (через `psutil.pid_exists` если local host, иначе assume dead); ставит `status='failed'` (finish_reason='zombie') или, при `--force-resume`, оставляет нетронутым.

## In scope

1. **`atm resume` command (`src/atm/experiment/cli.py`):**
   - Signature: `atm resume --run-id <uuid> [--force]`.
   - Reads `runs` row → reconstructs `ExperimentConfig` from `experiments.config_snapshot` + `runs.seed` + `runs.models_by_role_json`. (Note: current `experiments.config_snapshot` schema is sparse — see Notes; expand if needed in this block.)
   - Calls a new `runner.resume_one(run_id: UUID, cfg: ExperimentConfig, *, force: bool) -> RunResult`:
     - Re-opens `checkpointer_scope(pg_dsn)`.
     - Re-builds topology with same `thread_id=str(run_id)` so LangGraph resumes from last checkpoint.
     - Updates `runs.status = 'running'` (was probably 'failed' or 'running' stale).
     - On success → `_update_run_success` as usual.
   - `--force` bypasses pid-alive check (allows resuming a run whose process is technically still alive but stuck).
   - Exit codes match `atm run`.

2. **`atm replay` command:**
   - Signature: `atm replay <run_id> [--mode deterministic|semantic] [--output-config-only]`.
   - **deterministic mode:**
     - Loads original run's config from `experiments.config_snapshot` + `runs.*`.
     - Overrides `cfg.model.default = "fake:replay"` and points `fake_fixtures` (or a dedicated `replay_source`) to `data/experiments/{exp_id}/runs/{run_id}/llm_calls.parquet`.
     - Verifies `runs.model_version_snapshot` matches the original — if mismatch, abort with informative error unless `--mode semantic`.
     - Calls a new `runner.replay_one(original_run_id: UUID, mode: Literal["deterministic","semantic"]) -> RunResult`.
     - New run row is INSERTED with `replay_of=<original_run_id>`.
     - Final `final_answer` and `messages` must be bit-identical (modulo timestamps/UUIDs).
   - **semantic mode:** uses real LLM with same model+seed; only verifies `quality_score` within tolerance. Lower bar — not blocking for M12 exit criteria, but plumb the flag.

3. **`FakeLLM(mode="replay")` glue:**
   - The class already supports replay via `REPLAY_SCHEMA` — see `src/atm/llm/fake.py:34`.
   - **Add to `build_llm`** (`src/atm/llm/factory.py`): when `model_id == "fake:replay"` and `replay_source` kwarg provided → load parquet via `pyarrow.parquet.read_table(replay_source)`, instantiate `FakeLLM(mode="replay", replay_table=table)`.
   - Update `_build_llm_wrappers` in `runner.py` (or `replay_one`) so the `fake:replay` provider gets `replay_source` from a per-role mapping that points to the original parquet.

4. **Reconcile:**
   - `src/atm/experiment/reconcile.py` (NEW FILE):
     ```python
     @dataclass
     class ZombieRow:
         run_id: UUID
         host: str | None
         pid: int | None
         reason: Literal["pid_dead", "host_mismatch", "no_pid"]

     @dataclass
     class ReconcileReport:
         scanned: int
         zombies: list[ZombieRow]
         actions: dict[UUID, Literal["marked_failed", "kept_force_resume"]]

     async def reconcile_zombies(
         session_factory,
         exp_id: UUID,
         *,
         allow_force_resume: bool = False,
         current_host: str | None = None,
     ) -> ReconcileReport: ...
     ```
   - Heuristic: if `host` does not match `socket.gethostname()` → mark zombie (can't verify pid). If host matches → `psutil.pid_exists(pid)`; if False → zombie.
   - `allow_force_resume=True` only logs but does not mutate.

5. **Tests:**
   - `tests/unit/experiment/test_reconcile.py`: mock session; rows with various host/pid combos; verify report classification.
   - `tests/integration/experiment/test_resume_sigkill.py` (PG, slow):
     - Spawn a `run_one` subprocess, send SIGKILL mid-run (after first checkpoint).
     - Run `atm resume --run-id <uuid>` in same test process.
     - Assert run reaches `completed`, `quality_score` matches a fresh run with same FakeLLM seed.
   - `tests/integration/experiment/test_replay_deterministic.py` (PG):
     - Run `run_one` with FakeLLM scripted fixtures end-to-end.
     - Call `replay_one(run_id, mode="deterministic")`.
     - Assert: new run has `replay_of` set; `final_answer` equal; `iterations` equal; `messages` parquet content matches (sorted by id).
   - `tests/unit/llm/test_factory_replay.py`: `build_llm("fake:replay", replay_source=...)` returns FakeLLM with table loaded.

## Out of scope

- `atm grid` itself calling reconcile — that's m12-grid-integration (final wiring). This block exposes the `reconcile_zombies` function and CLI command (`atm reconcile --exp-id <uuid>` for ops).
- `atm replay --mode semantic` real-LLM evaluation pipeline beyond stub — full semantic equivalence is M13 territory.
- Cross-host PID verification (would need a heartbeat table) — host mismatch → zombie, period.

## Inputs / preconditions

- m12-config-schema **must be merged**: needs `runs.replay_of`, `runs.host`, `runs.process_pid` columns.
- LangGraph PG checkpointer (`checkpointer_scope`) — already in `src/atm/storage/checkpointer.py`.
- FakeLLM replay mode — already exists; `build_llm` factory just needs the routing case.
- `experiments.config_snapshot` schema **may need expansion** to be sufficient to reconstruct full `ExperimentConfig` for resume/replay. Current snapshot is `{name, topology, task_name, seed}` (see `_ensure_experiment` in runner.py). Decide one of:
  - (a) widen `config_snapshot` to full dump of `cfg.model_dump()`.
  - (b) require `--config <path>` on `atm resume` / `atm replay`.

   **Pick (a)** — implementation should widen the snapshot. Document the change in this block's deliverable.

## Outputs / contract

```python
# Public API
from atm.experiment.runner import resume_one, replay_one
from atm.experiment.reconcile import reconcile_zombies, ReconcileReport, ZombieRow

# CLI
$ atm resume --run-id <uuid>          # exits 0 on success
$ atm replay <run_id> --mode deterministic   # writes new run with replay_of=<orig>
$ atm reconcile --exp-id <uuid> [--dry-run]
```

`reconcile_zombies` is the function m12-grid-integration calls at grid start.

`runs.replay_of` is set ONLY by `replay_one` — never by `resume_one`.

## References

- `/home/cactustim/agents/feat/m12/arch/PLAN.md` §M12 lines 658-663 (G5).
- `/home/cactustim/agents/feat/m12/arch/arch.md` §11.3 (two-pools for checkpointer), §14.4 (reproducibility bundle).
- `/home/cactustim/agents/feat/m12/src/atm/llm/fake.py` — FakeLLM replay mode (already implemented).
- `/home/cactustim/agents/feat/m12/src/atm/storage/checkpointer.py` — `checkpointer_scope`.
- `/home/cactustim/agents/feat/m12/src/atm/experiment/runner.py` — `run_one` (template for `resume_one`/`replay_one`).
- `/home/cactustim/agents/feat/m12/tests/integration/llm/test_llm_layer_contract.py` — existing replay round-trip test (template).

## Depends On

- `m12-config-schema` — needs `runs.replay_of/host/process_pid` columns.

## Suggested run-task class

**complex** — touches checkpointer scope, LangGraph thread continuation, FakeLLM wiring, PG schema reading. SIGKILL-resume integration test is non-trivial. ~800-1100 LOC including tests.

## Notes / open questions

- **Decision:** widen `experiments.config_snapshot` to `cfg.model_dump()` (option a) — avoids requiring users to keep YAML around for resume.
- **Decision:** reconcile is opt-in via `atm grid` flag (default ON). `atm resume` does NOT auto-reconcile sibling runs; it operates only on the target run_id.
- **Decision:** `--force` on `atm resume` bypasses pid-alive check; useful when the process is stuck but not dead.
- `psutil` dependency — check `pyproject.toml`; if not present, add as optional or use `os.kill(pid, 0)` fallback (POSIX-only; M12 dev env is Linux per `env`).
- Assumption: bit-identical = identical `final_answer`, `iterations`, and ordered `messages` content (excluding `id`, `timestamps`, `latency_ms`). Document tolerance in test docstring.
- Open: should `replay_of` enforce `exp_id` parent equality? Probably yes — replay-run lives in same experiment as original. Implement and test.
