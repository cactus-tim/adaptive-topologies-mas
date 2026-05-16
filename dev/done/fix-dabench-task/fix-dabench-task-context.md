# fix-dabench-task — Context

## SESSION PROGRESS (2026-05-16)

### COMPLETED
- Step 1.3: Extended `conf/agents/executor.yaml` system_prompt with DABench data-analysis paragraph after the `solution.py` paragraph. Includes: CSV detection via "Dataset file" / `[Task metadata]` block, `code_run` + pandas guidance, `@metric_name[value]` format instruction, Answer format and Constraints line references. YAML parse verified; all 36 agent config tests pass.
- Step 1.1: Added `import shutil`, `from urllib.request import urlopen`, `_DABENCH_TABLES_URL_TEMPLATE`, `_DEFAULT_TABLES_CACHE`, and `stage_workspace_for` to `src/atm/tasks/dabench.py`. Added to `__all__`. All smoke checks pass (import, mypy clean, ruff clean, 64 task unit tests green).
- Step 1.2: Added `_augment_task_input_with_metadata` module-level helper and restructured `_build_initial_state` to accept optional `spec=` kwarg and always resolve spec when not provided. In `run_one`, spec is resolved once as `_run_spec` before `_build_initial_state` and reused at evaluation, keeping `resolve_spec` call count at 1 (preserving existing test contracts). mypy clean; 198 experiment unit tests pass.
- Step 2.1: Added module-level `from atm.tasks.dabench import stage_workspace_for` import to `runner.py`; added `_pre_stage_workspace(cfg, run_id, workspace_path, *, spec=None)` helper near `_build_initial_state` (line 522); moved `_run_spec = resolve_spec(cfg.task)` earlier in `run_one` (before workspace.mkdir) and passed it as `spec=_run_spec` so total `resolve_spec` call count stays at 1; added second call site in resume/replay path (~line 1814). mypy clean; all 198 experiment unit tests pass.

### IN PROGRESS
- Step 3.1 (Wave 3): Add 4 augmentation/resume tests to `tests/unit/experiment/test_runner_initial_state.py`

### BLOCKERS
- None

## Quick Resume
1. Read this file
2. Check fix-dabench-task-tasks.md for what's next
3. Read fix-dabench-task-plan.md Phase 1 for strategy
4. Start with: 1.1 — add `stage_workspace_for` to `src/atm/tasks/dabench.py`

## Key Files

**`src/atm/tasks/dabench.py`**
- Role: DABench task loader — downloads questions/labels from GitHub raw URLs, caches as Parquet
- Planned change: add `import shutil`, `_DABENCH_TABLES_URL_TEMPLATE` constant, `_DEFAULT_TABLES_CACHE` constant, `stage_workspace_for` function, `__all__` entry
- Status: DONE (Step 1.1)

**`src/atm/experiment/runner.py`**
- Role: Core experiment runner — builds initial shared state, runs agent graph, handles resume/replay
- Planned change (1.2): restructure `_build_initial_state` (lines 395-460) to always resolve `_spec`; add private `_augment_task_input_with_metadata` helper
- Planned change (2.1): add module-level `from atm.tasks.dabench import stage_workspace_for` (guarded by `try/except ImportError`); add `_pre_stage_workspace` helper; call it at two workspace-mkdir sites (~line 858 and ~line 1864)
- Status: DONE for 1.2 (awaiting 2.1)

**`conf/agents/executor.yaml`**
- Role: Executor agent configuration — system prompt, tools list, context budget
- Planned change: append dabench-aware paragraph to `system_prompt` literal block after the `solution.py` paragraph
- Status: DONE (Step 1.3)

**`tests/unit/experiment/test_runner_initial_state.py`**
- Role: Unit tests for `_build_initial_state` and related runner helpers
- Planned change: add 4 new tests (3 augmentation + 1 resume-path staging)
- Status: NOT STARTED

**`tests/unit/tasks/test_dabench.py`**
- Role: Unit tests for DABench task module
- Planned change: add 4 new staging tests
- Status: NOT STARTED

## Decisions

**Always resolve `_spec` in `_build_initial_state`**
- Decision: restructure so `_spec = resolve_spec(cfg.task)` runs unconditionally (outside the `if not task_input:` branch), wrapped in `try/except Exception`
- Rationale: the current `if not task_input:` guard means augmentation can never fire when `cfg.task.input` is non-empty; this is the root cause of bug (a)

**Augmentation trigger includes `spec_id.startswith("dabench/")` guard**
- Decision: trigger condition is `spec_id.startswith("dabench/") and metadata and (metadata.get("format") or metadata.get("file_name"))`
- Rationale: combining the M1 back-compat check (HumanEval/GSM8K/CommonGen verified to lack `format`/`file_name`) with the A2 defensive guard future-proofs against unrelated task families adding a `format` key

**Byte cache separate from Parquet cache**
- Decision: CSV staging uses `data/cache/dabench_tables/` (simple `Path.exists()` semantics), not `_cache.py`
- Rationale: `_cache.py` is designed for Parquet blobs; forcing raw CSV bytes through it would require API surgery and breaks the existing contract

**Staging helper uses module-level import for testability**
- Decision: `from atm.tasks.dabench import stage_workspace_for` at module level in `runner.py` (guarded by `try/except ImportError`), not imported inline inside `run_one`
- Rationale: an inline `try/import` inside a function body makes `monkeypatch.setattr("atm.experiment.runner.stage_workspace_for", ...)` impossible (no stable module attribute); the module-level import creates a real, patchable attribute

**Resume/replay path must be patched**
- Decision: both `run_one` (~line 858) and the resume/replay workspace block (~line 1864) call `_pre_stage_workspace`; a single helper function keeps both sites in sync
- Rationale: missing the resume path would leave dabench resume/replay runs broken; Step 3.1 resume-path test guards against regression

**Catch list for network errors: `(URLError, OSError, TimeoutError)`**
- Decision: do NOT add `HTTPError` separately — it is a subclass of `URLError`
- Rationale: reviewer note R6; redundant catch is confusing

**User amendment applied: Step 4(c) test uses Option A**
- Decision: the resume-path test (`test_pre_stage_workspace_invoked_on_resume_path`) calls `_pre_stage_workspace` directly with `stage_workspace_for` as the patched target
- Rationale: direct call to `_pre_stage_workspace` is simpler and more targeted than threading through the full `run_one` code path; reviewer preferred Option A

## Constraints

- `TaskCfg.input: str = ""` is non-Optional — test fixtures must use `input=""`, never `None`
- `import shutil` is NOT currently in `dabench.py` — must be added
- `data/` is gitignored project-wide — byte cache at `data/cache/dabench_tables/` is safe and consistent with `_cache.py` convention
- `ATM_DABENCH_OFFLINE=1` must short-circuit staging to a no-op (mirrors `DABenchLoader.load` behavior)
- No DB schema change. No Alembic migration. No new public package export beyond `atm.tasks.dabench.stage_workspace_for`
- All 1766 existing unit tests must remain green; suite grows to 1774
