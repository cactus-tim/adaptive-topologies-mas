# fix-dabench-task — Plan

## Executive Summary

DABench cells produce `quality_score=0.0` on every run due to two diagnosed
bugs: (a) `_build_initial_state` silently discards `TaskSpec.metadata`
(`format`, `constraints`, `file_name`), so agents never see how to emit
`@name[value]` answers; (b) the CSV tables referenced by DABench questions are
never staged into the executor workspace. The fix is three-pronged: augment
`task_input` with metadata when present; pre-stage CSV tables into the per-run
workspace via a new `stage_workspace_for` helper in `dabench.py` using a
byte-oriented cache; extend `executor.yaml` with dabench-aware instructions.
Back-compat is mandatory — HumanEval / GSM8K / CommonGen behavior must not
change.

## Current State

- `_build_initial_state` (runner.py:395-460) binds `_spec` only inside
  `if not task_input:`, so `_spec.metadata` is always discarded and the
  augmentation branch can never reach it when `cfg.task.input` is non-empty.
- No code copies CSV tables into `tools_workspace` before the graph runs.
- `executor.yaml` describes only the `solution.py` pattern for programming
  tasks; no dabench emission guidance exists.

## Proposed Approach

Three self-contained changes, all in parallel (Wave 1), followed by wiring
(Wave 2) and tests (Wave 3):

1. Add `stage_workspace_for` to `dabench.py` — downloads CSV from GitHub raw,
   caches in `data/cache/dabench_tables/`, copies to per-run workspace.
2. Restructure `_build_initial_state` to always resolve `_spec` and call a new
   private `_augment_task_input_with_metadata` helper.
3. Extend `executor.yaml` system prompt with dabench-specific emission guidance.
4. Wire the staging helper into `run_one` and resume/replay paths via
   `_pre_stage_workspace`.
5. Add 8 new unit tests across two test files.

Back-compat guard: `_augment_task_input_with_metadata` fires only when
`spec_id.startswith("dabench/")` AND `metadata` contains `format` or
`file_name` — safe for all existing task families (verified none carry these
keys).

## Implementation Phases

### Phase 1: Core implementation — Wave 1 (parallel) (~1.5h)
**Goal:** Land the three disjoint file changes that fix the two bugs.

- [ ] 1.1 Add `stage_workspace_for` + constants + `import shutil` to `dabench.py`
  - File: `src/atm/tasks/dabench.py`
  - Acceptance: `from atm.tasks.dabench import stage_workspace_for` works;
    calling it with a non-dabench spec returns `[]`; calling it with a
    `dabench/*` spec (monkeypatched urlopen) returns `[workspace/titanic.csv]`
    and the file exists; second call does not refetch.

- [ ] 1.2 Restructure `_build_initial_state` + add `_augment_task_input_with_metadata`
  - File: `src/atm/experiment/runner.py`
  - Acceptance: For a dabench spec with `format`/`constraints`/`file_name` in
    metadata, `state["shared"]["task_input"]` contains `[Task metadata]` and
    all three rendered lines. For HumanEval/GSM8K/CommonGen specs, `task_input`
    is byte-identical to `_spec.input`.

- [ ] 1.3 Extend `executor.yaml` system prompt with dabench paragraph
  - File: `conf/agents/executor.yaml`
  - Acceptance: `python -c "import yaml; yaml.safe_load(open('conf/agents/executor.yaml'))"` succeeds; new paragraph is present after the `solution.py` paragraph.

### Phase 2: Wiring — Wave 2 (~0.5h)
**Goal:** Wire `stage_workspace_for` into both run paths via `_pre_stage_workspace`.

- [ ] 2.1 Add module-level `stage_workspace_for` import + `_pre_stage_workspace` helper + two call sites
  - File: `src/atm/experiment/runner.py`
  - Acceptance: `atm.experiment.runner.stage_workspace_for` exists as a
    patchable module attribute; `_pre_stage_workspace` is defined; it is called
    immediately after `tools_workspace.mkdir` at both `run_one` (line ~858)
    and resume/replay (line ~1864) insertion points.

### Phase 3: Tests — Wave 3 (~1h)
**Goal:** Achieve 8 new passing tests; no regressions in the 1766-test suite.

- [ ] 3.1 Add 4 augmentation/resume tests to `test_runner_initial_state.py`
  - File: `tests/unit/experiment/test_runner_initial_state.py`
  - Acceptance: `test_build_initial_state_augments_dabench_metadata`,
    `test_build_initial_state_no_metadata_unchanged`,
    `test_build_initial_state_irrelevant_metadata_unchanged`,
    `test_pre_stage_workspace_invoked_on_resume_path` — all pass.

- [ ] 3.2 Add 4 staging tests to `test_dabench.py`
  - File: `tests/unit/tasks/test_dabench.py`
  - Acceptance: `test_stage_workspace_for_non_dabench_spec_returns_empty`,
    `test_stage_workspace_for_downloads_and_caches`,
    `test_stage_workspace_for_offline_returns_empty`,
    `test_stage_workspace_for_network_error_returns_empty_and_logs` — all pass.

- [ ] 3.3 Full regression check
  - File: `tests/unit/` (all)
  - Acceptance: `pytest tests/unit/ -q` exits 0; total count is 1774.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/tasks/dabench.py` | Add `import shutil`, `_DABENCH_TABLES_URL_TEMPLATE`, `_DEFAULT_TABLES_CACHE`, `stage_workspace_for` function, `__all__` entry | Implements CSV download, byte cache, and workspace copy |
| `src/atm/experiment/runner.py` | Restructure `_build_initial_state`; add `_augment_task_input_with_metadata`, module-level `stage_workspace_for` import, `_pre_stage_workspace` helper, two call sites | Fixes metadata discard bug; wires staging into both run paths |
| `conf/agents/executor.yaml` | Extend `system_prompt` with dabench-aware paragraph | Executor must know to emit `@name[value]` and load CSV via `code_run` |
| `tests/unit/experiment/test_runner_initial_state.py` | Add 4 new tests | Cover augmentation logic and resume-path staging |
| `tests/unit/tasks/test_dabench.py` | Add 4 new tests | Cover staging happy path, cache hit, offline, network error, non-dabench short-circuit |

## Dependencies & Order Constraints

- Wave 1 (steps 1.1, 1.2, 1.3) — fully parallel, disjoint files.
- Wave 2 (step 2.1) — must follow Wave 1. Imports `stage_workspace_for`
  from `dabench.py` (step 1.1) and touches the same `runner.py` as step 1.2;
  cannot run in parallel with step 1.2.
- Wave 3 (step 3.x) — must follow Wave 2. Tests need all implementation to
  exist and be importable; the resume-path test specifically requires
  `_pre_stage_workspace` (Wave 2) to be a real module attribute.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `resolve_spec` called more frequently (now unconditional in `_build_initial_state`) | Low | Low | Function is cached; `try/except` wrapper absorbs any failure |
| Network flakiness in staging download | Medium | Medium | `ATM_DABENCH_OFFLINE=1` short-circuit; broad `(URLError, OSError, TimeoutError)` catch returns `[]` without raising |
| YAML indentation error in `executor.yaml` | Low | High (breaks all agent tests) | Very loud immediate CI failure; trivially caught before merge |
| Duplicate workspace block at line ~1856 missed in Wave 2 | Low | Medium | Step 3.1 resume-path test guards this; `_pre_stage_workspace` helper makes both sites a one-line addition |
| Race condition on shared byte cache file | Very low | Low | `os.replace` on Linux is atomic; concurrent writes both succeed |
| Future task type adds `format` to metadata | Very low | Low | `spec_id.startswith("dabench/")` guard prevents augmentation for non-dabench tasks |

## Out of Scope

- Curated offline CSV fixtures (`tests/fixtures/tasks/dabench_tables/`) — if
  needed for diploma offline reproduction, this is a follow-up.
- Integration tests requiring real Cerebras keys in CI — the acceptance signal
  is a manual smoke run post-merge.
- Adding `code_run` / `file_read` hints to `task_input` — if the manual smoke
  shows the agent struggles with tool selection, this is a follow-up patch.
- Adaptive topology specific changes — `task_input` is forwarded uniformly to
  all sub-topologies; no change required.

## Timeline
- Total: ~3h
- Created: 2026-05-16
