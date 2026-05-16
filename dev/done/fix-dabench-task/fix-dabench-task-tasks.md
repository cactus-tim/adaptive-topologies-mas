# fix-dabench-task — Tasks

## Phase 1: Core implementation (Wave 1 — parallel) NOT STARTED

- [x] 1.1 Add `stage_workspace_for` + constants + `import shutil` to `dabench.py` — `src/atm/tasks/dabench.py`
  - Acceptance: `from atm.tasks.dabench import stage_workspace_for` works; non-dabench spec returns `[]`; dabench spec with monkeypatched urlopen returns `[workspace/titanic.csv]` and file exists; second call does not refetch

- [x] 1.2 Restructure `_build_initial_state` + add `_augment_task_input_with_metadata` — `src/atm/experiment/runner.py`
  - Acceptance: dabench spec with `format`/`constraints`/`file_name` metadata produces `task_input` containing `[Task metadata]` block with all three lines in order; HumanEval/GSM8K/CommonGen specs are byte-identical to before

- [x] 1.3 Extend `executor.yaml` system prompt with dabench paragraph — `conf/agents/executor.yaml`
  - Acceptance: `python -c "import yaml; yaml.safe_load(open('conf/agents/executor.yaml'))"` exits 0; new paragraph present after `solution.py` paragraph; existing `solution.py` instruction untouched

## Phase 2: Wiring (Wave 2) NOT STARTED

- [x] 2.1 Add module-level `stage_workspace_for` import + `_pre_stage_workspace` helper + two call sites — `src/atm/experiment/runner.py`
  - Acceptance: `atm.experiment.runner.stage_workspace_for` exists as a patchable attribute; `_pre_stage_workspace` defined near `_build_initial_state`; called immediately after `tools_workspace.mkdir` at both run (~line 858) and resume/replay (~line 1864) insertion points

## Phase 3: Tests (Wave 3) NOT STARTED

- [x] 3.1 Add 4 tests to `test_runner_initial_state.py` — `tests/unit/experiment/test_runner_initial_state.py`
  - Tests: `test_build_initial_state_augments_dabench_metadata`, `test_build_initial_state_no_metadata_unchanged`, `test_build_initial_state_irrelevant_metadata_unchanged`, `test_pre_stage_workspace_invoked_on_resume_path`
  - Acceptance: all 4 pass; patch target for resume test is `atm.experiment.runner.stage_workspace_for`, called via `_pre_stage_workspace` directly (Option A)

- [x] 3.2 Add 4 staging tests to `test_dabench.py` — `tests/unit/tasks/test_dabench.py`
  - Tests: `test_stage_workspace_for_non_dabench_spec_returns_empty`, `test_stage_workspace_for_downloads_and_caches`, `test_stage_workspace_for_offline_returns_empty`, `test_stage_workspace_for_network_error_returns_empty_and_logs`
  - Acceptance: all 4 pass; patch target is `atm.tasks.dabench.urlopen`; second call in cache-hit test does not invoke urlopen again

- [x] 3.3 Full regression check — `tests/unit/`
  - Acceptance: `pytest tests/unit/ -q` exits 0; total test count is 1774 (up from 1766)

---
## Stats
- Total: 7 tasks · ~3h
- Done: 0 / 7

## How to Update
Check off tasks with [x] and update context.md SESSION PROGRESS after each milestone.

## Verification Commands

```bash
# After Phase 1.1
mypy src/atm/tasks/dabench.py
ruff check src/atm/tasks/dabench.py

# After Phase 1.2
mypy src/atm/experiment/runner.py

# After Phase 1.3
python -c "import yaml; yaml.safe_load(open('conf/agents/executor.yaml'))"

# After Phase 3
pytest tests/unit/experiment/test_runner_initial_state.py tests/unit/tasks/test_dabench.py -v
pytest tests/unit/ -q
```
