# M11 Resync — Context

## SESSION PROGRESS (2026-05-12)

### COMPLETED
- Step 1: Merged `origin/feat/m10` into `feat/m11` (merge commit 197e75a).
  Resolved 2 conflicts: `src/atm/tasks/__init__.py` (took theirs + re-added `resolve_spec`),
  `dev/codebase-map.md` (manually merged M10 + M11 sections).
  Pulled audit-driven `arch/PLAN.md` from `origin/feat/m9` (commit 00c95c0) to add G2/G6/G7/G8/G10.
  All 4 post-merge invariants verified.

### COMPLETED
- Step 4: Created `src/atm/core/seed.py::seed_all(seed)` seeding `random`, `numpy` (optional), `torch` (optional).
  Re-exported from `src/atm/core/__init__.py`. Wired `seed_all(cfg.seed)` into `src/atm/experiment/runner.py::run_one`
  before `_ensure_experiment`. 9 new unit tests in `tests/unit/core/test_seed.py`. 1122/1122 unit tests pass.

### COMPLETED
- Step 5: Added `_extract_model_version(response_metadata)` helper and `last_model_version: str | None`
  attribute to `LLMWrapper`. Captures Anthropic `response_metadata['model']` and OpenAI
  `response_metadata['system_fingerprint']` (with `model_name` / `model` fallback) after each
  successful `ainvoke`. Added `model_version_snapshot` UPDATE in `finally` block of `runner.run_one`
  (collects non-None snapshots from all role wrappers; only fires when at least one entry exists).
  16 new unit tests in `tests/unit/llm/test_model_version_snapshot.py`. 1138/1138 unit tests pass.
  UPDATE call: `src/atm/experiment/runner.py:773`.

### COMPLETED
- Step 6: Added `image_digest: str | None` instance attribute to `DockerSandbox` (captured in
  `__init__` via `client.images.get(image_map["python"]).id`, wrapped in
  `except (DockerException, ImageNotFound, OSError, KeyError) -> None`; outer `docker.from_env()`
  also wrapped in `except (DockerException, OSError)` to handle unavailable daemon).
  Added class-level `image_digest: ClassVar[str | None] = None` to `SubprocessSandbox`.
  Added `sandbox_image_digest` UPDATE in `finally` block of `runner.run_one` at
  `src/atm/experiment/runner.py:794` (only fires when `sandbox.image_digest` is non-None;
  truncates to 80 chars for String(80) column).
  12 new unit tests in `tests/unit/tools/test_sandbox_digest.py`. 1150/1150 unit tests pass.

### COMPLETED
- Step 10: Created `.github/workflows/ci.yml` with lint, unit, and integration jobs.
  Python matrix `['3.11', '3.12']` for lint and unit; integration runs on 3.12 only with
  postgres:16 service container. `astral-sh/setup-uv@v3` pinned with `cache-dependency-glob: 'uv.lock'`
  and `enable-cache: true`. Concurrency block cancels in-progress runs on same branch.
  YAML validated: `python3 -c "import yaml; yaml.safe_load(...)"` exits 0.

### COMPLETED
- Step 3: Replaced MMLU e2e with GSM8K e2e.
  Deleted `tests/fixtures/llm/m11_mmlu_e2e_executor.yaml`.
  Created `tests/fixtures/llm/m11_gsm8k_e2e_executor.yaml` (executor emits "The answer is 42."; GSM8KMatcher last-numeric-token strategy matches expected="42").
  Rewrote `tests/integration/evaluation/test_aggregator_e2e.py`: `_MMLU_SPEC` → `_GSM8K_SPEC` (type="reasoning", evaluator_key="gsm8k_numeric"), task.name="gsm8k", test renamed to `test_gsm8k_run_quality_score_is_one`.
  Audited `test_m6_e2e.py`: no mmlu/qa references found — no changes needed.
  Integration test `test_gsm8k_run_quality_score_is_one` passes (1 passed, 2 skipped).

### IN PROGRESS
- Step 2 (Wave 2): Rewrite `evaluation/ground_truth.py` using `_DEPS` table

### BLOCKERS
- None

## Quick Resume

1. Read this file.
2. Check `m11-resync-tasks.md` for what is next.
3. Read `m11-resync-plan.md` Phase 1 for strategy.
4. Start with: Step 1 — `git fetch origin && git merge origin/feat/m10`.

## Key Files

**`src/atm/evaluation/ground_truth.py`**
- Role: Dispatches `score_ground_truth(spec, answer, ...)` to the registered evaluator for a given `TaskSpec`.
- Planned change: Full rewrite replacing flat if/elif with a literal `_DEPS` table + `EVALUATORS.get(spec.evaluator_key)` dispatch.
- Status: NOT STARTED

**`src/atm/evaluation/tlx.py`**
- Role: NASA-TLX scoring utilities.
- Planned change: Add `persist_tlx(session, interaction_id, raw_score)` idempotent UPDATE writer.
- Status: NOT STARTED

**`src/atm/evaluation/__init__.py`**
- Role: Public re-export surface for the evaluation package.
- Planned change: Add `persist_tlx` export; remove any stale MMLU imports post-merge.
- Status: NOT STARTED

**`src/atm/experiment/runner.py`**
- Role: Orchestrates a single experiment run end-to-end.
- Planned change: Three sequential edits — (a) `seed_all(cfg.seed)` at top of `run_one`, (b) `model_version_snapshot` UPDATE in `finally` block, (c) `sandbox_image_digest` UPDATE after sandbox start.
- Status: NOT STARTED

**`src/atm/llm/wrapper.py`**
- Role: LangChain LLM wrapper with usage tracking.
- Planned change: Add `_last_model_version: str | None`; parse `response_metadata['model']` (Anthropic) and `response_metadata['system_fingerprint']` with model-name fallback (OpenAI) after each successful `ainvoke`.
- Status: NOT STARTED

**`src/atm/tools/base.py`**
- Role: `ToolRegistry` — register/get/ainvoke tools.
- Planned change: Add `tools_for(role: str) -> list[Tool]` with lazy YAML policy loading; add `tools_policy_path: Path | None` to `__init__`.
- Status: NOT STARTED

**`src/atm/tools/sandbox/docker_sandbox.py`**
- Role: Docker-based code execution sandbox.
- Planned change: Add `image_digest` property that fetches `image_map["python"]` digest via docker-py SDK wrapped in `except (DockerException, OSError)`.
- Status: NOT STARTED

**`src/atm/tools/sandbox/subprocess_sandbox.py`**
- Role: Subprocess-based code execution sandbox.
- Planned change: Declare `image_digest: str | None = None` (Protocol compliance).
- Status: NOT STARTED

**`src/atm/tools/sandbox/base.py`**
- Role: `CodeSandbox` Protocol definition.
- Planned change: Add `image_digest: str | None` property to the Protocol.
- Status: NOT STARTED

**`src/atm/__init__.py`**
- Role: Package entry point; currently re-exports TASKS/EVALUATORS.
- Planned change: Append structlog bootstrap (`filter_secrets` processor) gated by `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` env var.
- Status: NOT STARTED

**`src/atm/core/seed.py`** (new)
- Role: Reproducibility seeding helper.
- Planned change: Create with `seed_all(seed: int) -> None` seeding `random`, `numpy`, optional `torch`.
- Status: DONE — seeds `random` + numpy (if installed) + torch (if installed); wired into runner.py.

**`src/atm/observability/log_processors.py`** (new)
- Role: Custom structlog processors.
- Planned change: Create with `filter_secrets` processor; regex `(?i).*(api_key|token|password|secret).*`; cyclic-dict guard via `id()`-based seen-set.
- Status: NOT STARTED

**`conf/tools_policy.yaml`** (new)
- Role: Role-based tool assignment policy.
- Planned change: Create with `global` list + `per_role` mapping using only currently-registered tools (no `arxiv_search`, no `search`).
- Status: NOT STARTED

**`.github/workflows/ci.yml`** (new)
- Role: GitHub Actions CI pipeline.
- Planned change: Create with lint, unit, and integration jobs; Python matrix `['3.11', '3.12']`; `astral-sh/setup-uv@v3` pinned with `cache-dependency-glob: 'uv.lock'`; postgres:16 service container.
- Status: DONE

**`tests/integration/evaluation/test_aggregator_e2e.py`**
- Role: E2e integration test for evaluation aggregation.
- Planned change: Rewrite from MMLU to GSM8K (`TaskSpec(type="reasoning", evaluator_key="gsm8k_numeric")`).
- Status: DONE — rewritten; test `test_gsm8k_run_quality_score_is_one` passes.

**`tests/fixtures/llm/m11_mmlu_e2e_executor.yaml`**
- Role: Old MMLU scripted executor fixture.
- Planned change: Delete entirely.
- Status: DONE — deleted.

**`tests/fixtures/llm/m11_gsm8k_e2e_executor.yaml`** (new)
- Role: Scripted executor fixture for GSM8K e2e.
- Planned change: Create with response containing a GSM8K-extractable answer.
- Status: DONE — created; emits "The answer is 42." (last numeric token = 42).

## Decisions

- **Literal `_DEPS` table over introspection (Step 2)**: `_DEPS` maps each evaluator
  key to its required constructor arguments. This avoids fragility from
  `inspect.signature` across `EvaluatorRegistry` contract changes (returns class
  vs instance). A unit test enumerates `EVALUATORS.names()` against `_DEPS.keys()`
  to catch drift.
  - Rationale: Reviewer refinement; `EVALUATORS.get(name)` confirmed to return a class.

- **`model_version_snapshot` UPDATE in `finally` block (Step 5)**: Ensures failed runs
  still record provider fingerprints. Anthropic uses `response_metadata['model']`;
  OpenAI uses `response_metadata['system_fingerprint']` with model-name fallback.
  - Rationale: Reviewer refinement; aligns with parquet flush placement.

- **Single-image digest capture (Step 6)**: `runs.sandbox_image_digest` is a
  `String(80)` single-string column; only `image_map["python"]` digest is captured,
  not a JSONB map. `SubprocessSandbox.image_digest = None`. `docker.from_env()`
  wrapped in `except (DockerException, OSError)`.
  - Rationale: Reviewer refinement; matches actual schema column type.

- **`ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` opt-out (Step 7)**: Downstream apps that
  manage their own structlog configuration can set this env var to suppress the
  bootstrap side-effect. The bootstrap is also idempotent (safe double-import).
  - Rationale: Reviewer refinement.

- **`tools_policy.yaml` with currently-registered tools only (Step 8)**: `arxiv_search`
  and `search` (bare) are not registered; the YAML only references tools that
  actually exist. `ToolError` is raised lazily at `tools_for()` call time, not at
  YAML-parse time, to allow partial policy files.
  - Rationale: Reviewer refinement; avoids blocking resync on missing tools.

- **Python matrix `['3.11', '3.12']` in CI (Step 10)**: Covers both the mypy target
  version and the current local dev environment. `astral-sh/setup-uv@v3` pinned
  with `cache-dependency-glob: 'uv.lock'` for reproducible lockfile-based caching.
  - Rationale: Reviewer refinement; user amendment.

- **Wave 3 strictly sequential (Steps 4→5→6)**: All three steps edit `runner.py`.
  Sequencing prevents merge conflicts; each targets a distinct function section
  (top of `run_one`, `finally` block, post-sandbox-init).
  - Rationale: Reviewer refinement + user amendment removing cross-parallel claims.

## Constraints

- **No schema migrations**: all required columns (`raw_tlx_score`,
  `model_version_snapshot`, `sandbox_image_digest`) already exist in `0001`.
- **`git_sha` already implemented**: `runner._get_git_sha()` exists; verify it
  survives the merge but do not rework it.
- **Docker unavailable in CI unit/integration jobs**: all docker-dependent code
  must gracefully fall back to `None` via `except (DockerException, OSError)`.
- **`torch` is optional**: `seed_all` must not fail if torch is not installed;
  use try/except ImportError.
- **`tools_policy.yaml` scope**: do not add stub tools (`arxiv_search`, `search`)
  in this resync; that is a future M-task.
- **M9.1/M9.2/M8.7 items**: leave any `human/` or `topology/` changes from the
  merge as-is; do not "finish" them in this resync.

## Amendments Applied

- Step 2: Changed from `inspect.signature` introspection to literal `_DEPS` table
  (reviewer refinement + user amendment confirmation).
- Steps 4/5/6 wave assignment: moved from Wave 2b (attempted parallel) to Wave 3
  (strictly sequential); removed cross-`Can-Parallel-With` claims among them.
  Step 5 `Depends On: 1, 4`; Step 6 `Depends On: 1, 5`.
- Step 5: `model_version_snapshot` UPDATE placed in `finally` block. Anthropic
  uses `response_metadata['model']`; OpenAI uses `response_metadata['system_fingerprint']`
  with model-name fallback (not just `system_fingerprint` alone).
- Step 6: Capture only `image_map["python"]` digest into single-string column.
  `SubprocessSandbox.image_digest = None`. `docker.from_env()` wrapped in
  `except (DockerException, OSError)`.
- Step 7: Added `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` env opt-out; added
  idempotency test and cyclic-dict test to acceptance criteria.
- Step 8: `tools_policy.yaml` ships with currently-registered tools only (no
  `arxiv_search`, no `search`). `ToolError` raised lazily at `tools_for()` call
  time, not at YAML-parse time.
- Step 10: Python matrix `['3.11', '3.12']`. Pinned `astral-sh/setup-uv@v3`
  with `cache-dependency-glob: 'uv.lock'`.
