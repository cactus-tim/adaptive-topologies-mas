# M11 Resync — Plan

## Executive Summary

Bring PR `feat/m11` (#12) into full alignment with the audit-driven `arch/PLAN.md`
update from commit `3738d61`. Two axes are out of sync:

1. **Block 1 — M10 resync**: M10 (PR #10) deleted `tasks/{mmlu,creative,analysis}.py`,
   added `tasks/{gsm8k,commongen,dabench}.py`, introduced `EvaluatorRegistry`
   singleton (`EVALUATORS`), changed `TaskSpec.type` to
   `Literal["programming","reasoning","creative","decision"]`, and rotated
   evaluator keys (`humaneval_pytest`, `gsm8k_numeric`, `commongen_rouge_coverage`,
   `dabench_numeric_exact`). M11 still references the old surface in
   `evaluation/ground_truth.py`, in the e2e test, and in fixtures.

2. **Block 2 — M11 catch-up tasks** (G2/G6/G7/G8/G10) added to PLAN.md but never
   landed on `feat/m11`:
   - **G6**: `raw_tlx_score` persistence path (`persist_tlx`, unit/integration
     tests). Schema column already exists — only the writer and tests are missing.
   - **G2**: reproducibility bundle — `core/seed.py::seed_all`,
     `model_version_snapshot` capture in `LLMWrapper`, `sandbox_image_digest`
     capture in `DockerSandbox`, `filter_secrets` structlog processor.
     `experiments.git_sha` capture already exists and does not need rework.
   - **G7**: `conf/tools_policy.yaml` + `ToolRegistry.tools_for(role)`.
   - **G8**: closed by Block 1 (rewriting `ground_truth.py`).
   - **G10**: `.github/workflows/ci.yml` (lint/unit/integration jobs).

**Final Class**: complex
**Needs Integration Tests**: yes

## Current State

- `tasks/{mmlu,creative,analysis}.py` still present on `feat/m11`; M10 deletes them.
- `evaluation/ground_truth.py` uses flat if/elif with deprecated keys
  (`mmlu_exact_match`, `creative_judge`, `analysis_hybrid`).
- `tests/integration/evaluation/test_aggregator_e2e.py` uses `TaskSpec(type="qa",
  evaluator_key="mmlu_exact_match")` and fixture `m11_mmlu_e2e_executor.yaml`.
- Schema already has `raw_tlx_score`, `model_version_snapshot`,
  `sandbox_image_digest` columns — no migrations needed.
- `runner._get_git_sha()` already implemented — no work needed.
- No `core/seed.py`, no `.github/workflows/`, no `conf/tools_policy.yaml`.
- `ToolRegistry` lacks `tools_for(role)` method.
- `structlog` is already a dependency; no global `configure()` call exists yet.

## Proposed Approach

Five waves of work, ordered to avoid same-file conflicts on `runner.py` and
`evaluation/__init__.py`:

1. Merge M10 into `feat/m11` as a blocking prerequisite.
2. Implement all catch-up items that touch independent file areas in parallel
   (`ground_truth.py`, `log_processors.py`, `tools_policy`, `tlx.py`).
3. Sequential runner edits: `seed_all` wiring → `model_version_snapshot` →
   `sandbox_image_digest`.
4. Replace MMLU e2e with GSM8K e2e once `ground_truth.py` is stable; ship CI
   workflow.
5. Final quality-gate sweep.

Key architectural decision: Step 2 uses a literal `_DEPS` table instead of
runtime introspection to avoid fragility across `EvaluatorRegistry` contracts.

## Implementation Phases

### Phase 1: Merge M10 (~1h)
**Goal:** Land `origin/feat/m10` cleanly so M11 has the correct task surface.

- [ ] 1.1 Run `git fetch origin && git merge origin/feat/m10`; resolve conflicts
       in `tasks/` (prefer M10), verify PLAN.md contains G2/G6/G7/G8/G10
       subsection; commit the merge.
  - Files: `src/atm/tasks/__init__.py`, `src/atm/tasks/base.py`, `arch/PLAN.md`
  - Acceptance: `ls src/atm/tasks/` shows `gsm8k.py`, `commongen.py`,
    `dabench.py`; `grep "G6\|G7\|G10" arch/PLAN.md` returns hits;
    `python -c "from atm.tasks import EVALUATORS; print(EVALUATORS.names())"` lists
    the four new keys.

---

### Phase 2: Independent catch-up items (~3h, Wave 2 — 4 steps parallel)
**Goal:** Deliver G6, G7, G8, filter_secrets without touching runner.py.

- [ ] 2.1 Rewrite `evaluation/ground_truth.py` using literal `_DEPS` table;
       extend `tests/unit/evaluation/test_ground_truth.py` with new evaluator
       key cases.
  - Files: `src/atm/evaluation/ground_truth.py`,
    `src/atm/evaluation/__init__.py`,
    `tests/unit/evaluation/test_ground_truth.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_ground_truth.py -q`
    green; no live code references `mmlu_exact_match`/`creative_judge`/
    `analysis_hybrid`.

- [ ] 2.2 Add `filter_secrets` structlog processor; wire bootstrap with
       `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` opt-out; add idempotency test and
       cyclic-dict test.
  - Files: `src/atm/observability/log_processors.py` (new),
    `src/atm/observability/__init__.py`, `src/atm/__init__.py`,
    `tests/unit/observability/test_log_processors.py` (new)
  - Acceptance: `uv run pytest tests/unit/observability/test_log_processors.py -q`
    green; idempotent double-import does not raise; opt-out env var respected.

- [ ] 2.3 Add `ToolRegistry.tools_for(role)` + `conf/tools_policy.yaml` with
       currently-registered tools only; `ToolError` raised lazily at call time.
  - Files: `conf/tools_policy.yaml` (new), `src/atm/tools/base.py`,
    `tests/unit/tools/test_tools_policy.py` (new)
  - Acceptance: unit tests green for happy path, missing-tool `ToolError`,
    env-var override, ctor-arg override, unknown-role fallback to global list.

- [ ] 2.4 Add `persist_tlx(session, interaction_id, raw_score)` to
       `evaluation/tlx.py`; add unit test (mock session) and integration test
       (PG-gated, idempotency round-trip).
  - Files: `src/atm/evaluation/tlx.py`, `src/atm/evaluation/__init__.py`,
    `tests/unit/evaluation/test_tlx_persist.py` (new),
    `tests/integration/evaluation/test_tlx_persist.py` (new)
  - Acceptance: unit test green; `ATM_INTEGRATION_PG=1 uv run pytest
    tests/integration/evaluation/test_tlx_persist.py -q` green; second call
    with different score overwrites (idempotent).

---

### Phase 3: Sequential runner.py edits (~2h, Wave 3 — Steps 4→5→6)
**Goal:** Wire reproducibility captures into runner without merge conflicts.

- [ ] 3.1 Create `src/atm/core/seed.py` with `seed_all(seed: int) -> None`
       (random + numpy + optional torch); wire into `runner.run_one` at top
       before `_ensure_experiment`; add unit tests.
  - Files: `src/atm/core/seed.py` (new), `src/atm/core/__init__.py`,
    `src/atm/experiment/runner.py`, `tests/unit/core/test_seed.py` (new)
  - Acceptance: `uv run pytest tests/unit/core/test_seed.py -q` green;
    `uv run mypy --strict src/atm/core/seed.py` clean.

- [ ] 3.2 Add `LLMWrapper.last_model_version: str | None`; parse
       `response_metadata['model']` for Anthropic, `response_metadata
       ['system_fingerprint']` with model-name fallback for OpenAI; UPDATE
       `runs.model_version_snapshot` in `finally` block of runner; add unit test.
  - Files: `src/atm/llm/wrapper.py`, `src/atm/experiment/runner.py`,
    `tests/unit/llm/test_model_version_snapshot.py` (new)
  - Acceptance: unit test asserts property updates; UPDATE issued only when at
    least one non-None entry exists; runs in `finally` so failed runs still record.

- [ ] 3.3 Add `image_digest: str | None` property to `CodeSandbox` Protocol and
       `DockerSandbox` (captures `image_map["python"]` digest only, wrapped in
       `except (DockerException, OSError)`); `SubprocessSandbox.image_digest =
       None`; UPDATE `runs.sandbox_image_digest` after sandbox start; add unit
       test mocking docker client.
  - Files: `src/atm/tools/sandbox/base.py`,
    `src/atm/tools/sandbox/docker_sandbox.py`,
    `src/atm/tools/sandbox/subprocess_sandbox.py`,
    `src/atm/experiment/runner.py`,
    `tests/unit/tools/test_sandbox_digest.py` (new)
  - Acceptance: unit test passes (mock `.id="sha256:abc123"`);
    `test_sandbox_digest_returns_none_when_docker_unavailable` passes;
    column NULL for SubprocessSandbox.

---

### Phase 4: E2E test replacement + CI workflow (~1.5h, Wave 4 — parallel)
**Goal:** Replace broken MMLU e2e with GSM8K; ship CI.

- [ ] 4.1 Delete `tests/fixtures/llm/m11_mmlu_e2e_executor.yaml`; create
       `m11_gsm8k_e2e_executor.yaml`; rewrite `test_aggregator_e2e.py` for GSM8K;
       audit `test_m6_e2e.py` for MMLU/qa references and fix.
  - Files: `tests/fixtures/llm/m11_gsm8k_e2e_executor.yaml` (new),
    `tests/fixtures/llm/m11_mmlu_e2e_executor.yaml` (delete),
    `tests/integration/evaluation/test_aggregator_e2e.py`,
    `tests/integration/experiment/test_m6_e2e.py`
  - Acceptance: `ATM_INTEGRATION_PG=1 uv run pytest
    tests/integration/evaluation/test_aggregator_e2e.py
    tests/integration/experiment/test_m6_e2e.py -q` green; no fixture
    references remain to `m11_mmlu_e2e_executor`.

- [ ] 4.2 Create `.github/workflows/ci.yml` with lint, unit, and integration
       jobs; Python matrix `['3.11', '3.12']`; pin `astral-sh/setup-uv@v3` with
       `cache-dependency-glob: 'uv.lock'`; postgres:16 service container for
       integration job.
  - Files: `.github/workflows/ci.yml` (new)
  - Acceptance: YAML is syntactically valid (`python -m yaml ci.yml`); on push
    all three jobs pass green.

---

### Phase 5: Final sweep (~0.5h, Wave 5)
**Goal:** Confirm the full quality gate passes end-to-end.

- [ ] 5.1 Run `ruff check`, `ruff format --check`, `mypy --strict`, `pytest
       tests/unit/ -q`, `pytest tests/integration/ -q` (PG-gated), grep sweep
       for stale MMLU references, `alembic downgrade base && upgrade head`
       round-trip; confirm test count ≥ 1160.
  - Files: none (verification only); fix in place if anything fails.
  - Acceptance: all commands exit 0; grep returns only docstring/comment lines;
    test count ≥ 1160.

---

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/evaluation/ground_truth.py` | Full rewrite | Replace deprecated if/elif with `_DEPS` + `EVALUATORS.get()` dispatch |
| `src/atm/evaluation/tlx.py` | Extend | Add `persist_tlx` writer (G6) |
| `src/atm/evaluation/__init__.py` | Extend | Re-export `persist_tlx`; verify no stale imports |
| `src/atm/experiment/runner.py` | Extend (3 sequential edits) | Wire `seed_all`, `model_version_snapshot` UPDATE (finally), `sandbox_image_digest` UPDATE |
| `src/atm/llm/wrapper.py` | Extend | Add `last_model_version` property and metadata parser |
| `src/atm/tools/base.py` | Extend | Add `tools_for(role)` method and policy loader |
| `src/atm/tools/sandbox/base.py` | Extend | Add `image_digest` to Protocol |
| `src/atm/tools/sandbox/docker_sandbox.py` | Extend | Implement `image_digest` with docker-py SDK |
| `src/atm/tools/sandbox/subprocess_sandbox.py` | Extend | Declare `image_digest = None` |
| `src/atm/__init__.py` | Extend | Bootstrap structlog with `filter_secrets` (opt-out via env var) |
| `src/atm/core/seed.py` | New | `seed_all(seed)` implementation |
| `src/atm/core/__init__.py` | Extend | Export `seed_all` |
| `src/atm/observability/log_processors.py` | New | `filter_secrets` processor with cyclic-dict guard |
| `src/atm/observability/__init__.py` | Extend | Export `filter_secrets` |
| `conf/tools_policy.yaml` | New | Role-based tool policy (currently-registered tools only) |
| `.github/workflows/ci.yml` | New | Three-job CI pipeline (lint, unit, integration) |
| `tests/fixtures/llm/m11_gsm8k_e2e_executor.yaml` | New | Scripted GSM8K fixture |
| `tests/fixtures/llm/m11_mmlu_e2e_executor.yaml` | Delete | Replaced by GSM8K |
| `tests/integration/evaluation/test_aggregator_e2e.py` | Rewrite | GSM8K e2e replaces MMLU |
| `tests/integration/experiment/test_m6_e2e.py` | Audit + adjust | Fix MMLU/qa references |
| `tests/unit/evaluation/test_ground_truth.py` | Extend | Add new evaluator key cases |
| `tests/unit/evaluation/test_tlx_persist.py` | New | Mock-session unit test for `persist_tlx` |
| `tests/integration/evaluation/test_tlx_persist.py` | New | PG-gated idempotency test |
| `tests/unit/core/test_seed.py` | New | Determinism assertions for `seed_all` |
| `tests/unit/observability/test_log_processors.py` | New | flat/nested/list/cyclic/no-op cases |
| `tests/unit/tools/test_tools_policy.py` | New | Happy path, ToolError, env-var, ctor, unknown role |
| `tests/unit/llm/test_model_version_snapshot.py` | New | FakeLLM metadata parse test |
| `tests/unit/tools/test_sandbox_digest.py` | New | Mock docker client + unavailable daemon |

## Dependencies & Order Constraints

- **Step 1 blocks everything** — merge must land before any file edit.
- **Wave 2 steps (2.1–2.4) are mutually independent** except that 2.1 and 2.4
  both append to `evaluation/__init__.py` — coordinate via clean `__all__`
  extension (run 2.4 after 2.1 within the wave if serialization is needed).
- **Wave 3 steps are strictly sequential** — all three edit `runner.py`:
  - 3.1 edits top of `run_one` (seed wiring)
  - 3.2 edits post-graph section and `finally` block
  - 3.3 edits post-sandbox-init section
- **Step 4.1 (GSM8K e2e) depends on 2.1** — `ground_truth.py` must dispatch
  `gsm8k_numeric` before the integration test can pass.
- **Step 4.2 (CI) depends on 1, 2.1, 4.1, 2.4** — CI should only be committed
  once the integration tests it will run are correct.
- **Step 5.1 depends on all preceding steps**.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Merge conflict in `tasks/` more complex than expected | Low | Medium | Use `-X theirs` for `tasks/` paths; `git merge --abort` + retry |
| GSM8K extractor requires `#### 42` not `The answer is 42.` | Medium | Low | Read `tasks/gsm8k.py` first; adjust fixture wording |
| `EvaluatorRegistry.get()` returns instance not class | Low | Medium | Read `tasks/base.py` first; if instance, dispatch directly without `__init__` |
| `docker.from_env()` raises in CI (no daemon) | High | Low | Already mitigated by `except (DockerException, OSError)` → None |
| `structlog.configure` double-call breaks downstream | Medium | Low | Idempotency guard + `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` opt-out |
| `filter_secrets` cyclic-dict infinite recursion | Low | High | `id()`-based seen-set guard; cyclic-dict unit test |
| `runner.py` sequential edits introduce conflicts between waves | Medium | Medium | Strictly sequence waves 3.1 → 3.2 → 3.3; each targets a distinct function section |
| `alembic downgrade base` fails (missing `op.drop_index`) | Low | Low | Fix in-place in Step 5.1 |

## Out of Scope

- Completing M9.1/M9.2/M8.7 items (human/, topology/ changes from merge).
- Adding `arxiv_search` or `search` stub tools (future M-task).
- CUDA-specific `torch` seeding semantics.
- Any schema migrations — all required columns already exist.
- Docker-in-docker or network-level CI tests.
- `torch.manual_seed` GPU seeding edge cases.

## Timeline

- Total: ~8h
- Wave 1: ~1h (Step 1)
- Wave 2: ~3h (Steps 2.1–2.4 parallel)
- Wave 3: ~2h (Steps 3.1→3.2→3.3 sequential)
- Wave 4: ~1.5h (Steps 4.1, 4.2 parallel)
- Wave 5: ~0.5h (Step 5.1)
- Created: 2026-05-12
