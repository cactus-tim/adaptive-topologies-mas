# M11 Resync — Tasks

## Wave 1: Merge M10 [Step 1] — COMPLETE

- [x] 1 [simple] Merge `origin/feat/m10` into `feat/m11` and resolve conflicts
  — `src/atm/tasks/`, `arch/PLAN.md`
  - Acceptance: `ls src/atm/tasks/` shows `gsm8k.py`, `commongen.py`, `dabench.py`;
    `grep "G6\|G7\|G10" arch/PLAN.md` returns hits; `python -c "from atm.tasks
    import EVALUATORS; print(EVALUATORS.names())"` lists four new evaluator keys.
  - Notes: Prefer M10 in all `tasks/` conflicts. Verify PLAN.md has G2/G6/G7/G8/G10
    subsection; if not, `git checkout origin/feat/m9 -- arch/PLAN.md`. Do NOT fix
    `ground_truth.py` here — leave it broken until Step 2.

---

## Wave 2: Independent catch-up items [Steps 2, 7, 8, 9] — NOT STARTED

- [ ] 2 [tdd] Rewrite `evaluation/ground_truth.py` using literal `_DEPS` table
  — `src/atm/evaluation/ground_truth.py`, `src/atm/evaluation/__init__.py`,
  `tests/unit/evaluation/test_ground_truth.py`
  - Acceptance: `uv run pytest tests/unit/evaluation/test_ground_truth.py -q` green;
    no live code references `mmlu_exact_match`/`creative_judge`/`analysis_hybrid`;
    `uv run mypy --strict src/atm/evaluation/ground_truth.py` clean.
  - Notes: `_DEPS = {"humaneval_pytest": ("sandbox",), "gsm8k_numeric": (),
    "commongen_rouge_coverage": ("judge_llm",), "dabench_numeric_exact": ()}` —
    verify `commongen_rouge_coverage` dep post-merge. Unit test enumerates
    `EVALUATORS.names()` vs `_DEPS.keys()` to catch drift. Depends On: 1.

- [ ] 7 [tdd] Add `filter_secrets` structlog processor and bootstrap wiring
  — `src/atm/observability/log_processors.py` (new),
  `src/atm/observability/__init__.py`, `src/atm/__init__.py`,
  `tests/unit/observability/test_log_processors.py` (new)
  - Acceptance: unit tests green for flat dict, nested dict, list-of-dicts,
    mixed-case keys, no-op clean event; idempotency test (double import does not
    raise or double-register); cyclic-dict test (self-referencing dict does not
    infinite-loop); `ATM_DISABLE_STRUCTLOG_BOOTSTRAP=1` suppresses `configure()`.
  - Notes: Compile `_SECRET_RE = re.compile(r"(?i).*(api_key|token|password|secret).*")`
    at module level. Return new dict (do not mutate input). Cyclic-dict guard via
    `id()`-based seen-set. Depends On: 1.

- [ ] 8 [tdd] Add `ToolRegistry.tools_for(role)` and `conf/tools_policy.yaml`
  — `conf/tools_policy.yaml` (new), `src/atm/tools/base.py`,
  `tests/unit/tools/test_tools_policy.py` (new)
  - Acceptance: unit tests green for (a) happy path per role, (b) missing-tool
    raises `ToolError` at call time (not YAML-parse time), (c) env-var override
    `ATM_TOOLS_POLICY_PATH`, (d) ctor-arg override, (e) unknown role returns just
    global list. YAML contains only currently-registered tools; no `arxiv_search`
    or `search`. Legacy callers using `registry.names()` unaffected.
  - Notes: Instance-level cache `self._policy: dict | None = None`. Path resolution:
    ctor arg → `os.environ["ATM_TOOLS_POLICY_PATH"]` → `Path.cwd() /
    "conf/tools_policy.yaml"`. YAML comment documents `duckduckgo_search`
    substitution. Depends On: 1.

- [ ] 9 [tdd] Add `persist_tlx` writer + unit + integration tests (G6)
  — `src/atm/evaluation/tlx.py`, `src/atm/evaluation/__init__.py`,
  `tests/unit/evaluation/test_tlx_persist.py` (new),
  `tests/integration/evaluation/test_tlx_persist.py` (new)
  - Acceptance: unit test (mock session) green; `ATM_INTEGRATION_PG=1 uv run pytest
    tests/integration/evaluation/test_tlx_persist.py -q` green; second call with
    different score overwrites correctly (idempotent); `raw_tlx_score` column
    written and readable. Integration test creates Run + HumanInteraction rows
    (FK chain respected).
  - Notes: Mirror style of `persist_quality` in `evaluation/aggregator.py`.
    `await session.commit()` after UPDATE. Signature:
    `async def persist_tlx(session: AsyncSession, interaction_id: UUID,
    raw_score: float) -> None`. Depends On: 1.

---

## Wave 3: Sequential runner.py edits [Steps 4, 5, 6] — NOT STARTED

- [x] 4 [tdd] Add `core/seed.py` with `seed_all` + unit tests + wire into runner
  — `src/atm/core/seed.py` (new), `src/atm/core/__init__.py`,
  `src/atm/experiment/runner.py`, `tests/unit/core/test_seed.py` (new)
  - Acceptance: `uv run pytest tests/unit/core/test_seed.py -q` green;
    determinism tests pass (two calls with same seed → same `random.random()` and
    same `numpy.random.rand()`); `uv run mypy --strict src/atm/core/seed.py` clean;
    `runner.run_one` calls `seed_all(cfg.seed)` before `_ensure_experiment`.
  - Notes: `torch.manual_seed` wrapped in `try/except ImportError`. Use
    `# type: ignore[import-not-found]` for torch. Depends On: 1.

- [x] 5 [tdd] Capture `model_version_snapshot` in `LLMWrapper`, persist on `Run`
  — `src/atm/llm/wrapper.py`, `src/atm/experiment/runner.py`,
  `tests/unit/llm/test_model_version_snapshot.py` (new)
  - Acceptance: unit test asserts `last_model_version` updates after `ainvoke` with
    patched `response_metadata`; UPDATE issued only when at least one non-None
    entry exists; UPDATE is in `finally` block (fires even on failed runs);
    `uv run mypy --strict src/atm/llm/wrapper.py` clean.
  - Notes: Anthropic parser: `response_metadata.get('model')`. OpenAI parser:
    `response_metadata.get('system_fingerprint') or response_metadata.get('model_name')`.
    Runner UPDATE: `sa.update(Run).values(model_version_snapshot={role: fp, ...})`.
    Collect `{role: wrapper.last_model_version for role, wrapper in wrappers.items()
    if wrapper.last_model_version is not None}`. Depends On: 1, 4.

- [x] 6 [tdd] Capture `sandbox_image_digest` in `DockerSandbox`, persist on `Run`
  — `src/atm/tools/sandbox/base.py`, `src/atm/tools/sandbox/docker_sandbox.py`,
  `src/atm/tools/sandbox/subprocess_sandbox.py`, `src/atm/experiment/runner.py`,
  `tests/unit/tools/test_sandbox_digest.py` (new)
  - Acceptance: unit test passes with mock `docker.from_env()` returning object with
    `.images.get(image).id = "sha256:abc123"`; `test_sandbox_digest_returns_none_when_docker_unavailable`
    passes (raises `DockerException` → returns None); column NULL for
    SubprocessSandbox; `sandbox_image_digest` UPDATE in runner only when non-None.
  - Notes: Capture only `image_map["python"]` digest (single string, not JSONB).
    `SubprocessSandbox.image_digest: str | None = None`. Wrap `docker.from_env()` in
    `except (DockerException, OSError)`. `docker.errors.ImageNotFound` also caught.
    Depends On: 1, 5.

---

## Wave 4: E2E replacement + CI [Steps 3, 10] — NOT STARTED

- [ ] 3 [tdd] Replace MMLU e2e test with GSM8K e2e; audit `test_m6_e2e.py`
  — `tests/fixtures/llm/m11_gsm8k_e2e_executor.yaml` (new),
  `tests/fixtures/llm/m11_mmlu_e2e_executor.yaml` (delete),
  `tests/integration/evaluation/test_aggregator_e2e.py`,
  `tests/integration/experiment/test_m6_e2e.py`
  - Acceptance: `ATM_INTEGRATION_PG=1 uv run pytest
    tests/integration/evaluation/test_aggregator_e2e.py
    tests/integration/experiment/test_m6_e2e.py -q` green; no file references
    remain to `m11_mmlu_e2e_executor`; test renamed to
    `test_gsm8k_run_quality_score_is_one`.
  - Notes: Read `src/atm/tasks/gsm8k.py` first to confirm extractor regex
    (`#### 42` vs `The answer is 42.`). `TaskSpec(id="gsm8k/e2e/0",
    type="reasoning", input="What is 6 * 7?", expected="42",
    evaluator_key="gsm8k_numeric")`. Audit `test_m6_e2e.py` for `"mmlu"`, `"qa"`,
    `"B"` references; switch task type to valid post-merge value. Depends On: 1, 2.

- [ ] 10 [simple] Add `.github/workflows/ci.yml` with lint, unit, integration jobs
  — `.github/workflows/ci.yml` (new)
  - Acceptance: YAML parses without error; on push, lint/unit/integration jobs all
    pass green; Python matrix `['3.11', '3.12']`; `astral-sh/setup-uv@v3` with
    `cache-dependency-glob: 'uv.lock'`; postgres:16 service container present in
    integration job.
  - Notes: `concurrency:` block for same-branch cancellation. Integration job runs
    `alembic upgrade head` before pytest. Environment variables for PG DSN set in
    job env. `enable-cache: true` on setup-uv action. Depends On: 1, 2, 3, 9.

---

## Wave 5: Final sweep [Step 11] — NOT STARTED

- [ ] 11 [simple] Final sanity sweep — run full lint + test matrix + grep sweep
  — (no file changes; fix in place if anything fails)
  - Acceptance: `ruff check src/atm tests` exit 0; `ruff format --check src/atm
    tests` exit 0; `mypy --strict src/atm` exit 0; `pytest tests/unit/ -q` exit 0;
    `ATM_INTEGRATION_PG=1 pytest tests/integration/ -q` exit 0; `grep -rn
    "mmlu\|m11_mmlu\|creative_judge\|analysis_hybrid" src/atm/evaluation/
    tests/integration/evaluation/` returns only docstring/comment lines; `alembic
    downgrade base && alembic upgrade head` exit 0; test count >= 1160.
  - Notes: If ruff autofixes needed, commit as `chore(m11): final formatting`.
    Update `dev/codebase-map.md` to reflect new files: `core/seed.py`,
    `observability/log_processors.py`, `conf/tools_policy.yaml`,
    `.github/workflows/ci.yml`. Depends On: 2, 3, 4, 5, 6, 7, 8, 9, 10.

---

## Stats

- Total: 11 tasks (~8h)
- Done: 0 / 11

## How to Update

Check off tasks with `[x]` and update `m11-resync-context.md` SESSION PROGRESS
after each milestone. Update phase headers:
- All tasks in a wave complete → replace "NOT STARTED" with "COMPLETE"
- Some tasks complete → replace "NOT STARTED" with "IN PROGRESS"
