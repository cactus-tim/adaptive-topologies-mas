# M4 — Tools Layer — Задачи

## Wave 1: Контракты + deps/conf (параллельно) ⏳ НЕ НАЧАТО

- [x] **Step 1** — Tool Protocol + ToolRegistry + ExecResult + safety/retry helpers
  - Тип: tdd | Зависит от: — | Параллельно с: 2 | Статус: `[x]`
  - Создать контрактный слой: `base.py` (Tool Protocol, ToolSchema с `returns`, ToolRegistry), `sandbox/base.py` (CodeSandbox, ExecResult, SandboxConfig с tmpfs /tmp), `_safety.py` (resolve_and_validate_url с SSRF defense), `_retry.py` (with_tool_retry → with_retry адаптер). ~25 unit-тестов.
  - Файлы: `src/atm/tools/base.py`, `src/atm/tools/_safety.py`, `src/atm/tools/_retry.py`, `src/atm/tools/sandbox/__init__.py`, `src/atm/tools/sandbox/base.py`, `src/atm/tools/__init__.py`, `tests/unit/tools/__init__.py`, `tests/unit/tools/test_base.py`, `tests/unit/tools/test_sandbox_base.py`, `tests/unit/tools/test_safety.py`, `tests/unit/tools/test_retry.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_base.py tests/unit/tools/test_sandbox_base.py tests/unit/tools/test_safety.py tests/unit/tools/test_retry.py -v && uv run mypy src/atm/tools/base.py src/atm/tools/sandbox/base.py src/atm/tools/_safety.py src/atm/tools/_retry.py`

- [x] **Step 2** — Dependencies bump + seccomp derivation pipeline + fixtures
  - Тип: simple | Зависит от: — | Параллельно с: 1 | Статус: `[x]`
  - Обновить `pyproject.toml` (docker, ddgs, httpx, numpy, faiss extras, markers, ruff pin). Написать и запустить `scripts/derive_seccomp.py` (moby upstream → deny-list → `conf/sandbox/seccomp.json`). `conf/sandbox/defaults.yaml` (с /work и /tmp tmpfs). Corpus fixtures. Drift-тест seccomp.
  - Файлы: `pyproject.toml`, `scripts/derive_seccomp.py`, `conf/sandbox/seccomp.json`, `conf/sandbox/seccomp.README.md`, `conf/sandbox/defaults.yaml`, `tests/fixtures/tools/__init__.py`, `tests/fixtures/tools/corpus/doc_1.txt`, `tests/fixtures/tools/corpus/doc_2.txt`, `tests/fixtures/tools/corpus/doc_3.txt`, `tests/unit/tools/test_seccomp_profile.py`
  - Приёмка: `uv sync && uv run pytest tests/unit/tools/test_seccomp_profile.py -v`

---

## Wave 2: Реализации инструментов (11 параллельно, после Wave 1) ⏳ НЕ НАЧАТО

- [x] **Step 3** — SubprocessSandbox (dev-mode, non-isolated)
  - Тип: tdd | Зависит от: 1, 2 | Параллельно с: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | Статус: `[x]`
  - `SubprocessSandbox`, `IS_ISOLATED=False`, async execute через asyncio.create_subprocess_exec + tempdir + cleanup. Docstring: "Dev-only. NOT isolated. DO NOT use in prod."
  - Файлы: `src/atm/tools/sandbox/subprocess_sandbox.py`, `tests/unit/tools/test_subprocess_sandbox.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_subprocess_sandbox.py -v` (~6-8 тестов: stdout, stderr, nonzero exit, timeout, multi-file)

- [x] **Step 4** — Global — CalculatorTool
  - Тип: tdd | Зависит от: 1 | Параллельно с: 3, 5, 6, 7, 8, 9, 10, 11, 12, 13 | Статус: `[x]`
  - AST-walker safe eval: whitelist BinOp/UnaryOp/Constant/math.* Call. Output: `{"value", "expression"}`. Пустой stub `global_/__init__.py`.
  - Файлы: `src/atm/tools/global_/__init__.py` (empty stub), `src/atm/tools/global_/calculator.py`, `tests/unit/tools/test_calculator.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_calculator.py -v` (~8 тестов: 2+2, sqrt, __import__ reject, Name-outside reject)

- [x] **Step 5** — Global — FileReadTool
  - Тип: tdd | Зависит от: 1 | Параллельно с: 3, 4, 6, 7, 8, 9, 10, 11, 12, 13 | Статус: `[x]`
  - `FileReadTool(workspace, max_bytes)`. Path validation через `is_relative_to(workspace.resolve())` до st_size. Output: `{"content", "path", "size"}`.
  - Файлы: `src/atm/tools/global_/file_read.py`, `tests/unit/tools/test_file_read.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_file_read.py -v` (~5-6 тестов: happy, ../../ reject, absolute reject, oversize, symlink outside)

- [x] **Step 6** — Global — DuckDuckGoSearchTool + integration conftest
  - Тип: tdd | Зависит от: 1, 2 | Параллельно с: 3, 4, 5, 7, 8, 9, 10, 11, 12, 13 | Статус: `[x]`
  - `DuckDuckGoSearchTool` через `asyncio.to_thread` + `with_tool_retry`. Output: `{"results": [...]}`. Integration conftest с `pytest_collection_modifyitems` (только этот hook, без `pytest.current_test`).
  - Файлы: `src/atm/tools/global_/search.py`, `tests/unit/tools/test_duckduckgo_search.py`, `tests/integration/tools/__init__.py`, `tests/integration/tools/conftest.py`, `tests/integration/tools/test_network_search.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_duckduckgo_search.py -v` (~5 unit-тестов с FakeDDGS mock; integration skipped без ATM_ENABLE_NETWORK_TESTS=1)

- [x] **Step 7** — Global — UrlFetchTool (SSRF-safe)
  - Тип: tdd | Зависит от: 1, 2 | Параллельно с: 3, 4, 5, 6, 8, 9, 10, 11, 12, 13 | Статус: `[x]`
  - `UrlFetchTool(allow_private, max_bytes, timeout)`. `resolve_and_validate_url` перед GET. `httpx.AsyncClient(follow_redirects=False)`. Size limit через aiter_bytes. Retry через `with_tool_retry`. Unit-тесты через `httpx.MockTransport`.
  - Файлы: `src/atm/tools/global_/url_fetch.py`, `tests/unit/tools/test_url_fetch.py`, `tests/integration/tools/test_network_url_fetch.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_url_fetch.py -v` (~10 тестов: 6 SSRF reject + 4 happy path)

- [x] **Step 8** — DockerSandbox (hardened) + live seccomp/isolation tests
  - Тип: tdd | Зависит от: 1, 2 | Параллельно с: 3, 4, 5, 6, 7, 9, 10, 11, 12, 13 | Статус: `[x]`
  - `DockerSandbox`, `IS_ISOLATED=True`. seccomp_json_str в конструкторе. `containers.run` с `read_only=True`, `cap_drop=["ALL"]`, `network_mode="none"`, `tmpfs={"/work":...,"/tmp":...}`, `user="1000:1000"`. `container.remove(force=True)` в finally. Unit-mock + live тесты (gated `@pytest.mark.docker`). Fork-bomb: `container.wait(timeout=20)["StatusCode"] != 0` ИЛИ wall-time < 30s. Per-syscall seccomp: 11 тестов (ptrace, mount, unshare, bpf, keyctl, pivot_root, chroot, setns, clone3, add_key, init_module). Isolation: rm -rf / → host file hash unchanged.
  - Файлы: `src/atm/tools/sandbox/docker_sandbox.py`, `tests/unit/tools/test_docker_sandbox_mock.py`, `tests/integration/tools/test_docker_sandbox_live.py`, `tests/integration/tools/test_docker_sandbox_seccomp.py`, `tests/integration/tools/test_docker_sandbox_isolation.py`
  - Приёмка unit-mock: `uv run pytest tests/unit/tools/test_docker_sandbox_mock.py -v`; live: `ATM_ENABLE_DOCKER_TESTS=1 ATM_AUTO_PULL_IMAGES=1 uv run pytest -m docker tests/integration/tools -v`

- [x] **Step 9** — Local — DiffTool + TodoWriteTool + PlanUpdateTool
  - Тип: tdd | Зависит от: 1 | Параллельно с: 3, 4, 5, 6, 7, 8, 10, 11, 12, 13 | Статус: `[x]`
  - `DiffTool` (difflib.unified_diff, output `{"diff": str}`). `TodoWriteTool` (output `{"state_update": {"shared": {"todos": ...}}}`, НЕ signals). `PlanUpdateTool` (output `{"state_update": {"shared": {"plan": ...}}}`). Пустой stub `local_/__init__.py`.
  - Файлы: `src/atm/tools/local_/__init__.py` (empty stub), `src/atm/tools/local_/diff.py`, `src/atm/tools/local_/todo_write.py`, `src/atm/tools/local_/plan_update.py`, `tests/unit/tools/test_diff.py`, `tests/unit/tools/test_todo_write.py`, `tests/unit/tools/test_plan_update.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_diff.py tests/unit/tools/test_todo_write.py tests/unit/tools/test_plan_update.py -v` (~8 тестов)

- [x] **Step 10** — Local — FileWriteTool (atomic, workspace-scoped)
  - Тип: tdd | Зависит от: 1 | Параллельно с: 3, 4, 5, 6, 7, 8, 9, 11, 12, 13 | Статус: `[x]`
  - `FileWriteTool(workspace)`. Path traversal protection. Atomic write: `.tmp` + `.rename`. Output: `{"bytes_written", "path"}`.
  - Файлы: `src/atm/tools/local_/file_write.py`, `tests/unit/tools/test_file_write.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_file_write.py -v` (~6-7 тестов: happy, traversal reject, exists reject, parent missing, create_parents, overwrite)

- [x] **Step 11** — Local — CodeRunTool + TestRunTool
  - Тип: tdd | Зависит от: 1, 3 | Параллельно с: 4, 5, 6, 7, 8, 9, 10, 12, 13 | Статус: `[x]`
  - `CodeRunTool` (делегирует в sandbox.execute, output 6 ключей). `TestRunTool` (runner=unittest, команда `python -m unittest test_solution 2>&1`, парсит последнюю непустую строку stdout как summary). Тесты через SubprocessSandbox.
  - Файлы: `src/atm/tools/local_/code_run.py`, `src/atm/tools/local_/test_run.py`, `tests/unit/tools/test_code_run.py`, `tests/unit/tools/test_test_run.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_code_run.py tests/unit/tools/test_test_run.py -v` (~8 тестов: print, multi-file, passing test, failing test, summary parse)

- [x] **Step 12** — Local — LintTool
  - Тип: tdd | Зависит от: 1 | Параллельно с: 3, 4, 5, 6, 7, 8, 9, 10, 11, 13 | Статус: `[x]`
  - `LintTool(include_pylint=False)`. ruff через asyncio.create_subprocess_exec со stdin. Optional pylint (FileNotFoundError → pylint_skipped). Output: `{"ruff", "pylint", "pylint_skipped"}`, ok=True всегда кроме ruff crash.
  - Файлы: `src/atm/tools/local_/lint.py`, `tests/unit/tools/test_lint.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_lint.py -v` (~5 тестов: clean, F811, pylint skipped, output shape)

- [x] **Step 13** — Local — SemanticSearchTool (numpy TF-IDF)
  - Тип: tdd | Зависит от: 1, 2 | Параллельно с: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | Статус: `[x]`
  - `SemanticSearchTool(corpus_dir, top_k)`. TF-IDF numpy (idf = log((1+N)/(1+df)) + 1, L2-normalize). argsort(kind='stable'). Empty/whitespace query → ok=False, error="empty query". Output: `{"hits": [{"doc_id", "score", "snippet"}]}`.
  - Файлы: `src/atm/tools/local_/semantic_search.py`, `tests/unit/tools/test_semantic_search.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_semantic_search.py -v` (~6 тестов: top-1 per doc, empty query, whitespace query, top_k)

---

## Wave 3: Финализация (после Wave 2) ✅ ВЫПОЛНЕНО

- [x] **Step 14** — Public API finalization + build_default_registry + integration acceptance
  - Тип: tdd | Зависит от: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | Параллельно с: нет | Статус: `[x]`
  - Финализировать `__init__.py` (tools, global_, local_). Добавить `build_default_registry` в base.py (prod_mode=True → reject IS_ISOLATED=False). Integration end-to-end (chain file_write→...→diff с SubprocessSandbox). Docker acceptance (gated). test_prod_mode_rejects_subprocess. Обновить codebase-map.md.
  - Файлы: `src/atm/tools/__init__.py`, `src/atm/tools/global_/__init__.py`, `src/atm/tools/local_/__init__.py`, `src/atm/tools/base.py` (build_default_registry), `tests/integration/tools/test_registry_endtoend.py`, `tests/integration/tools/test_docker_acceptance.py`, `tests/integration/tools/test_build_registry.py`, `dev/codebase-map.md`
  - Приёмка:
    - `uv run pytest tests/unit/tools tests/integration/tools -v` — все PASSED (non-docker/non-network)
    - `ATM_ENABLE_DOCKER_TESTS=1 uv run pytest -m docker -v` — все PASSED
    - `uv run mypy src/atm/tools` — clean
    - `uv run ruff check src/atm/tools tests/unit/tools tests/integration/tools` — clean

---

## Статистика

- Всего задач: **14 steps**
- Выполнено: 14 / 14
- Ориентировочно: ~7h wall-time (~4h при полном параллелизме Wave 2)

## Как обновлять

- Пометить выполненные задачи: `[ ]` → `[x]`.
- Обновить заголовок фазы: все задачи done → `✅ ВЫПОЛНЕНО`; часть done → `🟡 В ПРОЦЕССЕ`.
- После каждой фазы обновить раздел `## ПРОГРЕСС СЕССИИ` в `m4-context.md`.
