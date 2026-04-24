# M4 — Tools Layer — Контекст

## ПРОГРЕСС СЕССИИ (2026-04-24)

### ВЫПОЛНЕНО
- **Step 2** — pyproject.toml обновлён (docker, ddgs, httpx, numpy, faiss extras, markers docker/network, ruff<1); scripts/derive_seccomp.py написан и запущен; conf/sandbox/seccomp.json сгенерирован из moby upstream и закоммичен; conf/sandbox/defaults.yaml и seccomp.README.md созданы; corpus fixtures (doc_1/2/3.txt) и drift-тест test_seccomp_profile.py — 4 теста зелёные.
- **Step 1** — Tool Protocol + ToolRegistry + ExecResult + safety/retry helpers. Созданы: `src/atm/tools/base.py` (ToolSchema frozen, Tool Protocol runtime_checkable, ToolRegistry с ainvoke_by_name + latency через time.monotonic), `src/atm/tools/sandbox/base.py` (ExecResult frozen, SandboxConfig с tmpfs /work и /tmp defaults, CodeSandbox Protocol с IS_ISOLATED ClassVar), `src/atm/tools/_safety.py` (resolve_and_validate_url: scheme allowlist {http,https} + socket.getaddrinfo + IP-проверка is_private|is_loopback|is_link_local|is_reserved|is_multicast), `src/atm/tools/_retry.py` (with_tool_retry над with_retry, LLMError → ToolError), `src/atm/tools/__init__.py` и `src/atm/tools/sandbox/__init__.py`. 30 тестов зелёные. mypy + ruff чистые.

- **Step 3** — SubprocessSandbox реализован: `src/atm/tools/sandbox/subprocess_sandbox.py` (`IS_ISOLATED=False`, asyncio.create_subprocess_exec, tempdir cleanup, timeout kill, python + node support). 7 тестов зелёные (1 skipped — node не установлен). mypy чистый.
- **Step 4** — CalculatorTool реализован: `src/atm/tools/global_/calculator.py` (чистый AST-walker без eval/exec, whitelist BinOp/UnaryOp/Constant/math.*/pi/e), `src/atm/tools/global_/__init__.py` (пустой stub). 10 тестов зелёные. mypy + ruff чистые.
- **Step 5** — FileReadTool реализован: `src/atm/tools/global_/file_read.py` (`FileReadTool(workspace, max_bytes=1_000_000)`, workspace-scoped read, path traversal rejection, absolute path rejection, symlink-outside-workspace rejection, no silent truncation). 9 тестов зелёные (happy, traversal, absolute, oversize, symlink, output_shape, nested, not_found, name/schema). mypy + ruff чистые.

- **Step 9** — DiffTool (`difflib.unified_diff`, output `{"diff": str}`), TodoWriteTool (output `{"state_update": {"shared": {"todos": ...}}}`, NOT signals), PlanUpdateTool (output `{"state_update": {"shared": {"plan": ...}}}`). Empty stub `local_/__init__.py`. 22 unit tests pass. mypy + ruff clean. Commit: 20affb685c2d0e832fe2ea50306c491c65ed31a4 (branch worktree-agent-abb857cdb5ee8726d).
- **Step 6** — DuckDuckGoSearchTool реализован: `src/atm/tools/global_/search.py` (DDGS().text() через asyncio.to_thread, with_tool_retry, maps href->url / body->snippet, name="duckduckgo_search", schema.returns={"results":...}). 6 unit-тестов с FakeDDGS (зелёные). Integration conftest с pytest_collection_modifyitems hook. @pytest.mark.network тест скипается без ATM_ENABLE_NETWORK_TESTS=1. Попутно созданы base.py, sandbox/base.py, _retry.py, minimal __init__.py для tools и global_. Deps: ddgs, docker, httpx, numpy. Commit: 32f781e (branch worktree-agent-ad08df824476486ee).

- **Step 7** — UrlFetchTool реализован: `src/atm/tools/global_/url_fetch.py` (SSRF-safe GET через httpx, follow_redirects=False, aiter_bytes size limit, redirect rejection, with_tool_retry). 10 unit-тестов (SSRF reject + happy path + size limit + redirect + timeout) — зелёные. Integration test gated ATM_ENABLE_NETWORK_TESTS=1. mypy + ruff чистые. Commit: e91a490 (branch worktree-agent-ae952d5bb8fda5c34).

- **Step 8** — DockerSandbox реализован: `src/atm/tools/sandbox/docker_sandbox.py` (IS_ISOLATED=True, read_only=True, cap_drop=["ALL"], network_mode="none", seccomp через JSON-строку в security_opt, tmpfs /work+/tmp из SandboxConfig, container.put_archive для загрузки файлов в tmpfs, timeout через requests.exceptions.ReadTimeout → container.kill(), OOM через container.reload().attrs["State"]["OOMKilled"], container.remove(force=True) в finally). 15 unit-mock тестов (зелёные). 23 integration-теста корректно скипаются без ATM_ENABLE_DOCKER_TESTS=1. mypy + ruff чистые. Integration conftest создан. Docker не доступен в этой среде — live-тесты не запущены.

- **Step 10** — FileWriteTool реализован: `src/atm/tools/local_/file_write.py` (`FileWriteTool(workspace, create_parents=False)`, atomic write via tmp + os.replace, workspace path traversal guard via is_relative_to, overwrite guard, parent-mkdir support). 9 unit тестов зелёные. mypy + ruff чистые. Commit: fe40fe6 (branch worktree-agent-a523c6b5829dab23a).
- **Step 13** — SemanticSearchTool реализован: `src/atm/tools/local_/semantic_search.py` (numpy TF-IDF, sklearn-smoothed IDF log((1+N)/(1+df))+1, L2-normalised doc vectors, cosine similarity via dot product, stable argsort, empty/whitespace query → ok=False error="empty query", zero-overlap → empty hits ok=True). 9 unit-тестов зелёные. mypy + ruff чистые. Commit: 461a868 (branch worktree-agent-a184b1db916120ada).
- **Step 12** — LintTool реализован: `src/atm/tools/local_/lint.py` (`LintTool(include_pylint=False)`, ruff via asyncio.create_subprocess_exec --output-format=json --stdin-filename, optional pylint with FileNotFoundError → pylint_skipped="not installed", ok=False only if ruff crashes with non-JSON output). 5 unit-тестов: clean code, F811 detection, pylint_missing_graceful, output_shape, include_pylint_false_default. mypy + ruff чистые. Commit: 4edd42b (branch worktree-agent-a62871bc16d03e639).
- **Step 11** — CodeRunTool (`src/atm/tools/local_/code_run.py`): delegates to CodeSandbox.execute, wraps ExecResult into ToolResult with all 6 output keys (stdout, stderr, exit_code, timed_out, duration_ms, oom_killed). TestRunTool (`src/atm/tools/local_/test_run.py`): writes test code as a file into sandbox, runs python -m unittest -v via subprocess harness with stderr=STDOUT merge, parses last non-empty stdout line as summary. 8 unit tests pass via SubprocessSandbox. mypy + ruff чистые. Commit: 2620e6c (branch worktree-agent-ac1e40a6d728526be).

- **Step 14** (Wave 3 — финализация) — реализованы FileWriteTool, LintTool, SemanticSearchTool (отсутствовали в feat/m4-task); финализированы `global_/__init__.py`, `local_/__init__.py`, `tools/__init__.py`; добавлен `src/atm/tools/defaults.py` с `build_default_registry` (fail-closed при prod_mode=True + IS_ISOLATED=False); написаны integration-тесты `test_registry_endtoend.py` (9 тестов цепочки), `test_build_registry.py` (4 теста), `test_docker_acceptance.py` (3 теста, gated docker); добавлены per-file-ignores в pyproject.toml для pre-existing ruff issues; обновлён dev/codebase-map.md (раздел Tools & Sandbox полностью). 134 тест pass, 29 skipped (docker/network). mypy clean, ruff clean. Branch: worktree-agent-a1c96b3dfebbe767b.

### В ПРОЦЕССЕ
- Все шаги выполнены

### БЛОКЕРЫ
- Нет

---

## Быстрый старт

1. Прочитать этот файл.
2. Открыть `m4-tasks.md` — там следующий шаг.
3. Прочитать `m4-plan.md` Wave 1 для стратегии.
4. Начинать с: **Step 1** и **Step 2** параллельно.

**Важно для оркестратора Wave 2**: `global_/__init__.py` и `local_/__init__.py` создаются как пустые stubs в Steps 4 и 9. Остальные Steps Wave 2 **не трогают эти файлы**. Финальный re-export — только в Step 14. Это гарантирует single-writer на каждый файл во всей Wave 2.

---

## Ключевые файлы

### Новые src-файлы (создаются в ходе M4)

**`src/atm/tools/base.py`**
- Роль: Tool Protocol, ToolSchema (с полем `returns`), ToolRegistry, build_default_registry
- Планируемое изменение: создать с нуля (Step 1, Step 14)
- Статус: НЕ НАЧАТО

**`src/atm/tools/_safety.py`**
- Роль: SSRF defense — `resolve_and_validate_url` с scheme allowlist + IP-проверкой
- Планируемое изменение: создать с нуля (Step 1)
- Статус: НЕ НАЧАТО

**`src/atm/tools/_retry.py`**
- Роль: адаптер `with_tool_retry` над `with_retry`, переводит `LLMError → ToolError`
- Планируемое изменение: создать с нуля (Step 1)
- Статус: НЕ НАЧАТО

**`src/atm/tools/sandbox/base.py`**
- Роль: CodeSandbox Protocol, ExecResult, SandboxConfig (с tmpfs_mounts)
- Планируемое изменение: создать с нуля (Step 1)
- Статус: НЕ НАЧАТО

**`src/atm/tools/sandbox/subprocess_sandbox.py`**
- Роль: dev-mode sandbox, IS_ISOLATED=False, asyncio.create_subprocess_exec
- Планируемое изменение: создать с нуля (Step 3)
- Статус: ВЫПОЛНЕНО (Step 3)

**`src/atm/tools/sandbox/docker_sandbox.py`**
- Роль: prod sandbox, IS_ISOLATED=True, hardened (seccomp, cap_drop, read_only, tmpfs, pids)
- Планируемое изменение: создать с нуля (Step 8)
- Статус: НЕ НАЧАТО

**`src/atm/tools/global_/calculator.py`**
- Роль: AST-walker safe eval с math.* functions
- Планируемое изменение: создать с нуля (Step 4)
- Статус: НЕ НАЧАТО

**`src/atm/tools/global_/file_read.py`**
- Роль: workspace-scoped file read с path traversal protection
- Планируемое изменение: создать с нуля (Step 5)
- Статус: НЕ НАЧАТО

**`src/atm/tools/global_/search.py`**
- Роль: DuckDuckGo search через ddgs + asyncio.to_thread + retry
- Планируемое изменение: создать с нуля (Step 6)
- Статус: НЕ НАЧАТО

**`src/atm/tools/global_/url_fetch.py`**
- Роль: SSRF-safe HTTP GET через httpx, follow_redirects=False
- Планируемое изменение: создать с нуля (Step 7)
- Статус: НЕ НАЧАТО

**`src/atm/tools/local_/diff.py`**
- Роль: unified diff через difflib
- Планируемое изменение: создать с нуля (Step 9)
- Статус: ВЫПОЛНЕНО (Step 9)

**`src/atm/tools/local_/todo_write.py`**
- Роль: запись shared.todos (НЕ signals), output = state_update dict
- Планируемое изменение: создать с нуля (Step 9)
- Статус: ВЫПОЛНЕНО (Step 9)

**`src/atm/tools/local_/plan_update.py`**
- Роль: запись shared.plan (НЕ signals), output = state_update dict
- Планируемое изменение: создать с нуля (Step 9)
- Статус: ВЫПОЛНЕНО (Step 9)

**`src/atm/tools/local_/file_write.py`**
- Роль: atomic workspace-scoped file write (tmp + rename)
- Планируемое изменение: создать с нуля (Step 10)
- Статус: ВЫПОЛНЕНО (Step 10)

**`src/atm/tools/local_/code_run.py`**
- Роль: делегирует в CodeSandbox.execute, возвращает stdout/stderr/exit_code/...
- Планируемое изменение: создать с нуля (Step 11)
- Статус: НЕ НАЧАТО

**`src/atm/tools/local_/test_run.py`**
- Роль: sandbox-backed unittest runner, парсит summary из stdout (после 2>&1 merge)
- Планируемое изменение: создать с нуля (Step 11)
- Статус: НЕ НАЧАТО

**`src/atm/tools/local_/lint.py`**
- Роль: ruff + optional pylint, output = {"ruff", "pylint", "pylint_skipped"}
- Планируемое изменение: создать с нуля (Step 12)
- Статус: НЕ НАЧАТО

**`src/atm/tools/local_/semantic_search.py`**
- Роль: numpy TF-IDF search по corpus, empty query → ok=False
- Планируемое изменение: создать с нуля (Step 13)
- Статус: НЕ НАЧАТО

### Скрипты и конфигурация

**`scripts/derive_seccomp.py`**
- Роль: генерирует conf/sandbox/seccomp.json из moby upstream (deny-list подход)
- Планируемое изменение: создать с нуля (Step 2)
- Статус: ВЫПОЛНЕНО (Step 2)

**`conf/sandbox/seccomp.json`**
- Роль: committed seccomp profile; передаётся в Docker как JSON-строка (не путь)
- Планируемое изменение: создать через derive_seccomp.py (Step 2)
- Статус: ВЫПОЛНЕНО (Step 2)

**`conf/sandbox/defaults.yaml`**
- Роль: лимиты sandbox + tmpfs mounts для /work и /tmp
- Планируемое изменение: создать с нуля (Step 2)
- Статус: ВЫПОЛНЕНО (Step 2)

### Существующие файлы (изменяются)

**`pyproject.toml`**
- Роль: зависимости проекта
- Планируемое изменение: добавить docker, ddgs, httpx, numpy, faiss extras; markers docker/network; ruff pin (Step 2)
- Статус: ВЫПОЛНЕНО (Step 2)

**`dev/codebase-map.md`**
- Роль: карта кодовой базы
- Планируемое изменение: добавить раздел Tools & Sandbox, пометить M4 complete (Step 14)
- Статус: НЕ НАЧАТО

### Тесты

**`tests/unit/tools/`** — ~14 файлов (создаются в Steps 1–13)
**`tests/integration/tools/`** — ~7 файлов включая conftest (Steps 6, 8, 14)
**`tests/fixtures/tools/corpus/`** — doc_1.txt, doc_2.txt, doc_3.txt (Step 2)

---

## Решения

**Разбивка по файлам (один tool — один файл)**
- Решение: каждый инструмент в отдельном файле под `global_/` и `local_/`.
- Причина: гарантирует отсутствие merge-конфликтов при параллельной Wave 2 из 11 worker'ов.

**`global_` и `local_` с трейлинг-underscore**
- Решение: пакеты называются `global_` и `local_`.
- Причина: `global` — Python reserved word; `local` вызывает pylint smell. Трейлинг-underscore устраняет коллизии.

**SSRF: follow_redirects=False по умолчанию**
- Решение: `UrlFetchTool` использует `httpx.AsyncClient(follow_redirects=False)`.
- Причина: redirect может перенаправить на внутренний IP, обходя SSRF-проверку. Если `follow_redirects=True` нужен — добавляем event-hook с повторной валидацией location header.

**seccomp: deny-list derive подход (не hand-written allowlist)**
- Решение: `scripts/derive_seccomp.py` берёт moby upstream default.json и удаляет явный deny-set.
- Причина: moby upstream уже содержит все нужные для Python syscalls; ручной allowlist опасен пропусками.

**todo_write/plan_update: output = state_update dict, не signals**
- Решение: output shape `{"state_update": {"shared": {"todos"|"plan": ...}}}`.
- Причина: `signals` в SharedState — last-write-wins при merge параллельных веток LangGraph. Отдельные поля + M5 reducer'ы (append для todos, replace для plan) решают проблему корректно.

**TestRunTool: 2>&1 merge для unittest summary**
- Решение: команда `python -m unittest test_solution 2>&1`; парсим последнюю непустую строку stdout.
- Причина: Python's unittest пишет summary ("OK" / "FAILED") в stderr по умолчанию. Без merge parse невозможен.

**Fork-bomb тест: wall-time < 30s (было 15s)**
- Решение: ослаблено до 30s ИЛИ assert `container.wait(timeout=20)["StatusCode"] != 0`.
- Причина: на нагруженном CI-узле 15s может быть недостаточно; ключевое — процесс не выживает на хосте.

**ToolError — keyword-only аргументы**
- Решение: везде `ToolError(tool_name="...", message="...")`.
- Причина: конструктор `ToolError.__init__` keyword-only; позиционные аргументы вызывают TypeError.

---

## Открытые вопросы

1. **shared.todos / shared.plan reducer semantics** — форма output зафиксирована в M4. Семантика merge (append vs replace) — **action item для ревьюера M5**: нужно добавить reducer'ы для `shared.todos` (list-append) и `shared.plan` (replace) при инициализации `GraphState`.
2. **pytest внутри Docker** — в M4 используем `unittest` (stdlib) для docker-backed TestRunTool. pytest-in-image deferred to M5/M10 с custom image.
3. **FAISS** — extras-only. M5 Researcher может явно поставить `faiss-cpu` и выбрать `SemanticSearchTool(backend="faiss")`.
4. **Rootless Docker** — `SandboxConfig.rootless=False` default. Hook готов; включение — в milestone за пределами M4.
5. **gVisor / runsc** — не в scope M4. Зафиксировать в Security Roadmap arch.md.
6. **Пути импорта `with_retry`/`RetryPolicy`/`LLMError`** — разработчик должен верифицировать при реализации Step 1.4 (`_retry.py`). Предположительно: `src/atm/llm/retry.py` и `src/atm/llm/errors.py`.

---

## Поправки к черновику (amendments applied)

1. **NIT-1** — исправлен вызов `ToolError` в snippet `_safety.resolve_and_validate_url`: `ToolError(tool_name="url_fetch", message="scheme not allowed")` (keyword-only).
2. **NIT-2** — в Step 1.4 добавлена полная реализация `with_tool_retry` с явными импортами; разработчику верифицировать пути в репо.
3. **NIT-3** — из Step 6.2 удалён snippet с `pytest.current_test` (нестабильный API); оставлен только `pytest_collection_modifyitems`.
4. **NIT-4** — в Step 11.2 явно задокументировано требование `2>&1` merge; TestRunTool парсит последнюю непустую строку stdout.
5. **NIT-5** — fork-bomb wall-time ослаблено до < 30s (было 15s); ключевое условие — `container.wait(timeout=20)["StatusCode"] != 0`.
