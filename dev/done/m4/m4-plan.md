# M4 — Tools Layer (Tool Protocol, Registry, Sandbox, Global/Local Tools) — План

## Краткое резюме

Реализовать слой инструментов `src/atm/tools/*`: протокол `Tool` + `ToolRegistry`, протокол `CodeSandbox` + две реализации (`DockerSandbox` с hardening по arch.md §5.2/§17#7 и `SubprocessSandbox` для dev-режима), набор глобальных (`calculator`, `duckduckgo_search`, `url_fetch`, `file_read`) и локальных (`code_run`, `file_write`, `test_run`, `diff`, `lint`, `semantic_search`, `todo_write`, `plan_update`) инструментов. Переиспользовать типы из M1 (`ToolCall`, `ToolResult`, `ToolError`, `SharedState`). Docker- и network-тесты гейтим env-var / pytest-marker; subprocess-sandbox и остальные unit-тесты запускаются всегда.

**Класс задачи**: complex
**Причина**: Два новых протокола-контракта (Tool, CodeSandbox), три хардненых подсистемы (Docker, subprocess, seccomp-derivation pipeline), 12 отдельных tool-реализаций, новые runtime-зависимости, live safety-проверка.

**Нужны интеграционные тесты**: да
**Причина**: `Tool` и `CodeSandbox` — published contracts для M5. Docker-isolation acceptance-тесты (`rm -rf /`, fork-bomb, per-syscall seccomp-block) не покрываемы unit-тестами — нужен живой Docker daemon. `prod_mode` gating — integration-level fail-closed контракт.

## Текущее состояние

- **M1 готов**: `src/atm/core/types.py` — `ToolCall`, `ToolResult` (`frozen=True` Pydantic v2); `ToolError(tool_name, message)` в `errors.py`; `SharedState` (TypedDict, total=False) с `signals: dict[str, Any]`.
- **M2 готов**: `src/atm/llm/retry.py` — `with_retry(fn, policy, provider, model)`, бросает `LLMError`; `src/atm/llm/errors.py` — `LLMError`. Используются как референс для адаптера `_retry.with_tool_retry`.
- **pyproject.toml**: Python 3.11+, pydantic>=2.7, pyarrow>=16.

## Предлагаемый подход

Каждый инструмент — отдельный файл (в `global_/` и `local_/`), чтобы Wave 2 шла параллельно без merge-конфликтов. Контракты (base, safety, retry) — Wave 1 вместе с deps/conf. Финальный `__init__.py` и integration acceptance — Wave 3 (Step 14).

**Ключевые архитектурные решения:**
- `global` — Python reserved word, поэтому пакет называется `global_` (трейлинг-underscore); `local_` — аналогично.
- `_safety.resolve_and_validate_url`: scheme allowlist + `socket.getaddrinfo` + IP-проверка (`is_private|is_loopback|is_link_local|is_reserved|is_multicast`). `UrlFetchTool` по умолчанию `follow_redirects=False`.
- `conf/sandbox/seccomp.json` генерируется из moby upstream скриптом `scripts/derive_seccomp.py` (deny-list подход), коммитится в репо; в CI используется результат, не скрипт.
- `SandboxConfig.tmpfs_mounts` по умолчанию включает `/work` и `/tmp` (Python tempfile/pyc требует `/tmp`).
- `build_default_registry(prod_mode=True, ...)` отказывается принимать sandbox с `IS_ISOLATED=False` — raise `ToolError` при сборке.
- `todo_write` / `plan_update` выводят `{"state_update": {"shared": {"todos"|"plan": ...}}}` — НЕ в `signals`, чтобы избежать last-write-wins при merge веток. Reducer-семантика (append vs replace) — ответственность M5.
- FAISS — extras-only (`[project.optional-dependencies] faiss = ["faiss-cpu>=1.11"]`); default backend `SemanticSearchTool` — numpy TF-IDF.

## Фазы реализации

### Wave 1 — Контракты + deps/conf (~2h суммарно, параллельно)

#### Step 1: Tool Protocol + ToolRegistry + ExecResult + safety/retry helpers (~1.5h)
**Тип**: tdd
**Цель**: создать весь контрактный слой, на который опираются все последующие шаги.

- [ ] 1.1 `src/atm/tools/base.py` — `ToolSchema` (Pydantic frozen; fields: `name`, `description`, `parameters: dict[str, Any]`, `returns: dict[str, Any]`), `Tool` Protocol (`runtime_checkable=True`, `name: ClassVar[str]`, `schema: ClassVar[ToolSchema]`, `async def ainvoke(args) -> ToolResult`), `ToolRegistry` (register/get/names/ainvoke-by-name с wrapping exceptions и измерением latency через `time.monotonic`).
  - Файл: `src/atm/tools/base.py`
  - Приёмка: `test_tool_schema_frozen_has_returns_field`, `test_registry_register_get_names`, `test_registry_duplicate_raises`, `test_registry_unknown_raises`, `test_registry_ainvoke_wraps_exception_in_toolresult`, `test_registry_ainvoke_measures_latency` — все зелёные.

- [ ] 1.2 `src/atm/tools/sandbox/base.py` — `ExecResult` (frozen Pydantic: stdout, stderr, exit_code, duration_ms, timed_out, oom_killed), `SandboxConfig` (cpu_quota, mem_limit, pids_limit, timeout_s, workdir_size, `tmpfs_mounts: dict[str,str]` с дефолтом `{"/work":"size=64m,noexec,nosuid,uid=1000,gid=1000", "/tmp":"size=64m,noexec,nosuid"}`; network, image_map, seccomp_path, rootless), `CodeSandbox` Protocol (`IS_ISOLATED: ClassVar[bool]`, `async def execute(...) -> ExecResult`).
  - Файлы: `src/atm/tools/sandbox/__init__.py`, `src/atm/tools/sandbox/base.py`
  - Приёмка: `test_sandbox_config_tmpfs_defaults_include_tmp`, `test_exec_result_frozen_fields` — зелёные.

- [ ] 1.3 `src/atm/tools/_safety.py` — `resolve_and_validate_url(url: str, allow_private: bool=False) -> str`.
  - Файл: `src/atm/tools/_safety.py`
  - Реализация:
    ```python
    def resolve_and_validate_url(url: str, allow_private: bool = False) -> str:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError(tool_name="url_fetch", message="scheme not allowed")
        host = parsed.hostname
        if not host:
            raise ToolError(tool_name="url_fetch", message="missing host")
        infos = socket.getaddrinfo(host, None)
        for family, _, _, _, sockaddr in infos:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast) and not allow_private:
                raise ToolError(tool_name="url_fetch", message=f"private/reserved address: {ip_str}")
        return url
    ```
  - Приёмка: `test_safety_rejects_metadata_ip` (169.254.169.254), `test_safety_rejects_loopback`, `test_safety_rejects_ipv6_localhost` ([::1]), `test_safety_rejects_rfc1918` (10.0.0.1), `test_safety_rejects_ftp_scheme`, `test_safety_happy_path_public` (mock via `monkeypatch.setattr(socket, "getaddrinfo", ...)`), `test_safety_allow_private_true_permits_localhost` — все зелёные.

- [ ] 1.4 `src/atm/tools/_retry.py` — `with_tool_retry(policy: RetryPolicy, tool_name: str)` декоратор.
  - Файл: `src/atm/tools/_retry.py`
  - Реализация (проверить точные пути импорта в репо — `with_retry`/`RetryPolicy` скорее всего в `src/atm/llm/retry.py`, `LLMError` в `src/atm/llm/errors.py`):
    ```python
    from functools import wraps
    from atm.llm.retry import with_retry, RetryPolicy
    from atm.llm.errors import LLMError
    from atm.core.errors import ToolError

    def with_tool_retry(policy: RetryPolicy, tool_name: str):
        def decorator(fn):
            @wraps(fn)
            async def wrapper(*args, **kwargs):
                try:
                    return await with_retry(
                        lambda: fn(*args, **kwargs),
                        policy=policy,
                        provider="tool",
                        model=tool_name,
                    )
                except LLMError as e:
                    raise ToolError(tool_name=tool_name, message=str(e)) from e
            return wrapper
        return decorator
    ```
  - Приёмка: `test_with_tool_retry_converts_llmerror_to_toolerror`, `test_with_tool_retry_preserves_ok_path` — зелёные.

- [ ] 1.5 `src/atm/tools/__init__.py` (минимальный, полный будет в Step 14) + `tests/unit/tools/__init__.py`.
  - Файлы: `src/atm/tools/__init__.py`, `tests/unit/tools/__init__.py`
  - Приёмка: `from atm.tools import Tool, ToolSchema, ToolRegistry, CodeSandbox, ExecResult, SandboxConfig` не падает.

**Верификация Step 1**: `uv run pytest tests/unit/tools/test_base.py tests/unit/tools/test_sandbox_base.py tests/unit/tools/test_safety.py tests/unit/tools/test_retry.py -v && uv run mypy src/atm/tools/base.py src/atm/tools/sandbox/base.py src/atm/tools/_safety.py src/atm/tools/_retry.py`

---

#### Step 2: Dependencies bump + seccomp derivation pipeline + fixtures (~0.5h)
**Тип**: simple
**Цель**: все зависимости, конфиг sandbox и тестовые фикстуры готовы до старта Wave 2.

- [ ] 2.1 `pyproject.toml` — добавить `docker>=7.1,<8`, `ddgs>=9.13,<10`, `httpx>=0.27`, `numpy>=1.26`; extras `faiss = ["faiss-cpu>=1.11"]`; markers `docker`, `network`; `ruff>=0.9,<1` в dev-group; узкие `filterwarnings`: `ignore::RuntimeWarning:ddgs`, `ignore::ResourceWarning:docker.*`.
  - Файл: `pyproject.toml`
  - Приёмка: `uv sync` зелёный, все пакеты резолвятся.

- [ ] 2.2 `scripts/derive_seccomp.py` — фетчит moby upstream default.json, удаляет deny-set, записывает `conf/sandbox/seccomp.json` (pretty-printed, `indent=2, sort_keys=True`). Запустить один раз и закоммитить результат.
  - Файлы: `scripts/derive_seccomp.py`, `conf/sandbox/seccomp.json`
  - Deny-set: `{ptrace, mount, umount, umount2, unshare, keyctl, bpf, pivot_root, chroot, setns, clone3, add_key, request_key, finit_module, init_module, delete_module, create_module, ioperm, iopl, kexec_load, kexec_file_load, reboot, swapon, swapoff, sysfs, lookup_dcookie}`.
  - Приёмка: `uv run python -c "import json; json.load(open('conf/sandbox/seccomp.json'))"` — успех.

- [ ] 2.3 `conf/sandbox/seccomp.README.md` — how-to по regenerate и список deny-set. `conf/sandbox/defaults.yaml` — лимиты и tmpfs mounts.
  - Файлы: `conf/sandbox/seccomp.README.md`, `conf/sandbox/defaults.yaml`
  - Приёмка: файлы существуют, `defaults.yaml` содержит `/work` и `/tmp` в `tmpfs_mounts`.

- [ ] 2.4 Тестовые фикстуры corpus — три файла с уникальными target-термами: `doc_1.txt` ("neural"), `doc_2.txt` ("carrot"), `doc_3.txt` ("Paris"). Drift-тест `test_seccomp_profile.py`.
  - Файлы: `tests/fixtures/tools/__init__.py`, `tests/fixtures/tools/corpus/doc_1.txt`, `doc_2.txt`, `doc_3.txt`, `tests/unit/tools/test_seccomp_profile.py`
  - Приёмка: `uv run pytest tests/unit/tools/test_seccomp_profile.py -v` — drift-тест зелёный (denied syscalls не попадают в allowed set, `defaultAction == SCMP_ACT_ERRNO`).

**Верификация Step 2**: `uv sync && uv run pytest tests/unit/tools/test_seccomp_profile.py -v`

---

### Wave 2 — Реализации инструментов (~4h, все параллельно после Wave 1)

#### Step 3: SubprocessSandbox (dev-mode, non-isolated) (~0.5h)
**Тип**: tdd
**Зависит от**: 1, 2
**Параллельно с**: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13

- [ ] 3.1 `src/atm/tools/sandbox/subprocess_sandbox.py` — `SubprocessSandbox`, `IS_ISOLATED=False`. `async def execute(lang, code, files=None, timeout=None) -> ExecResult`: tempdir + write code + `asyncio.create_subprocess_exec`; timeout через `asyncio.wait_for`; cleanup в finally; `sys.executable` для python; `shutil.which("node")` для node; `env={"PATH": os.environ.get("PATH", "")}`. Docstring: "Dev-only. NOT isolated. DO NOT use in prod."
  - Файл: `src/atm/tools/sandbox/subprocess_sandbox.py`
  - Приёмка: `print(1+1)` → stdout "2\n", nonzero exit, timeout (sleep 10 с timeout=0.3), multi-file — все зелёные.

**Верификация Step 3**: `uv run pytest tests/unit/tools/test_subprocess_sandbox.py -v`

---

#### Step 4: Global — CalculatorTool (~0.5h)
**Тип**: tdd
**Зависит от**: 1
**Параллельно с**: 3, 5, 6, 7, 8, 9, 10, 11, 12, 13

- [ ] 4.1 `src/atm/tools/global_/calculator.py` — AST-walker safe eval: whitelist `BinOp`, `UnaryOp`, `Constant(int|float)`, `Call` только на `math.*` через `_SAFE_FUNCS`. Output: `{"value": <numeric>, "expression": <str>}`. `ToolSchema.returns` задан явно.
  - Файл: `src/atm/tools/global_/calculator.py`
  - `_SAFE_FUNCS`: `{sqrt, sin, cos, log, log10, exp, pow, floor, ceil, abs, min, max}`.
  - `eval(compiled, {"__builtins__": {}}, _SAFE_FUNCS)`.
  - Приёмка: `2+2 → value=4`, `sqrt(16) → 4.0`, `__import__('os')... → ok=False`, `x+1 → ok=False`, output содержит оба ключа.

- [ ] 4.2 `src/atm/tools/global_/__init__.py` — создать как пустой stub (финализируется в Step 14).
  - Файл: `src/atm/tools/global_/__init__.py`

**Верификация Step 4**: `uv run pytest tests/unit/tools/test_calculator.py -v`

---

#### Step 5: Global — FileReadTool (~0.5h)
**Тип**: tdd
**Зависит от**: 1
**Параллельно с**: 3, 4, 6, 7, 8, 9, 10, 11, 12, 13

- [ ] 5.1 `src/atm/tools/global_/file_read.py` — `FileReadTool(workspace: Path, max_bytes: int = 1_000_000)`. Проверка пути через `is_relative_to(workspace.resolve())` до `st_size`. При превышении — `ok=False, error="file too large"` (не silent-truncate). Output: `{"content": str, "path": str (relative), "size": int}`.
  - Файл: `src/atm/tools/global_/file_read.py`
  - Приёмка: happy path, reject `../../etc/passwd`, reject абсолютный путь вне workspace, reject oversize, symlink outside workspace (resolve+check) — все зелёные.

**Верификация Step 5**: `uv run pytest tests/unit/tools/test_file_read.py -v`

---

#### Step 6: Global — DuckDuckGoSearchTool (~0.5h)
**Тип**: tdd
**Зависит от**: 1, 2
**Параллельно с**: 3, 4, 5, 7, 8, 9, 10, 11, 12, 13

- [ ] 6.1 `src/atm/tools/global_/search.py` — `DuckDuckGoSearchTool`. `ddgs.DDGS().text(query, max_results)` через `asyncio.to_thread`. Retry через `with_tool_retry`. Output: `{"results": [{"title": str, "href": str, "body": str}, ...]}`. Unit-тесты через monkeypatch mock; integration-тест `@pytest.mark.network`.
  - Файл: `src/atm/tools/global_/search.py`
  - Приёмка: unit-тест с FakeDDGS (context manager) — зелёный; integration skipped без `ATM_ENABLE_NETWORK_TESTS=1`.

- [ ] 6.2 `tests/integration/tools/conftest.py` — **только** `pytest_collection_modifyitems` hook (без `pytest.current_test`):
  - Файл: `tests/integration/tools/conftest.py`, `tests/integration/tools/__init__.py`
  - Реализация:
    ```python
    def pytest_collection_modifyitems(config, items):
        import os
        if os.environ.get("ATM_ENABLE_DOCKER_TESTS") != "1":
            skip_docker = pytest.mark.skip(reason="Docker tests disabled (set ATM_ENABLE_DOCKER_TESTS=1)")
            for item in items:
                if "docker" in item.keywords:
                    item.add_marker(skip_docker)
        if os.environ.get("ATM_ENABLE_NETWORK_TESTS") != "1":
            skip_network = pytest.mark.skip(reason="Network tests disabled (set ATM_ENABLE_NETWORK_TESTS=1)")
            for item in items:
                if "network" in item.keywords:
                    item.add_marker(skip_network)
    ```
  - Приёмка: тесты с `@pytest.mark.docker` скипаются без env-var, пропускаются с ним.

**Верификация Step 6**: `uv run pytest tests/unit/tools/test_duckduckgo_search.py -v`

---

#### Step 7: Global — UrlFetchTool (SSRF-safe) (~0.75h)
**Тип**: tdd
**Зависит от**: 1, 2
**Параллельно с**: 3, 4, 5, 6, 8, 9, 10, 11, 12, 13

- [ ] 7.1 `src/atm/tools/global_/url_fetch.py` — `UrlFetchTool(allow_private: bool=False, max_bytes: int=1_000_000, timeout: float=10.0)`. Вызов `resolve_and_validate_url` перед каждым GET. `httpx.AsyncClient(follow_redirects=False)`. Ограничение размера через `aiter_bytes`. Retry через `with_tool_retry`. Output: `{"status": int, "content": str, "content_type": str, "url": str}`. Unit-тесты через `httpx.MockTransport`.
  - Файл: `src/atm/tools/global_/url_fetch.py`
  - SSRF-тесты (mock `socket.getaddrinfo`): `http://169.254.169.254/` → ok=False, `http://127.0.0.1:8000/` → ok=False, `http://[::1]/` → ok=False, `http://10.0.0.1/` → ok=False, `ftp://example.com/` → ok=False "scheme not allowed".
  - Happy path через `httpx.MockTransport` с `allow_private=True`.
  - Приёмка: ~10 unit-тестов зелёные; integration skipped без `ATM_ENABLE_NETWORK_TESTS=1`.

**Верификация Step 7**: `uv run pytest tests/unit/tools/test_url_fetch.py -v`

---

#### Step 8: DockerSandbox (hardened) + live seccomp/isolation tests (~1.5h)
**Тип**: tdd
**Зависит от**: 1, 2
**Параллельно с**: 3, 4, 5, 6, 7, 9, 10, 11, 12, 13

- [ ] 8.1 `src/atm/tools/sandbox/docker_sandbox.py` — `DockerSandbox`, `IS_ISOLATED=True`. Конструктор принимает `SandboxConfig` + `prefetch: bool=False` (или `ATM_AUTO_PULL_IMAGES=1`). Загружает seccomp JSON как строку один раз в конструкторе. `execute` через `asyncio.to_thread` → sync `_execute_sync`. `containers.run(image, command, detach=True, read_only=True, tmpfs=cfg.tmpfs_mounts, security_opt=["no-new-privileges", f"seccomp={seccomp_str}"], cap_drop=["ALL"], network_mode="none", mem_limit=..., nano_cpus=..., pids_limit=..., user="1000:1000", working_dir="/work")`. Файлы через `container.put_archive`. `container.wait(timeout=...)`. OOM через `container.reload()["State"]["OOMKilled"]`. `container.remove(force=True)` в finally.
  - Файл: `src/atm/tools/sandbox/docker_sandbox.py`
  - Unit-mock приёмка: `containers.run` вызван с `read_only=True`, `cap_drop=["ALL"]`, `network_mode="none"`, `tmpfs={"/work":...,"/tmp":...}`, `user="1000:1000"`.

- [ ] 8.2 Live integration-тесты (gated `@pytest.mark.docker`):
  - `test_hello_world`: stdout "2\n", exit 0.
  - `test_readonly_rootfs_rejects_write`: non-zero exit при `open("/etc/foo","w")`.
  - `test_tmp_writable_via_tmpfs`: `open("/tmp/foo","w")` → exit 0 (проверяет tmpfs).
  - `test_network_disabled`: `socket.create_connection(("1.1.1.1",53), timeout=2)` → non-zero.
  - `test_timeout_killed`: `while True: pass` → `timed_out=True`.
  - `test_pids_limit_fork_bomb`: `while True: os.fork()` → non-zero exit ИЛИ `container.wait(timeout=20)["StatusCode"] != 0` (wall-time < 30s; ключевое: процесс не выживает на хосте).
  - `test_oom_killed`: `x=[0]*(10**9)` → `oom_killed=True`.
  - Файлы: `tests/unit/tools/test_docker_sandbox_mock.py`, `tests/integration/tools/test_docker_sandbox_live.py`

- [ ] 8.3 Per-syscall seccomp live-тесты (11 штук, gated `@pytest.mark.docker`):
  - `test_seccomp_ptrace_blocked`, `test_seccomp_mount_blocked`, `test_seccomp_unshare_blocked`, `test_seccomp_bpf_blocked`, `test_seccomp_keyctl_blocked`, `test_seccomp_pivot_root_blocked`, `test_seccomp_chroot_blocked`, `test_seccomp_setns_blocked`, `test_seccomp_clone3_blocked`, `test_seccomp_add_key_blocked`, `test_seccomp_init_module_blocked`.
  - Каждый: Python snippet через `ctypes.libc.<syscall>(...)`, проверяет `errno == EPERM (1)` или `ENOSYS (38)`.
  - Файл: `tests/integration/tools/test_docker_sandbox_seccomp.py`

- [ ] 8.4 Isolation тест (gated `@pytest.mark.docker`):
  - `rm -rf /` в контейнере; после — `hashlib.sha256(host_file.read_bytes())` идентичен pre-state.
  - Файл: `tests/integration/tools/test_docker_sandbox_isolation.py`

**Верификация Step 8**:
- Unit-mock: `uv run pytest tests/unit/tools/test_docker_sandbox_mock.py -v`
- Live: `ATM_ENABLE_DOCKER_TESTS=1 ATM_AUTO_PULL_IMAGES=1 uv run pytest -m docker tests/integration/tools -v`

---

#### Step 9: Local — DiffTool + TodoWriteTool + PlanUpdateTool (~0.5h)
**Тип**: tdd
**Зависит от**: 1
**Параллельно с**: 3, 4, 5, 6, 7, 8, 10, 11, 12, 13

- [ ] 9.1 `src/atm/tools/local_/diff.py` — `DiffTool`: `difflib.unified_diff`. Output: `{"diff": str}`.
  - Файл: `src/atm/tools/local_/diff.py`

- [ ] 9.2 `src/atm/tools/local_/todo_write.py` — `TodoWriteTool`. Args: `{todos: list[Todo]}` (Todo = `{"id": str, "text": str, "status": Literal["open","done"]}`). Output: `{"state_update": {"shared": {"todos": args["todos"]}}}`. **НЕ** в `signals`.
  - Файл: `src/atm/tools/local_/todo_write.py`
  - Docstring: `"Output contains 'state_update' dict; ToolNode in M5 merges it into GraphState."`.

- [ ] 9.3 `src/atm/tools/local_/plan_update.py` — `PlanUpdateTool`. Args: `{plan: list[PlanStep]}`. Output: `{"state_update": {"shared": {"plan": args["plan"]}}}`.
  - Файл: `src/atm/tools/local_/plan_update.py`

- [ ] 9.4 `src/atm/tools/local_/__init__.py` — создать как пустой stub (финализируется в Step 14).
  - Файл: `src/atm/tools/local_/__init__.py`

- Приёмка: diff (header `@@`, `+`/`-` lines); todo_write (ключ `shared.todos`, НЕ `signals`); plan_update (ключ `shared.plan`).

**Верификация Step 9**: `uv run pytest tests/unit/tools/test_diff.py tests/unit/tools/test_todo_write.py tests/unit/tools/test_plan_update.py -v`

---

#### Step 10: Local — FileWriteTool (atomic, workspace-scoped) (~0.5h)
**Тип**: tdd
**Зависит от**: 1
**Параллельно с**: 3, 4, 5, 6, 7, 8, 9, 11, 12, 13

- [ ] 10.1 `src/atm/tools/local_/file_write.py` — `FileWriteTool(workspace: Path)`. Args: `{path, content, overwrite=False, create_parents=False}`. Safety: `resolved.is_relative_to(workspace.resolve())`. Atomic write: `path.with_suffix(path.suffix + ".tmp") → .rename(path)`. Output: `{"bytes_written": int, "path": str}`.
  - Файл: `src/atm/tools/local_/file_write.py`
  - Приёмка: happy (write new file), reject path traversal, reject exists без overwrite, reject parent missing без create_parents, create_parents=True, overwrite=True — все зелёные.

**Верификация Step 10**: `uv run pytest tests/unit/tools/test_file_write.py -v`

---

#### Step 11: Local — CodeRunTool + TestRunTool (~0.75h)
**Тип**: tdd
**Зависит от**: 1, 3
**Параллельно с**: 4, 5, 6, 7, 8, 9, 10, 12, 13

- [ ] 11.1 `src/atm/tools/local_/code_run.py` — `CodeRunTool(sandbox: CodeSandbox, lang="python")`. Output: `{"stdout", "stderr", "exit_code", "timed_out", "oom_killed", "duration_ms"}` (все 6 ключей).
  - Файл: `src/atm/tools/local_/code_run.py`

- [ ] 11.2 `src/atm/tools/local_/test_run.py` — `TestRunTool(sandbox: CodeSandbox, runner="unittest")`. Пишет `solution.py` и `test_solution.py` в sandbox files. Runner `unittest`: `python -m unittest test_solution 2>&1` (merge stderr в stdout для парсинга summary). Парсит последнюю непустую строку stdout как summary ("OK" / "FAILED (failures=N, errors=M)"). Output: `{"passed": bool, "exit_code": int, "stdout": str, "stderr": str, "summary": str}`.
  - Файл: `src/atm/tools/local_/test_run.py`
  - Важно: команда ДОЛЖНА использовать `2>&1` merge (или `stream=sys.stdout` в harness), чтобы unittest summary попала в stdout.

- Тесты через `SubprocessSandbox`:
  - `print(1+1)` → stdout "2\n".
  - multi-file (solution.py + helper.py).
  - test_run passing: `class T(unittest.TestCase): def test_ok(self): self.assertEqual(1+1, 2)` → `passed=True`.
  - test_run failing: `self.assertEqual(1+1, 3)` → `passed=False`, non-zero exit.
  - summary parse: последняя непустая строка stdout.

**Верификация Step 11**: `uv run pytest tests/unit/tools/test_code_run.py tests/unit/tools/test_test_run.py -v`

---

#### Step 12: Local — LintTool (~0.5h)
**Тип**: tdd
**Зависит от**: 1
**Параллельно с**: 3, 4, 5, 6, 7, 8, 9, 10, 11, 13

- [ ] 12.1 `src/atm/tools/local_/lint.py` — `LintTool(include_pylint: bool=False)`. `ruff check --output-format=json --stdin-filename input.py -` через `asyncio.create_subprocess_exec`. При `include_pylint=True` — пробуем pylint; `FileNotFoundError` → `pylint_skipped="not installed"`. Output: `{"ruff": list, "pylint": list | None, "pylint_skipped": str | None}`. `ok=True` даже если pylint missing.
  - Файл: `src/atm/tools/local_/lint.py`
  - Приёмка: `print("ok")` → ruff=[], pylint=None; `import os\nimport os\n` → ruff содержит F811; `include_pylint=True` + not installed → ok=True, pylint_skipped="not installed".

**Верификация Step 12**: `uv run pytest tests/unit/tools/test_lint.py -v`

---

#### Step 13: Local — SemanticSearchTool (numpy TF-IDF) (~0.75h)
**Тип**: tdd
**Зависит от**: 1, 2
**Параллельно с**: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

- [ ] 13.1 `src/atm/tools/local_/semantic_search.py` — `SemanticSearchTool(corpus_dir: Path, top_k: int=3)`. TF-IDF numpy (formula: `idf = log((1+N)/(1+df)) + 1`, L2-normalize). Cosine similarity, `argsort(kind='stable')`. Empty/whitespace query → `ok=False, error="empty query"`. Output: `{"hits": [{"doc_id": str, "score": float, "snippet": str (первые 120 chars)}, ...]}`.
  - Файл: `src/atm/tools/local_/semantic_search.py`
  - Приёмка: "neural network" → top-1 doc_1; "carrot cake" → top-1 doc_2; "capital of France Paris" → top-1 doc_3; `""` → ok=False; `"   "` → ok=False; top_k соблюдается.

**Верификация Step 13**: `uv run pytest tests/unit/tools/test_semantic_search.py -v`

---

### Wave 3 — Финализация (~1h, последовательно после Wave 2)

#### Step 14: Public API + build_default_registry + integration acceptance (~1h)
**Тип**: tdd
**Зависит от**: все предыдущие
**Параллельно с**: нет

- [ ] 14.1 Финализировать `src/atm/tools/__init__.py` — полный публичный API.
- [ ] 14.2 Финализировать `src/atm/tools/global_/__init__.py` (re-export calculator/search/url_fetch/file_read) и `src/atm/tools/local_/__init__.py` (re-export всех 8 local tools).
- [ ] 14.3 `build_default_registry(workspace, sandbox, prod_mode=True)` в `src/atm/tools/base.py` — регистрирует все 12 tools; при `prod_mode=True` и `sandbox.IS_ISOLATED is False` → `raise ToolError(tool_name="registry", message="non-isolated sandbox in prod mode")`.
- [ ] 14.4 Integration end-to-end (non-docker): `ToolRegistry + SubprocessSandbox`, chain file_write → file_read → code_run → lint → calculator → todo_write → plan_update → semantic_search → diff.
  - Файл: `tests/integration/tools/test_registry_endtoend.py`
- [ ] 14.5 Integration docker acceptance (gated): registry с DockerSandbox, `code_run` hello-world + `test_run(unittest)` → end-to-end.
  - Файл: `tests/integration/tools/test_docker_acceptance.py`
- [ ] 14.6 `test_prod_mode_rejects_subprocess`: `build_default_registry(prod_mode=True, sandbox=SubprocessSandbox())` → `ToolError`.
  - Файл: `tests/integration/tools/test_build_registry.py`
- [ ] 14.7 Обновить `dev/codebase-map.md`: раздел "Tools & Sandbox", M4 complete, полный layout, exports list.

**Верификация Step 14**:
- `uv run pytest tests/unit/tools tests/integration/tools -v` (non-docker/non-network — все PASSED).
- `ATM_ENABLE_DOCKER_TESTS=1 uv run pytest -m docker -v` (все PASSED).
- `uv run mypy src/atm/tools` — clean.
- `uv run ruff check src/atm/tools tests/unit/tools tests/integration/tools` — clean.

---

## Ключевые файлы

| Файл | Изменение | Зачем |
|------|-----------|-------|
| `src/atm/tools/base.py` | новый | Tool Protocol, ToolRegistry, build_default_registry |
| `src/atm/tools/_safety.py` | новый | SSRF defense — resolve_and_validate_url |
| `src/atm/tools/_retry.py` | новый | адаптер with_tool_retry над with_retry |
| `src/atm/tools/sandbox/base.py` | новый | CodeSandbox Protocol, ExecResult, SandboxConfig |
| `src/atm/tools/sandbox/subprocess_sandbox.py` | новый | dev-mode sandbox, IS_ISOLATED=False |
| `src/atm/tools/sandbox/docker_sandbox.py` | новый | prod sandbox, IS_ISOLATED=True, hardened |
| `src/atm/tools/global_/calculator.py` | новый | AST-safe calculator |
| `src/atm/tools/global_/file_read.py` | новый | workspace-scoped file read |
| `src/atm/tools/global_/search.py` | новый | DuckDuckGo search |
| `src/atm/tools/global_/url_fetch.py` | новый | SSRF-safe HTTP fetch |
| `src/atm/tools/local_/diff.py` | новый | unified diff |
| `src/atm/tools/local_/todo_write.py` | новый | writes to shared.todos |
| `src/atm/tools/local_/plan_update.py` | новый | writes to shared.plan |
| `src/atm/tools/local_/file_write.py` | новый | atomic workspace-scoped file write |
| `src/atm/tools/local_/code_run.py` | новый | sandbox-backed code execution |
| `src/atm/tools/local_/test_run.py` | новый | sandbox-backed unittest runner |
| `src/atm/tools/local_/lint.py` | новый | ruff + optional pylint |
| `src/atm/tools/local_/semantic_search.py` | новый | numpy TF-IDF search |
| `scripts/derive_seccomp.py` | новый | генерирует seccomp.json из moby upstream |
| `conf/sandbox/seccomp.json` | новый (generated) | committed seccomp profile |
| `conf/sandbox/defaults.yaml` | новый | sandbox defaults с tmpfs mounts |
| `pyproject.toml` | изменён | новые deps + pytest markers |
| `dev/codebase-map.md` | изменён | раздел Tools & Sandbox |

## Зависимости и порядок

- Wave 1 (Steps 1, 2) — параллельно, нет внешних зависимостей.
- Wave 2 (Steps 3–13) — параллельно после Wave 1. Step 11 дополнительно зависит от Step 3 (SubprocessSandbox), но файлы не пересекаются.
- Wave 3 (Step 14) — строго после Wave 2. Единственная точка сборки `__init__.py`.
- `global_/__init__.py` и `local_/__init__.py` создаются как пустые stubs в Steps 4 и 9; финальный re-export — только в Step 14 (single-writer).

## Риски

| Риск | Вероятность | Влияние | Митигация |
|------|------------|---------|-----------|
| seccomp слишком строгий (hello-world падает) | низкая | высокое | moby upstream уже включает все нужные syscalls; drift-тест гарантирует только deny-set |
| `oom_killed` флапает на CI | средняя | низкое | pin `mem_limit` + гарантированно-превышающая аллокация |
| `fork_bomb` превышает wall-time | средняя | низкое | ослаблено до < 30s; ключевое — `container.wait(timeout=20)["StatusCode"] != 0` |
| `socket.getaddrinfo` в unit-тестах требует сеть | низкая | высокое | всегда мокаем через `monkeypatch.setattr(socket, "getaddrinfo", fake)` |
| upstream moby недоступен при derive | средняя | низкое | derive-скрипт только у разработчика; CI использует закоммиченный JSON |
| ruff JSON format drift | низкая | среднее | pin `<1` в pyproject |
| `state_update` shape расходится с M5 reducer | средняя | высокое | pin форму сейчас, action item для ревьюера M5 |
| seccomp JSON передаётся как строка (не path) | низкая | высокое | docker-py не раскрывает пути; проверяется на первом live-тесте |

## Вне scope

- pytest как runner в `TestRunTool` в Docker-образе (deferred to M5/M10 с custom image).
- FAISS backend (hook есть, реализация — extras-only, numpy default).
- Rootless Docker (SandboxConfig.rootless=False, hook готов).
- gVisor / runsc runtime (Security Roadmap в arch.md).
- M5 reducer'ы для `shared.todos` (append) и `shared.plan` (replace).

## Временная шкала

- Wave 1: ~2h
- Wave 2: ~4h (параллельно, max 11 worker'ов)
- Wave 3: ~1h
- **Итого**: ~7h (wall-time при полном параллелизме ~4h)
- Создан: 2026-04-24

## Поправки (amendments applied)

1. **NIT-1** — в `_safety.resolve_and_validate_url` исправлен вызов `ToolError`: `ToolError(tool_name="url_fetch", message="scheme not allowed")` (keyword-only аргументы).
2. **NIT-2** — в Step 1.4 добавлена полная реализация `with_tool_retry` с явными импортами (проверить пути в репо) и `except LLMError → raise ToolError`.
3. **NIT-3** — из Step 6.2 удалён промежуточный snippet с `pytest.current_test` (нестабильный публичный API); оставлен только `pytest_collection_modifyitems` hook.
4. **NIT-4** — в Step 11.2 явно документировано требование `2>&1` merge для `python -m unittest` (unittest пишет summary в stderr); TestRunTool парсит последнюю непустую строку из stdout после merge.
5. **NIT-5** — в Step 8.2 ослаблено условие fork-bomb теста: `host wall-time < 30s` (вместо 15s) ИЛИ `container.wait(timeout=20)["StatusCode"] != 0`; ключевое — процесс не выживает на хосте.

## Поправки к code-review (M4 review amendments)

6. **CR-6 — Todo schema: `text`→`content`; добавлен статус `in_progress`** (Step 9.2). Реализация использует поле `content` вместо `text` (более явное человекочитаемое имя) и добавляет статус `in_progress` (нужен для pipeline-состояния агента). Итоговая форма: `{"id": str, "content": str, "status": Literal["open","done","in_progress"]}`. M5 prompt templates должны использовать `content` и поддерживать `in_progress`.
7. **CR-7 — PlanStep: `text`→`title`; расширенные статусы** (Step 9.3). Реализация использует поле `title` (семантически точнее для шагов плана) и статусы `open|done|in_progress|blocked|skipped`. Форма: `{"id": str, "title": str, "status": Literal["open","done","in_progress","blocked","skipped"]}`. M5 должен использовать ключ `plan` при вызове `plan_update` (после CR-5/fix #5 — args key теперь `plan`, не `steps`).
8. **CR-8 — SemanticSearchTool snippet 200 chars vs 120** (Step 13.1). Оставлен 200-символьный сниппет: для коротких корпусов (≤ 10 документов) больший контекст улучшает полезность без значимых накладных расходов. M5 может обрезать при необходимости.
9. **CR-9 — SemanticSearchTool default top_k=5 vs 3** (Step 13.1). Оставлен `top_k=5`: более широкий набор результатов по умолчанию снижает риск пропустить релевантный документ. M5 может переопределить через аргумент `k`.
10. **CR-14 — Calculator `tan`** (Step 4.1). `tan` добавлен в `_SAFE_FUNCS` как дополнение к `sin`/`cos` (математически тривиальное расширение). Обновлённый whitelist: `{sqrt, sin, cos, tan, log, log10, exp, pow, floor, ceil, abs, min, max}`.
