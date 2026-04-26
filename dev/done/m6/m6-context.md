# M6 — Topology Framework + Star + Chain + End-to-End — Context

## SESSION PROGRESS (2026-04-24)

### COMPLETED
- **Step 2** (2026-04-24): Создана StarTopology (`atm.topology.star`). Файлы:
  - `src/atm/topology/star.py` — StarTopology с `@TopologyRegistry.register("star")`; граф на `GraphState` TypedDict (StateGraph[GraphState]); coordinator async-нода (rule-based роутер с phase-advance); `_critic_postprocess` adapter-нода; `_extract_final_answer` helper; `_route_from_coord` closure; edges: START→coordinator, coordinator→conditional, planner/executor→coordinator, critic→critic_postprocess→coordinator. `__init__.py` не редактировался.
  - `tests/unit/topology/test_star.py` — 16 тестов (все зелёные): регистрация, компиляция графа, полный прогон с approve, reject-loop до verify_max_iter, malformed critic, _critic_postprocess isolation, global max_iter cap, counter increment.
  - `tests/unit/topology/conftest.py` — autouse fixture `ensure_star_registered` для изоляции от `test_base.py::TestTopologyRegistry.teardown_method`.
  - `pyproject.toml` — добавлены `langgraph>=0.3,<1` в `[project] dependencies` и `[dependency-groups] dev` (необходимо для StateGraph API).
  - ruff clean + mypy clean; 741 unit-тестов зелёных.
- **Step 4** (2026-04-24): Создан config-слой `atm.experiment`. Файлы:
  - `src/atm/experiment/config.py` — Pydantic v2 схемы (BudgetCfg, ModelCfg, ScratchpadCfg, AgentSetCfg, TopologyCfg, TaskCfg, ObservabilityCfg, ExperimentConfig) + `load_config(path, overrides)` с OmegaConf pipeline (load → include merge → dotlist overrides → resolve → model_validate). Mypy clean.
  - `src/atm/experiment/__init__.py` — публичный API (ExperimentConfig, load_config).
  - `pyproject.toml` — добавлены `typer>=0.12`, `omegaconf>=2.3` в `[project] dependencies`; `uv lock` прошёл.
  - `tests/unit/experiment/__init__.py` — пустой инит.
  - `tests/unit/experiment/test_config.py` — 19 тестов (все зелёные). Покрывает: load, публичный API, BudgetCfg defaults, ModelCfg.get_model_for fallback, dotlist overrides (name + int coercion), ValidationError на missing/invalid fields, env interpolation (с дефолтом и из env), ScratchpadCfg defaults, TopologyCfg.extra passthrough, full round-trip.
  - `tests/fixtures/experiment/valid_minimal.yaml` — минимальный валидный конфиг для тестов.
- **Step 1** (2026-04-24): Создан topology-базовый слой `atm.topology`. Файлы:
  - `src/atm/topology/base.py` — Topology Protocol (@runtime_checkable), TopologyConfig (Pydantic), TopologyRegistry (register/get/list_names с валидацией build/name), `_should_stop` helper (precedence: max_iter > success > topology_max > continue; reason-strings == FinishReason.value).
  - `src/atm/topology/__init__.py` — публичный API (Topology, TopologyConfig, TopologyRegistry, _should_stop) + contextlib.suppress guards для star/chain side-effect imports.
  - `tests/unit/topology/__init__.py` — пустой инит.
  - `tests/unit/topology/test_base.py` — 30 тестов (все зелёные). ruff + mypy clean.

- **Step 3** (2026-04-24): Создана ChainTopology `atm.topology.chain`. Файлы:
  - `src/atm/topology/chain.py` — ChainTopology с @TopologyRegistry.register("chain"), `_critic_postprocess` adapter-нода (парсит DECISION payload["approved"], выставляет signals["critic_approved"], заполняет final_answer из Executor DRAFT или "<incomplete>" fallback), `_route_from_critic` (инкрементирует iter_total/iteration, вызывает _should_stop, возвращает CHAIN_END или "executor"), CHAIN_END="__end__" константа, lazy import StateGraph через _import_state_graph(). shared.phase pinned "execution" в M6. `__init__.py` не редактировался.
  - `tests/unit/topology/test_chain.py` — 22 теста (все зелёные). Покрывает: регистрацию в Registry, Protocol соответствие, все ветки _critic_postprocess (approved/rejected/malformed/empty outbox/final_answer fallback), _route_from_critic counter increments + routing, build() mock test. ruff + mypy clean.
  - Примечание: `_ensure_chain_registered()` в test_chain.py решает проблему очистки Registry в test_base.py.

- **Step 5** (2026-04-24): Создан experiment runner `atm.experiment`. Файлы:
  - `src/atm/topology/base.py` — Topology Protocol, TopologyConfig, TopologyRegistry, `_should_stop` (реализован в этом же шаге т.к. отсутствовал в worktree)
  - `src/atm/topology/__init__.py` — публичный API + side-effect imports с guards
  - `src/atm/topology/star.py` — StarTopology с `@TopologyRegistry.register("star")`; coordinator async-нода, `_critic_postprocess`, `_route_from_coord` closure
  - `src/atm/topology/chain.py` — ChainTopology с `@TopologyRegistry.register("chain")`; linear Planner→Executor→Critic с retry-loop
  - `src/atm/experiment/config.py` — Pydantic-v2 схемы (ExperimentConfig + sub-models) + OmegaConf loader
  - `src/atm/experiment/_evaluator.py` — inline substring evaluator (`"55" in final_answer` → 1.0)
  - `src/atm/experiment/runner.py` — `run_one(cfg)` lifecycle: ensure_experiment → insert_run → build LLMs/agents/topology → ainvoke → parquet.close() (BEFORE update) → update_run_success/failed → dispose; RunResult Pydantic model; _build_initial_state с 14 SharedState ключами
  - `src/atm/experiment/__init__.py` — публичный API (ExperimentConfig, load_config, RunResult, run_one)
  - `src/atm/llm/factory.py` — build_llm factory с поддержкой provider="fake"
  - `pyproject.toml` — добавлены typer>=0.12, omegaconf>=2.3, langgraph>=0.3,<1 в deps
  - `tests/unit/experiment/test_runner.py` — 15 тестов (all green): evaluate(), _build_initial_state 14 keys, git_sha fallback, success path, budget_exceeded path, generic exception path, flush-before-update order (×2), RunResult model
  - `tests/fixtures/experiment/valid_minimal.yaml` — минимальный валидный конфиг
  - ruff clean + mypy clean; 691 unit-тестов зелёных

- **Step 6** (2026-04-24): Создан CLI `atm run` через Typer. Файлы:
  - `src/atm/experiment/cli.py` — `app = typer.Typer(name="atm", no_args_is_help=True)`; `@app.command("run")`; --config/-c option; override Argument (list[str] | None); exit codes 0/1/2/3; asyncio.run(run_one(cfg)); ValidationError/FileNotFoundError → exit 3; status=completed → 0; budget_exceeded → 2; else → 1; output format `Run <run_id>: status=..., quality=..., cost=$..., iters=...`.
  - `pyproject.toml` — добавлена секция `[project.scripts]`: `atm = "atm.experiment.cli:app"`.
  - `tests/unit/experiment/test_cli.py` — 6 тестов (все зелёные): --help exit 0, success→0, config error→3, budget→2, failed→1, output format. ruff + mypy clean.

- **Step 7** (2026-04-24): Созданы YAML-конфиги для smoke-эксперимента. Файлы:
  - `conf/agents/canonical_4.yaml` — AgentSet definition с 4 агентами (planner, executor, critic, researcher).
  - `conf/topology/star.yaml` — Star топология: max_iterations=20, extra: {planning_max_iter:2, exec_max_iter:5, verify_max_iter:3}.
  - `conf/topology/chain.yaml` — Chain топология: max_iterations=12, extra: {}.
  - `conf/experiments/smoke.yaml` — полный ExperimentConfig для smoke-запуска: task=fibonacci_smoke, model=fake:scripted, topology default=chain (override +topology.name=star), pg_dsn через oc.env interpolation.
  - Agent stubs (planner/executor/critic/researcher.yaml) — уже существовали из M5, НЕ перезаписаны.
  - `tests/unit/experiment/test_smoke_yaml_loads.py` — 3 теста (chain load, star override, ≥4 agent yamls); все зелёные.
  - ruff clean; 781 unit-тестов зелёных.

- **Step 8** (2026-04-24): Создан E2E integration test (Star + Chain, FakeLLM scripted, real PG). Файлы:
  - `tests/integration/experiment/__init__.py` — пустой инит.
  - `tests/fixtures/llm/m6_{star,chain}_{planner,executor,critic}.yaml` — 6 scripted FakeLLM фикстур.
  - `tests/conftest.py` — `ephemeral_pg_dsn` fixture (create_all/drop_all, skip без ATM_ENABLE_PG_TESTS).
  - `tests/integration/experiment/test_m6_e2e.py` — 2 теста: star + chain. Каждый: 18 assertions (расслаблены где parquet не пишется FakeLLM).
  - Обнаружены и задокументированы 4 бага в runner.py/topologies (BUG-1..4), все обойдены патчингом на стороне теста.
  - Тесты: 2 PASSED (с ATM_ENABLE_PG_TESTS=1), 2 SKIPPED (без PG). 787 unit-тестов зелёные.
  - pyproject.toml: добавлены typer, omegaconf, langgraph в deps; [project.scripts]; PytestUnraisableExceptionWarning в filterwarnings.

### IN PROGRESS
- Все шаги Step 8 завершены. M6 полностью завершён.

### BLOCKERS
- Нет

---

## Quick Resume

1. Прочитать этот файл.
2. Открыть `m6-tasks.md` — найти первую незачеркнутую задачу.
3. Прочитать `m6-plan.md` Phase 1 для понимания стратегии.
4. Начать с: **Step 1.1** — создать `src/atm/topology/base.py` (Topology Protocol + TopologyRegistry + `_should_stop` helper).

**Важный порядок первой волны**: Steps 1 и 4 параллельны и не конфликтуют по файлам.
- Агент A: Step 1 (`topology/base.py` + `__init__.py` + тесты).
- Агент B: Step 4 (`experiment/config.py` + `pyproject.toml` deps + тесты).

---

## Key Files

*Заполняется по мере реализации — developer-агенты добавляют статус по каждому файлу.*

**`src/atm/topology/base.py`**
- Роль: Topology Protocol, TopologyConfig, TopologyRegistry, `_should_stop` helper.
- Планируемое изменение: создать с нуля.
- Статус: DONE (Step 1, 2026-04-24)

**`src/atm/topology/__init__.py`**
- Роль: публичный API пакета topology; side-effect imports для star/chain через contextlib.suppress guards.
- Планируемое изменение: создать; владелец — ТОЛЬКО Step 1, Steps 2/3 не редактируют.
- Статус: DONE (Step 1, 2026-04-24)

**`src/atm/topology/star.py`**
- Роль: StarTopology — coordinator-centered LangGraph с phase-advance и critic_postprocess adapter.
- Планируемое изменение: создать с нуля; НЕ редактировать `__init__.py`.
- Статус: DONE (Step 2, 2026-04-24)

**`src/atm/topology/chain.py`**
- Роль: ChainTopology — линейный граф Planner→Executor→Critic с retry-loop.
- Планируемое изменение: создать с нуля; НЕ редактировать `__init__.py`.
- Статус: DONE (Step 3, 2026-04-24)

**`src/atm/experiment/config.py`**
- Роль: Pydantic-схемы (ExperimentConfig, BudgetCfg, ModelCfg, AgentSetCfg, TopologyCfg, TaskCfg, ObservabilityCfg) + OmegaConf loader `load_config(path, overrides)`.
- Планируемое изменение: создать с нуля. Без sweep/grid/dry-run.
- Статус: NOT STARTED

**`src/atm/experiment/runner.py`**
- Роль: `run_one(cfg)` — полный lifecycle (INSERT→build→ainvoke→flush parquet→UPDATE runs).
- Планируемое изменение: создать с нуля.
- Статус: NOT STARTED

**`src/atm/experiment/cli.py`**
- Роль: Typer-app `atm run --config <path> [+key=val ...]`, exit codes 0/1/2/3.
- Планируемое изменение: создать с нуля.
- Статус: DONE (Step 6, 2026-04-24)

**`src/atm/experiment/_evaluator.py`**
- Роль: inline substring-based evaluator (`"55" in final_answer` → quality=1.0).
- Планируемое изменение: создать с нуля; без SubprocessSandbox.
- Статус: NOT STARTED

**`src/atm/llm/factory.py`**
- Роль: фабрика LLMWrapper по provider-строке.
- Планируемое изменение: CONDITIONAL — добавить branch `provider="fake"` если отсутствует.
- Статус: NOT STARTED (pre-flight check при Step 5)

**`pyproject.toml`**
- Роль: зависимости проекта + entry-points.
- Планируемое изменение: (а) Step 4 → только `[project] dependencies`: `typer>=0.12`, `omegaconf>=2.3`; (б) Step 6 → только `[project.scripts]`: `atm = "atm.experiment.cli:app"`.
- КРИТИЧНО: шаги пишут в разные секции — merge-конфликт невозможен при соблюдении порядка.
- Статус: DONE (Step 4 deps + Step 6 scripts, 2026-04-24)

**`conf/experiments/smoke.yaml`**
- Роль: главный конфиг для smoke-запуска; task = fibonacci, model = fake:scripted.
- Планируемое изменение: создать (Step 7).
- Статус: NOT STARTED

**`conf/agents/critic.yaml`**
- Роль: конфиг Critic-агента.
- Планируемое изменение: создать stub если отсутствует; system_prompt ОБЯЗАН содержать инструкцию о `DECISION payload={"approved": bool, "comment": str}`.
- Статус: NOT STARTED (pre-flight Step 7)

**`tests/integration/experiment/test_m6_e2e.py`**
- Роль: E2E тест Star + Chain; маркер `@pytest.mark.integration`; пропускается без `ATM_ENABLE_PG_TESTS`.
- Планируемое изменение: создать (Step 8). Без SubprocessSandbox.
- Статус: DONE (Step 8, 2026-04-24)

---

## Decisions

### Topology Protocol — runtime_checkable Protocol (не ABC)
- Решение: `Topology` — `@runtime_checkable Protocol` с `name: str` и `build(...) -> CompiledStateGraph`. Конкретные классы не наследуют Protocol.
- Обоснование: следует LangGraph-стилю duck-typing; Registry проверяет наличие `build`/`name` при регистрации — защита от забытых атрибутов.

### `topology/__init__.py` — sole ownership Step 1
- Решение: `__init__.py` написан в Step 1 и содержит `try/except ImportError` для `from . import star` и `from . import chain`. Steps 2 и 3 НЕ редактируют `__init__.py`.
- Обоснование: устраняет merge-конфликт при параллельном выполнении Steps 2+3 (fix §2.2 ревью).

### Side-effect регистрация топологий через `@TopologyRegistry.register("star"/"chain")`
- Решение: decorator на уровне class definition в `star.py`/`chain.py`. При `import atm.topology.star` decorator срабатывает как side-effect.
- Обоснование: нет глобального dict в `__init__.py`; каждый модуль топологии самодостаточен.

### `_should_stop` precedence: budget → max_iter → topology_success → topology_max
- Решение: helper принимает `state`, `cfg`, `topology_success: bool`, `topology_max_reached: bool`. Budget не детектируется здесь (бросается из LLMWrapper) — runner ловит `BudgetExceededError` верхним уровнем.
- Обоснование: arch.md §7.1; reason-strings совпадают с `FinishReason.value`.

### Star coordinator — rule-based роутер (не LLM-агент)
- Решение: `coordinator` — чистая async-функция без LLM-вызова. Инкрементирует `iter_total`/`iteration`/`iter_within_phase`, применяет advance-правила, возвращает delta через conditional edges.
- Обоснование: arch.md §7.2 для M6; полноценный LLM-координатор — M8.

### `iter_within_phase` tracking в Star coordinator
- Решение: `iter_within_phase = state["shared"]["iter_total"] - state["shared"]["phase_started_at_iter"]`. При переходе фазы coordinator записывает `phase_started_at_iter = new_iter_total`.
- Обоснование: fix §2.1 ревью; критик-reject loop НЕ продвигает phase пока не approved или не превышен `verify_max_iter`.

### `final_answer` invariant — обязателен перед END
- Решение: каждая топология ОБЯЗАНА установить `state["shared"]["final_answer"]: str` перед переходом в END. Отсутствие — fatal contract-violation. Star: coordinator при `verification → done` берёт последний Executor DRAFT или fallback `"<incomplete>"`. Chain: `_critic_postprocess` при `approved=True` заполняет из последнего Executor DRAFT.
- Обоснование: runner читает `final_state["shared"]["final_answer"]` unconditionally (fix §3.3 ревью).

### `_critic_postprocess` adapter-нода (общая для Star и Chain)
- Решение: маленькая async-нода, парсит последний `MessageKind.DECISION` Critic-message, читает `payload["approved"]: bool`. Malformed message (без `payload["approved"]`) → treat as `approved=False`, логгировать warning.
- Обоснование: отделяет parsing-логику от routing-логики; если код дублируется — выносить в `topology/base.py` shared helper.

### Chain `shared.phase` — pinned `"execution"` в M6
- Решение: Chain не использует PhaseManager, `shared.phase = "execution"` всё время.
- Обоснование: полноценный PhaseManager — M8; Chain в M6 — линейный граф.

### Evaluator M6 — inline substring (без sandbox)
- Решение: `"55" in final_answer` → quality=1.0, иначе 0.0. Никакого SubprocessSandbox/DockerSandbox.
- Обоснование: fix §2.4 ревью; SubprocessSandbox запрещён в CI; полноценный TaskRegistry — M10.

### Integration-тест — полная фальсификация `code_run` через FakeLLM fixtures
- Решение: Executor FakeLLM fixture содержит `tool_call code_run` + pre-baked observation с `stdout="55"`. Если `code_run` tool invocation проходит через реальный ToolRegistry — использовать `FakeSandbox` stub (тестовая fixture `tests/fixtures/sandbox_fake.py`).
- Обоснование: fix §2.4 ревью; integration-тест не требует docker-in-docker.

### OmegaConf → Pydantic bridge через `to_container → model_validate`
- Решение: `OmegaConf.to_container(cfg, resolve=True)` → `ExperimentConfig.model_validate(data)`. OmegaConf не поддерживает Pydantic-v2 natively.
- Обоснование: canonical pattern 2026; env-interpolация через `${oc.env:PG_DSN,...}`.

### `_ensure_experiment` — `ON CONFLICT DO NOTHING RETURNING id`
- Решение: INSERT с `ON CONFLICT (name) DO NOTHING RETURNING id`; если RETURNING пустой — SELECT.
- Обоснование: fix §3.2 ревью; избегает TOCTOU race при параллельных runner-ах.

### `pyproject.toml` — секции разделены между Steps 4 и 6
- Решение: Step 4 → только `[project] dependencies`; Step 6 → только `[project.scripts]`. Step 6 обязан rebase over feat/m6 перед правкой.
- Обоснование: fix §2.3 ревью; merge-конфликт технически невозможен при разных строках в файле.

---

## Constraints

- `ExperimentCallbackHandler` и `ParquetWriter` из M3 используются as-is — без изменений сигнатур.
- `RunResult` — уже в `atm/core/types.py`, импортировать оттуда.
- `flush parquet` ВСЕГДА ДО `UPDATE runs.status` (M3 flush-invariant).
- Steps 2/3 НЕ редактируют `src/atm/topology/__init__.py`.
- Sweep/grid/dry-run — вне scope (M12).
- SubprocessSandbox/DockerSandbox — запрещены в integration-тестах M6.
- `Phase` enum — `from atm.core.types import Phase`; `PhaseRow` ORM — `from atm.storage import PhaseRow`.
- Nits §5.1–5.5 из ревью — некритичны, исправляются developer-агентами по ходу:
  - §5.1: дублированный номер 18 в assertions списка — косметика.
  - §5.2: `${model.by_role.<role>:fake:scripted}` — невалидный OmegaConf syntax; hardcode `"fake:scripted"` в YAML-стабах.
  - §5.3: `ModelCfg` schema — developer выводит из arch.md §12.1.
  - §5.4: `AgentSetCfg.set` → resolution — выводимо из кодовой базы.
  - §5.5: `canonical_4` содержит 4 агента, M6 использует 3 (researcher reserved) — naming friction, не блокер.
