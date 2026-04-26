# M6 — Topology Framework + Star + Chain + End-to-End — Tasks

> Формат каждой задачи:
> - Тип: `tdd` (сначала тест, потом реализация) / `simple` (реализация без TDD-цикла)
> - Depends On: номер шага, который должен быть завершён ДО начала этого
> - Can-Parallel-With: номера шагов, которые можно выполнять одновременно
>
> После завершения каждой задачи: отметить `[x]`, обновить `m6-context.md` раздел SESSION PROGRESS.

---

## Wave 1 — Базовый контракт и конфиг-слой (параллельно: Step 1 + Step 4)

### Step 1: topology/base.py — Protocol + Registry + _should_stop + registry-aware __init__.py COMPLETE
**Тип**: tdd
**Depends On**: —
**Can-Parallel-With**: Step 4

- [x] 1.1 Создать `src/atm/topology/base.py` — Topology Protocol, TopologyConfig, TopologyRegistry, `_should_stop(state, cfg, *, topology_success, topology_max_reached) -> tuple[bool, str]`
  - Файл: `src/atm/topology/base.py`
  - Приёмочный критерий: `_should_stop` корректно обрабатывает все 5 веток (budget-не-здесь, max_iter, topology_success, topology_max_reached, продолжение); Registry проверяет наличие `build`/`name` при регистрации; reason-strings совпадают с `FinishReason.value`.

- [x] 1.2 Создать `src/atm/topology/__init__.py` с try/except ImportError guards для star и chain
  - Файл: `src/atm/topology/__init__.py`
  - Приёмочный критерий: `from atm.topology import Topology, TopologyRegistry, _should_stop` работает; guard не падает при отсутствии `star.py`/`chain.py`; Steps 2 и 3 НЕ редактируют этот файл.

- [x] 1.3 Создать `tests/unit/topology/__init__.py` (пустой)
  - Файл: `tests/unit/topology/__init__.py`
  - Приёмочный критерий: файл существует.

- [x] 1.4 Написать unit-тесты `tests/unit/topology/test_base.py` (≥10 тестов)
  - Файл: `tests/unit/topology/test_base.py`
  - Приёмочный критерий: тесты на приоритет reasons (budget не детектируется здесь, max_iter > topology_success > topology_max), регистрацию/lookup, `_should_stop` для всех 5 веток; `uv run pytest tests/unit/topology/ -v` зелёный; ruff + mypy clean.

---

### Step 4: experiment/config.py — Pydantic schemas + OmegaConf loader + pyproject deps ✅ COMPLETE
**Тип**: simple
**Depends On**: —
**Can-Parallel-With**: Step 1

- [x] 4.1 Создать `src/atm/experiment/__init__.py` (минимальный, экспорт ExperimentConfig + load_config)
  - Файл: `src/atm/experiment/__init__.py`
  - Приёмочный критерий: файл существует; экспортирует `ExperimentConfig`, `load_config`.

- [x] 4.2 Создать `src/atm/experiment/config.py` — схемы BudgetCfg, ModelCfg, AgentSetCfg, TopologyCfg, TaskCfg, ObservabilityCfg, ExperimentConfig + `load_config(path, overrides)`
  - Файл: `src/atm/experiment/config.py`
  - Приёмочный критерий: `load_config("conf/experiments/smoke.yaml")` возвращает `ExperimentConfig`; invalid YAML бросает `ValidationError`; OmegaConf pipeline: load → include merge → dotlist overrides → resolve → to_container → model_validate; без sweep/grid/dry-run.

- [x] 4.3 Обновить `pyproject.toml` — добавить `typer>=0.12` и `omegaconf>=2.3` в `[project] dependencies`
  - Файл: `pyproject.toml`
  - Приёмочный критерий: только секция `[project] dependencies`; `[project.scripts]` НЕ трогать; `uv lock` проходит.

- [x] 4.4 Создать `tests/unit/experiment/__init__.py` (пустой)
  - Файл: `tests/unit/experiment/__init__.py`
  - Приёмочный критерий: файл существует.

- [x] 4.5 Написать unit-тесты `tests/unit/experiment/test_config.py` (≥8 тестов)
  - Файл: `tests/unit/experiment/test_config.py`
  - Приёмочный критерий: тесты на валидацию схем, dotlist overrides, env-интерполяцию (`${oc.env:PG_DSN,...}`), invalid YAML; ruff + mypy clean.

- [x] 4.6 Создать `tests/fixtures/experiment/valid_minimal.yaml`
  - Файл: `tests/fixtures/experiment/valid_minimal.yaml`
  - Приёмочный критерий: загружается через `load_config` без ошибок; содержит все обязательные поля ExperimentConfig.

---

## Wave 2 — Реализации топологий (параллельно: Step 2 + Step 3; оба зависят от Step 1)

### Step 2: topology/star.py — Coordinator-centered LangGraph ✅ COMPLETE
**Тип**: tdd
**Depends On**: Step 1
**Can-Parallel-With**: Step 3

- [x] 2.1 Создать `src/atm/topology/star.py` — StarTopology с @TopologyRegistry.register("star")
  - Файл: `src/atm/topology/star.py`
  - Приёмочный критерий: НЕ редактировать `__init__.py`; граф: coordinator → {planner|executor|critic} → coordinator → ... → END; entry_point="coordinator"; compile(checkpointer=cp) работает.

- [x] 2.2 Реализовать coordinator async-ноду (rule-based роутер)
  - Файл: `src/atm/topology/star.py`
  - Приёмочный критерий: инкрементирует `iter_total`/`iteration`; вычисляет `iter_within_phase = iter_total - phase_started_at_iter`; при transition пишет `phase_started_at_iter = new_iter_total`; при `verification → done` заполняет `shared.final_answer` (из последнего Executor DRAFT или fallback `"<incomplete>"`).

- [x] 2.3 Реализовать `_route_from_coord(state) -> str` с полной логикой phase-advance и _should_stop
  - Файл: `src/atm/topology/star.py`
  - Приёмочный критерий: planning: `ready_for_execution` или `iter_within_phase >= planning_max_iter` → executor; execution: `ready_for_verification` или `iter_within_phase >= exec_max_iter` → critic; verification: `critic_approved` → END через done; `iter_within_phase >= verify_max_iter` → END через done; иначе → critic (loop); `_should_stop` по max_iter → END.

- [x] 2.4 Реализовать `_critic_postprocess` adapter-ноду (общая с Chain если дублируется)
  - Файл: `src/atm/topology/star.py`
  - Приёмочный критерий: парсит `MessageKind.DECISION payload["approved"]`; malformed → `approved=False` + warning; выставляет `shared.signals["critic_approved"]`.

- [x] 2.5 Написать unit-тесты `tests/unit/topology/test_star.py` (≥10 тестов)
  - Файл: `tests/unit/topology/test_star.py`
  - Приёмочный критерий: граф компилируется; mocked-граф: reject-loop делает `verify_max_iter` итераций → END; approved → END на первой итерации; `final_answer` заполнен в обоих случаях; max_iter триггерит END; ruff + mypy clean.

---

### Step 3: topology/chain.py — Linear Planner→Executor→Critic с retry-loop ✅ COMPLETE
**Тип**: tdd
**Depends On**: Step 1
**Can-Parallel-With**: Step 2

- [x] 3.1 Создать `src/atm/topology/chain.py` — ChainTopology с @TopologyRegistry.register("chain")
  - Файл: `src/atm/topology/chain.py`
  - Приёмочный критерий: НЕ редактировать `__init__.py`; порядок нод: START → planner → executor → critic → critic_postprocess → [END | executor]; `shared.phase` pinned `"execution"` в M6; compile(checkpointer=cp) работает.

- [x] 3.2 Реализовать `_critic_postprocess` и `_route_from_critic(state) -> str` для Chain
  - Файл: `src/atm/topology/chain.py`
  - Приёмочный критерий: `_critic_postprocess` при `approved=True` заполняет `shared.final_answer` из последнего Executor DRAFT; `_route_from_critic`: `_should_stop` → END, `critic_approved=True` → END, иначе → "executor"; инкрементирует `iter_total`.

- [x] 3.3 Написать unit-тесты `tests/unit/topology/test_chain.py` (≥8 тестов)
  - Файл: `tests/unit/topology/test_chain.py`
  - Приёмочный критерий: approved → END на первой итерации; `final_state["shared"]["final_answer"]` не пустой; rejected → loop до max_iterations с fallback final_answer; ruff + mypy clean.

---

## Wave 3 — Runner (sequential; зависит от Steps 2, 3, 4)

### Step 5: experiment/runner.py — run_one() оркестратор ✅ COMPLETE
**Тип**: tdd
**Depends On**: Step 2, Step 3, Step 4
**Can-Parallel-With**: —

- [x] 5.1 Pre-flight: проверить `src/atm/llm/factory.py` на наличие branch `provider="fake"`; добавить если отсутствует
  - Файл: `src/atm/llm/factory.py` (CONDITIONAL)
  - Приёмочный критерий: `_build_llm(model_id="fake:scripted", ...)` не падает с `KeyError`; существующая логика factory не меняется.

- [x] 5.2 Создать `src/atm/experiment/_evaluator.py` — inline substring evaluator
  - Файл: `src/atm/experiment/_evaluator.py`
  - Приёмочный критерий: `evaluate(task_cfg, final_answer)` → `1.0` если `"55" in final_answer`, иначе `0.0`; никакого SubprocessSandbox; чистый Python.

- [x] 5.3 Реализовать `run_one(cfg: ExperimentConfig) -> RunResult` в `src/atm/experiment/runner.py`
  - Файл: `src/atm/experiment/runner.py`
  - Приёмочный критерий: lifecycle: `_ensure_experiment` (ON CONFLICT DO NOTHING RETURNING id) → `_insert_run` → build LLMs by role → build agents → build topology → `ainvoke(initial_state, config={callbacks, configurable:{thread_id}})` → `parquet_writer.close()` → `_update_run_success/failed` → `engine.dispose()`; flush parquet ВСЕГДА ДО UPDATE runs; `BudgetExceededError` ловится отдельно; `Exception` ловится как fallback.

- [x] 5.4 Реализовать `_build_initial_state(cfg, run_id) -> dict` — полный GraphState с 14 ключами в shared
  - Файл: `src/atm/experiment/runner.py`
  - Приёмочный критерий: все 14 ключей присутствуют: task_id, task_input, phase, iteration, iter_total, active_topology, final_answer, signals, phase_started_at_iter, topology_started_at_iter, topology_history, topology_switch_count, phase_history, human_requests, human_responses, broadcast_bus; agents dict по всем agent_id.

- [x] 5.5 Обновить `src/atm/experiment/__init__.py` — добавить экспорт `run_one`
  - Файл: `src/atm/experiment/__init__.py`
  - Приёмочный критерий: `from atm.experiment import run_one` работает.

- [x] 5.6 Написать unit-тесты `tests/unit/experiment/test_runner.py` (≥6 тестов с моками)
  - Файл: `tests/unit/experiment/test_runner.py`
  - Приёмочный критерий: mocked LLM / AsyncMock session_factory / noop callback; тесты на success-путь, BudgetExceededError, generic Exception, порядок flush→UPDATE; ruff + mypy clean.

---

## Wave 4 — CLI и конфиги (параллельно: Step 6 + Step 7; оба зависят от Step 5)

### Step 6: experiment/cli.py — Typer CLI `atm run` + pyproject.toml scripts ✅ COMPLETE
**Тип**: simple
**Depends On**: Step 5
**Can-Parallel-With**: Step 7

- [x] 6.1 Создать `src/atm/experiment/cli.py` — Typer app с командой `atm run --config <path> [+key=val ...]`
  - Файл: `src/atm/experiment/cli.py`
  - Приёмочный критерий: `app = typer.Typer(name="atm")`; exit codes: 0 completed / 1 failed / 2 budget_exceeded / 3 config error; output: `Run <run_id>: status=<status>, quality=<score>, cost=$<cost>, iters=<iters>`; `asyncio.run(run_one(cfg))`.

- [x] 6.2 Обновить `pyproject.toml` — добавить `[project.scripts] atm = "atm.experiment.cli:app"`
  - Файл: `pyproject.toml`
  - Приёмочный критерий: только секция `[project.scripts]`; `[project] dependencies` НЕ трогать; перед правкой убедиться что deps из Step 4.3 присутствуют (rebase over feat/m6 если нужно); `grep -q 'typer' pyproject.toml && grep -q 'omegaconf' pyproject.toml` — не потеряли.

- [x] 6.3 Написать unit-тесты `tests/unit/experiment/test_cli.py` (≥4 теста через CliRunner)
  - Файл: `tests/unit/experiment/test_cli.py`
  - Приёмочный критерий: `typer.testing.CliRunner`; тесты: `--help` выдаёт подсказку, success-путь exit 0, config error exit 3, budget error exit 2; ruff + mypy clean.

---

### Step 7: Configs — smoke.yaml + topology YAMLs + canonical_4 + agent stubs ✅ COMPLETE
**Тип**: simple
**Depends On**: Step 4
**Can-Parallel-With**: Step 6

- [x] 7.1 Pre-flight: проверить `conf/agents/` и создать недостающие из {planner,executor,critic,researcher}.yaml
  - Файлы: `conf/agents/planner.yaml`, `conf/agents/executor.yaml`, `conf/agents/critic.yaml`, `conf/agents/researcher.yaml`
  - Приёмочный критерий: файлы существуют после шага; создавать ТОЛЬКО если отсутствуют; critic.yaml system_prompt содержит `{approved: bool, comment: str}` instruction; model: hardcode `"fake:scripted"` (НЕ OmegaConf interpolation `${model.by_role...}`).

- [x] 7.2 Создать `conf/agents/canonical_4.yaml` — AgentSet definition
  - Файл: `conf/agents/canonical_4.yaml`
  - Приёмочный критерий: ссылается на 4 agent config-файла; загружается через `AgentSetCfg`.

- [x] 7.3 Создать `conf/topology/star.yaml` и `conf/topology/chain.yaml`
  - Файлы: `conf/topology/star.yaml`, `conf/topology/chain.yaml`
  - Приёмочный критерий: star.yaml — `name: star`, `max_iterations: 20`, `extra: {planning_max_iter: 2, exec_max_iter: 5, verify_max_iter: 3}`; chain.yaml — `name: chain`, `max_iterations: 12`, `extra: {}`.

- [x] 7.4 Создать `conf/experiments/smoke.yaml` с include-механизмом
  - Файл: `conf/experiments/smoke.yaml`
  - Приёмочный критерий: task: `{name: fibonacci_smoke, input: "Write a Python function fib(n)..."}`; model.by_role все роли = `"fake:scripted"`; `pg_dsn: "${oc.env:PG_DSN,...}"`; `parquet_dir: "data/experiments"`; topology default = chain (override через `+topology.name=star`).

- [x] 7.5 Написать unit-тесты `tests/unit/experiment/test_smoke_yaml_loads.py` (≥3 теста)
  - Файл: `tests/unit/experiment/test_smoke_yaml_loads.py`
  - Приёмочный критерий: smoke с Chain загружается OK; smoke с Star override (`+topology.name=star`) загружается OK; `ls conf/agents/*.yaml | wc -l` ≥ 4 (assert в тесте).

---

## Wave 5 — Integration-тест (sequential; зависит от всего)

### Step 8: Integration test — E2E fibonacci Star + Chain (FakeLLM, no sandbox) ✅ COMPLETE
**Тип**: tdd
**Depends On**: Step 5, Step 6, Step 7
**Can-Parallel-With**: —

- [x] 8.1 Создать `tests/integration/experiment/__init__.py` (пустой)
  - Файл: `tests/integration/experiment/__init__.py`
  - Приёмочный критерий: файл существует.

- [x] 8.2 Создать FakeLLM scripted fixtures для Star (3 файла: planner, executor, critic)
  - Файлы: `tests/fixtures/llm/m6_star_planner.yaml`, `tests/fixtures/llm/m6_star_executor.yaml`, `tests/fixtures/llm/m6_star_critic.yaml`
  - Приёмочный критерий: planner: response — "Plan: write fib(n); compute fib(10)", kind=draft; executor: response 1 — tool_call `code_run(code=...)`, response 2 — DRAFT с text "fib(10)=55" и `payload["draft"]="fib(10)=55"` (pre-baked observation "55"); critic: DECISION с `payload={"approved": true}`, text "APPROVE".

- [x] 8.3 Создать FakeLLM scripted fixtures для Chain (3 файла: planner, executor, critic)
  - Файлы: `tests/fixtures/llm/m6_chain_planner.yaml`, `tests/fixtures/llm/m6_chain_executor.yaml`, `tests/fixtures/llm/m6_chain_critic.yaml`
  - Приёмочный критерий: аналогично Star fixtures; executor содержит pre-baked observation "55".

- [x] 8.4 Обновить `tests/conftest.py` — добавить fixture `ephemeral_pg_dsn` если отсутствует
  - Файл: `tests/conftest.py`
  - Приёмочный критерий: fixture делает полный reset PG-схемы; пропускает тест если `ATM_ENABLE_PG_TESTS` не задан.

- [x] 8.5 Написать `tests/integration/experiment/test_m6_e2e.py` — 2 test-кейса (star, chain)
  - Файл: `tests/integration/experiment/test_m6_e2e.py`
  - Приёмочный критерий: маркер `@pytest.mark.integration`; пропуск при отсутствии `ATM_ENABLE_PG_TESTS`; никакого SubprocessSandbox; ≥18 assertions на кейс (перечислены ниже).

**Полный список assertions для каждого test-кейса** (Star: 18, Chain: 18):

| # | Assertion |
|---|-----------|
| 1 | `RunResult.status == "completed"` |
| 2 | `RunResult.metrics["quality_score"] == 1.0` |
| 3 | PG: строка в `runs` существует; `finish_reason == "success"` |
| 4 | PG: `runs.quality_score IS NOT NULL` |
| 5 | PG: `runs.budget_spent_usd > 0` |
| 6 | PG: `runs.iterations > 0` и `<= cfg.topology.max_iterations` |
| 7 | PG: `runs.finished_at IS NOT NULL` |
| 8 | PG: строка в `experiments`; `config_snapshot` содержит `name="m6_smoke"` |
| 9 | PG: `topology_transitions` — ровно один row (`to_topology=star\|chain`) |
| 10 | Parquet: `llm_calls.parquet` существует, содержит ≥ 2 calls |
| 11 | Parquet: `sum(llm_calls.cost_usd) ≈ runs.budget_spent_usd` с ε=0.001 |
| 12 | Parquet: `messages.parquet` существует, содержит ≥ 3 messages |
| 13 | Parquet: `tool_calls.parquet` существует, содержит ≥ 1 `code_run` invocation |
| 14 | Parquet: `scratchpad/<agent_id>.parquet` существует для всех агентов |
| 15 | `final_state["shared"]["final_answer"]` содержит "55" |
| 16 | `"55" in final_state["shared"]["final_answer"]` |
| 17 | `state["shared"]["active_topology"] == cfg.topology.name` |
| 18 (Star) | `phase_history` содержит transitions planning→execution→verification→done |
| 18 (Chain) | `iter_total == 1` при first-approve scenario |

---

## Stats

- Итого задач: 39
- По Steps: 4 (Step 1) + 6 (Step 4) + 5 (Step 2) + 3 (Step 3) + 6 (Step 5) + 3 (Step 6) + 5 (Step 7) + 5 (Step 8) = 37 основных + 2 пустых __init__ = 39
- Выполнено: 0 / 39
- Оценка времени: ~12h общих / ~7h реального (при 2 параллельных агентах)

## Как обновлять

1. Отметить завершённые задачи `[x]`.
2. Обновить заголовок фазы: все задачи done → `COMPLETE`, хотя бы одна in progress → `IN PROGRESS`.
3. Обновить `m6-context.md` раздел SESSION PROGRESS после каждого milestone (завершения Step).
4. В KEY FILES обновить статус соответствующего файла.
