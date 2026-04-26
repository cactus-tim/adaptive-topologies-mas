# M6 — Topology Framework + Star + Chain + End-to-End — Plan

## Executive Summary

Реализовать каркас топологий (`Topology` Protocol + `TopologyRegistry` + общий `_should_stop` helper
согласно stopping-precedence arch.md §7.1), две конкретные статические топологии (Star, Chain),
минимальный `experiment.Runner` (`load config → build agents → build topology → ainvoke → persist`),
Typer-CLI `atm run --config`, набор YAML-конфигов и end-to-end integration-тест (FakeLLM),
прогоняющий задачу «написать fibonacci» через Star и Chain с проверкой записей в PG (`runs`,
`phases`) и Parquet (`llm_calls`, `messages`, `tool_calls`, `scratchpad`) и корректным
`runs.quality_score`.

Это первый milestone, где агенты (M5), LLM (M2), tools (M4), storage/callback (M3) склеиваются
в один рабочий pipeline.

**Класс задачи**: complex — 5 новых модулей, первый публичный контракт `Topology.build(...)`,
первый CLI entry-point `atm run`, интеграция LangGraph + AsyncPostgresSaver + Parquet.

## Current State

- M5 (агенты), M4 (tools), M3 (storage/callback), M2 (LLM) завершены и зелёные.
- `GraphState`, `BudgetExceededError`, `FinishReason` уже определены.
- `ExperimentCallbackHandler`, `ParquetWriter`, `checkpointer_scope` из M3 используются as-is.
- Пакета `atm.topology` и `atm.experiment` не существует.
- `typer` и `omegaconf` ещё не в зависимостях.
- Всего 488+ unit-тестов зелёные; регрессии недопустимы.

## Proposed Approach

Реализуем снизу вверх по слоям:

1. **Контракт топологии** (`topology/base.py`) — Protocol + Registry + `_should_stop` helper
   с precedence budget → max_iter → topology_success → topology_max. `__init__.py` содержит
   `try/except ImportError` guards для side-effect-регистрации `star`/`chain`.

2. **Две топологии** (`star.py`, `chain.py`) — параллельно, оба только создают свои файлы,
   не редактируют `__init__.py`. Регистрируются через `@TopologyRegistry.register(...)` на уровне
   модуля (side-effect import из `__init__.py`).

3. **Конфиг-слой** (`experiment/config.py`) — минимальные Pydantic-схемы + OmegaConf loader
   (параллельно со Step 1). Без sweep/grid/dry-run — это M12.

4. **Runner** (`experiment/runner.py`) — оркестрирует все слои: создаёт engine, INSERT
   experiments/runs, строит агентов, строит topology, `ainvoke`, flush parquet (ДО UPDATE runs),
   UPDATE runs.

5. **CLI** (`experiment/cli.py`) — Typer `atm run --config <path> [+key=val ...]`, exit codes
   0/1/2/3, Typer `CliRunner` unit-тесты.

6. **YAML-конфиги** + pre-flight agent stubs.

7. **Integration-тест** (FakeLLM scripted, no sandbox) — 18 assertions на Star, 18 на Chain.

Ключевое решение по `topology/__init__.py`: Step 1 владеет файлом целиком, Steps 2/3 его НЕ
трогают — устраняет merge-конфликт при параллельном исполнении.

Ключевое решение по `code_run` в integration-тесте: полная фальсификация через FakeLLM fixtures
(observation hardcoded в fixture), без SubprocessSandbox/DockerSandbox.

## Implementation Phases

### Phase 1: Контракт топологии и конфиг-слой (~3h)

**Goal:** Заложить базовый Protocol, Registry, `_should_stop` helper и Pydantic config-схемы.

- [ ] 1.1 Реализовать `Topology` Protocol, `TopologyConfig`, `TopologyRegistry`, `_should_stop` helper
  - Файл: `src/atm/topology/base.py`
  - Приёмочный критерий: `_should_stop` возвращает правильный reason для всех 5 веток; фейковая топология регистрируется и резолвится по имени; 100% coverage.

- [ ] 1.2 Создать `src/atm/topology/__init__.py` с `try/except ImportError` guards
  - Файл: `src/atm/topology/__init__.py`
  - Приёмочный критерий: `from atm.topology import Topology, TopologyRegistry` работает; `from . import star` внутри guard не падает даже без `star.py`.

- [ ] 1.3 Unit-тесты на topology/base
  - Файл: `tests/unit/topology/test_base.py`
  - Приёмочный критерий: ≥10 тестов, все зелёные; `uv run ruff check` + `uv run mypy` clean.

- [ ] 1.4 Создать минимальные Pydantic-схемы + OmegaConf loader
  - Файл: `src/atm/experiment/config.py`
  - Приёмочный критерий: `load_config("conf/experiments/smoke.yaml")` возвращает `ExperimentConfig`; invalid YAML бросает `ValidationError`.

- [ ] 1.5 Добавить `typer>=0.12`, `omegaconf>=2.3` в `pyproject.toml [project] dependencies`
  - Файл: `pyproject.toml`
  - Приёмочный критерий: `uv lock` проходит без конфликтов; НЕ трогать `[project.scripts]`.

- [ ] 1.6 Unit-тесты на config
  - Файл: `tests/unit/experiment/test_config.py`
  - Приёмочный критерий: ≥8 тестов — валидация, overrides, env-интерполяция.

- [ ] 1.7 Fixture для тестов конфига
  - Файл: `tests/fixtures/experiment/valid_minimal.yaml`
  - Приёмочный критерий: файл загружается через `load_config` без ошибок.

### Phase 2: Реализации топологий (~3h, параллельно Step 2 + Step 3)

**Goal:** Собрать два рабочих `CompiledStateGraph` с корректными invariants.

- [ ] 2.1 Реализовать `StarTopology` с coordinator-нодой и conditional edges
  - Файл: `src/atm/topology/star.py`
  - Приёмочный критерий: `StarTopology().build(agents, cfg, checkpointer=cp)` компилируется; mocked-граф: critic-reject-loop делает `verify_max_iter` итераций, approved → END с заполненным `final_answer`.

- [ ] 2.2 Unit-тесты на Star
  - Файл: `tests/unit/topology/test_star.py`
  - Приёмочный критерий: ≥10 тестов; ruff + mypy clean.

- [ ] 2.3 Реализовать `ChainTopology` с critic retry-loop и `_critic_postprocess` adapter
  - Файл: `src/atm/topology/chain.py`
  - Приёмочный критерий: `ChainTopology().build(...)` компилируется; approved → END на первой итерации, `final_answer` заполнен; rejected → loop до `max_iterations`.

- [ ] 2.4 Unit-тесты на Chain
  - Файл: `tests/unit/topology/test_chain.py`
  - Приёмочный критерий: ≥8 тестов; invariant-тест: `final_state["shared"]["final_answer"]` не пустой при approved.

### Phase 3: Runner + CLI (~3h)

**Goal:** Рабочий `run_one()` и CLI-команда `atm run`.

- [ ] 3.1 Реализовать `run_one(cfg)` с полным lifecycle: INSERT→build→ainvoke→flush→UPDATE
  - Файл: `src/atm/experiment/runner.py`
  - Приёмочный критерий: unit-тесты проходят с mocked LLM/DB/callback; flush parquet ДО UPDATE runs (порядок в finally); `_ensure_experiment` использует `ON CONFLICT DO NOTHING RETURNING id`.

- [ ] 3.2 Pre-flight проверка FakeLLM в factory
  - Файл: `src/atm/llm/factory.py` (CONDITIONAL UPDATE — если `provider="fake"` ещё не поддержан)
  - Приёмочный критерий: `_build_llm(model_id="fake:scripted", ...)` не падает с `KeyError`.

- [ ] 3.3 Inline evaluator (substring-based)
  - Файл: `src/atm/experiment/_evaluator.py`
  - Приёмочный критерий: `"55" in final_answer` → `quality=1.0`, иначе `0.0`; никакого реального исполнения кода.

- [ ] 3.4 Unit-тесты на runner
  - Файл: `tests/unit/experiment/test_runner.py`
  - Приёмочный критерий: ≥6 тестов с mocked postgres/callback.

- [ ] 3.5 Реализовать CLI `atm run --config` через Typer
  - Файл: `src/atm/experiment/cli.py`
  - Приёмочный критерий: exit codes 0/1/2/3; `typer.testing.CliRunner` unit-тесты проходят; `uv run atm run --help` выдаёт подсказку.

- [ ] 3.6 Добавить `[project.scripts] atm = "atm.experiment.cli:app"` в `pyproject.toml`
  - Файл: `pyproject.toml`
  - Приёмочный критерий: НЕ трогать `[project] dependencies`; deps от Step 1.5 должны присутствовать в worktree перед правкой.

- [ ] 3.7 Unit-тесты на CLI
  - Файл: `tests/unit/experiment/test_cli.py`
  - Приёмочный критерий: ≥4 тестов через `CliRunner`.

### Phase 4: Конфиги и pre-flight agent stubs (~1h)

**Goal:** Все YAML-конфиги загружаются; все agent configs существуют.

- [ ] 4.1 Pre-flight: проверить и создать недостающие `conf/agents/{planner,executor,critic,researcher}.yaml`
  - Файлы: `conf/agents/planner.yaml`, `conf/agents/executor.yaml`, `conf/agents/critic.yaml`, `conf/agents/researcher.yaml`
  - Приёмочный критерий: `ls conf/agents/*.yaml | wc -l` ≥ 4; critic.yaml system_prompt содержит `{approved: bool, comment: str}` instruction; создавать ТОЛЬКО если файл отсутствует.

- [ ] 4.2 Создать `conf/experiments/smoke.yaml`, `conf/topology/{star,chain}.yaml`, `conf/agents/canonical_4.yaml`
  - Файлы: `conf/experiments/smoke.yaml`, `conf/topology/star.yaml`, `conf/topology/chain.yaml`, `conf/agents/canonical_4.yaml`
  - Приёмочный критерий: `load_config("conf/experiments/smoke.yaml")` возвращает валидный `ExperimentConfig` для Star и Chain (через override `+topology.name=star`).

- [ ] 4.3 Unit-тесты на загрузку YAML-конфигов
  - Файл: `tests/unit/experiment/test_smoke_yaml_loads.py`
  - Приёмочный критерий: ≥3 теста — smoke с Star, smoke с Chain, pre-flight agent configs.

### Phase 5: Integration-тест end-to-end (~2h)

**Goal:** Полный pipeline Star + Chain с реальным PG и FakeLLM — зелёный.

- [ ] 5.1 Создать FakeLLM scripted fixtures для Star и Chain
  - Файлы: `tests/fixtures/llm/m6_star_{planner,executor,critic}.yaml`, `tests/fixtures/llm/m6_chain_{planner,executor,critic}.yaml`
  - Приёмочный критерий: executor fixture содержит `tool_call code_run` + pre-baked observation "55"; critic fixture содержит `DECISION payload={"approved": true}`.

- [ ] 5.2 Написать integration-тест (2 test-кейса: star, chain)
  - Файл: `tests/integration/experiment/test_m6_e2e.py`
  - Приёмочный критерий: ≥18 assertions на каждый тест-кейс (см. список в разделе Steps); пропускается при отсутствии `ATM_ENABLE_PG_TESTS`; никакого SubprocessSandbox.

- [ ] 5.3 Обновить `tests/conftest.py` — добавить fixture `ephemeral_pg_dsn` если отсутствует
  - Файл: `tests/conftest.py`
  - Приёмочный критерий: `ephemeral_pg_dsn` делает полный reset PG-схемы после каждого теста.

## Key Files Affected

| Файл | Изменение | Зачем |
|------|-----------|-------|
| `src/atm/topology/base.py` | NEW | Topology Protocol, Registry, `_should_stop` |
| `src/atm/topology/__init__.py` | NEW | Публичный API пакета + side-effect imports с guards |
| `src/atm/topology/star.py` | NEW | StarTopology — coord-centered LangGraph |
| `src/atm/topology/chain.py` | NEW | ChainTopology — linear + critic retry |
| `src/atm/experiment/config.py` | NEW | Pydantic-схемы + OmegaConf loader |
| `src/atm/experiment/runner.py` | NEW | `run_one()` — основной оркестратор |
| `src/atm/experiment/cli.py` | NEW | Typer CLI `atm run` |
| `src/atm/experiment/_evaluator.py` | NEW | Substring-based inline evaluator |
| `src/atm/experiment/__init__.py` | NEW | Публичный API пакета |
| `src/atm/llm/factory.py` | CONDITIONAL UPDATE | `provider="fake"` branch если отсутствует |
| `pyproject.toml` | UPDATE (2 секции) | `typer`, `omegaconf` в deps; `atm` entry-point в scripts |
| `conf/experiments/smoke.yaml` | NEW | Главный конфиг smoke-запуска |
| `conf/topology/star.yaml` | NEW | Star-specific extra params |
| `conf/topology/chain.yaml` | NEW | Chain-specific params |
| `conf/agents/canonical_4.yaml` | NEW | AgentSet definition |
| `conf/agents/{planner,executor,critic,researcher}.yaml` | NEW (если отсутствуют) | Agent role configs |
| `tests/unit/topology/test_base.py` | NEW | ≥10 тестов на базовый контракт |
| `tests/unit/topology/test_star.py` | NEW | ≥10 тестов на Star |
| `tests/unit/topology/test_chain.py` | NEW | ≥8 тестов на Chain |
| `tests/unit/experiment/test_config.py` | NEW | ≥8 тестов на config |
| `tests/unit/experiment/test_runner.py` | NEW | ≥6 тестов на runner |
| `tests/unit/experiment/test_cli.py` | NEW | ≥4 теста на CLI |
| `tests/unit/experiment/test_smoke_yaml_loads.py` | NEW | ≥3 теста на YAML |
| `tests/integration/experiment/test_m6_e2e.py` | NEW | E2E Star+Chain, ≥18 assertions каждый |
| `tests/fixtures/llm/m6_*.yaml` | NEW (6 файлов) | Scripted FakeLLM для integration |
| `tests/fixtures/experiment/valid_minimal.yaml` | NEW | Fixture для unit config-тестов |
| `tests/conftest.py` | UPDATE | `ephemeral_pg_dsn` fixture если отсутствует |

## Dependencies & Order Constraints

```
Wave 1 (параллельно): Step 1 [topology/base] + Step 4 [experiment/config]
        ↓
Wave 2 (параллельно): Step 2 [topology/star] + Step 3 [topology/chain]
        ↓
Wave 3 (sequential):  Step 5 [experiment/runner]
        ↓
Wave 4 (параллельно): Step 6 [experiment/cli] + Step 7 [configs + agent stubs]
        ↓
Wave 5 (sequential):  Step 8 [integration test]
```

Критические ограничения порядка:
- Steps 2/3 зависят от Step 1 (нужен `Topology` Protocol и `_should_stop`).
- Step 5 зависит от Steps 2, 3, 4 (нужны topology + config для runner).
- Step 6 зависит от Step 5 (CLI вызывает `run_one`).
- Step 6 при правке `pyproject.toml [project.scripts]` ОБЯЗАН убедиться, что deps из Step 4 уже в worktree (rebase over feat/m6 если работает в отдельной ветке).
- Step 8 зависит от всего (Steps 5, 6, 7).

## Risks

| Риск | Вероятность | Влияние | Смягчение |
|------|-------------|---------|-----------|
| Merge-конфликт в `topology/__init__.py` при параллельном Steps 2+3 | Низкая | Высокое | `__init__.py` owned только Step 1; Steps 2/3 НЕ редактируют его |
| Merge-конфликт в `pyproject.toml` между Steps 4/6 | Средняя | Среднее | Step 4 пишет только `[project] dependencies`, Step 6 только `[project.scripts]`; Step 6 rebase перед правкой |
| Infinite loop в Star при неверной phase-advance логике | Средняя | Высокое | `_should_stop` по `max_iterations` — глобальный guard |
| FakeLLM fixture step_idx рассинхрон | Средняя | Среднее | Точные fixtures; integration-тест проверяет конкретный путь |
| `provider="fake"` отсутствует в LLM factory | Высокая | Среднее | Pre-flight check в Step 5; минимальная добавка если нет |
| `parquet_writer.close()` не вызван до UPDATE | Средняя | Высокое | Явный `try/finally`; порядок flush→UPDATE документирован |
| Транзитивные зависимости typer/omegaconf конфликтуют | Низкая | Среднее | Pin minor-версий; `uv lock` проверяет |
| Integration-тест нестабилен из-за PG-состояния | Средняя | Низкое | `ephemeral_pg_dsn` делает полный reset |
| `final_answer` не установлен топологией перед END | Средняя | Высокое | Инвариант задокументирован; unit-тесты проверяют invariant |

## Out of Scope

- Sweep/grid/dry-run конфиги — M12.
- `atm resume` — M12.
- `PhaseManager`/`SignalBus` — M8.
- TaskRegistry с реальной проверкой кода — M10.
- SubprocessSandbox/DockerSandbox в integration-тестах — M10 (CI-compatible).
- `PhasesCfg`, `HumanCfg`, `SweepCfg` — M8/M12.
- Adaptive topology switching — M7.
- Per-topology typing для `TopologyConfig.extra` — M7.

## Timeline

- Итого: ~12h
- Из них параллельные волны: ~7h реального времени при 2 агентах одновременно.
- Создан: 2026-04-24
