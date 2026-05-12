# M9 — Human Gateway + LLM-simulator (HITL infrastructure) — Plan

## Executive Summary

Реализовать первый этап HITL-инфраструктуры: внешний `HumanGateway` Protocol с тремя слоями — контракт/модели (`gateway.py`), LLM-симулятор (`llm_simulated.py`) с системными промптами для 5 ролей, минимальный отладочный `cli_gateway.py`. Персистенция в таблицу `human_interactions` (PG) идёт строго через `ExperimentCallbackHandler` (единственный writer, arch.md §10.2). Интеграция с Chain-топологией через LangGraph `interrupt()` / `Command(resume=...)` с поддержкой `timeout` + `fallback`-policy. Реальный UI (Streamlit) откладывается до M14.

## Current State

- `HumanContext`, `HumanResponse`, `HumanRole` — уже определены в `src/atm/core/types.py` (lines 235–261), frozen Pydantic v2.
- `HumanInteraction` SQLAlchemy-модель — уже в `src/atm/storage/models.py:266–326` с полями `run_id`, `role`, `context_json`, `response_json`, `request_id: VARCHAR(64) nullable`. Уникального constraint по `(run_id, request_id)` НЕТ — нужна миграция.
- `checkpointer_scope(dsn)` — готов в `src/atm/storage/checkpointer.py`.
- `ExperimentCallbackHandler.on_custom_event` — обрабатывает `message_emit`, `phase_transition`, `topology_transition`; нужны две новые ветви.
- `Runner._build_initial_state` — кладёт 14+ ключей в `state["shared"]`, но `run_id` отсутствует — блокер для `human_reviewer` node, правится одной строкой.
- `Chain.build` — линейный граф `START → planner → executor → critic → ...`; точка вставки Reviewer: между executor и critic.
- `HumanCfg` в `ExperimentConfig` — отсутствует; `src/atm/human/` — отсутствует как модуль.

## Proposed Approach

1. **Контракт без дублирования**: `human/gateway.py` реэкспортирует модели из `core.types`, добавляет только `HumanGateway` Protocol и `TimeoutPolicy` тип-алиас.
2. **Два gateway-реализации**: `LLMSimulatedGateway` (основной, для тестов и экспериментов) + `CLIGateway` (отладочный, stdin/stdout, без внешних зависимостей).
3. **In-process idempotency**: оба gateway держат `dict[(UUID, str), HumanResponse]` + `asyncio.Lock` — второй вызов с тем же `(run_id, request_id)` возвращает кэшированный результат без LLM-вызова.
4. **PG-idempotency через constraint**: Alembic-миграция добавляет UNIQUE CONSTRAINT `(run_id, request_id)` на `human_interactions`. Callback использует `INSERT ... ON CONFLICT DO NOTHING` и `UPDATE ... WHERE response_json IS NULL`.
5. **Callback — единственный writer**: узел Chain эмитит два `adispatch_custom_event` (`human_request` перед interrupt, `human_response` после resume); callback пишет в PG. Прямой DB-доступ из узла отсутствует.
6. **Timeout/fallback policy**: pure-async функция `request_with_timeout` в `human/_timeout.py` — обёртка над `asyncio.wait_for` с тремя политиками (`fail`, `llm_fallback`, `skip`).
7. **Resume-loop orchestration**: `human/runner.py::run_with_human` — независимый helper, управляет циклом `ainvoke → interrupt → gateway.request → Command(resume=...)`.
8. **HumanCfg via kwarg**: `ExperimentConfig.human: HumanCfg | None = None` → Runner передаёт `human_cfg=cfg.human` в `topology.build(**kwargs)` — Chain читает через `kwargs.get("human_cfg")`.

## Implementation Phases

### Phase 1: Foundation — Wave A (~2h)
**Goal:** Заложить контракт и DB-constraint — всё остальное зависит от этих двух артефактов.

- [ ] 1.1 Gateway Protocol + per-role system prompts module
  - Files: `src/atm/human/gateway.py`, `src/atm/human/prompts.py`, `src/atm/human/__init__.py`, `tests/unit/human/__init__.py`, `tests/unit/human/test_gateway_protocol.py`, `tests/unit/human/test_prompts.py`
  - Acceptance: `from atm.human import HumanGateway, HumanContext, HumanResponse, HumanRole, TimeoutPolicy` работает; все 5 ролей дают непустой prompt; Protocol-runtime-check корректен.
  - Verification: `uv run pytest tests/unit/human/test_gateway_protocol.py tests/unit/human/test_prompts.py -q && uv run mypy src/atm/human/gateway.py src/atm/human/prompts.py`

- [ ] 1.2 Alembic migration — UNIQUE CONSTRAINT `(run_id, request_id)`
  - Files: `alembic/versions/0002_human_interactions_idempotency.py`
  - Acceptance: `alembic upgrade head` создаёт constraint `uq_human_interactions_run_request`; `INSERT ... ON CONFLICT ON CONSTRAINT uq_human_interactions_run_request DO NOTHING` отрабатывает; `alembic downgrade -1 && alembic upgrade head` идемпотентно.
  - Verification: на dev-БД `psql -c "\d human_interactions"` показывает новый constraint.

### Phase 2: Core implementations — Wave B (~4h)
**Goal:** Реализовать оба gateway, timeout-wrapper, callback-handlers и HumanCfg — все параллельно.

- [ ] 2.1 LLMSimulatedGateway
  - Files: `src/atm/human/llm_simulated.py`, `tests/unit/human/test_llm_simulated_gateway.py`, `tests/fixtures/llm/m9_human_reviewer_approve.yaml`, `tests/fixtures/llm/m9_human_reviewer_reject.yaml`, `tests/fixtures/llm/m9_human_invalid_json.yaml`
  - Acceptance: `isinstance(LLMSimulatedGateway(fake_wrapper), HumanGateway) is True`; 6+ unit-тестов зелёных (approve/reject/invalid-JSON/action-not-in-allowed/idempotency/cache-key).
  - Verification: `uv run pytest tests/unit/human/test_llm_simulated_gateway.py -q`

- [ ] 2.2 CLIGateway
  - Files: `src/atm/human/cli_gateway.py`, `tests/unit/human/test_cli_gateway.py`
  - Acceptance: `isinstance(CLIGateway(), HumanGateway) is True`; тесты зелёные; без новых зависимостей; `asyncio.to_thread(input, ...)` патчится в тестах.
  - Verification: `uv run pytest tests/unit/human/test_cli_gateway.py -q`

- [ ] 2.3 Observability callback — handlers `human_request` / `human_response`
  - Files: `src/atm/observability/callbacks.py` (modify), `tests/unit/observability/test_callbacks_human.py` (create)
  - Acceptance: callback обрабатывает 2 новых event-name; INSERT idempotent (ON CONFLICT); UPDATE идемпотентен (WHERE response_json IS NULL); 6+ unit-тестов зелёных.
  - Verification: `uv run pytest tests/unit/observability/ -q && uv run mypy src/atm/observability/callbacks.py`

- [ ] 2.4 HumanCfg + YAML configs
  - Files: `src/atm/experiment/config.py` (modify), `conf/human/llm_simulated.yaml`, `conf/human/cli.yaml`, `tests/unit/experiment/test_config_human.py`
  - Acceptance: `load_config("conf/experiments/smoke.yaml").human is None` (back-compat); с подмешанным `conf/human/llm_simulated.yaml` — `HumanCfg` инстанцируется корректно.
  - Verification: `uv run pytest tests/unit/experiment/test_config_human.py -q`

- [ ] 2.5 Timeout wrapper + fallback policy logic
  - Files: `src/atm/human/_timeout.py`, `tests/unit/human/test_timeout_policy.py`
  - Acceptance: 3 политики обработаны (`fail`, `llm_fallback`, `skip`), source-метки корректны; 5+ unit-тестов зелёных; `policy="llm_fallback"` + `llm_fallback_gateway=None` → `ValueError`.
  - Verification: `uv run pytest tests/unit/human/test_timeout_policy.py -q`

### Phase 3: Integration wiring — Waves C + D + E (~3h)
**Goal:** Подключить HITL к Runner и Chain-топологии, реализовать orchestration helper.

- [ ] 3.1 Runner — `run_id` в shared state + проброс `human_cfg` в topology.build (Wave C)
  - Files: `src/atm/experiment/runner.py` (modify), `tests/unit/experiment/test_runner_initial_state.py`
  - Acceptance: `state["shared"]["run_id"]` присутствует; `Chain.build` получает `human_cfg=cfg.human` kwarg; 3 unit-теста (run_id present, back-compat 14 ключей, human_cfg kwarg).
  - Note: если `SharedState` TypedDict с `total=True` — добавить `run_id: UUID` в `src/atm/core/state.py`.
  - Verification: `uv run pytest tests/unit/experiment/test_runner_initial_state.py -q`

- [ ] 3.2 Chain topology — `human_reviewer` node с `interrupt()` и `dispatch_custom_event` (Wave D)
  - Files: `src/atm/topology/chain.py` (modify), `tests/unit/topology/test_chain_human.py` (create)
  - Acceptance: без `human_cfg` — граф точно как раньше; с `enabled=True` — узел вставлен между executor и critic; порядок dispatch → interrupt → dispatch корректен; 7 unit-тестов зелёных.
  - Verification: `uv run pytest tests/unit/topology/ -q && uv run mypy src/atm/topology/chain.py`

- [ ] 3.3 Orchestration helper — resume loop с timeout/fallback (Wave E)
  - Files: `src/atm/human/runner.py` (create), `tests/unit/human/test_runner_resume_loop.py` (create)
  - Acceptance: single/multi interrupt scenarios; timeout triggers fallback (source="fallback"); max_interactions guard; >1 interrupt в одном result → RuntimeError; 6+ unit-тестов зелёных.
  - Verification: `uv run pytest tests/unit/human/test_runner_resume_loop.py -q`

### Phase 4: Finalization — Wave F + Integration Tests (~2h)
**Goal:** Собрать публичный API модуля, обновить codebase-map, написать integration-тесты.

- [ ] 4.1 Module wiring + public exports + codebase-map update (Wave F)
  - Files: `src/atm/human/__init__.py` (finalize), `dev/codebase-map.md` (modify)
  - Acceptance: `from atm.human import HumanGateway, LLMSimulatedGateway, CLIGateway, run_with_human` работает; `uv run pytest tests/ -q` — full suite green; codebase-map содержит секцию M9.
  - Verification: `uv run python -c "from atm.human import HumanGateway, LLMSimulatedGateway, CLIGateway, run_with_human"`

- [ ] 4.2 Integration тест — Reviewer end-to-end (PG required)
  - Files: `tests/integration/human/test_hitl_chain_e2e.py`
  - Acceptance: Chain с `human.enabled=true, gateway=llm_simulated, role=reviewer` → ровно 1 запись в `human_interactions` с `response_json->>'source' = 'llm_sim'`; `SELECT COUNT(*) = 1`.

- [ ] 4.3 Integration тест — Timeout + fallback
  - Files: `tests/integration/human/test_hitl_timeout.py`
  - Acceptance: mock CLIGateway sleeps 5s, `timeout_s=1`, `policy=llm_fallback` → run завершается, `response_json->>'source' = 'fallback'`.

- [ ] 4.4 Integration тест — Idempotency на resume
  - Files: `tests/integration/human/test_hitl_idempotency.py`
  - Acceptance: simulated node re-execution в одном thread_id → `SELECT COUNT(*) = 1 FROM human_interactions WHERE run_id=...`.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `src/atm/human/gateway.py` | CREATE | HumanGateway Protocol + TimeoutPolicy alias |
| `src/atm/human/prompts.py` | CREATE | ROLE_SYSTEM_PROMPTS + build_role_prompt helper |
| `src/atm/human/llm_simulated.py` | CREATE | LLMSimulatedGateway реализация |
| `src/atm/human/cli_gateway.py` | CREATE | CLIGateway реализация (отладочная) |
| `src/atm/human/_timeout.py` | CREATE | request_with_timeout — 3 policy |
| `src/atm/human/runner.py` | CREATE | run_with_human — resume loop orchestrator |
| `src/atm/human/__init__.py` | CREATE/MODIFY | публичный API модуля |
| `src/atm/observability/callbacks.py` | MODIFY | +2 ветви on_custom_event: human_request/human_response |
| `src/atm/experiment/config.py` | MODIFY | +HumanCfg, +ExperimentConfig.human field |
| `src/atm/experiment/runner.py` | MODIFY | +run_id в shared, +human_cfg kwarg в topology.build |
| `src/atm/topology/chain.py` | MODIFY | +human_reviewer node, conditional insertion |
| `alembic/versions/0002_human_interactions_idempotency.py` | CREATE | UNIQUE CONSTRAINT (run_id, request_id) |
| `conf/human/llm_simulated.yaml` | CREATE | YAML конфиг LLMSimulatedGateway |
| `conf/human/cli.yaml` | CREATE | YAML конфиг CLIGateway |
| `dev/codebase-map.md` | MODIFY | секция M9 |
| `src/atm/core/state.py` | MODIFY (conditional) | +run_id: UUID в SharedState если TypedDict total=True |

## Dependencies & Order Constraints

```
Step 1 (gateway.py + prompts.py)
  └─► Step 2 (llm_simulated.py)
  └─► Step 3 (cli_gateway.py)
  └─► Step 5 (timeout wrapper)
  └─► Step 6 (HumanCfg)

Step 4 (alembic migration)
  └─► Step 5-callback (observability callbacks)

Step 6 (HumanCfg)
  └─► Step 8 (Runner changes)

Steps 1 + 5-callback + 8
  └─► Step 9 (Chain topology)

Steps 5-timeout + 9 (interrupt payload contract)
  └─► Step 10 (orchestration helper)

Steps 2 + 3 + 5-timeout + 7-timeout + 9 + 10
  └─► Step 11 (module wiring + exports)
```

**Waves:**
- Wave A: Steps 1, 4 (нет зависимостей, файлы disjoint)
- Wave B: Steps 2, 3, 5-callback, 6, 7-timeout (зависят только от Wave A)
- Wave C: Step 8 (зависит от Step 6 — ждёт HumanCfg тип)
- Wave D: Step 9 (зависит от Steps 1, 5-callback, 8)
- Wave E: Step 10 (зависит от Steps 7-timeout + 9)
- Wave F: Step 11 (зависит от всех)

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LangGraph `interrupt` / `Command` API breaking change | low | high | Верифицировать import path до Step 9; pin версию в pyproject |
| `adispatch_custom_event` недоступен в node context | medium | high | `uv run python -c "from langchain_core.callbacks import adispatch_custom_event"` перед Step 9 |
| `SharedState` TypedDict total=True блокирует добавление run_id | medium | medium | grep перед Step 8; добавить `run_id: UUID` в state.py при необходимости |
| `asyncio.to_thread(input)` не прерывается при CancelledError | high | low | Задокументировано в docstring CLIGateway и _timeout.py; в CI используется только LLMSimulatedGateway |
| Существующие дубль-записи в `human_interactions` блокируют миграцию | low | medium | `SELECT COUNT(*)` перед миграцией; на свежей БД ОК |
| `pg_insert(...).on_conflict_do_nothing(constraint=...)` — строка должна совпадать с именем в БД | medium | high | Вынести `CONSTRAINT_NAME = "uq_human_interactions_run_request"` в константу; одно место изменения |
| Неверный `down_revision` в миграции | medium | high | Прочитать `alembic/versions/0001_*.py` до написания 0002 |
| Dispatch `human_request` ПОСЛЕ interrupt — race condition | low | high | Строгий порядок в Step 9: dispatch → interrupt → (resume) → dispatch; unit-тест на ordered tracking |
| Циклические импорты `human ↔ core.types` | low | low | core.types не импортирует human; однонаправленный граф |

## Out of Scope

- Streamlit UI (откладывается до M14)
- Star, Mesh, Debate, Hierarchical, Adaptive топологии — только Chain в M9
- NASA-TLX заполнение (M11)
- StreamlitGateway (M14) — только Protocol-контракт достаточно широк для будущей реализации
- Заполнение `tlx_scores` — остаётся NULL в M9

## Timeline

- Total: ~11h
- Phase 1 (Wave A): ~2h
- Phase 2 (Wave B): ~4h (параллельно: 5 agentов)
- Phase 3 (Waves C+D+E): ~3h (последовательно по зависимостям)
- Phase 4 (Wave F + integration): ~2h
- Created: 2026-05-12
