# M14 — Streamlit HITL UI, NASA-TLX, Proctoring Protocol — Plan

## Executive Summary

M14+ — второй этап диплома (адаптивные топологии MAS с HITL). Реализуем web-UI на Streamlit
для подключения живых участников user-study вместо `LLMSimulatedGateway`. В той же транзакции,
что и ответ человека, собираем NASA-TLX (6 шкал). Добавляем протокол прокторированных сессий
(consent, participant_id, proctor notes, события start/pause/resume/end). 100% backward-compat
с M9.1/M9.2: при `human.gateway: llm_simulated` (дефолт) поведение полностью идентично текущему.

## Current State

- Шесть топологий (chain, star, mesh, debate, hierarchical, adaptive) используют `HumanGateway`
  через inline-вызовы `await gateway.request(...)` — без LangGraph `interrupt()`.
- Реализованы два gateway-а: `LLMSimulatedGateway` (дефолт) и `CLIGateway`.
- Callback `_handle_human_response` пишет `human_interactions` в Postgres, но не имеет полей
  `tlx_scores`, `raw_tlx_score`, `study_session_id`.
- Таймаут-fallback реализован в 8 местах; все читают `getattr(gateway, "_llm", None)` —
  это сломается при `StreamlitHumanGateway` (у которого нет `._llm`).
- `HumanCfg.gateway` поддерживает только `"llm_simulated"` и `"cli"`.
- Alembic head: `0004_m12_runs_replay_and_pid`.

## Proposed Approach

1. **DB layer** (Steps 1-2): новая миграция 0005 + ORM-модели для `study_sessions`,
   `human_request_queue`, расширение `human_interactions`.
2. **PG-queue helper** (Step 3): async-абстракция `atm.human._queue.HumanRequestQueue`
   над таблицей. Polling (1s); LISTEN/NOTIFY отложен до M14.1.
3. **Config extension** (Step 4): `HumanCfg` добавляет `"streamlit"` как третье значение
   `gateway`, плюс поля `queue_dsn`, `participant_id`, `study_session_id`, `shared_secret`,
   `fallback_llm_model`.
4. **StreamlitHumanGateway** (Step 5): новый gateway + расширение `_handle_human_response`
   для записи TLX-полей в обе ветки INSERT/UPDATE.
5. **Streamlit UI** (Step 6): пакет `atm.ui` со страницами Login, Queue, Response form,
   TLX form, Proctor panel; `nest_asyncio.apply()` в bootstrap.
6. **Runner wiring** (Step 7): централизованная фабрика `_build_human_gateway` в runner.py;
   новый kwarg `human_gateway` во все `.build(...)` вызовы 6 топологий; хирургический патч
   fallback-блоков в 8 сайтах через `fallback_llm` параметр (top-level factories) или
   closure-capture (inline closures).
7. **Packaging & infra** (Steps 8-9): optional-dependency `[ui]` в pyproject.toml,
   docker-compose профиль `ui`.
8. **Integration test** (Step 10): end-to-end Chain + FakeStreamlitClient через реальный PG.
9. **Docs** (Step 11): README, proctor-protocol, participant-consent.

## Implementation Phases

### Phase 1: DB & Config Foundation (~3h)
**Goal:** Поднять все DB-структуры и расширить конфиг — независимо от остального кода.

- [x] 1.1 Alembic migration 0005 — `study_sessions`, `human_request_queue`, `human_interactions.study_session_id`
  - File: `alembic/versions/0005_m14_streamlit_hitl.py`
  - File: `tests/integration/storage/test_migration_0005.py`
  - Acceptance: `alembic upgrade head && alembic downgrade -1` проходит без ошибок; integration-тест зелёный.

- [ ] 1.2 SQLAlchemy ORM — `StudySession`, `HumanRequestQueue`, расширение `HumanInteraction`
  - File: `src/atm/storage/models.py`
  - File: `tests/unit/storage/test_models.py`
  - Acceptance: `mypy strict` + `pytest tests/unit/storage/test_models.py` зелёный; минимум 2 новых теста.

- [ ] 1.3 Расширить `HumanCfg` + создать `conf/human/streamlit.yaml`
  - File: `src/atm/experiment/config.py`
  - File: `conf/human/streamlit.yaml`
  - File: `tests/unit/experiment/test_config_human.py`
  - Acceptance: `mypy strict` + unit-тесты зелёные; `streamlit` как валидное значение `gateway`.

- [ ] 1.4 pyproject — optional-dependency `[ui]`, mypy overrides для streamlit/nest_asyncio
  - File: `pyproject.toml`
  - Acceptance: `uv lock --check`; `uv pip install -e ".[ui]" && python -c "import streamlit, nest_asyncio"`.

### Phase 2: Queue Helper (~2h)
**Goal:** Реализовать async PG-очередь с полным жизненным циклом запроса.

- [ ] 2.1 `atm.human._queue.HumanRequestQueue`
  - File: `src/atm/human/_queue.py`
  - File: `tests/unit/human/test_queue.py`
  - File: `tests/integration/human/test_queue_pg.py`
  - Acceptance: unit + integration (`requires_postgres`) зелёные; `mypy strict`; идемпотентность `enqueue` подтверждена тестом.

### Phase 3: Gateway + Callback (~2.5h)
**Goal:** Реализовать StreamlitHumanGateway и расширить callback для TLX-полей.

- [ ] 3.1 `StreamlitHumanGateway` + `_handle_human_response` TLX extension
  - File: `src/atm/human/streamlit_gateway.py`
  - File: `src/atm/human/__init__.py`
  - File: `src/atm/observability/callbacks.py`
  - File: `tests/unit/human/test_streamlit_gateway.py`
  - File: `tests/unit/observability/test_human_response_tlx.py`
  - Acceptance: unit-тесты для обеих ветвей callback; существующие тесты без TLX не сломаны; `mypy strict`.

### Phase 4: Streamlit UI (~3h)
**Goal:** Рабочий Streamlit-UI с login, queue, response form, TLX form, proctor panel.

- [ ] 4.1 Пакет `atm.ui` — все страницы и вспомогательные модули
  - File: `src/atm/ui/__init__.py`
  - File: `src/atm/ui/app.py`
  - File: `src/atm/ui/auth.py`
  - File: `src/atm/ui/views.py`
  - File: `src/atm/ui/tlx.py`
  - File: `src/atm/ui/_state.py`
  - File: `tests/unit/ui/test_tlx_form.py`
  - File: `tests/unit/ui/test_auth.py`
  - Acceptance: `pytest tests/unit/ui/` зелёный; `streamlit run src/atm/ui/app.py` поднимается (manual smoke).

### Phase 5: Runner Wiring (~4h)
**Goal:** Централизовать создание gateway в runner и распространить `human_gateway` kwarg во все 6 топологий с корректным patching fallback-логики.

- [ ] 5.1 `_build_human_gateway` фабрика в runner.py + оба wiring-сайта
  - File: `src/atm/experiment/runner.py`
  - File: `tests/unit/experiment/test_runner_human_gateway.py`
  - Acceptance: 4 сценария (llm_simulated, cli, streamlit, disabled) зелёные; `mypy strict`.

- [ ] 5.2 Gateway dispatch patch для всех 6 топологий (D7 canon `human_gateway` kwarg)
  - File: `src/atm/topology/chain.py`
  - File: `src/atm/topology/star.py`
  - File: `src/atm/topology/mesh.py`
  - File: `src/atm/topology/debate.py`
  - File: `src/atm/topology/hierarchical.py`
  - File: `src/atm/topology/adaptive.py`
  - File: `tests/unit/topology/test_topology_streamlit_dispatch.py`
  - Acceptance: 6 dispatch-тестов (один на топологию) зелёные; 6 grep-чеков из Implementation Notes верифицированы.

- [ ] 5.3 Fallback-блоки: patch 8 сайтов + `fallback_llm` в top-level factories + `_node_factory.py`
  - File: `src/atm/human/_node_factory.py`
  - File: `src/atm/topology/chain.py` (сайт B)
  - File: `src/atm/topology/mesh.py` (сайт C)
  - File: `src/atm/topology/debate.py` (сайты D, E)
  - File: `src/atm/topology/hierarchical.py` (сайты F, G + `_build_subgraph_with_human`)
  - File: `src/atm/topology/adaptive.py` (сайт H)
  - Acceptance: тест с `force-timeout` + `FakeGateway` (без `._llm`) не кидает `ValueError`; `mypy strict`.

### Phase 6: Infra & Integration (~3h)
**Goal:** Docker UI-профиль, acceptance-тест e2e, документация.

- [ ] 6.1 docker-compose профиль `ui` + `Dockerfile.ui`
  - File: `docker-compose.yml`
  - File: `Dockerfile.ui`
  - File: `.env.example`
  - Acceptance: `docker compose --profile ui up --build`; `curl -I http://localhost:8501` возвращает 200.

- [ ] 6.2 Integration test — Chain + StreamlitGateway end-to-end
  - File: `tests/integration/human/test_streamlit_e2e.py`
  - File: `tests/integration/human/conftest.py`
  - Acceptance: `ATM_ENABLE_PG_TESTS=1 pytest tests/integration/human/test_streamlit_e2e.py -m requires_postgres -v` зелёный; `human_interactions` содержит все 3 новых поля.

- [ ] 6.3 Документация сессий
  - File: `dev/active/m14/README.md`
  - File: `dev/active/m14/proctor-protocol.md`
  - File: `dev/active/m14/participant-consent.md`
  - Acceptance: ручной review; SELECT-пример латентности присутствует; чеклист proctoring-протокола.

## Key Files Affected

| File | Change | Why |
|------|--------|-----|
| `alembic/versions/0005_m14_streamlit_hitl.py` | New | study_sessions + human_request_queue + human_interactions FK |
| `src/atm/storage/models.py` | Extend | ORM для двух новых таблиц + study_session_id на HumanInteraction |
| `src/atm/experiment/config.py` | Extend HumanCfg | Третий вариант gateway + 5 новых полей |
| `conf/human/streamlit.yaml` | New | Конфиг для user-study сессий |
| `src/atm/human/_queue.py` | New (~250 LoC) | Async PG-очередь между runner и UI |
| `src/atm/human/streamlit_gateway.py` | New (~200 LoC) | StreamlitHumanGateway: enqueue + wait |
| `src/atm/human/__init__.py` | Extend __all__ | Экспорт нового gateway |
| `src/atm/human/_node_factory.py` | Modify | Новый параметр fallback_llm в build_human_node_factory |
| `src/atm/observability/callbacks.py` | Modify | Обе ветки INSERT/UPDATE — 3 новых поля TLX |
| `src/atm/ui/__init__.py` | New | Пакет atm.ui |
| `src/atm/ui/app.py` | New | Streamlit app + nest_asyncio bootstrap |
| `src/atm/ui/auth.py` | New | Shared-secret auth |
| `src/atm/ui/views.py` | New | Queue, Response, TLX, Proctor страницы |
| `src/atm/ui/tlx.py` | New | NASA-TLX 6-шкальная форма |
| `src/atm/ui/_state.py` | New | _run_sync helper + session state |
| `src/atm/experiment/runner.py` | Modify | _build_human_gateway фабрика + 2 wiring-сайта |
| `src/atm/topology/chain.py` | Modify | human_gateway dispatch + fallback_llm |
| `src/atm/topology/star.py` | Modify | human_gateway dispatch + fallback_llm caller |
| `src/atm/topology/mesh.py` | Modify | human_gateway dispatch (reuse existing var) + closure fallback |
| `src/atm/topology/debate.py` | Modify | human_gateway + legacy "gateway" key compat + 2 fallback sites |
| `src/atm/topology/hierarchical.py` | Modify | human_gateway dispatch + 2 factories + _build_subgraph_with_human |
| `src/atm/topology/adaptive.py` | Modify | human_gateway dispatch (reuse _gateway) + closure fallback |
| `pyproject.toml` | Extend | [project.optional-dependencies] ui + mypy overrides |
| `docker-compose.yml` | Extend | Профиль ui |
| `Dockerfile.ui` | New | Streamlit контейнер |

## Dependencies & Order Constraints

```
Wave 1 (parallel): Step 1.1 (migration), Step 1.3 (config), Step 1.4 (pyproject)
Wave 2: Step 1.2 (ORM) — depends on 1.1
Wave 3: Step 2.1 (queue) — depends on 1.2
Wave 4: Step 3.1 (gateway+callback) — depends on 2.1 AND 1.3
Wave 5 (parallel): Step 4.1 (UI) + Step 5.x (runner wiring) — disjoint files
Wave 6 (parallel): Step 6.1 (docker) + Step 6.2 (e2e test) + Step 6.3 (docs)
```

Критический путь: 1.1 → 1.2 → 2.1 → 3.1 → 5.x → 6.2 (~14.5h последовательно).

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Двойной Alembic head из-за bc5f66dd0897_initial.py | Low | High (deploy blocker) | Проверить `alembic heads` до коммита; down_revision должен быть строго "0004" |
| Конфликт имени `human_gateway` в mesh.py | Medium | Medium | Переиспользование (не переименование) существующего локала — не добавлять новый |
| Тихий fallback при пропущенном caller `build_human_node_factory` | Low | Medium | Per-topology dispatch-тесты с force-timeout и FakeGateway без `._llm` |
| Race condition в `submit_response` (двойная запись) | Low | High | Guard `AND response_json IS NULL` в UPDATE; integration-тест |
| Streamlit 1.33+ event-loop конфликт | Medium | High | `nest_asyncio.apply()` первой строкой; `_run_sync` через `loop.run_until_complete` |
| Регрессия M9.1/M9.2 тестов | Low | High | Все используют `gateway: llm_simulated`; `human_gateway=None` → старый path |
| debate.py старый ключ `"gateway"` в тестах | Medium | Medium | `or kwargs.get("gateway")` сохраняет back-compat |

## Out of Scope

- LISTEN/NOTIFY вместо polling — отложено до M14.1
- Многопользовательский режим (>1 участника на UI-инстанс) — один за раз
- Sandbox безопасности — покрыто M4-DockerSandbox
- LangGraph `interrupt()`/resume — не нужен, inline gateway работает
- Auth сложнее shared-secret — в M14.x

## Timeline

- Total: ~17.5h
- Phases: 1 (~3h) → 2 (~2h) → 3 (~2.5h) → 4 (~3h) → 5 (~4h) → 6 (~3h)
- Created: 2026-05-17
