# M14 — Streamlit HITL UI, NASA-TLX, Proctoring Protocol — Tasks

## Phase 1: DB & Config Foundation (~3h) — IN PROGRESS

- [x] 1.1 Alembic migration 0005 — `alembic/versions/0005_m14_streamlit_hitl.py`
  - type: tdd | depends-on: — | can-parallel-with: 1.3, 1.4
  - Три структурных изменения: `study_sessions`, `human_request_queue`, расширение `human_interactions`
  - Acceptance: upgrade/downgrade round-trip без ошибок; `pytest tests/integration/storage/test_migration_0005.py -m requires_postgres` зелёный

- [ ] 1.2 ORM-модели `StudySession`, `HumanRequestQueue`, расширение `HumanInteraction` — `src/atm/storage/models.py`
  - type: tdd | depends-on: 1.1 | can-parallel-with: —
  - Acceptance: `pytest tests/unit/storage/test_models.py` (min 2 новых теста); `mypy src/atm/storage/models.py` strict

- [ ] 1.3 Расширить `HumanCfg` + `conf/human/streamlit.yaml` — `src/atm/experiment/config.py`
  - type: simple | depends-on: — | can-parallel-with: 1.1, 1.4, 2.1
  - gateway: Literal["llm_simulated", "cli", "streamlit"]; поля queue_dsn, participant_id, study_session_id, shared_secret, fallback_llm_model
  - Acceptance: `pytest tests/unit/experiment/test_config_human.py`; `mypy` strict

- [ ] 1.4 pyproject optional-dependency `[ui]` + mypy overrides — `pyproject.toml`
  - type: simple | depends-on: — | can-parallel-with: 1.1, 1.3
  - streamlit>=1.36,<2; nest_asyncio>=1.5,<2; overrides для streamlit.* и nest_asyncio
  - Acceptance: `uv lock --check`; `python -c "import streamlit, nest_asyncio"` без ошибок

## Phase 2: Queue Helper (~2h) — NOT STARTED

- [ ] 2.1 `atm.human._queue.HumanRequestQueue` — `src/atm/human/_queue.py`
  - type: tdd | depends-on: 1.2 | can-parallel-with: —
  - API: enqueue (idempotent ON CONFLICT DO NOTHING), fetch_pending, claim (atomic UPDATE RETURNING), submit_response (AND response_json IS NULL), wait_for_response (polling 1s + TODO M14.1), cancel
  - Acceptance: `pytest tests/unit/human/test_queue.py` + `pytest tests/integration/human/test_queue_pg.py -m requires_postgres`; `mypy` strict

## Phase 3: Gateway + Callback (~2.5h) — NOT STARTED

- [ ] 3.1 `StreamlitHumanGateway` — `src/atm/human/streamlit_gateway.py`
  - type: tdd | depends-on: 2.1, 1.3 | can-parallel-with: 4.1
  - Enqueue → wait_for_response → HumanResponse(source="human", tlx_scores=..., payload={"study_session_id": ...})
  - Acceptance: `pytest tests/unit/human/test_streamlit_gateway.py`; `mypy` strict

- [ ] 3.2 Расширить `_handle_human_response` — `src/atm/observability/callbacks.py`
  - type: tdd | depends-on: 3.1 | can-parallel-with: 4.1
  - Обе ветки INSERT (561-579) и UPDATE (580-588): добавить tlx_scores JSONB, raw_tlx_score FLOAT, study_session_id UUID
  - Acceptance: `pytest tests/unit/observability/test_human_response_tlx.py` (обе ветви); существующие тесты без TLX не сломаны

## Phase 4: Streamlit UI (~3h) — NOT STARTED

- [ ] 4.1 Пакет `atm.ui` — все модули — `src/atm/ui/`
  - type: simple | depends-on: 3.1 | can-parallel-with: 5.1
  - Модули: app.py (nest_asyncio.apply() первой), auth.py (shared-secret), views.py (Login/Queue/Response/Proctor), tlx.py (6-шкальная форма), _state.py (_run_sync helper)
  - Acceptance: `pytest tests/unit/ui/` зелёный; manual smoke `streamlit run src/atm/ui/app.py` поднимается

## Phase 5: Runner Wiring (~4h) — NOT STARTED

- [ ] 5.1 `_build_human_gateway` фабрика в runner.py — `src/atm/experiment/runner.py`
  - type: tdd | depends-on: 3.1 | can-parallel-with: 4.1
  - Функция возвращает (primary_gateway | None, fallback_llm | None); оба wiring-сайта (lines ~1187, ~2156) обновляются: передают human_gateway=human_gateway kwarg
  - Acceptance: `pytest tests/unit/experiment/test_runner_human_gateway.py` — 4 сценария (llm_simulated, cli, streamlit, disabled); `mypy` strict

- [ ] 5.2 Gateway dispatch patch для 6 топологий — D7 канон `human_gateway` kwarg
  - type: tdd | depends-on: 5.1 | can-parallel-with: —
  - chain.py, star.py: добавить kwargs.get("human_gateway") как приоритетный источник
  - mesh.py: переиспользовать существующий локал human_gateway (НЕ переименовывать)
  - debate.py: kwargs.get("human_gateway") or kwargs.get("gateway") (legacy back-compat)
  - hierarchical.py, adaptive.py: аналогично chain с учётом своих локалов (_gateway)
  - Acceptance: 6 dispatch-тестов в test_topology_streamlit_dispatch.py зелёные; все 6 grep-чеков из plan верифицированы

- [ ] 5.3 Fallback-блоки: patch 8 сайтов
  - type: tdd | depends-on: 5.2 | can-parallel-with: —
  - Сайт A (_node_factory.py): новый параметр fallback_llm в build_human_node_factory; patch closure
  - Сайт B (chain.py): новый параметр fallback_llm в _build_human_reviewer_node; patch caller
  - Сайт C (mesh.py): closure-capture gateway_llm (уже в scope после 5.2)
  - Сайт D (debate.py): новый параметр fallback_llm в _build_human_judge_node; patch caller
  - Сайт E (debate.py): closure-capture _gateway_llm
  - Сайт F (hierarchical.py): новый параметр fallback_llm в _build_human_top_reviewer_node
  - Сайт G (hierarchical.py): fallback_llm через _build_subgraph_with_human цепочку
  - Сайт H (adaptive.py): closure-capture llm_wrapper
  - Все guard: `LLMSimulatedGateway(llm=_fb_llm) if _fb_llm is not None else None`
  - Acceptance: тест с FakeGateway (без ._llm) + force-timeout не поднимает AttributeError; `mypy` strict

## Phase 6: Infra & Integration (~3h) — NOT STARTED

- [ ] 6.1 docker-compose профиль `ui` + `Dockerfile.ui`
  - type: simple | depends-on: 4.1, 1.4 | can-parallel-with: 6.2, 6.3
  - Acceptance: `docker compose --profile ui up --build`; `curl -I http://localhost:8501` → 200

- [ ] 6.2 Integration test — Chain + StreamlitGateway e2e — `tests/integration/human/test_streamlit_e2e.py`
  - type: tdd | depends-on: 3.2, 5.3 | can-parallel-with: 6.1, 6.3
  - FakeStreamlitClient: asyncio.gather(run_one(cfg), fake_client.run()); conftest cleanup truncate 4 таблиц
  - Acceptance: `ATM_ENABLE_PG_TESTS=1 pytest tests/integration/human/test_streamlit_e2e.py -m requires_postgres -v` зелёный; human_interactions содержит tlx_scores + raw_tlx_score + study_session_id

- [ ] 6.3 Документация сессий — `dev/active/m14/README.md`, `proctor-protocol.md`, `participant-consent.md`
  - type: simple | depends-on: 1.1, 3.1, 4.1 | can-parallel-with: 6.1, 6.2
  - SELECT-пример латентности; чеклист proctoring-протокола; ссылки D1-D7
  - Acceptance: ручной review пройден

---

## Stats

- Total: 14 tasks across 6 phases · ~17.5h
- Done: 1 / 14

## Verification Checklist (Step 5.2)

После патча топологий обязательно прогнать 6 grep-чеков:

```bash
# 1. Каждый if cli/else LLMSim — внутри блока `if <var> is None`
grep -rn 'gateway == "cli"\|gateway_kind == "cli"\|gateway_type == "cli"' src/atm/topology/

# 2. Ровно 6 чтений "human_gateway" из kwargs (по одному на файл топологии)
grep -rn 'kwargs.get("human_gateway")' src/atm/topology/

# 3. Старый "gateway" key — ТОЛЬКО debate.py (через `or kwargs.get("gateway")`)
grep -rn 'kwargs.get("gateway")' src/atm/topology/

# 4. Все top-level factories принимают fallback_llm (min 6+ совпадений)
grep -rn 'fallback_llm' src/atm/human/_node_factory.py src/atm/topology/chain.py src/atm/topology/debate.py src/atm/topology/hierarchical.py

# 5. Closure-capture sites используют локальную переменную (3 совпадения: mesh, debate, adaptive)
grep -rn 'gateway_llm if gateway_llm is not None\|_gateway_llm if _gateway_llm is not None\|llm_wrapper if llm_wrapper is not None' src/atm/topology/

# 6. Все fallback-блоки guard-ят None
grep -rn 'LLMSimulatedGateway(llm=_fb_llm)\|_LLMSimulatedGateway(llm=_fb_llm)' src/atm/
```

## How to Update

После каждого завершённого шага:
1. Поменять `[ ]` на `[x]` в этом файле
2. Обновить `SESSION PROGRESS` в `m14-context.md`
3. Если фаза завершена полностью — обновить заголовок фазы: `NOT STARTED` → `IN PROGRESS` → `COMPLETE`
4. Обновить `Stats.Done` счётчик
