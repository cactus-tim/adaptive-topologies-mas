# M14 — Streamlit HITL UI, NASA-TLX, Proctoring Protocol — Context

## SESSION PROGRESS (2026-05-17)

### COMPLETED
- Step 1.1: Alembic migration 0005 — создана миграция `alembic/versions/0005_m14_streamlit_hitl.py`
  (study_sessions, human_request_queue, human_interactions.study_session_id FK);
  тест `tests/integration/storage/test_migration_0005.py` написан и зелёный (1 passed);
  upgrade/downgrade round-trip работает без ошибок.

### IN PROGRESS
- Step 1.2: ORM-модели StudySession, HumanRequestQueue, расширение HumanInteraction

### BLOCKERS
- None

## Quick Resume

1. Прочитать этот файл
2. Открыть `m14-tasks.md` — посмотреть первый незавершённый чекбокс
3. Прочитать `m14-plan.md` Phase 1 для стратегии
4. Начать с: **Step 1.1** — `alembic/versions/0005_m14_streamlit_hitl.py` (миграция)

## Key Files

**`alembic/versions/0005_m14_streamlit_hitl.py`**
- Role: DB-миграция — создание study_sessions, human_request_queue, расширение human_interactions
- Planned change: New migration (down_revision="0004")
- Status: DONE (2026-05-17)

**`tests/integration/storage/test_migration_0005.py`**
- Role: Integration test — round-trip upgrade/downgrade для миграции 0005
- Planned change: New test file
- Status: DONE (2026-05-17)

**`src/atm/storage/models.py`**
- Role: SQLAlchemy ORM — все таблицы Postgres
- Planned change: Добавить StudySession, HumanRequestQueue; расширить HumanInteraction (study_session_id FK + relationship)
- Status: NOT STARTED

**`src/atm/experiment/config.py`**
- Role: Pydantic-конфиг всего эксперимента; HumanCfg — конфиг HITL
- Planned change: HumanCfg.gateway добавляет "streamlit"; новые поля queue_dsn, participant_id, study_session_id, shared_secret, fallback_llm_model
- Status: NOT STARTED

**`conf/human/streamlit.yaml`**
- Role: Стандартный конфиг для прокторированных user-study сессий
- Planned change: New file
- Status: NOT STARTED

**`src/atm/human/_queue.py`**
- Role: Async PG-очередь между runner-процессом и Streamlit-процессом
- Planned change: New module (~250 LoC); API: enqueue, fetch_pending, claim, submit_response, wait_for_response, cancel
- Status: NOT STARTED

**`src/atm/human/streamlit_gateway.py`**
- Role: HumanGateway реализация через PG-очередь
- Planned change: New module (~200 LoC); enqueue запроса → polling ответа → возврат HumanResponse с TLX
- Status: NOT STARTED

**`src/atm/human/__init__.py`**
- Role: Public API пакета atm.human
- Planned change: Добавить StreamlitHumanGateway в __all__
- Status: NOT STARTED

**`src/atm/human/_node_factory.py`**
- Role: build_human_node_factory — фабрика LangGraph-нод для HITL
- Planned change: Новый keyword-only параметр fallback_llm в сигнатуру; patch fallback-блока сайта A
- Status: NOT STARTED

**`src/atm/observability/callbacks.py`**
- Role: ExperimentCallbackHandler — пишет human_interactions в PG
- Planned change: Расширить _handle_human_response: обе ветки (INSERT строка 561-579, UPDATE строка 580-588) добавляют tlx_scores, raw_tlx_score, study_session_id
- Status: NOT STARTED

**`src/atm/ui/app.py`**
- Role: Streamlit entry-point (главный файл UI)
- Planned change: New; nest_asyncio.apply() первой строкой; роутинг страниц
- Status: NOT STARTED

**`src/atm/ui/auth.py`**
- Role: Shared-secret аутентификация участника
- Planned change: New; проверка ATM_UI_SHARED_SECRET через st.session_state
- Status: NOT STARTED

**`src/atm/ui/views.py`**
- Role: Streamlit-страницы: Login, Queue, Response form, Proctor panel
- Planned change: New
- Status: NOT STARTED

**`src/atm/ui/tlx.py`**
- Role: NASA-TLX форма (6 шкал)
- Planned change: New; render_tlx_form() → dict[str, int]; raw_tlx_score = mean
- Status: NOT STARTED

**`src/atm/ui/_state.py`**
- Role: _run_sync helper, st.session_state обёртки
- Planned change: New; _run_sync(coro) через loop.run_until_complete
- Status: NOT STARTED

**`src/atm/experiment/runner.py`**
- Role: Главный runner эксперимента; wires топологии
- Planned change: Добавить _build_human_gateway фабрику; оба wiring-сайта (lines 1187-1195 и 2156-2164) передают human_gateway kwarg
- Status: NOT STARTED

**`src/atm/topology/chain.py`**
- Role: Chain-топология
- Planned change: Dispatch patch (human_gateway kwarg); _build_human_reviewer_node + fallback_llm параметр (сайт B)
- Status: NOT STARTED

**`src/atm/topology/star.py`**
- Role: Star-топология
- Planned change: Dispatch patch; caller build_human_node_factory передаёт fallback_llm=gateway_llm
- Status: NOT STARTED

**`src/atm/topology/mesh.py`**
- Role: Mesh-топология
- Planned change: Dispatch patch с переиспользованием существующего локала human_gateway; closure fallback patch (сайт C)
- Status: NOT STARTED

**`src/atm/topology/debate.py`**
- Role: Debate-топология
- Planned change: Dispatch — kwargs.get("human_gateway") or kwargs.get("gateway") (back-compat с тестами); _build_human_judge_node + fallback_llm (сайт D); closure fallback (сайт E)
- Status: NOT STARTED

**`src/atm/topology/hierarchical.py`**
- Role: Hierarchical-топология
- Planned change: Dispatch patch; _build_human_top_reviewer_node + fallback_llm (сайт F); _build_human_sub_reviewer_node + fallback_llm (сайт G); _build_subgraph_with_human принимает fallback_llm; 2 caller-а обновляются
- Status: NOT STARTED

**`src/atm/topology/adaptive.py`**
- Role: Adaptive-топология
- Planned change: Dispatch patch с переиспользованием существующего _gateway; closure fallback patch (сайт H)
- Status: NOT STARTED

**`pyproject.toml`**
- Role: Package definition, dependencies, mypy config
- Planned change: [project.optional-dependencies] ui = ["streamlit>=1.36,<2", "nest_asyncio>=1.5,<2"]; mypy overrides для streamlit.* и nest_asyncio
- Status: NOT STARTED

**`docker-compose.yml`**
- Role: Local development orchestration
- Planned change: Добавить сервис ui с профилем ui
- Status: NOT STARTED

**`Dockerfile.ui`**
- Role: Streamlit контейнер
- Planned change: New; FROM python:3.11-slim; uv install -e ".[ui]"
- Status: NOT STARTED

## Decisions

**D1 — Polling vs LISTEN/NOTIFY**
- Decision: Polling с интервалом 1 секунда
- Rationale: LISTEN/NOTIFY отложен до M14.1 для упрощения M14

**D2 — Multi-participant**
- Decision: Один участник за раз на одну Streamlit-инстанс
- Rationale: Лабораторный сценарий; shared-secret достаточен

**D3 — TLX per-decision vs post-task**
- Decision: Per-decision (после каждого HITL-запроса)
- Rationale: Rubio et al. (2004) — higher granularity, lower recall bias

**D4 — HumanResponse.source для UI**
- Decision: "human" (не расширяем Literal)
- Rationale: Согласованность с CLIGateway; source уже различает "human" и "llm_sim"

**D5 — study_session_id пробрасывание**
- Decision: Через HumanResponse.payload["study_session_id"]
- Rationale: Не меняет контракт HumanResponse; payload уже есть и сериализуется

**D6 — Backward compat по умолчанию**
- Decision: _build_human_gateway для llm_simulated возвращает (None, llm_factory()); human_gateway kwarg = None; топология строит LLMSimulatedGateway через старую логику
- Rationale: Нулевые изменения поведения для всех существующих конфигов M9.1/M9.2

**D7 — Канонические имена kwargs/локалов gateway dispatch**
- Decision: Runner-side kwarg key = "human_gateway"; fallback LLM kwarg = "human_gateway_llm" (без изменений); локал после kwargs.get = human_gateway / gateway_llm
- Rationale: Единое соглашение по всем 6 топологиям; debate.py читает также старый "gateway" key для back-compat с тестами

## Constraints

- Streamlit-процесс и runner-процесс используют одну Postgres-инстанцию (DSN из ATM_PG_DSN)
- Streamlit 1.33+ требует nest_asyncio.apply() для async-кода в UI event loop
- mypy strict на весь src/atm/ — нельзя добавлять Any без обоснования
- Все 8 fallback-сайтов должны guard-овать `if _fb_llm is not None else None` — иначе LLMSimulatedGateway(llm=None) создаётся тихо и падает позже
- `alembic heads` должен показывать единственный head "0004" до миграции 0005
- Python 3.11+; async/await обязателен для всего IO
- При `gateway: streamlit` ATM_PG_DSN или human.queue_dsn обязателен — иначе ValueError на старте
