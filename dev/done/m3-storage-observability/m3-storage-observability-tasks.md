# M3 — Storage & Observability — Задачи

## Wave 1: Зависимости (solo) — NOT STARTED

- [ ] **1.1** Обновить `pyproject.toml` — runtime deps для M3 — `pyproject.toml`
  - Type: simple
  - Depends On: —
  - Can-Parallel-With: —
  - Acceptance: `uv sync --dev` чистый; `uv run python -c "import sqlalchemy; import asyncpg; import langgraph.checkpoint.postgres.aio; import psycopg; import structlog"` OK; `uv run pytest` (существующие тесты) зелёный; `uv run python -c "from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver"` не падает.
  - Детали: переместить `sqlalchemy[asyncio]>=2.0`, `asyncpg>=0.29` из `dev` в `dependencies`; добавить в runtime: `langgraph-checkpoint-postgres>=3.0,<4`, `psycopg[binary,pool]>=3.2`, `structlog>=24`; добавить в dev: `pandas>=2.2`; оставить в dev: `alembic>=1.13`. После правки выполнить `uv sync --dev` и закоммитить `uv.lock`.

---

## Wave 2: Ядро хранилища (параллельно, после Wave 1) — NOT STARTED

- [ ] **2.1** Создать SQLAlchemy-модели 6 бизнес-таблиц — `src/atm/storage/models.py` (TDD)
  - Type: tdd
  - Depends On: 1.1
  - Can-Parallel-With: 2.2, 2.3, 2.4
  - Acceptance: `Base.metadata.sorted_tables` → 6 таблиц (`budget_events`, `experiments`, `human_interactions`, `phases`, `runs`, `topology_transitions`); все reproducibility fields присутствуют (`models_by_role_json`, `model_version_snapshot`, `sandbox_image_digest`); `Phase.from_phase`, `Phase.ended_at` nullable; `HumanInteraction.raw_tlx_score` nullable Float; `FinishReason("success").value == "success"`; `uv run pytest tests/unit/storage/test_models.py -v` зелёный; `uv run mypy src/atm/storage/models.py` чистый.
  - Детали: 6 моделей — `Experiment` (root), `Run`, `Phase`, `HumanInteraction`, `BudgetEvent`, `TopologyTransition`. `FinishReason` StrEnum. Все relationships `lazy="raise"`. JSONB `server_default=text("'{}'::jsonb")`. ARRAY(String) `server_default=text("'{}'::text[]")`. **НЕ трогать `src/atm/storage/__init__.py`**.

- [ ] **2.2** Создать async session factory — `src/atm/storage/session.py` (simple)
  - Type: simple
  - Depends On: 1.1
  - Can-Parallel-With: 2.1, 2.3, 2.4
  - Acceptance: `create_engine("postgresql+asyncpg://fake/db", pool_pre_ping=False)` → `AsyncEngine`; `create_session_factory(engine).kw["expire_on_commit"] is False`; `session_scope` — async context manager; `uv run pytest tests/unit/storage/test_session.py` зелёный; `uv run mypy src/atm/storage/session.py` чистый.
  - Детали: `create_engine(dsn, *, echo, pool_size, max_overflow, pool_pre_ping)`, `create_session_factory(engine)` → `async_sessionmaker`, `session_scope(factory)` → commit/rollback CM. **НЕ трогать `src/atm/storage/__init__.py`**.

- [ ] **2.3** Создать PyArrow schemas + ParquetWriter — `src/atm/storage/schemas.py` + `parquet_writer.py` (TDD)
  - Type: tdd
  - Depends On: 1.1
  - Can-Parallel-With: 2.1, 2.2, 2.4
  - Acceptance: все 6 pa.schema с `pa.timestamp("us", tz="UTC")` на timestamp-колонках; `write_llm_call * 3 → flush() → pa.parquet.read_table().num_rows == 3`; auto-flush при `buffer_rows=2, 3 writes`; `close()` идемпотентен; tz-aware datetime round-trip (`tzinfo is not None`); schema-mismatch → `pa.lib.ArrowInvalid`; multi-stream concurrency без коллизий; `uv run pytest tests/unit/storage/test_schemas.py tests/unit/storage/test_parquet_writer.py -v` зелёный.
  - Детали: `ParquetWriter(root, run_id, exp_id, *, buffer_rows, buffer_seconds, compression)`. Путь файла: `root/experiments/{exp_id}/runs/{run_id}/{stream}.parquet`. Scratchpad: `.../scratchpads/{agent_id}.parquet`. asyncio.Lock per stream. **НЕ трогать `src/atm/storage/__init__.py`**.

- [ ] **2.4** Создать checkpointer wrapper — `src/atm/storage/checkpointer.py` (simple)
  - Type: simple
  - Depends On: 1.1
  - Can-Parallel-With: 2.1, 2.2, 2.3
  - Acceptance: `_to_psycopg_dsn("postgresql+asyncpg://a:b@h/d") == "postgresql://a:b@h/d"` (idempotent); mock-based lifecycle-тест: `pool.open()` → `saver.setup()` → `yield` → `pool.close()` ordering confirmed; `from atm.storage.checkpointer import checkpointer_scope, build_checkpointer` без ошибок; `uv run pytest tests/unit/storage/test_checkpointer.py -v` зелёный.
  - Детали: `checkpointer_scope(dsn, *, max_size, min_size)` — primary `@asynccontextmanager`; `build_checkpointer(dsn, *, max_size, min_size)` — low-level; `_to_psycopg_dsn` helper. psycopg pool с `autocommit=True, row_factory=dict_row, prepare_threshold=0`. **НЕ трогать `src/atm/storage/__init__.py`**.

---

## Wave 3: Alembic env (после 2.1) — NOT STARTED

- [ ] **3.1** Подключить `target_metadata` в Alembic env.py — `alembic/env.py` (simple)
  - Type: simple
  - Depends On: 2.1
  - Can-Parallel-With: —
  - Acceptance: строки 27–29 в `alembic/env.py` заменены на `from atm.storage.models import Base; target_metadata = Base.metadata`; TODO-комментарий M3 удалён; `uv run alembic check` не падает; временная `alembic revision --autogenerate` (удалить после проверки) видит 6 таблиц.

---

## Wave 4: Alembic migration (после 2.1 + 3.1) — NOT STARTED

- [ ] **3.2** Сгенерировать и curate начальную Alembic-миграцию — `alembic/versions/0001_initial_business_schema.py` (simple)
  - Type: simple
  - Depends On: 2.1, 3.1
  - Can-Parallel-With: —
  - Acceptance: `alembic upgrade head` на чистой БД → 6 таблиц; `\d runs` содержит `models_by_role_json jsonb`, `model_version_snapshot jsonb`, `sandbox_image_digest varchar(80)`, `finish_reason varchar(32)`; `\d human_interactions` содержит `raw_tlx_score double precision`, `request_id varchar(64)`; `\d phases` содержит `from_phase varchar(32)`, `ended_at timestamptz`; `grep "::jsonb" alembic/versions/0001_*.py` — ≥ N JSONB-колонок; `grep -E "server_default=sa\.text\(\"'{}'\"" alembic/versions/0001_*.py` — пусто; `alembic downgrade base && alembic upgrade head` — идемпотентно.
  - Детали: `down_revision = None`. Docstring: "Initial business schema — arch.md §3.4 DDL as of 2026-04-24. Checkpoint tables NOT here." `downgrade()` в обратном FK-порядке: `topology_transitions, budget_events, human_interactions, phases, runs, experiments`. Checkpoint-таблиц в миграции НЕТ.

---

## Wave 5: Сериализаторы (после 2.3) — NOT STARTED

- [ ] **4.1** Создать observability serializers — `src/atm/observability/serializers.py` (simple)
  - Type: simple
  - Depends On: 2.3
  - Can-Parallel-With: 2.4
  - Acceptance: 6 функций (`llm_response_to_row`, `message_to_row`, `tool_call_to_row`, `phase_transition_to_row`, `topology_transition_to_row`, `scratchpad_entry_to_row`); для каждой `set(row.keys()) == set(SCHEMA.names)`; `pa.Table.from_pylist([row], schema=SCHEMA)` без ошибок; tz-aware datetime не теряет tzinfo; `uv run pytest tests/unit/observability/test_serializers.py -v` зелёный; `uv run mypy src/atm/observability/serializers.py` чистый.
  - Детали: чистые функции без side-effects. `_dumps` helper: `json.dumps(obj, default=str, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`. Ключи dict точно по schemas.py. **НЕ трогать `src/atm/observability/__init__.py`**.

---

## Wave 6: Callback handler (после 2.1 + 2.2 + 2.3 + 4.1) — NOT STARTED

- [ ] **4.2** Создать ExperimentCallbackHandler — `src/atm/observability/callbacks.py` (TDD)
  - Type: tdd
  - Depends On: 2.1, 2.2, 2.3, 4.1
  - Can-Parallel-With: 2.4
  - Acceptance: ordering-тест `flush_idx < insert_idx` для `phase_transition` и `topology_transition` — зелёный; atomic `UPDATE ... RETURNING` + budget warn/exceed branching — зелёный; exception в хуке не пробрасывается наружу; root-chain detection через metadata-flag работает; `uv run pytest tests/unit/observability/test_callbacks.py -v` (7+ тестов) зелёный; `uv run mypy src/atm/observability/callbacks.py` чистый.
  - Детали: `ExperimentCallbackHandler(run_id, exp_id, session_factory, parquet_writer, *, budget_warn_threshold, budget_exceed_threshold, logger)`. Базовый класс `langchain_core.callbacks.AsyncCallbackHandler`. Structlog bind `run_id + exp_id`. Root-chain: `parent_run_id is None AND metadata.get("is_root_run") is True` с fallback на первый seen. Разделение: `on_llm_end` → Parquet llm_calls + PG atomic UPDATE; `on_tool_end/error` → Parquet tool_calls; custom_event `message_emit` → Parquet messages; `phase_transition/topology_transition` → flush() FIRST → PG INSERT → Parquet. **НЕ трогать `src/atm/observability/__init__.py`**.

---

## Wave 7: Integration smoke (после 3.2 + 2.3 + 2.4 + 4.1 + 4.2) — NOT STARTED

- [ ] **5.1** Написать integration smoke-тест — `tests/integration/storage/test_smoke_run.py` (TDD)
  - Type: tdd
  - Depends On: 3.2, 2.3, 2.4, 4.1, 4.2
  - Can-Parallel-With: —
  - Acceptance: `ATM_INTEGRATION_PG=1 uv run pytest tests/integration/storage/ -v` зелёный; `select(Phase).where(...)` → ≥ 1 строка; `pandas.read_parquet(.../messages.parquet).shape[0] >= 1`; `pandas.read_parquet(.../llm_calls.parquet).shape[0] == 1`; `pandas.read_parquet(.../phases.parquet).shape[0] >= 1`; `select(BudgetEvent).where(...)` → ≥ 1 row; checkpoint `saver.alist(...)` → count > 0; без `ATM_INTEGRATION_PG=1` → `pytest.skip`; `test_alembic_equivalence` проверяет присутствие 6 таблиц + reproducibility columns.
  - Детали: два fixture'а — `pg_engine_fast` (function scope, `create_all/drop_all`) и `pg_engine_alembic` (session scope, `alembic upgrade head`). Fake `StateGraph` с 1-2 нодами, dispatch custom events. Manual `on_llm_end` для llm_calls + budget_events. `asyncio.sleep(0.1)` после `ainvoke` перед assert'ами. Marker `@pytest.mark.integration`.

---

## Wave 8: Финализация (после всего) — NOT STARTED

- [ ] **5.2** Финализировать публичный API + обновить codebase-map — `src/atm/storage/__init__.py`, `src/atm/observability/__init__.py`, `dev/codebase-map.md` (simple)
  - Type: simple
  - Depends On: 2.1, 2.2, 2.3, 2.4, 3.2, 4.1, 4.2, 5.1
  - Can-Parallel-With: —
  - Acceptance: `from atm.storage import ParquetWriter, build_checkpointer, checkpointer_scope, Base, Experiment, Run, FinishReason, create_engine, create_session_factory, session_scope` — OK; `from atm.observability import ExperimentCallbackHandler, llm_response_to_row` — OK; `uv run pytest` — весь suite зелёный; `uv run mypy src/atm` чистый; `dev/codebase-map.md` содержит TODO-block reproducibility fields M5+.
  - Детали: `storage/__init__.py` — re-export всех публичных символов из models, session, parquet_writer, schemas, checkpointer. `observability/__init__.py` — re-export ExperimentCallbackHandler + 6 сериализаторов. codebase-map: M3 статус "complete", 4 flush-инварианта, two-pools, atomic UPDATE ... RETURNING, TODO для M5+ (Runner/LLMWrapper/DockerSandbox ownership).

---

## Статистика

- Итого: 11 задач в 8 волнах
- Выполнено: 0 / 11
- TDD-задачи: 4 (2.1, 2.3, 4.2, 5.1)
- Simple-задачи: 7 (1.1, 2.2, 2.4, 3.1, 3.2, 4.1, 5.2)
- Max параллелизм: 4 (Wave 2)
- Оценка времени: ~14–18 ч

---

## Как обновлять этот файл

После каждой завершённой задачи:
1. Заменить `[ ]` на `[x]` у выполненной задачи.
2. Обновить заголовок Wave: если все задачи в волне — `COMPLETE`, если часть — `IN PROGRESS`.
3. Обновить счётчик "Выполнено: N / 11".
4. Обновить раздел SESSION PROGRESS в `m3-storage-observability-context.md`.
