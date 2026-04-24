# M3 — Storage & Observability — Контекст

## SESSION PROGRESS (2026-04-24)

### ЗАВЕРШЕНО
- (ничего пока — задача не начата)

### В ПРОЦЕССЕ
- Не начато

### БЛОКЕРЫ
- Нет

---

## Быстрое возобновление

1. Прочитать этот файл.
2. Открыть `m3-storage-observability-tasks.md` — найти первую незавершённую задачу.
3. Прочитать `m3-storage-observability-plan.md` Phase 1 для стратегии.
4. Начать с: Step 1.1 — обновить `pyproject.toml` (переместить deps в runtime, добавить новые).
5. После `uv sync --dev` — Wave 2 можно запускать параллельно (Steps 2.1, 2.2, 2.3, 2.4).

---

## Ключевые файлы

### Новые файлы (создаются в M3)

**`src/atm/storage/models.py`**
- Роль: 6 SQLAlchemy-моделей (`Experiment`, `Run`, `Phase`, `HumanInteraction`, `BudgetEvent`, `TopologyTransition`) + `FinishReason` StrEnum.
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`src/atm/storage/session.py`**
- Роль: `create_engine`, `create_session_factory`, `session_scope` — async session factory.
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`src/atm/storage/schemas.py`**
- Роль: 6 PyArrow pa.schema констант для parquet-файлов. Все timestamps `pa.timestamp("us", tz="UTC")`.
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`src/atm/storage/parquet_writer.py`**
- Роль: буферизованный async `ParquetWriter` — lazy open, auto-flush по buffer_rows/buffer_seconds, explicit flush, close().
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`src/atm/storage/checkpointer.py`**
- Роль: `checkpointer_scope` (primary `@asynccontextmanager`) + `build_checkpointer` (low-level) для `AsyncPostgresSaver` через psycopg pool.
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`src/atm/observability/serializers.py`**
- Роль: 6 чистых функций `pydantic-model → dict-row по pa.schema`: `llm_response_to_row`, `message_to_row`, `tool_call_to_row`, `phase_transition_to_row`, `topology_transition_to_row`, `scratchpad_entry_to_row`.
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`src/atm/observability/callbacks.py`**
- Роль: `ExperimentCallbackHandler(AsyncCallbackHandler)` — LangGraph/LangChain callback handler. Пишет в PG (atomic UPDATE ... RETURNING) и Parquet. Обеспечивает 4 flush-инварианта arch.md §10.3.
- Запланированное изменение: создать с нуля.
- Статус: НЕ НАЧАТО

**`alembic/versions/0001_initial_business_schema.py`**
- Роль: начальная Alembic-миграция с 6 бизнес-таблицами + индексами.
- Запланированное изменение: autogenerate + curate (проверить JSONB `::jsonb` cast).
- Статус: НЕ НАЧАТО

### Тестовые файлы (новые)

**`tests/unit/storage/test_models.py`**
- Роль: unit-тесты моделей без БД (sa.inspect, FK graph, FinishReason enum).
- Статус: НЕ НАЧАТО

**`tests/unit/storage/test_session.py`**
- Роль: unit-тесты session factory (сигнатуры без реального connect).
- Статус: НЕ НАЧАТО

**`tests/unit/storage/test_schemas.py`**
- Роль: unit-тесты pa.schema констант (tz="UTC" на каждой timestamp-колонке).
- Статус: НЕ НАЧАТО

**`tests/unit/storage/test_parquet_writer.py`**
- Роль: unit-тесты ParquetWriter (auto-flush, idempotent close, concurrency, tz-aware round-trip).
- Статус: НЕ НАЧАТО

**`tests/unit/storage/test_checkpointer.py`**
- Роль: unit-тесты `_to_psycopg_dsn` + mock-based lifecycle-тест (`open → setup → yield → close` ordering).
- Статус: НЕ НАЧАТО

**`tests/unit/observability/test_serializers.py`**
- Роль: unit-тесты сериализаторов (присутствие ключей, `pa.Table.from_pylist` round-trip).
- Статус: НЕ НАЧАТО

**`tests/unit/observability/test_callbacks.py`**
- Роль: unit-тесты callback handler (ordering invariant, atomic UPDATE RETURNING, budget branching, exception swallowing).
- Статус: НЕ НАЧАТО

**`tests/integration/storage/test_smoke_run.py`**
- Роль: integration smoke — fake LangGraph run, PG rows + parquet readable by pandas. Exit criteria M3.
- Статус: НЕ НАЧАТО

**`tests/integration/storage/conftest.py`**
- Роль: два pg_engine fixtures — `pg_engine_fast` (function scope, create_all/drop_all) и `pg_engine_alembic` (session scope, alembic upgrade head).
- Статус: НЕ НАЧАТО

### Модифицируемые файлы

**`pyproject.toml`**
- Роль: конфигурация зависимостей проекта.
- Запланированное изменение: переместить `sqlalchemy[asyncio]>=2.0`, `asyncpg>=0.29` из `dev` в `dependencies`; добавить в runtime `langgraph-checkpoint-postgres>=3.0,<4`, `psycopg[binary,pool]>=3.2`, `structlog>=24`; добавить в dev `pandas>=2.2`.
- Статус: НЕ НАЧАТО

**`alembic/env.py`**
- Роль: Alembic environment script (async skeleton).
- Запланированное изменение: строка 27–29 — заменить `target_metadata = None` на `from atm.storage.models import Base; target_metadata = Base.metadata`.
- Статус: НЕ НАЧАТО

**`src/atm/storage/__init__.py`**
- Роль: публичный API storage слоя.
- Запланированное изменение: overwrite с полными re-exports. ВАЖНО: трогать ТОЛЬКО в Step 5.2 (Wave 8).
- Статус: НЕ НАЧАТО

**`src/atm/observability/__init__.py`**
- Роль: публичный API observability слоя.
- Запланированное изменение: overwrite с полными re-exports. ВАЖНО: трогать ТОЛЬКО в Step 5.2 (Wave 8).
- Статус: НЕ НАЧАТО

**`dev/codebase-map.md`**
- Роль: карта кодовой базы проекта.
- Запланированное изменение: M3 статус "complete", перечень модулей и экспортов, TODO для reproducibility fields M5+.
- Статус: НЕ НАЧАТО

---

## Архитектурные решения

### Two-pools pattern (arch.md §11.3)
- **Решение:** бизнес-БД использует SQLAlchemy `AsyncEngine` (asyncpg); checkpointer — отдельный `psycopg.AsyncConnectionPool` с `autocommit=True, row_factory=dict_row, prepare_threshold=0`.
- **Обоснование:** `langgraph-checkpoint-postgres` требует psycopg-specific API; смешивание с SQLAlchemy pool невозможно. Два изолированных pool'а исключают cross-pool deadlock.

### 4 flush-инварианта (arch.md §10.3)
- **Решение:** (a) root `on_chain_end` → `parquet_writer.close()`; (b) `phase/topology_transition` → `flush()` СТРОГО ДО PG INSERT; (c) root `on_chain_error` → `parquet_writer.close()`; (d) buffer overflow → auto-flush внутри writer.
- **Обоснование:** crash между parquet-flush и PG-commit оставляет parquet-сегменты полными (PG rows можно reconcile через LangGraph replay; parquet сегменты — нет).

### Atomic UPDATE ... RETURNING (arch.md §4.2)
- **Решение:** `UPDATE runs SET budget_spent_usd = budget_spent_usd + :delta RETURNING budget_spent_usd` — одна операция без read-then-write.
- **Обоснование:** grid-параллелизм (M12) — несколько воркеров могут UPDATE одновременно; non-atomic read-then-write теряет cost-записи.

### pa.timestamp("us", tz="UTC") везде (minor correction vs arch.md §3.5)
- **Решение:** все timestamp-колонки в parquet-схемах с `tz="UTC"`, а не tz-naive.
- **Обоснование:** `core/types._utcnow()` возвращает tz-aware datetime. При `pa.Table.from_pylist` с tz-naive схемой и tz-aware datetime pytest `filterwarnings=["error"]` поймает `UserWarning` как ошибку.

### best-effort exception в callback hooks (arch.md §14.2)
- **Решение:** любое исключение внутри hook'а — catch + log через structlog, не reraise.
- **Исключение:** в `on_chain_end` корневого chain PG UPDATE падает — swallow + log CRITICAL.
- **Обоснование:** упавший callback не должен роняет основной граф LangGraph.

### root-chain detection (non-blocking fix из ревью)
- **Решение:** root chain = `parent_run_id is None AND metadata.get("is_root_run") is True`. Fallback: первый `run_id` с `parent_run_id is None` в `on_chain_start`.
- **Обоснование:** LangGraph subgraph может создавать вложенные цепочки с `parent_run_id is None` для промежуточных узлов; metadata-flag — explicit marker от Runner.

### `__init__.py` дисциплина
- **Решение:** Steps 2.1, 2.2, 2.3, 2.4 не трогают `src/atm/storage/__init__.py` / `src/atm/observability/__init__.py`. Финализация только в Step 5.2.
- **Обоснование:** Wave 2 выполняется параллельно; конкурентная запись в один файл создаёт merge-конфликты.

---

## Ограничения

### Из зависимостей
- `langgraph-checkpoint-postgres` Pin `>=3.0,<4` — API `AsyncPostgresSaver(conn=pool)` подтверждён для 3.0.5 (WebSearch март 2026).
- `prepare_threshold=0` для psycopg pool — обязательно (langgraph#2755); если забыть, integration smoke поймает `InvalidSqlStatementName`.

### Из рисков
- Alembic autogenerate часто не добавляет `::jsonb` суффикс к server_default — обязательный grep-check после генерации.
- `filterwarnings=["error"]` может поймать DeprecationWarning из asyncpg/alembic в integration-тестах — расслаблять per-test через `@pytest.mark.filterwarnings("default")` по факту.

### Из Open Questions плана
- budget thresholds (`budget_warn_threshold`, `budget_exceed_threshold`) передаются в конструктор `ExperimentCallbackHandler`; TODO для M5+ — интеграция с `Experiment.config_snapshot`.
- `human_interactions.request_id` — колонка проиндексирована, но ON CONFLICT — M9.
