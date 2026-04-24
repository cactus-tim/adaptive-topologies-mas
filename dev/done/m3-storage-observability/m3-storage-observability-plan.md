# M3 — Storage & Observability — План

## Краткое резюме

Реализуем слой персистентности эксперимента: SQLAlchemy-модели 6 бизнес-таблиц (arch.md §3.4), начальную Alembic-миграцию, async session factory, буферизованный PyArrow `ParquetWriter` (arch.md §11.2), LangGraph `ExperimentCallbackHandler` (arch.md §10.1–§10.3) и обёртку над `langgraph-checkpoint-postgres` `AsyncPostgresSaver` с отдельным psycopg-pool (arch.md §11.3, §17/#3).

Exit criteria: один fake-run создаёт корректные строки в PG + parquet-файлы читаются `pandas`-ом.

**Класс задачи**: complex
**Итераций ревью**: 2 (все 6 блокеров закрыты, план одобрен)

---

## Текущее состояние

- `docker-compose.yml` с Postgres 16 + healthcheck — уже есть.
- `alembic/env.py` — async-скелет, `target_metadata = None` с TODO "M3: set to Base.metadata".
- `alembic/versions/` — пуст.
- `src/atm/core/types.py` — все Pydantic-модели (`Message`, `LLMResponse`, `HumanContext`, `BudgetEvent`, `PhaseTransition`, `TopologyTransition`). Все timestamp через `_utcnow()` → tz-aware UTC.
- `src/atm/storage/__init__.py`, `src/atm/observability/__init__.py` — пустые skeleton'ы.
- `pyproject.toml`: `sqlalchemy[asyncio]` и `asyncpg` — в `dependency-groups.dev` (BLOCKER, надо в runtime).

---

## Подход к реализации

### Архитектурные принципы (arch.md §2.2)
- `storage/` зависит только от `core/`.
- `observability/` зависит от `core/` и `storage/`.
- Слои выше не импортируют `observability/` напрямую — подключение через LangChain `config={"callbacks":[...]}`.

### Two-pools pattern (arch.md §11.3)
- Бизнес-БД — SQLAlchemy async engine (`asyncpg`).
- Checkpointer — `psycopg.AsyncConnectionPool` с `autocommit=True, row_factory=dict_row, prepare_threshold=0`.
- Два изолированных pool'а на процесс.

### 4 flush-инварианта callback-handler'а (arch.md §10.3)
1. Root `on_chain_end` → `parquet_writer.close()`.
2. `phase_transition` / `topology_transition` → `parquet_writer.flush()` **строго ДО** PG INSERT.
3. Root `on_chain_error` → `parquet_writer.close()`.
4. Buffer overflow (`len(buffer) >= buffer_rows`) → auto-flush внутри `ParquetWriter.write_*`.

### Race-safety (arch.md §4.2)
- Инкремент `runs.budget_spent_usd` и `experiments.total_cost_usd` — строго через `UPDATE ... SET col = col + :delta ... RETURNING col` (не read-then-write).

### PyArrow tz-aware (minor correction vs arch.md §3.5)
- Все timestamp-колонки в parquet-схемах используют `pa.timestamp("us", tz="UTC")` вместо tz-naive `pa.timestamp("us")`. Требование `filterwarnings=["error"]` в pytest.

### Дисциплина `__init__.py`
- Steps 2, 3, 6, 7 **не трогают** `src/atm/storage/__init__.py` и `src/atm/observability/__init__.py`.
- Финализация публичного API — только в Step 11.

---

## Фазы реализации

### Phase 1: Инфраструктура (Wave 1 — solo)

**Цель:** выровнять зависимости `pyproject.toml` под runtime M3.

- [ ] 1.1 Обновить `pyproject.toml` — переместить `sqlalchemy[asyncio]`, `asyncpg` из dev в runtime; добавить в runtime `langgraph-checkpoint-postgres>=3.0,<4`, `psycopg[binary,pool]>=3.2`, `structlog>=24`; добавить в dev `pandas>=2.2`.
  - Файл: `pyproject.toml`
  - Acceptance: `uv sync --dev` чистый; `uv run python -c "import sqlalchemy; import asyncpg; import langgraph.checkpoint.postgres.aio; import psycopg; import structlog"` — OK.

---

### Phase 2: Ядро хранилища (Wave 2 — параллельно после Phase 1)

**Цель:** реализовать 4 независимых модуля storage слоя без затрагивания `__init__.py`.

- [ ] 2.1 Создать `src/atm/storage/models.py` — 6 SQLAlchemy-моделей + `FinishReason` StrEnum (TDD).
  - Файл: `src/atm/storage/models.py`, `tests/unit/storage/test_models.py`, `tests/unit/storage/__init__.py`
  - Acceptance: `Base.metadata.sorted_tables` → 6 таблиц; все reproducibility fields (`models_by_role_json`, `model_version_snapshot`, `sandbox_image_digest`) присутствуют; unit-тесты зелёные.

- [ ] 2.2 Создать `src/atm/storage/session.py` — `create_engine`, `create_session_factory`, `session_scope`.
  - Файл: `src/atm/storage/session.py`, `tests/unit/storage/test_session.py`
  - Acceptance: unit-тест сигнатур без реального connect — зелёный; mypy strict чистый.

- [ ] 2.3 Создать `src/atm/storage/schemas.py` + `src/atm/storage/parquet_writer.py` — 6 pa.schema с `tz="UTC"` + буферизованный `ParquetWriter` с 4 flush-инвариантами (TDD).
  - Файл: `src/atm/storage/schemas.py`, `src/atm/storage/parquet_writer.py`, `tests/unit/storage/test_schemas.py`, `tests/unit/storage/test_parquet_writer.py`
  - Acceptance: все timestamps в схемах `pa.timestamp("us", tz="UTC")`; auto-flush по buffer_rows; close() идемпотентен; tz-aware round-trip — зелёный.

- [ ] 2.4 Создать `src/atm/storage/checkpointer.py` — `checkpointer_scope` (primary CM) + `build_checkpointer` (low-level).
  - Файл: `src/atm/storage/checkpointer.py`, `tests/unit/storage/test_checkpointer.py`
  - Acceptance: mock-based lifecycle-тест (`open → setup → yield → close` ordering) — зелёный; `_to_psycopg_dsn` unit-тест; mypy strict чистый.

---

### Phase 3: Alembic (Wave 3–4, последовательно после 2.1)

**Цель:** настроить Alembic и сгенерировать начальную миграцию.

- [ ] 3.1 Обновить `alembic/env.py` — `target_metadata = Base.metadata` (Simple).
  - Файл: `alembic/env.py`
  - Acceptance: `uv run alembic check` не падает; autogenerate видит 6 таблиц.

- [ ] 3.2 Сгенерировать и curate `alembic/versions/0001_initial_business_schema.py` (Simple).
  - Файл: `alembic/versions/0001_initial_business_schema.py`
  - Acceptance: `alembic upgrade head` на чистой БД → 6 таблиц; все JSONB server_default с `::jsonb` cast; `alembic downgrade base` → clean rollback.

---

### Phase 4: Observability (Wave 5–6, после Phase 2)

**Цель:** реализовать сериализаторы и callback-handler.

- [ ] 4.1 Создать `src/atm/observability/serializers.py` — 6 чистых функций `pydantic → dict-row` (Simple).
  - Файл: `src/atm/observability/serializers.py`, `tests/unit/observability/test_serializers.py`, `tests/unit/observability/__init__.py`
  - Acceptance: `pa.Table.from_pylist([row], schema=SCHEMA)` без ошибок для каждого сериализатора; `set(row.keys()) == set(SCHEMA.names)` — OK.

- [ ] 4.2 Создать `src/atm/observability/callbacks.py` — `ExperimentCallbackHandler` с 4 flush-инвариантами и атомарным `UPDATE ... RETURNING` (TDD).
  - Файл: `src/atm/observability/callbacks.py`, `tests/unit/observability/test_callbacks.py`
  - Acceptance: ordering-тест `flush_idx < insert_idx` для `phase_transition` — зелёный; atomic UPDATE ... RETURNING + budget warn/exceed branching — зелёный; 7+ unit-тестов.

---

### Phase 5: Integration + финализация (Wave 7–8)

**Цель:** проверить exit criteria M3, зафиксировать публичный API.

- [ ] 5.1 Написать integration smoke-тест `tests/integration/storage/test_smoke_run.py` (TDD).
  - Файл: `tests/integration/storage/test_smoke_run.py`, `tests/integration/storage/conftest.py`, `tests/integration/storage/__init__.py`, `tests/integration/__init__.py`
  - Acceptance: `ATM_INTEGRATION_PG=1 uv run pytest tests/integration/storage/ -v` зелёный; PG rows в `experiments`, `runs`, `phases`, `budget_events`; parquet-файлы читаются `pandas.read_parquet`; checkpoint-записи есть; alembic-equivalence тест проходит.

- [ ] 5.2 Финализировать `src/atm/storage/__init__.py` и `src/atm/observability/__init__.py`, обновить `dev/codebase-map.md` (Simple).
  - Файл: `src/atm/storage/__init__.py`, `src/atm/observability/__init__.py`, `dev/codebase-map.md`
  - Acceptance: `from atm.storage import ParquetWriter, build_checkpointer, checkpointer_scope, Base, Experiment, Run, FinishReason` — OK; `from atm.observability import ExperimentCallbackHandler` — OK; `uv run mypy src/atm` чистый; `uv run pytest` — весь suite зелёный.

---

## Ключевые файлы

| Файл | Изменение | Причина |
|------|-----------|---------|
| `pyproject.toml` | modify | Переместить SQLAlchemy/asyncpg в runtime; добавить новые deps |
| `src/atm/storage/models.py` | new | 6 SQLAlchemy-моделей + FinishReason |
| `src/atm/storage/session.py` | new | Async session factory |
| `src/atm/storage/schemas.py` | new | PyArrow pa.schema constants (6 схем, все timestamps tz="UTC") |
| `src/atm/storage/parquet_writer.py` | new | Буферизованный async ParquetWriter с 4 flush-инвариантами |
| `src/atm/storage/checkpointer.py` | new | checkpointer_scope (CM) + build_checkpointer (low-level) |
| `src/atm/storage/__init__.py` | overwrite | Публичные re-exports (только Step 11) |
| `src/atm/observability/serializers.py` | new | 6 чистых pydantic→dict-row функций |
| `src/atm/observability/callbacks.py` | new | ExperimentCallbackHandler + 4 flush-инварианта |
| `src/atm/observability/__init__.py` | overwrite | Публичные re-exports (только Step 11) |
| `alembic/env.py` | modify | target_metadata = Base.metadata |
| `alembic/versions/0001_initial_business_schema.py` | new | Начальная миграция 6 таблиц |
| `tests/unit/storage/test_models.py` | new | Unit-тесты моделей без БД |
| `tests/unit/storage/test_session.py` | new | Unit-тесты session factory |
| `tests/unit/storage/test_schemas.py` | new | Unit-тесты pa.schema constants |
| `tests/unit/storage/test_parquet_writer.py` | new | Unit-тесты ParquetWriter |
| `tests/unit/storage/test_checkpointer.py` | new | Unit-тесты + mock lifecycle |
| `tests/unit/observability/test_serializers.py` | new | Unit-тесты сериализаторов |
| `tests/unit/observability/test_callbacks.py` | new | Unit-тесты callback handler |
| `tests/integration/storage/test_smoke_run.py` | new | Integration smoke (exit criteria M3) |
| `tests/integration/storage/conftest.py` | new | pg_engine_fast + pg_engine_alembic fixtures |
| `dev/codebase-map.md` | modify | M3 статус "complete" + TODO reproducibility fields |

---

## Зависимости и порядок выполнения

```
Wave 1: [1.1]
Wave 2: [2.1, 2.2, 2.3, 2.4]  — параллельно, все ждут Wave 1
Wave 3: [3.1]                  — ждёт 2.1
Wave 4: [3.2]                  — ждёт 2.1 + 3.1
Wave 5: [4.1]                  — ждёт 2.3 (schemas.py)
Wave 6: [4.2]                  — ждёт 2.1 + 2.2 + 2.3 + 4.1
Wave 7: [5.1]                  — ждёт 3.2 + 2.3 + 2.4 + 4.1 + 4.2
Wave 8: [5.2]                  — ждёт всё
```

**Критическое ограничение:** Steps 2.1, 2.2, 2.3, 2.4 **не трогают** `src/atm/storage/__init__.py` и `src/atm/observability/__init__.py`. Файлы финализируются только в Step 5.2 (Wave 8).

---

## Риски

| Риск | Вероятность | Impact | Митигация |
|------|-------------|--------|-----------|
| Конфликт версий `langgraph-checkpoint-postgres` с `langgraph-core` | Средняя | Средний | Pin `<4`; если solver падает — добавить `langgraph>=0.2` в runtime |
| autogenerate Alembic упускает JSONB `::jsonb` cast | Высокая | Высокий | grep-check + manual diff после autogenerate |
| `filterwarnings=["error"]` ловит DeprecationWarning asyncpg/alembic в integration | Средняя | Низкий | Расслабить через `@pytest.mark.filterwarnings("default")` per-test если нужно |
| `lazy="raise"` на relationships неожиданно ломает тест | Низкая | Низкий | Принудительная дисциплина; async explicit load |
| Race на `budget_spent_usd` при grid-параллелизме (M12) | Применимо в M12 | Высокий | Atomic `UPDATE ... RETURNING` pattern — реализован в M3 |
| LangGraph subgraph — первый `parent_run_id is None` — не global root | Средняя | Средний | metadata-flag `is_root_run=True` + fallback |
| `prepare_threshold=0` забыт для psycopg pool | Средняя | Высокий | Зафиксировано в `build_checkpointer`; integration smoke Step 5.1 поймает ошибку |

---

## Вне скоупа

- Checkpoint-таблицы в Alembic-миграции (управляются `await AsyncPostgresSaver.setup()`, arch.md §11.4).
- Заполнение `models_by_role_json`, `model_version_snapshot`, `sandbox_image_digest` — это задача Runner/LLMWrapper/DockerSandbox (M5+).
- `ON CONFLICT (run_id, request_id) DO NOTHING` для `human_interactions.request_id` — M9 доделает.
- Grid-параллелизм (M12) — M3 только закладывает атомарные UPDATE-паттерны.
- HTTP/gRPC/pub-sub контракты — M3 не вводит внешних API.

---

## Timeline

- Итого: ~14–18 ч (8 волн, max parallelism 4)
- Создан: 2026-04-24
