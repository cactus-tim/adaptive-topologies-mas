# M9 — Human Gateway + LLM-simulator (HITL infrastructure) — Context

## SESSION PROGRESS (2026-05-12)

### COMPLETED
- Step 2.1 (LLMSimulatedGateway): создан `src/atm/human/llm_simulated.py` — LLM-driven HumanGateway с in-process idempotency cache, retry on invalid JSON, source='llm_sim'. 11 тестов зелёных, mypy clean. Созданы 3 YAML fixtures: approve, reject, invalid_json.
- Step 2.2 (CLIGateway): создан `src/atm/human/cli_gateway.py` — stdin/stdout gateway с asyncio.to_thread, idempotency cache, fallback на first allowed_action при невалидном вводе. 14 тестов зелёных. Примечание: `source` установлен в `'human'` (не `'cli'`), так как HumanResponse.source Literal не включает 'cli'.
- Step 2.5 (Timeout wrapper): создан `src/atm/human/_timeout.py` — `request_with_timeout` с тремя политиками (fail/llm_fallback/skip). `timeout_s=None` → прямой вызов без wait_for. `llm_fallback` без fallback_gateway → ValueError. Fallback всегда overrides source='fallback'. 10 тестов зелёных, mypy clean.
- Step 2.3 (Observability callbacks): расширен `src/atm/observability/callbacks.py` — два elif в `on_custom_event` + два private метода `_handle_human_request` (pg_insert + on_conflict_do_nothing constraint=CONSTRAINT_NAME) и `_handle_human_response` (UPDATE WHERE response_json IS NULL). CONSTRAINT_NAME константа на уровне модуля. 8 тестов зелёных (57/57 всего), mypy clean.
- Step 2.4 (HumanCfg + YAML configs): добавлен `HumanCfg` (frozen Pydantic v2) в `src/atm/experiment/config.py` и поле `human: HumanCfg | None = None` в `ExperimentConfig`. Созданы `conf/human/llm_simulated.yaml`, `conf/human/cli.yaml`. 17 тестов зелёных, mypy clean, существующие тесты (21/21) без изменений.

- Step 3.1 (Runner wiring): `_build_initial_state` теперь добавляет `"run_id": run_id` в `state["shared"]`; `topology_instance.build(...)` получает `human_cfg=cfg.human` как явный kwarg. SharedState total=False — state.py не изменялся. 3 новых теста + 68/68 существующих зелёных, mypy clean.

### IN PROGRESS
- Wave D: Step 4.1 Chain topology — human_reviewer node (depends on 1.1, 2.3, 3.1)

### BLOCKERS
- Нет

---

## Quick Resume

1. Прочитать этот файл целиком
2. Открыть `m9-tasks.md` — найти первую незачёркнутую задачу
3. Прочитать `m9-plan.md` Phase 1 для стратегии Wave A
4. Начать с: **Step 1 (gateway.py + prompts.py)** и **Step 4 (Alembic migration)** параллельно — Wave A, файлы disjoint

---

## Key Files

**`src/atm/core/types.py` (lines 235–261)**
- Роль: содержит `HumanContext`, `HumanResponse`, `HumanRole` — frozen Pydantic v2, уже готовы
- Плановое изменение: НЕ изменять — только реэкспортировать из `human/gateway.py`
- Статус: СТАБИЛЬНО (не трогать)

**`src/atm/human/gateway.py`**
- Роль: `HumanGateway` Protocol (`@runtime_checkable`), `TimeoutPolicy` alias, реэкспорт из core.types
- Плановое изменение: CREATE в Step 1 (Wave A)
- Статус: НЕ НАЧАТО

**`src/atm/human/prompts.py`**
- Роль: `ROLE_SYSTEM_PROMPTS: dict[HumanRole, str]` для 5 ролей + `build_role_prompt(role, ctx) -> tuple[list[Message], dict]`
- Плановое изменение: CREATE в Step 1 (Wave A)
- Статус: НЕ НАЧАТО

**`src/atm/human/llm_simulated.py`**
- Роль: `LLMSimulatedGateway(HumanGateway)` — основной LLM-based gateway с in-process idempotency cache
- Плановое изменение: CREATE в Step 2 (Wave B)
- Статус: ГОТОВО (Step 2.1)

**`src/atm/human/cli_gateway.py`**
- Роль: `CLIGateway(HumanGateway)` — отладочный stdin/stdout gateway, stdlib only
- Плановое изменение: CREATE в Step 2.2 (Wave B)
- Статус: ГОТОВО (14/14 тестов, mypy clean)

**`src/atm/human/_timeout.py`**
- Роль: `request_with_timeout(...)` — pure-async обёртка с 3 политиками (fail/llm_fallback/skip)
- Плановое изменение: CREATE в Step 7 (Wave B)
- Статус: НЕ НАЧАТО

**`src/atm/human/runner.py`**
- Роль: `run_with_human(graph, ...)` — orchestration helper, управляет циклом ainvoke → interrupt → resume
- Плановое изменение: CREATE в Step 10 (Wave E)
- Статус: НЕ НАЧАТО

**`src/atm/human/__init__.py`**
- Роль: публичный API модуля `atm.human` — полный `__all__`
- Плановое изменение: CREATE в Step 1, FINALIZE в Step 11 (Wave F)
- Статус: НЕ НАЧАТО

**`src/atm/observability/callbacks.py` (lines 347–466)**
- Роль: `ExperimentCallbackHandler.on_custom_event` — сейчас обрабатывает message_emit, phase_transition, topology_transition
- Плановое изменение: MODIFY в Step 5 (Wave B) — добавить 2 ветви: `human_request` (INSERT) и `human_response` (UPDATE)
- Статус: НЕ НАЧАТО

**`src/atm/experiment/config.py`**
- Роль: `ExperimentConfig` — верхнеуровневый конфиг эксперимента
- Плановое изменение: MODIFY в Step 6 (Wave B) — добавить `HumanCfg` модель и поле `human: HumanCfg | None = None`
- Статус: ГОТОВО (Step 2.4) — `HumanCfg` добавлен, `ExperimentConfig.human: HumanCfg | None = None`

**`conf/human/llm_simulated.yaml`**
- Роль: YAML конфиг для LLMSimulatedGateway
- Статус: ГОТОВО (Step 2.4)

**`conf/human/cli.yaml`**
- Роль: YAML конфиг для CLIGateway
- Статус: ГОТОВО (Step 2.4)

**`src/atm/experiment/runner.py` (lines 282–309, 542–546)**
- Роль: `Runner._build_initial_state` — формирует initial state с 15 ключами в `shared`
- Плановое изменение: MODIFY в Step 3.1 (Wave C) — (a) добавить `"run_id": run_id` в shared, (b) добавить `human_cfg=cfg.human` в вызов `topology.build(...)`
- Статус: ГОТОВО (Step 3.1) — run_id в shared, human_cfg kwarg пробрасывается

**`src/atm/topology/chain.py` (lines 195–283)**
- Роль: Chain-топология `START → planner → executor → critic → ...`
- Плановое изменение: MODIFY в Step 9 (Wave D) — вставить `human_reviewer` node между executor и critic при `human_cfg.enabled=True`
- Статус: НЕ НАЧАТО

**`alembic/versions/0002_human_interactions_idempotency.py`**
- Роль: Alembic-миграция — UNIQUE CONSTRAINT `uq_human_interactions_run_request` на `(run_id, request_id)`
- Плановое изменение: CREATE в Step 4 (Wave A)
- Статус: НЕ НАЧАТО

**`src/atm/storage/models.py` (lines 266–326)**
- Роль: `HumanInteraction` SQLAlchemy-модель — уже содержит все нужные поля
- Плановое изменение: НЕ изменять — constraint добавляется только через миграцию
- Статус: СТАБИЛЬНО (не трогать)

**`src/atm/core/state.py`**
- Роль: `SharedState` TypedDict — определяет ключи общего состояния
- Плановое изменение: CONDITIONAL MODIFY — добавить `run_id: UUID` если TypedDict total=True
- Статус: СТАБИЛЬНО (total=False — изменение не требуется)

**`dev/codebase-map.md`**
- Роль: карта кодовой базы проекта
- Плановое изменение: MODIFY в Step 11 (Wave F) — добавить секцию M9
- Статус: НЕ НАЧАТО

---

## Decisions

### Callback — единственный writer для human_interactions (arch.md §10.2)
- Decision: узел `human_reviewer` НЕ пишет в БД напрямую; он эмитит два `adispatch_custom_event` (`human_request` перед interrupt, `human_response` после resume). `ExperimentCallbackHandler` делает INSERT/UPDATE.
- Rationale: invariant §10.2 — весь I/O через callback-слой; упрощает тестирование узла (не нужна реальная БД); идемпотентность на PG-уровне.

### Реэкспорт моделей, а не дублирование
- Decision: `HumanContext`, `HumanResponse`, `HumanRole` — только в `core/types.py`; `human/gateway.py` только реэкспортирует.
- Rationale: модели уже определены (lines 235–261); дублирование создаст расхождение типов в mypy.

### Полный UNIQUE CONSTRAINT, не partial index
- Decision: `op.create_unique_constraint("uq_human_interactions_run_request", "human_interactions", ["run_id", "request_id"])` без WHERE.
- Rationale: PostgreSQL не поддерживает `WHERE` в `ADD CONSTRAINT UNIQUE`; NULL-значения в `request_id` PostgreSQL считает distinct по умолчанию → записи без request_id не конфликтуют между собой → constraint безопасен.

### HumanCfg через kwarg, не через cfg.extra
- Decision: Runner передаёт `human_cfg=cfg.human` как явный kwarg в `topology.build(...)`; Chain читает через `kwargs.get("human_cfg")`.
- Rationale: явный механизм, тот же паттерн что `checkpointer`; `cfg.extra` — не типизирован, нет mypy-проверки.

### LLMSimulatedGateway — in-process idempotency cache
- Decision: `dict[tuple[UUID, str], HumanResponse]` + `asyncio.Lock` — проверка ДО LLM-вызова, сохранение ПОСЛЕ.
- Rationale: при resume node вызывается повторно → без кэша было бы два LLM-запроса. Кэш + PG ON CONFLICT = двухуровневая idempotency.

### Single interrupt per node (arch.md §9.3)
- Decision: один `interrupt()` вызов на один human_node; при >1 interrupt в result — `RuntimeError`.
- Rationale: обход langgraph#6663; упрощает resume-логику; достаточно для всех 5 ролей.

### CLIGateway: asyncio.to_thread(input) не прерывается при cancel
- Decision: документируем ограничение; в CI integration-тестах CLIGateway не используется (только LLMSimulatedGateway).
- Rationale: системный thread продолжает ждать ввода даже после CancelledError на asyncio-задаче — это ограничение Python stdlib, не обходится без OS-level API.

### Dispatch порядок в human_reviewer node (критично)
- Decision: `dispatch("human_request")` → `interrupt()` → (resume) → `dispatch("human_response")`.
- Rationale: dispatch ПЕРЕД interrupt гарантирует, что INSERT в human_interactions запишется до checkpoint; иначе при crash между interrupt и callback будет ghost row отсутствовать.

---

## Constraints

- LangGraph ≥0.3 — обязательно для `interrupt()`, `Command(resume=...)`, `adispatch_custom_event`.
- Postgres ≥16 — JSONB, NULL-distinct UNIQUE constraint.
- Python 3.11+ — PEP 604, StrEnum, `asyncio.to_thread`.
- `ExperimentCallbackHandler` — единственный writer для `human_interactions` (arch.md §10.2 invariant).
- Один `interrupt()` per node (arch.md §9.3).
- В M9 трогаем только Chain-топологию; Star/Mesh/Debate/Hierarchical/Adaptive — не затрагиваются.
- `HumanInteraction.id` — UUID, без DB default; явно генерить `uuid.uuid4()` в INSERT.
- Узел передаёт нативные `UUID` и `datetime` в event payload (не json-dumped строки); pg-driver сериализует сам. JSONB-поля — dict-ы.

---

## Open Questions (для developer agent)

1. **`SharedState` TypedDict total=True/False?** — проверить grep'ом `src/atm/core/state.py` перед Step 8; если total=True — добавить `run_id: UUID` в definition.
2. **Точный import path для `adispatch_custom_event`** — верифицировать: `uv run python -c "from langchain_core.callbacks import adispatch_custom_event"` перед Step 9.
3. **Точный ключ interrupt-результата** — `__interrupt__` или `interrupts` — проверить smoke-test'ом до Step 10.
