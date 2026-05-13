# M9 — Human Gateway + LLM-simulator (HITL infrastructure) — Tasks

---

## >> CURRENT WAVE: Wave F (COMPLETE — integration tests 6.2–6.4 pending PG) <<

---

## Wave A — Foundation (параллельно) NOT STARTED

- [x] 1.1 Gateway Protocol + per-role system prompts module
  - Type: tdd
  - Files: `src/atm/human/gateway.py`, `src/atm/human/prompts.py`, `src/atm/human/__init__.py` (initial), `tests/unit/human/__init__.py`, `tests/unit/human/test_gateway_protocol.py`, `tests/unit/human/test_prompts.py`
  - Depends On: ничего
  - Can-Parallel-With: 1.2
  - Acceptance: `from atm.human import HumanGateway, HumanContext, HumanResponse, HumanRole, TimeoutPolicy` работает; `isinstance(stub, HumanGateway)` корректен; все 5 ролей дают непустой prompt; `allowed_actions` упоминается в system-prompt
  - Verification: `uv run pytest tests/unit/human/test_gateway_protocol.py tests/unit/human/test_prompts.py -q && uv run mypy src/atm/human/gateway.py src/atm/human/prompts.py`
  - Notes: НЕ дублировать `HumanContext`/`HumanResponse`/`HumanRole` — только реэкспорт из `atm.core.types`. Protocol сигнатура: `async def request(self, ctx: HumanContext, *, request_id: str) -> HumanResponse`

- [x] 1.2 Alembic migration — UNIQUE CONSTRAINT `(run_id, request_id)`
  - Type: simple
  - Files: `alembic/versions/0002_human_interactions_idempotency.py`
  - Depends On: ничего
  - Can-Parallel-With: 1.1
  - Acceptance: `alembic upgrade head` создаёт constraint `uq_human_interactions_run_request`; `INSERT ... ON CONFLICT ON CONSTRAINT uq_human_interactions_run_request DO NOTHING` работает; round-trip downgrade/upgrade идемпотентен
  - Verification: `alembic upgrade head && psql -c "\d human_interactions"` → видно constraint; `alembic downgrade -1 && alembic upgrade head` — без ошибок
  - Notes: СНАЧАЛА прочитать `alembic/versions/0001_*.py` для down_revision. Использовать `op.create_unique_constraint(...)` (не `op.create_index`). Downgrade: `op.drop_constraint("uq_human_interactions_run_request", "human_interactions", type_="unique")`

---

## Wave B — Core implementations (параллельно) NOT STARTED

- [x] 2.1 LLMSimulatedGateway
  - Type: tdd
  - Files: `src/atm/human/llm_simulated.py`, `tests/unit/human/test_llm_simulated_gateway.py`, `tests/fixtures/llm/m9_human_reviewer_approve.yaml`, `tests/fixtures/llm/m9_human_reviewer_reject.yaml`, `tests/fixtures/llm/m9_human_invalid_json.yaml`
  - Depends On: 1.1
  - Can-Parallel-With: 2.2, 2.3, 2.4, 2.5
  - Acceptance: `isinstance(LLMSimulatedGateway(fake_wrapper), HumanGateway) is True`; 6+ тестов зелёных: approve/reject happy path, invalid JSON → fallback, action не в allowed → fallback, idempotency (2 вызова → 1 LLM-call), cache-key проверка
  - Verification: `uv run pytest tests/unit/human/test_llm_simulated_gateway.py -q`
  - Notes: `messages, _meta = build_role_prompt(ctx.role, ctx)` → `await self._llm.ainvoke(messages)`. Кэш `_cache: dict[tuple[UUID, str], HumanResponse]` + `_lock: asyncio.Lock`. Regex для снятия ```code fence```. `source="llm_sim"` всегда

- [x] 2.2 CLIGateway
  - Type: tdd
  - Files: `src/atm/human/cli_gateway.py`, `tests/unit/human/test_cli_gateway.py`
  - Depends On: 1.1
  - Can-Parallel-With: 2.1, 2.3, 2.4, 2.5
  - Acceptance: `isinstance(CLIGateway(), HumanGateway) is True`; 3+ тестов: happy path, invalid action → fallback, idempotency; без новых зависимостей
  - Verification: `uv run pytest tests/unit/human/test_cli_gateway.py -q`
  - Notes: `asyncio.to_thread(input, prompt_str)` — обязательно. Stdlib only (no rich). В тестах патчить `asyncio.to_thread`, не `builtins.input`. Неизвестный action → `allowed_actions[0]` + `comment=f"invalid_action_{input}"`

- [x] 2.3 Observability callback — handlers `human_request` / `human_response`
  - Type: tdd
  - Files: `src/atm/observability/callbacks.py` (modify: +2 elif + 2 private methods), `tests/unit/observability/test_callbacks_human.py` (create)
  - Depends On: 1.2
  - Can-Parallel-With: 2.1, 2.2, 2.4, 2.5
  - Acceptance: callback обрабатывает 2 новых event; INSERT idempotent (ON CONFLICT); UPDATE через `WHERE response_json IS NULL`; 6+ тестов зелёных
  - Verification: `uv run pytest tests/unit/observability/ -q && uv run mypy src/atm/observability/callbacks.py`
  - Notes: вынести `CONSTRAINT_NAME = "uq_human_interactions_run_request"` в module-level константу. `_handle_human_request`: `pg_insert(...).on_conflict_do_nothing(constraint=CONSTRAINT_NAME)`. `_handle_human_response`: `UPDATE ... WHERE response_json IS NULL`. `uuid.uuid4()` явно в INSERT. Узел передаёт нативные UUID/datetime — не json-строки

- [x] 2.4 HumanCfg + YAML configs
  - Type: simple
  - Files: `src/atm/experiment/config.py` (modify), `conf/human/llm_simulated.yaml`, `conf/human/cli.yaml`, `tests/unit/experiment/test_config_human.py`
  - Depends On: 1.1
  - Can-Parallel-With: 2.1, 2.2, 2.3, 2.5
  - Acceptance: `load_config("conf/experiments/smoke.yaml").human is None` (back-compat); HumanCfg из YAML инстанцируется корректно; все поля с дефолтами работают
  - Verification: `uv run pytest tests/unit/experiment/test_config_human.py -q`
  - Notes: `HumanCfg` поля: `enabled: bool = False`, `gateway: Literal["llm_simulated","cli"] = "llm_simulated"`, `role: HumanRole = HumanRole.REVIEWER`, `timeout_s: int | None = 900`, `timeout_policy: Literal["fail","llm_fallback","skip"] = "llm_fallback"`, `llm_sim_model_id: str | None = None`. `ExperimentConfig.human: HumanCfg | None = None` — default=None обязателен

- [x] 2.5 Timeout wrapper + fallback policy
  - Type: tdd
  - Files: `src/atm/human/_timeout.py`, `tests/unit/human/test_timeout_policy.py`
  - Depends On: 1.1
  - Can-Parallel-With: 2.1, 2.2, 2.3, 2.4
  - Acceptance: 3 политики обработаны; source-метки корректны (`timeout`/`fallback`/`timeout`); 5+ тестов: no timeout, timeout+fail, timeout+llm_fallback (source="fallback"), timeout+skip, missing fallback → ValueError
  - Verification: `uv run pytest tests/unit/human/test_timeout_policy.py -q`
  - Notes: сигнатура: `async def request_with_timeout(gateway, ctx, *, request_id, timeout_s, policy, llm_fallback_gateway=None) -> HumanResponse`. `timeout_s is None` → direct await. `asyncio.wait_for(..., timeout=timeout_s)` → TimeoutError → ветвление по policy. `policy="llm_fallback"` + `llm_fallback_gateway is None` → ValueError

---

## Wave C — Runner wiring (sequential) COMPLETE

- [x] 3.1 Runner — `run_id` в shared + `human_cfg` kwarg в topology.build
  - Type: tdd
  - Files: `src/atm/experiment/runner.py` (modify: 2 точки), `tests/unit/experiment/test_runner_initial_state.py` (create or modify), `src/atm/core/state.py` (conditional modify — только если SharedState TypedDict total=True)
  - Depends On: 2.4 (нужен HumanCfg тип)
  - Can-Parallel-With: ничего (Wave C — одиночная)
  - Acceptance: `state["shared"]["run_id"]` присутствует; `Chain.build` получает `human_cfg=cfg.human` kwarg; 3 unit-теста зелёных
  - Verification: `uv run pytest tests/unit/experiment/test_runner_initial_state.py -q`
  - Notes: (a) в `_build_initial_state` добавить `"run_id": run_id` в `initial_state["shared"]`. (b) grep `grep -n "\.build(" src/atm/experiment/runner.py` — найти вызов topology.build; добавить `human_cfg=cfg.human`. ПЕРЕД началом — проверить `grep -n "SharedState" src/atm/core/state.py` на total=True/False

---

## Wave D — Chain topology (sequential) COMPLETE

- [x] 4.1 Chain topology — `human_reviewer` node с interrupt() и dispatch
  - Type: simple
  - Files: `src/atm/topology/chain.py` (modify), `tests/unit/topology/test_chain_human.py` (create)
  - Depends On: 1.1, 2.3, 3.1
  - Can-Parallel-With: ничего (Wave D — одиночная)
  - Acceptance: без human_cfg — граф как раньше; с enabled=True — node вставлен, порядок dispatch→interrupt→dispatch корректен; 7 тестов зелёных
  - Verification: `uv run pytest tests/unit/topology/ -q && uv run mypy src/atm/topology/chain.py`
  - Notes: `human_cfg = kwargs.get("human_cfg")`. Conditional insertion: если enabled → add_node("human_reviewer"), edge executor→human_reviewer→critic; иначе прямой edge executor→critic. ПЕРЕД началом верифицировать: `uv run python -c "from langchain_core.callbacks import adispatch_custom_event"`. Node: (1) читает run_id из shared или raise RuntimeError, (2) строит HumanContext, (3) вычисляет request_id=`f"chain:reviewer:{iteration}"`, (4) dispatch(human_request), (5) interrupt(payload), (6) HumanResponse.model_validate(resp), (7) dispatch(human_response), (8) return state-delta. Dispatch ПЕРЕД interrupt — критично

---

## Wave E — Orchestration helper (sequential) NOT STARTED

- [x] 5.1 run_with_human — resume loop orchestrator
  - Type: tdd
  - Files: `src/atm/human/runner.py` (create), `tests/unit/human/test_runner_resume_loop.py` (create)
  - Depends On: 2.5, 4.1
  - Can-Parallel-With: ничего (Wave E — одиночная)
  - Acceptance: single/multi interrupt scenarios; timeout → fallback (source="fallback"); max_interactions guard; >1 interrupt в result → RuntimeError; 6+ тестов зелёных
  - Verification: `uv run pytest tests/unit/human/test_runner_resume_loop.py -q`
  - Notes: `from langgraph.types import Command`. Interrupt-detection: `isinstance(result, dict) and result.get("__interrupt__")`. ПЕРЕД началом проверить фактический ключ smoke-test'ом. Interrupt payload формат (контракт от Step 4.1): `{"ctx": <model_dump>, "request_id": <str>}`. Восстановление: `HumanContext.model_validate(payload["ctx"])`. Resume: `graph.ainvoke(Command(resume=response.model_dump(mode="json")), config={"configurable": {"thread_id": str(run_id)}})`. `max_interactions=10` default

---

## Wave F — Finalization (sequential) COMPLETE

- [x] 6.1 Module wiring + public exports + codebase-map update
  - Type: simple
  - Files: `src/atm/human/__init__.py` (finalize with full `__all__`), `dev/codebase-map.md` (modify: секция M9)
  - Depends On: 2.1, 2.2, 2.5, 4.1, 5.1
  - Can-Parallel-With: ничего
  - Acceptance: `from atm.human import HumanGateway, LLMSimulatedGateway, CLIGateway, run_with_human` работает; full suite green
  - Verification: `uv run python -c "from atm.human import HumanGateway, LLMSimulatedGateway, CLIGateway, run_with_human" && uv run pytest tests/ -q`
  - Notes: явный `__all__: list[str]`; НЕ экспортировать `persist_human_*` функций (их нет). В codebase-map: новые файлы + `on_custom_event` обрабатывает `human_request`/`human_response`

- [x] 6.2 Integration тест — Reviewer end-to-end (PG required)
  - Type: tdd
  - Files: `tests/integration/human/test_hitl_chain_e2e.py`
  - Depends On: 6.1
  - Can-Parallel-With: 6.3, 6.4
  - Acceptance: Chain с `human.enabled=true, gateway=llm_simulated, role=reviewer` → `SELECT COUNT(*) = 1 AND response_json->>'source' = 'llm_sim' FROM human_interactions`

- [x] 6.3 Integration тест — Timeout + fallback (PG required)
  - Type: tdd
  - Files: `tests/integration/human/test_hitl_timeout.py`
  - Depends On: 6.1
  - Can-Parallel-With: 6.2, 6.4
  - Acceptance: mock CLIGateway sleeps 5s, `timeout_s=1`, `policy=llm_fallback` → run завершается, `response_json->>'source' = 'fallback'`

- [x] 6.4 Integration тест — Idempotency на resume (PG required)
  - Type: tdd
  - Files: `tests/integration/human/test_hitl_idempotency.py`
  - Depends On: 6.1
  - Can-Parallel-With: 6.2, 6.3
  - Acceptance: simulated node re-execution → `SELECT COUNT(*) = 1 FROM human_interactions WHERE run_id=...`

---

## Stats

- Всего задач: 15
- Unit-тестовых шагов (tdd): 8 (1.1, 2.1, 2.2, 2.3, 2.5, 3.1, 5.1, integration)
- Простых шагов (simple): 4 (1.2, 2.4, 4.1, 6.1)
- Integration-тестов: 3 (6.2, 6.3, 6.4)
- Готово: 0 / 15
- Расчётное время: ~11h

---

## Как обновлять

После выполнения задачи:
1. Поставить `[x]` вместо `[ ]`
2. Обновить заголовок волны: `NOT STARTED` → `IN PROGRESS` / `COMPLETE`
3. Обновить `>> CURRENT WAVE <<` маркер
4. Обновить `m9-context.md` раздел SESSION PROGRESS (COMPLETED / IN PROGRESS / BLOCKERS)
5. Обновить статус файлов в разделе Key Files (context.md)
