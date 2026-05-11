# m8-routing — TopologyRouter (3 режима) + SwitchGuards

## Goal
Реализовать TopologyRouter в трёх режимах (Rule / LLM / Oracle) и набор SwitchGuards с декоратором `GuardedRouter` — это M8.3 + M8.4 из PLAN. После этого блока meta-граф из m8-adaptive сможет в каждом тике принимать решение «какую топологию использовать» с защитой от thrashing.

## Scope
- `src/atm/phases/topology_router.py`:
  - `Protocol TopologyRouter` с методом `decide(state) -> TopologyDecision`
  - `RuleBasedTopologyRouter` — статическая таблица (phase × signals) → topology, копируется из `arch.md §7.7`; детерминированный приоритет правил при коллизии
  - `LLMTopologyRouter` — prompt template + Pydantic-валидация ответа; учёт `router_cost_usd` (пишется в `TopologyDecision`); fallback на Rule при невалидном ответе
  - `OracleTopologyRouter` — читает `oracle_table.json` (путь — параметр конструктора), ключ — `(task_id или task_type, phase)`; **без зависимости от M8.7** (тестируется на mock-таблице)
- `src/atm/phases/guards.py`:
  - `SwitchGuards` — конфиг-объект с параметрами `min_dwell`, `cooldown`, `max_per_run`, `max_per_phase`
  - все 4 guard-функции (чистые: state + proposed_decision → bool blocked)
  - `GuardedRouter(inner: TopologyRouter, guards: SwitchGuards)` — декоратор, который вызывает inner, прогоняет через guards; при блокировке возвращает решение «остаться» с `decided_by='guard_override'`, при этом `considered_alternatives` содержит изначально выбранную альтернативу
- Unit-тесты:
  - Rule: фиксированный state + signals → ожидаемая topology (≥3 кейса на ключевые правила из §7.7)
  - LLM: FakeLLM возвращает валидный JSON → правильный TopologyDecision; невалидный → fallback на Rule
  - Oracle: фикстура `tests/fixtures/oracle_table.json` → стабильный детерминированный выбор
  - SwitchGuards: каждый из 4 guard'ов блокирует в ожидаемом сценарии; `considered_alternatives` сохраняется в финальном decision

## Out-of-scope
- Любые изменения `SharedState` или новых reducer'ов → блок **m8-foundation**
- SignalBus и emission helpers → блок **m8-signals**
- Сборка meta-графа, TransitionGate, dispatch событий → блок **m8-adaptive**
- Реальный oracle pipeline (`build_leave_one_out_oracle`, manual yaml) — M8.7 после E1.

## Inputs / Preconditions
- Блок **m8-foundation** завершён и смержен:
  - Доступны `TopologyDecision`, `PhaseDecision`, `TopologyTransition` из `atm.core.types`
  - `SharedState` уже содержит `signals`, `topology_switch_count`, `topology_started_at_iter`, `phase_started_at_iter`, `iter_total`
- Существуют статические топологии из M7 — но в этом блоке мы только возвращаем их имена (`"linear" | "supervisor" | "mesh" | "debate" | "hierarchical"`), без их инстанцирования.

## Outputs / Contract
- Импорты для m8-adaptive:
  - `from atm.phases.topology_router import TopologyRouter, RuleBasedTopologyRouter, LLMTopologyRouter, OracleTopologyRouter`
  - `from atm.phases.guards import SwitchGuards, GuardedRouter`
- API: `router.decide(state: SharedState) -> TopologyDecision`
- `GuardedRouter` гарантирует: если guard блокирует — `decision.decided_by == 'guard_override'`, `decision.chosen == state.current_topology`, `decision.considered_alternatives` непустой.
- Формат фикстуры `oracle_table.json`:
  ```json
  {
    "by_task_type": {"code_fix": {"plan": "linear", "execute": "mesh", "verify": "debate"}},
    "by_task_id": {}
  }
  ```

## References
- `arch/PLAN.md` строки 481–491 (M8.3 + M8.4 чек-лист)
- `arch/arch.md` §7.7 — каноническая таблица phase × signals → topology (источник правды для RuleBasedTopologyRouter)
- `arch/arch.md` §8 — Adaptive L2 контекст
- `dev/dec/m8-foundation.md` — типы и расширенный SharedState

## Depends On
- `m8-foundation` (типы `TopologyDecision` + расширения `SharedState`)
- **Может выполняться параллельно с `m8-signals`** — файлы не пересекаются (этот блок трогает `phases/topology_router.py` и `phases/guards.py`; m8-signals трогает `phases/signals.py` и `agents/*`).

## Suggested run-task class
**complex** — три имплементации Protocol + декоратор guards, 4 разных guard-правила, валидация LLM-ответа с fallback, ≥10 unit-тестов.

## Notes
- `OracleTopologyRouter` тестируется на mock-таблице — фикстуру положить в `tests/fixtures/oracle_table.json`. Реальные данные подтянутся в M8.7 после E1-pilot — НЕ блокируем M8 на M8.7.
- `router_cost_usd` в LLMTopologyRouter — складывается с usage от LLM-клиента (см. как считается в LLMPhaseRouter из m8-foundation для единообразия).
- Таблицу §7.7 копируем дословно — если в `arch.md` есть неоднозначность, отметить в notes к PR и не молча додумывать.
- Приоритет правил при коллизиях в Rule-роутере: документировать явно в docstring класса.
