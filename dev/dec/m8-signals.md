# m8-signals — SignalBus + emission helpers + обновления агентов

## Goal
Реализовать M8.5: конвенции ключей сигналов, helper `emit_signal`, и обновить трёх агентов (Critic, Executor, Planner) так, чтобы они эмитили сигналы, на которые опирается TopologyRouter (`stuck`, `rejected_count`, `critic_approved`, `ready_for_execution`, `ready_for_verification`).

## Scope
- `src/atm/phases/signals.py`:
  - Конвенции ключей как константы (Enum или `Literal` строки): `STUCK`, `REJECTED_COUNT`, `NEEDS_DEBATE`, `READY_FOR_EXECUTION`, `READY_FOR_VERIFICATION`, `CRITIC_APPROVED`
  - `emit_signal(state: SharedState, key: str, value: Any) -> SharedState` — чистая функция: возвращает обновлённый state, вызывает `dispatch_custom_event("signal_emit", {"key": key, "value": value, "at_iter": state.iter_total})`
  - Семантика инкремента: для счётчиков (`rejected_count`) — отдельный helper `increment_signal(state, key)` или соглашение, что `emit_signal` с числом просто пишет значение, а инкремент делает caller
- `src/atm/agents/critic.py`:
  - При reject → инкремент `signals['rejected_count']`
  - При approve → `signals['critic_approved'] = True`
- `src/atm/agents/executor.py`:
  - После N неудачных `code_run` подряд (N — параметр конфига, дефолт 3) → `signals['stuck'] = True`
  - После первого успешного `code_run` → `signals['ready_for_verification'] = True`
- `src/atm/agents/planner.py`:
  - При финализации плана → `signals['ready_for_execution'] = True`
- Unit-тесты для каждого агента:
  - Critic: подать стабом ответ "reject" → `signals['rejected_count']` инкрементировался; "approve" → `critic_approved == True`
  - Executor: сценарий с 3 failed run'ами → `stuck == True`; первый успешный → `ready_for_verification == True`
  - Planner: финализация → `ready_for_execution == True`
- Unit-тесты `emit_signal`:
  - State обновляется неподлокально (immutable update — соответствует существующей конвенции в проекте)
  - `dispatch_custom_event` вызывается ровно один раз с правильным payload (mock на dispatcher)

## Out-of-scope
- Любые изменения routers/guards → блок **m8-routing**
- Расширение `SharedState` слотом `signals` → блок **m8-foundation** (он уже должен быть готов)
- Сборка meta-графа и реакция TopologyRouter на сигналы → блок **m8-adaptive**

## Inputs / Preconditions
- Блок **m8-foundation** завершён: `SharedState.signals: dict[str, Any]` доступен.
- Существуют агенты Critic / Executor / Planner из M5/M6 — смотри в `src/atm/agents/`.
- Существует механизм `dispatch_custom_event` (LangGraph встроенный или собственный wrapper — посмотреть в codebase-map).

## Outputs / Contract
- Импорт: `from atm.phases.signals import emit_signal, STUCK, REJECTED_COUNT, READY_FOR_EXECUTION, READY_FOR_VERIFICATION, CRITIC_APPROVED, NEEDS_DEBATE`
- Гарантия: после прогона Planner→Executor→Critic в типовом happy-path сценарии в `state.signals` появятся ключи `ready_for_execution`, `ready_for_verification`, `critic_approved` (или `rejected_count` если reject).
- Event stream получает события `signal_emit` с payload `{key, value, at_iter}` — на это позже подпишется аналитика в M11.

## References
- `arch/PLAN.md` строки 493–499 (M8.5 чек-лист)
- `arch/arch.md` §7.7 — какие сигналы куда диспатчатся (`stuck → mesh`, `rejected_count≥3 → debate`)
- `arch/arch.md` §8 — общий контекст Adaptive L2
- `dev/dec/m8-foundation.md` — слот `signals` в `SharedState`
- `src/atm/agents/critic.py`, `src/atm/agents/executor.py`, `src/atm/agents/planner.py` — текущие реализации

## Depends On
- `m8-foundation` (нужен `SharedState.signals`)
- **Может выполняться параллельно с `m8-routing`** — этот блок трогает `phases/signals.py` и `agents/*.py`, не пересекается с `phases/topology_router.py` и `phases/guards.py`.

## Suggested run-task class
**standard** — небольшой новый модуль `signals.py` + правки в 3 существующих файлах агентов + ~5 unit-тестов. Меньше сложности, чем foundation/routing/adaptive.

## Notes
- Конвенция immutable-обновления state: использовать тот же стиль, что и в существующих агентах M5/M6 (вероятно `state.model_copy(update={"signals": {**state.signals, key: value}})` или эквивалент — смотри code-map).
- Параметр N (порог `stuck`) для Executor — вынести в config-параметр агента, дефолт `3`. Не хардкодить.
- `rejected_count` — int, дефолт 0; `stuck`, `critic_approved`, `ready_for_*` — bool, дефолт False. Дефолты обеспечивает m8-foundation через factory `signals` или ленивые `.get(...)`.
- `dispatch_custom_event` — если в проекте уже есть wrapper, использовать его; иначе — LangGraph API напрямую.
